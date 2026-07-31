#!/usr/bin/env bash
set -euo pipefail

ASSETS_DIR="${1:?usage: $0 ASSETS_DIR [RELEASE_TAG] [REPOSITORY] [PUBLIC_CODE_REF]}"
RELEASE_TAG="${2:-v1.1.0}"
REPOSITORY="${3:-valeoai/Franca}"
PUBLIC_CODE_REF="${4:-}"

command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
gh auth status >/dev/null

required=(
  franca_vitb14_In21K_518.pth
  franca_vitb14_In21K_518_rasa.pth
  franca_vitb14_Dinov2_In21K_518.pth
  franca_vitb14_Dinov2_In21K_518_rasa.pth
  franca_vitb14_Dinov2_In21K.pth
  franca_vitl14_Dinov2_In21K_518.pth
  franca_vitl14_Dinov2_In21K_518_rasa.pth
  franca_vitl14_Dinov2_In21K.pth
  franca_vitl14_Laion_518.pth
  franca_vitl14_Laion_518_rasa.pth
  CHECKPOINTS.json
  SHA256SUMS
)

assets=()
for name in "${required[@]}"; do
  path="$ASSETS_DIR/$name"
  [[ -f "$path" ]] || { echo "Missing release asset: $path" >&2; exit 1; }
  assets+=("$path")
done

(cd "$ASSETS_DIR" && sha256sum --check SHA256SUMS)

if ! gh release view "$RELEASE_TAG" --repo "$REPOSITORY" >/dev/null 2>&1; then
  if [[ -z "$PUBLIC_CODE_REF" ]]; then
    echo "Release $RELEASE_TAG does not exist. Pass the public branch or commit containing the v1.1.0 loader as argument 4." >&2
    exit 1
  fi
  gh release create "$RELEASE_TAG" --repo "$REPOSITORY" --draft --title "Franca $RELEASE_TAG checkpoints" \
    --target "$PUBLIC_CODE_REF" \
    --notes "Adds Franca and DINOv2 baseline checkpoints trained or continued at 518px. See CHECKPOINTS.json for provenance."
fi

release_info="$(gh release view "$RELEASE_TAG" --repo "$REPOSITORY" --json databaseId,isDraft,targetCommitish)"
release_id="$(printf '%s' "$release_info" | jq --raw-output '.databaseId')"
release_target="$(printf '%s' "$release_info" | jq --raw-output '.targetCommitish')"
is_draft="$(printf '%s' "$release_info" | jq --raw-output '.isDraft')"

if [[ "$is_draft" != "true" ]]; then
  echo "Release $RELEASE_TAG is already published; refusing to modify it" >&2
  exit 1
fi

if [[ -n "$PUBLIC_CODE_REF" && "$release_target" != "$PUBLIC_CODE_REF" ]]; then
  gh release edit "$RELEASE_TAG" --repo "$REPOSITORY" --target "$PUBLIC_CODE_REF"
  release_target="$PUBLIC_CODE_REF"
fi

if ! gh api "repos/$REPOSITORY/contents/franca/hub/backbones.py?ref=$release_target" --jq '.content' \
  | base64 --decode | grep --quiet 'IN21K_518'; then
  echo "Release target $release_target does not contain the v1.1.0 loader changes; publish the code first" >&2
  exit 1
fi

for path in "${assets[@]}"; do
  name="$(basename "$path")"
  local_size="$(stat --format='%s' "$path")"
  local_digest="sha256:$(sha256sum "$path" | cut --delimiter=' ' --fields=1)"
  remote_info="$(
    gh api "repos/$REPOSITORY/releases/$release_id" \
      --jq ".assets[] | select(.name == \"$name\") | [.size, .digest] | @tsv"
  )"
  if [[ -n "$remote_info" ]]; then
    IFS=$'\t' read -r remote_size remote_digest <<< "$remote_info"
    if [[ "$remote_size" != "$local_size" || "$remote_digest" != "$local_digest" ]]; then
      echo "Remote asset $name does not match the local size and SHA-256 digest; refusing to overwrite it" >&2
      exit 1
    fi
    echo "Already uploaded and verified: $name"
    continue
  fi

  gh release upload "$RELEASE_TAG" "$path" --repo "$REPOSITORY"
  remote_info="$(
    gh api "repos/$REPOSITORY/releases/$release_id" \
      --jq ".assets[] | select(.name == \"$name\") | [.size, .digest] | @tsv"
  )"
  IFS=$'\t' read -r remote_size remote_digest <<< "$remote_info"
  if [[ "$remote_size" != "$local_size" || "$remote_digest" != "$local_digest" ]]; then
    echo "Uploaded asset $name failed remote size or SHA-256 verification" >&2
    exit 1
  fi
  echo "Uploaded and verified: $name"
done

echo "All assets are present with the expected sizes and SHA-256 digests in draft release $REPOSITORY@$RELEASE_TAG"
