# Preparing the 518px checkpoint release

The v1.1.0 release contains four 518px backbone/RASA pairs and republishes the missing legacy DINOv2-B/L 224px
backbones. Backbone assets use the existing `{"teacher": state_dict}` format; new RASA assets are standalone
10-tensor state dictionaries.

## Build and verify

Run from the repository root in an environment containing PyTorch:

```bash
python scripts/prepare_release_checkpoints.py \
  --output-dir /hnvme/workspace/v115be14-ws/checkpoints/franca/release_v1.1.0

python scripts/verify_release_checkpoints.py \
  /hnvme/workspace/v115be14-ws/checkpoints/franca/release_v1.1.0
```

The exporter removes training-only DINO/iBOT heads, separates embedded RASA tensors, validates architecture and
position-embedding shapes, and writes `CHECKPOINTS.json` plus `SHA256SUMS`.

## Publish

An isolated environment containing Python 3.9 and GitHub CLI 2.97.0 is available at
`/hnvme/workspace/v115be14-ws/envs/franca-release`. Activate and authenticate it with:

```bash
source /hnvme/workspace/v115be14-ws/envs/franca-release/bin/activate
gh auth login --hostname github.com --web
gh auth status
```

Publishing requires write access to `valeoai/Franca`. First commit this repository's changes and move or merge that
commit into a public Franca branch. Then run:

```bash
scripts/upload_gh_ckpt.sh \
  /hnvme/workspace/v115be14-ws/checkpoints/franca/release_v1.1.0 \
  v1.1.0 valeoai/Franca PUBLIC_CODE_REF
```

Replace `PUBLIC_CODE_REF` with the public branch or commit containing these loader changes. The script validates every
checksum, creates a draft v1.1.0 release at that exact ref when necessary, and uploads the required assets without
replacing existing ones. Test one clean `torch.hub.load` download from the tag, then publish the draft release.

## Provenance limits

- Franca-B: ImageNet training at 518px initialized from ImageNet-21K weights. Its RASA head is
  the existing standalone B-518 checkpoint.
- DINOv2-B/L: checkpoint names identify them as In21K 518px HRFT baselines. No adjacent training config was retained.
- Franca-L: trained at 518px on the same LAION-600M dataset as the other LAION Franca models.
- Legacy B-In21K and L-LAION RASA assets have eight preprocessing projections; the new assets have nine. The loader
  infers this count, so both formats work.
- The published v1.0.0 L-LAION backbone has a 518px positional grid even though the old README listed 224px. The
  loader now constructs it at 518px; patch-size-compatible inference resolutions still use positional interpolation.
- The retained local `franca_vitl14_In21K.pth` is identical to DINOv2-L and is therefore not published as Franca-L.
- The v1.0.0 G-In21K release is incomplete and no valid local source was found, so that combination is not exposed.
