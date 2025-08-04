# In the following directory, we should store all ViT-B/L checkpoints.
# The names should have the following format:
# franca_vitb14_In21K.pth, i.e., franca_{compact_arch_name}{patch_size}_{pretraining_dataset}.pth
# dataset names are In21K or Laion600M.
# you can check it out in franca/hub/backbones.py
ALL_FINAL_WEIGHTS_DIR="$HOME/iveco/shashank/logfiles/franca_release_weights"

# In the following directory, we should store all ViT-G checkpoints.
# They should have the same following format as described above.
GIGA_WEIGHTS_DIR="$HOME/iveco/shashank/logfiles/giga_models"

_GITHUB_MAX_SIZE="1900MB"

# We split the giga model for both Ink21K and LAION into chunks of 1900MB each.
# This is because GitHub has a limit of 2GB per file.
tar czf - -C $GIGA_WEIGHTS_DIR franca_vitg14_In21K.pth | split -b $_GITHUB_MAX_SIZE - $ALL_FINAL_WEIGHTS_DIR/franca_vitg14_In21K_chunked.tar.gz.part_
tar czf - -C $GIGA_WEIGHTS_DIR franca_vitg14_Laion600M.pth | split -b $_GITHUB_MAX_SIZE - $ALL_FINAL_WEIGHTS_DIR/franca_vitg14_Laion600M_chunked.tar.gz.part_


# Here we use GitHub CLI to upload the files to the Franca release.
# Here are the instructions to install GitHub CLI:
# https://github.com/cli/cli#installation
# Make sure you have authenticated with GitHub CLI before running this script.
# You can authenticate by running:
# gh auth login
# see: https://cli.github.com/manual/gh_auth_login
find $ALL_FINAL_WEIGHTS_DIR -type f -name "*.pth" -o -name "*.tar.gz*" | while read -r filepath; do
    # Copy the file to the destination with the new name
    echo "[ ] Uploading: $filepath"
    gh release upload v1.0.0 $filepath --clobber --repo valeoai/Franca
    echo "[x] Uploaded: $filepath"
done
