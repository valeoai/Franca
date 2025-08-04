import os
import subprocess
import sys
import tempfile
import warnings
from typing import Any, List, Optional
from urllib.parse import urlparse

import torch
from torch.hub import download_url_to_file, get_dir
from torch.serialization import MAP_LOCATION

_TEMPDIR = os.getenv("TMPDIR", "/tmp")


def extract_tar_file(tar_dir: str, outdir: str) -> str:
    with tempfile.TemporaryDirectory(dir=_TEMPDIR) as tmpdirname:
        # Concatenate tar parts
        part_name = os.path.join(tar_dir, "*.tar.gz.part_*")
        tmp_tar = os.path.join(tmpdirname, "tempfile.tar.gz")
        subprocess.run(f"cat {part_name} > {tmp_tar}", shell=True)

        # Extract tar file
        os.makedirs(outdir, exist_ok=True)
        log = subprocess.run(f"tar xzfv {tmp_tar} -C {outdir}", shell=True, capture_output=True, text=True)

    extracted_files = [line.strip() for line in log.stdout.split("\n") if line.strip()]
    return extracted_files[0]


# Overwriting torch.hub.load_state_dict_from_url to handle possible multiple
# tar files created by chunking the model weights into 1900MB parts.
def load_state_dict_from_url(
    url: str | List[str],
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
) -> dict[str, Any]:
    if isinstance(url, str):
        return torch.hub.load_state_dict_from_url(
            url=url,
            model_dir=model_dir,
            map_location=map_location,
            progress=progress,
            check_hash=check_hash,
            file_name=file_name,
            weights_only=weights_only,
        )

    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead",
            stacklevel=2,
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        model_parts_dir = os.path.join(hub_dir, "checkpoints", "parts")
    else:
        model_parts_dir = os.path.join(model_dir, "parts")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_parts_dir, exist_ok=True)

    final_url = url[0].split("_chunked.tar.gz")[0] + ".pth"
    parts = urlparse(final_url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        for _url in url:
            sys.stdout.write(f'Downloading: "{_url}" to {cached_file} parts\n')
            _parts = urlparse(_url)
            _filename = os.path.basename(_parts.path)
            _cached_file = os.path.join(model_parts_dir, _filename)
            download_url_to_file(_url, _cached_file, None, progress=progress)

        extract_tar_file(model_parts_dir, model_dir)

    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)
