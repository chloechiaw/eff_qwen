"""Standalone: download Qwen3.5-4B into the adaptfm-weights Modal volume.

This is intentionally separate from app.py so it doesn't drag serve() along
for the ride. Run it once, then weights live in the volume permanently.

    modal run download.py::download

Uses hf_transfer (Rust, multi-threaded) so ~8 GB downloads in 1–2 min on
Modal's network. Without it, snapshot_download is single-threaded and slow.
"""

import modal

WEIGHTS_DIR = "/opt/ml/model"
HF_REPO = "Qwen/Qwen3.5-4B"

weights = modal.Volume.from_name("adaptfm-weights", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("huggingface_hub", "hf_transfer")

app = modal.App("adaptfm-download", image=image)


@app.function(volumes={WEIGHTS_DIR: weights}, timeout=60 * 60)
def download():
    import os

    # 10x faster downloads via Rust-based parallel transfer.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from huggingface_hub import snapshot_download

    if os.path.isdir(WEIGHTS_DIR) and any(
        f.endswith(".safetensors") for f in os.listdir(WEIGHTS_DIR)
    ):
        print(f"already present at {WEIGHTS_DIR}")
        return

    print(f"downloading {HF_REPO} -> {WEIGHTS_DIR}")
    snapshot_download(repo_id=HF_REPO, local_dir=WEIGHTS_DIR)
    weights.commit()

    total = sum(
        os.path.getsize(os.path.join(WEIGHTS_DIR, f))
        for f in os.listdir(WEIGHTS_DIR)
        if os.path.isfile(os.path.join(WEIGHTS_DIR, f))
    )
    print(f"done. wrote {total / 1e9:.1f} GB")
