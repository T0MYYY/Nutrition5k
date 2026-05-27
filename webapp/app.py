from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import web_app


def _download_if_needed(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[ckpt] Reusing existing file: {dst}")
        return
    print(f"[ckpt] Downloading {url} -> {dst}")
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        f.write(response.read())
    print(f"[ckpt] Ready: {dst} ({dst.stat().st_size} bytes)")


def main() -> None:
    rgb_url = os.getenv("RGB_CKPT_URL", "").strip()
    rgbd_url = os.getenv("RGBD_CKPT_URL", "").strip()

    rgb_path = Path("outputs_food101_4passes/checkpoints/best.pt")
    rgbd_path = Path("outputs_train_rgbd_food101/checkpoints/best.pt")

    if rgb_url:
        _download_if_needed(rgb_url, rgb_path)
    if rgbd_url:
        _download_if_needed(rgbd_url, rgbd_path)

    args = ["app.py"]
    if rgb_path.exists():
        args += ["--checkpoint_rgb", str(rgb_path)]
    if rgbd_path.exists():
        args += ["--checkpoint_rgbd", str(rgbd_path)]
    args += ["--host", "0.0.0.0", "--port", os.getenv("PORT", "7860")]

    if "--checkpoint_rgb" not in args and "--checkpoint_rgbd" not in args:
        raise RuntimeError(
            "No checkpoints found. Set RGB_CKPT_URL / RGBD_CKPT_URL in Space variables."
        )

    sys.argv = args
    web_app.main()


if __name__ == "__main__":
    main()
