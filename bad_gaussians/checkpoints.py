"""Select model checkpoints without treating controller sidecars as models."""

from pathlib import Path
import re


def latest_checkpoint_step(directory: Path) -> int:
    steps = []
    for path in Path(directory).iterdir():
        match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
        if path.is_file() and match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No step-*.ckpt model checkpoints in {directory}")
    return max(steps)
