"""Prepare one SpikeGS real scene for the USP-Gaussian data pipeline.

The source scene remains untouched.  The prepared directory contains symlinks
to the supplied images and COLMAP point cloud plus a manifest pointing to the
single continuous raw spike stream.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _link(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_scene(
    scene_dir: Path, raw_spike: Path, output_dir: Path, window_frames: int, spatial_reduce: int
) -> None:
    scene_dir = scene_dir.resolve()
    raw_spike = raw_spike.resolve()
    output_dir = output_dir.resolve()
    if not (scene_dir / "images").is_dir():
        raise FileNotFoundError(scene_dir / "images")
    if not (scene_dir / "tfp").is_dir():
        raise FileNotFoundError(scene_dir / "tfp")
    if not (scene_dir / "sparse/0").is_dir():
        raise FileNotFoundError(scene_dir / "sparse/0")
    if not raw_spike.is_file():
        raise FileNotFoundError(raw_spike)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    # colmap_to_json writes transforms.json and sparse_pc.ply.  It is imported
    # lazily so the manifest logic remains inspectable without Nerfstudio.
    from nerfstudio.process_data.colmap_utils import colmap_to_json

    _link(scene_dir / "images", output_dir / "images")
    _link(scene_dir / "tfp", output_dir / "tfp")
    colmap_to_json(scene_dir / "sparse/0", output_dir, keep_original_world_coordinate=False)

    transforms_path = output_dir / "transforms.json"
    transforms = json.loads(transforms_path.read_text())
    frame_pattern = re.compile(r"r_(\d+)\.png$")
    for frame in transforms["frames"]:
        match = frame_pattern.search(Path(frame["file_path"]).name)
        if match is None:
            raise ValueError(f"cannot infer spike center from {frame['file_path']}")
        frame["spike_frame_index"] = int(match.group(1))
    transforms_path.write_text(json.dumps(transforms, indent=2) + "\n")

    manifest = {
        "format": "packed_binary_spike_stream",
        "path": str(raw_spike),
        "width": 1000,
        "height": 1000,
        "window_frames": window_frames,
        "spatial_reduce": spatial_reduce,
        "value_scale": float(spatial_reduce * spatial_reduce),
        "reverse_vertical": False,
        "bitorder": "little",
        "bayer_pattern": "GBRG",
        "boundary_mode": "edge",
        "tfp_folder": "tfp",
    }
    (output_dir / "spike_stream.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--raw-spike", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=177)
    parser.add_argument("--spatial-reduce", type=int, default=2)
    args = parser.parse_args()
    if args.window_frames <= 0 or args.window_frames % 2 == 0:
        parser.error("--window-frames must be a positive odd number")
    if args.spatial_reduce <= 0:
        parser.error("--spatial-reduce must be positive")
    prepare_scene(
        args.scene_dir, args.raw_spike, args.output_dir, args.window_frames, args.spatial_reduce
    )


if __name__ == "__main__":
    main()
