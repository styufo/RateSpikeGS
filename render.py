"""Export evaluation views using the same poses and exposure convention as training."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml
from nerfstudio.utils.eval_utils import eval_load_checkpoint
from bad_gaussians.checkpoints import latest_checkpoint_step


def save_image(path, tensor):
    array = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Trusted training config.yml (contains Python YAML objects).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Override dataset location.")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--indices", nargs="+", type=int, help="Zero-based indices in the saved evaluation split."
    )
    args = parser.parse_args()
    # Nerfstudio serializes Python configuration objects. Never load untrusted YAML.
    config = yaml.load(args.config.read_text(), Loader=yaml.Loader)
    if args.data is not None:
        config.pipeline.datamanager.dataparser.data = args.data.resolve()
    config.load_dir = args.checkpoint_dir or args.config.parent / config.relative_model_dir
    config.load_step = (
        args.step if args.step is not None else latest_checkpoint_step(config.load_dir)
    )
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required by the Gaussian rasterizer.")
    pipeline = config.pipeline.setup(device="cuda", test_mode="test")
    pipeline.eval()
    checkpoint, step = eval_load_checkpoint(config, pipeline)
    loader = pipeline.datamanager.fixed_indices_eval_dataloader
    indices = args.indices if args.indices is not None else list(range(len(loader)))
    if not indices or any(i < 0 or i >= len(loader) for i in indices):
        parser.error("Indices must be nonempty and within the evaluation split.")
    is_real = config.pipeline.datamanager.use_real
    rows = {}
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in indices:
            camera, batch = loader.get_camera(index)
            outputs = pipeline.model.get_outputs_for_camera(
                camera, mode="mid", spike=batch["spike"]
            )
            target = args.output / f"{index:04d}"
            target.mkdir(exist_ok=True)
            save_image(target / "gs.png", pipeline.post_process(outputs["rgb"]))
            save_image(target / "recon.png", pipeline.post_process(outputs["spike_blur"]))
            save_image(target / "input.png", batch["degraded"][..., :3])
            save_image(target / ("proxy.png" if is_real else "gt.png"), batch["image"][..., :3])
            if not is_real:
                metrics = pipeline.model.get_metrics_rgb_spike(outputs, batch)
                rows[f"{index:04d}"] = {key: float(value) for key, value in metrics.items()}
            print(f"Exported view {index:04d}", flush=True)
    means = {
        key: sum(row[key] for row in rows.values()) / len(rows)
        for key in next(iter(rows.values()), {})
    }
    result = {
        "step": step,
        "checkpoint": str(checkpoint),
        "real_data": is_real,
        "indices": indices,
        "per_view": rows,
        "mean": means,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
