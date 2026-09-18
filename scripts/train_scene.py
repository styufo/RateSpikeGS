"""Launch the released synthetic presets without machine-specific paths."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "baseline": dict(use_physical_spike_loss=False, spike_densify_mode="joint", use_skipgs=False),
    "mfrc": dict(use_physical_spike_loss=True, spike_densify_mode="joint", use_skipgs=False),
    "mfrc_dc": dict(
        use_physical_spike_loss=True, spike_densify_mode="curriculum", use_skipgs=False
    ),
    "full": dict(use_physical_spike_loss=True, spike_densify_mode="curriculum", use_skipgs=True),
}


def build_command(args):
    config = json.loads(args.config.read_text())
    config.update(PRESETS[args.preset])
    config.update(
        data_path=str(args.data.resolve()),
        data_name=args.name or args.data.name,
        exp_name=args.preset,
        output_dir=str(args.output.resolve()),
        max_num_iterations=args.steps + 1,
    )
    if args.resume:
        config["load_dir"] = str(args.resume.resolve())
    if args.load_step is not None:
        config["load_step"] = args.load_step
    cmd = [sys.executable, str(ROOT / "train.py")]
    for key, value in config.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no-{key}")
        else:
            cmd.append(f"--{key}")
            cmd.extend(map(str, value if isinstance(value, list) else [value]))
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--name")
    parser.add_argument("--preset", choices=PRESETS, default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/synthetic.json")
    parser.add_argument("--steps", type=int, default=30000, help="Absolute final step (inclusive).")
    parser.add_argument("--resume", type=Path, help="Checkpoint directory, not a checkpoint file.")
    parser.add_argument("--load-step", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.load_step is not None and args.resume is None:
        parser.error("--load-step requires --resume")
    cmd = build_command(args)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        if not (args.data / "transforms.json").is_file():
            parser.error("Dataset must contain transforms.json")
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
