from argparse import Namespace
from pathlib import Path

import pytest
from scripts.train_scene import build_command, ROOT


@pytest.mark.parametrize(
    "preset,mfrc,dc,skip",
    [
        ("baseline", False, "joint", False),
        ("mfrc", True, "joint", False),
        ("mfrc_dc", True, "curriculum", False),
        ("full", True, "curriculum", True),
    ],
)
def test_component_presets(preset, mfrc, dc, skip):
    args = Namespace(
        config=ROOT / "configs/synthetic.json",
        preset=preset,
        data=Path("data/factory"),
        output=Path("outputs"),
        name=None,
        steps=30000,
        resume=None,
        load_step=None,
    )
    command = build_command(args)
    assert ("--use_physical_spike_loss" in command) == mfrc
    assert ("--use_skipgs" in command) == skip
    assert command[command.index("--spike_densify_mode") + 1] == dc
    assert command[command.index("--max_num_iterations") + 1] == "30001"
    assert "--no-reset_seed_after_setup" in command
