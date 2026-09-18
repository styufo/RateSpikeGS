import pytest
from bad_gaussians.checkpoints import latest_checkpoint_step


def test_latest_checkpoint_ignores_sidecars(tmp_path):
    for name in (
        "step-000000007.ckpt",
        "step-000000080.ckpt",
        "skipgs-000000080.pt",
        "skipgs_stats.json",
        "readme.txt",
    ):
        (tmp_path / name).touch()
    assert latest_checkpoint_step(tmp_path) == 80


def test_empty_checkpoint_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_checkpoint_step(tmp_path)
