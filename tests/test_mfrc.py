import pytest
import torch
from bad_gaussians.spike_sensor import SpikeResponseModel, interpolate_temporal_sequence


def make_sensor():
    return SpikeResponseModel(
        observation_domain="rate",
        spatial_downsample=1,
        count_scales=(4, 8, 16, 32),
        overlapping_counts=True,
        cumulative_weight=0.1,
    )


def test_identical_rates_have_zero_loss():
    rates = torch.rand(1, 97, 1, 4, 4)
    loss, metrics = make_sensor()(rates, rates)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    assert all(f"count_loss_{w}" in metrics for w in (4, 8, 16, 32))


def test_loss_and_gradients_are_finite_and_nonzero():
    rates = torch.full((1, 97, 1, 4, 4), 0.25, requires_grad=True)
    loss, _ = make_sensor()(rates, torch.zeros_like(rates))
    loss.backward()
    assert loss > 0 and torch.isfinite(rates.grad).all() and rates.grad.abs().sum() > 0


def test_temporal_interpolation_preserves_endpoints():
    x = torch.rand(1, 13, 1, 3, 3, requires_grad=True)
    y = interpolate_temporal_sequence(x, 97)
    torch.testing.assert_close(y[:, 0], x[:, 0])
    torch.testing.assert_close(y[:, -1], x[:, -1])
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_curriculum_excludes_only_late_mfrc_gradient():
    from bad_gaussians.bad_gaussians import curriculum_densification_gradients

    total = torch.tensor([[3.0, 4.0]])
    physical = torch.tensor([[3.0, 0.0]])
    assert curriculum_densification_gradients(total, physical, 999, 1000).item() == 5
    assert curriculum_densification_gradients(total, physical, 1000, 1000).item() == 4
