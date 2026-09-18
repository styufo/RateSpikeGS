"""Differentiable spike-camera observation models."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_sensor_signal(rgb, weights=(0.299, 0.587, 0.114)):
    """Convert rendered RGB to a calibrated single-channel sensor signal."""
    if rgb.shape[-1] != 3:
        raise ValueError("rgb must have a final RGB channel")
    weights_tensor = rgb.new_tensor(weights)
    if weights_tensor.numel() != 3 or not torch.isfinite(weights_tensor).all():
        raise ValueError("sensor weights must contain three finite values")
    if torch.any(weights_tensor < 0) or float(weights_tensor.sum()) <= 0:
        raise ValueError("sensor weights must be non-negative and non-zero")
    weights_tensor = weights_tensor / weights_tensor.sum()
    return (rgb * weights_tensor).sum(dim=-1, keepdim=True)


class _SurrogateThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_minus_threshold, slope):
        ctx.save_for_backward(voltage_minus_threshold)
        ctx.slope = slope
        return (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (distance,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + ctx.slope * distance.abs()).pow(2)
        return grad_output * surrogate, None


def surrogate_threshold(voltage_minus_threshold, slope=5.0):
    return _SurrogateThreshold.apply(voltage_minus_threshold, slope)


def interpolate_temporal_sequence(sequence, target_steps):
    """Interpolate a [B, T, C, H, W] sequence only along time."""
    if sequence.ndim != 5:
        raise ValueError("sequence must have shape [B, T, C, H, W]")
    if target_steps < 2:
        raise ValueError("target_steps must be at least 2")
    sequence = sequence.permute(0, 2, 1, 3, 4)
    sequence = F.interpolate(
        sequence,
        size=(target_steps, sequence.shape[-2], sequence.shape[-1]),
        mode="trilinear",
        align_corners=True,
    )
    return sequence.permute(0, 2, 1, 3, 4)


class SpikeResponseModel(nn.Module):
    """Self-calibrating integrate-and-fire model with phase-robust count loss."""

    def __init__(
        self,
        initial_gain=0.5,
        threshold=1.0,
        count_scales=(4, 8, 16, 32),
        spatial_downsample=4,
        charbonnier_epsilon=1e-3,
        timing_weight=0.0,
        observation_domain: Literal["intensity", "rate"] = "intensity",
        overlapping_counts=False,
        cumulative_weight=0.0,
        calibrate_rate_gain=False,
    ):
        super().__init__()
        if initial_gain <= 0.0:
            raise ValueError("initial_gain must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if not count_scales or any(scale < 1 for scale in count_scales):
            raise ValueError("count_scales must contain positive integers")
        if spatial_downsample < 1:
            raise ValueError("spatial_downsample must be positive")
        if timing_weight < 0.0:
            raise ValueError("timing_weight must be non-negative")
        if observation_domain not in ("intensity", "rate"):
            raise ValueError("observation_domain must be 'intensity' or 'rate'")
        if observation_domain == "rate" and timing_weight > 0.0:
            raise ValueError("integrate-and-fire timing loss requires the intensity domain")
        if cumulative_weight < 0.0:
            raise ValueError("cumulative_weight must be non-negative")
        self.log_gain = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_gain)), dtype=torch.float32)
        )
        self.initial_phase_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))
        self.count_scales = tuple(int(scale) for scale in count_scales)
        self.spatial_downsample = spatial_downsample
        self.charbonnier_epsilon = charbonnier_epsilon
        self.timing_weight = timing_weight
        self.observation_domain = observation_domain
        self.calibrate_rate_gain = bool(calibrate_rate_gain)
        self.overlapping_counts = overlapping_counts
        self.cumulative_weight = cumulative_weight
        if observation_domain == "rate" and not self.calibrate_rate_gain:
            self.log_gain.requires_grad_(False)
            self.initial_phase_logit.requires_grad_(False)
        elif observation_domain == "rate":
            self.initial_phase_logit.requires_grad_(False)

    @property
    def gain(self):
        return F.softplus(self.log_gain)

    @property
    def initial_phase(self):
        return torch.sigmoid(self.initial_phase_logit)

    @staticmethod
    def _canonicalize_observed(observed_spikes):
        if observed_spikes.ndim == 4:
            observed_spikes = observed_spikes.unsqueeze(2)
        if observed_spikes.ndim != 5 or observed_spikes.shape[2] != 1:
            raise ValueError("observed_spikes must have shape [B, T, H, W] or [B, T, 1, H, W]")
        return observed_spikes

    def _spatial_pool(self, sequence):
        if self.spatial_downsample == 1:
            return sequence
        sequence = sequence.permute(0, 2, 1, 3, 4)
        sequence = F.avg_pool3d(
            sequence,
            kernel_size=(1, self.spatial_downsample, self.spatial_downsample),
            stride=(1, self.spatial_downsample, self.spatial_downsample),
        )
        return sequence.permute(0, 2, 1, 3, 4)

    def integrate_and_fire(self, rendered_intensity):
        """Generate binary spikes with a surrogate gradient through the threshold."""
        if rendered_intensity.ndim != 5 or rendered_intensity.shape[2] != 1:
            raise ValueError("rendered_intensity must have shape [B, T, 1, H, W]")
        membrane = self.initial_phase * self.threshold * torch.ones_like(rendered_intensity[:, 0])
        spikes = []
        for step in range(rendered_intensity.shape[1]):
            voltage = membrane + self.gain * rendered_intensity[:, step]
            spike = surrogate_threshold(voltage - self.threshold)
            membrane = voltage - spike * self.threshold
            spikes.append(spike)
        return torch.stack(spikes, dim=1), membrane

    def _expected_rate(self, rendered_signal):
        if self.observation_domain == "rate":
            if self.calibrate_rate_gain:
                return self.gain * rendered_signal
            return rendered_signal
        return self.gain * rendered_signal / self.threshold

    def multiscale_count_loss(self, rendered_intensity, observed_spikes):
        """Compare expected and observed counts while marginalizing unknown spike phase."""
        if rendered_intensity.ndim != 5 or rendered_intensity.shape[2] != 1:
            raise ValueError("rendered_intensity must have shape [B, T, 1, H, W]")
        observed_spikes = self._canonicalize_observed(observed_spikes).to(rendered_intensity)
        if rendered_intensity.shape[:2] != observed_spikes.shape[:2]:
            raise ValueError(
                "rendered intensity and observed spikes must share batch/time dimensions"
            )
        if rendered_intensity.shape[-2:] != observed_spikes.shape[-2:]:
            raise ValueError("rendered intensity and observed spikes must share spatial dimensions")

        rendered_intensity = self._spatial_pool(rendered_intensity.clamp_min(0.0))
        observed_spikes = self._spatial_pool(observed_spikes)
        expected_rate = self._expected_rate(rendered_intensity)
        rendered_intensity = rendered_intensity.permute(0, 2, 1, 3, 4)
        expected_rate = expected_rate.permute(0, 2, 1, 3, 4)
        observed_spikes = observed_spikes.permute(0, 2, 1, 3, 4)

        losses = []
        metrics = {
            "gain": (
                self.gain.detach()
                if self.observation_domain == "intensity" or self.calibrate_rate_gain
                else self.threshold.detach()
            ),
            "observed_rate": observed_spikes.mean().detach(),
            "expected_rate": expected_rate.mean().detach(),
        }
        for scale in self.count_scales:
            if scale > expected_rate.shape[2]:
                continue
            kernel = (scale, 1, 1)
            stride = (1, 1, 1) if self.overlapping_counts else kernel
            expected_count = F.avg_pool3d(expected_rate, kernel_size=kernel, stride=stride) * scale
            observed_count = (
                F.avg_pool3d(observed_spikes, kernel_size=kernel, stride=stride) * scale
            )
            normalized_error = (expected_count - observed_count) / torch.sqrt(observed_count + 1.0)
            epsilon = self.charbonnier_epsilon
            scale_loss = (torch.sqrt(normalized_error.square() + epsilon**2) - epsilon).mean()
            losses.append(scale_loss)
            metrics[f"count_loss_{scale}"] = scale_loss.detach()
        if not losses:
            raise ValueError("all count scales exceed the temporal sequence length")
        return torch.stack(losses).mean(), metrics

    def phase_invariant_cumulative_loss(self, rendered_signal, observed_spikes):
        """Match cumulative firing-rate shape while marginalizing a constant initial phase."""
        observed_spikes = self._canonicalize_observed(observed_spikes).to(rendered_signal)
        rendered_signal = self._spatial_pool(rendered_signal.clamp_min(0.0))
        observed_spikes = self._spatial_pool(observed_spikes)
        expected_cumulative = self._expected_rate(rendered_signal).cumsum(dim=1)
        observed_cumulative = observed_spikes.cumsum(dim=1)
        residual = expected_cumulative - observed_cumulative
        residual = residual - residual.mean(dim=1, keepdim=True)
        normalized = residual / torch.sqrt(observed_cumulative + 1.0)
        epsilon = self.charbonnier_epsilon
        return (torch.sqrt(normalized.square() + epsilon**2) - epsilon).mean()

    def forward(self, rendered_intensity, observed_spikes):
        count_loss, metrics = self.multiscale_count_loss(rendered_intensity, observed_spikes)
        loss = count_loss
        if self.cumulative_weight > 0.0:
            cumulative_loss = self.phase_invariant_cumulative_loss(
                rendered_intensity, observed_spikes
            )
            metrics["cumulative_loss"] = cumulative_loss.detach()
            loss = loss + self.cumulative_weight * cumulative_loss
        if self.timing_weight == 0.0:
            return loss, metrics

        observed_spikes = self._canonicalize_observed(observed_spikes).to(rendered_intensity)
        timing_intensity = self._spatial_pool(rendered_intensity.clamp_min(0.0))
        timing_observed = self._spatial_pool(observed_spikes)
        predicted_spikes, _ = self.integrate_and_fire(timing_intensity)
        timing_loss = torch.abs(predicted_spikes - timing_observed).mean()
        metrics["timing_loss"] = timing_loss.detach()
        metrics["predicted_binary_rate"] = predicted_spikes.mean().detach()
        metrics["initial_phase"] = self.initial_phase.detach()
        return loss + self.timing_weight * timing_loss, metrics
