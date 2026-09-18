"""
RateSpikeGS model, extending the USP-Gaussian / BAD-Gaussians backbone.
"""

from __future__ import annotations
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SH2RGB, SplatfactoModel, SplatfactoModelConfig
from nerfstudio.model_components import renderers
from bad_gaussians.bad_camera_optimizer import (
    BadCameraOptimizer,
    BadCameraOptimizerConfig,
    TrajSamplingMode,
)
from bad_gaussians.bad_losses import EdgeAwareVariationLoss
from bad_gaussians.network import SpkRecon_Net, Multi_SpkRecon_Net
from bad_gaussians.spike_sensor import (
    SpikeResponseModel,
    interpolate_temporal_sequence,
    rgb_to_sensor_signal,
)


def curriculum_densification_gradients(
    total_gradient: torch.Tensor, physical_gradient: torch.Tensor, step: int, stop_step: int
) -> torch.Tensor:
    """Use joint split evidence early, then exclude physical gradients after a cutoff."""
    if step < stop_step:
        return total_gradient.norm(dim=-1)
    return (total_gradient - physical_gradient).norm(dim=-1)


@dataclass
class BadGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: BadGaussiansModel)
    "The target class to be instantiated."
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    '\n    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This\n    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which\n    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors\n    and apply them to the opacities of gaussians to preserve the total integrated density of splats.\n\n    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that\n    were implemented for classic mode can not render antialiased mode PLY properly without modifications.\n    Refs:\n    1. https://github.com/nerfstudio-project/gsplat/pull/117\n    2. https://github.com/nerfstudio-project/nerfstudio/pull/2888\n    3. Yu, Zehao, et al. "Mip-Splatting: Alias-free 3D Gaussian Splatting." arXiv preprint arXiv:2311.16493 (2023).\n    '
    camera_optimizer: BadCameraOptimizerConfig = field(default_factory=BadCameraOptimizerConfig)
    "Config of the camera optimizer to use"
    cull_alpha_thresh: float = 0.005
    "Threshold for alpha to cull gaussians. Default: 0.1 in splatfacto, 0.005 in splatfacto-big."
    densify_grad_thresh: float = 0.0004
    "[IMPORTANT] Threshold for gradient to densify gaussians. Default: 4e-4. Tune it smaller with complex scenes."
    continue_cull_post_densification: bool = False
    "Whether to continue culling after densification. Default: True in splatfacto, False in splatfacto-big."
    resolution_schedule: int = 250
    "training starts at 1/d resolution, every n steps this is doubled.\n    Default: 250. Use 3000 with high resolution images (e.g. higher than 1920x1080).\n    "
    num_downscales: int = 0
    "at the beginning, resolution is 1/2^d, where d is this number. Default: 0. Use 2 with high resolution images."
    enable_absgrad: bool = False
    "Whether to enable absgrad for gaussians. (It affects param tuning of densify_grad_thresh)\n    Default: False. Ref: (https://github.com/nerfstudio-project/nerfstudio/pull/3113)\n    "
    tv_loss_lambda: Optional[float] = None
    "weight of total variation loss"
    use_3dgs: bool = True
    "use 3DGS reblur optimization loss"
    use_spike: bool = True
    "use spike-net optimization loss"
    use_flip: bool = False
    "use flip operation to align pose and result"
    use_multi_net: bool = False
    "use multi-input spike-net"
    use_multi_reblur: bool = False
    "use multi-reblur loss function"
    weight_3dgs: float = 1
    "3dgs loss weight"
    weight_spike: float = 1
    "spike loss weight"
    weight_joint: float = 1
    "mutual loss weight"
    use_physical_spike_loss: bool = False
    "Supervise rendered intensity directly with raw spike counts."
    physical_spike_weight: float = 1.0
    "Weight of the raw-spike physical observation loss."
    physical_spike_initial_gain: float = 0.5
    "Initial photoelectric response gain relative to a unit firing threshold."
    physical_spike_scales: Tuple[int, ...] = (4, 8, 16, 32)
    "Temporal count windows used by the phase-robust physical loss."
    physical_spike_downsample: int = 4
    "Spatial downsampling used by the initial physical observation prototype."
    physical_spike_start_step: int = 0
    "Delay raw-spike supervision until blur-only geometry has stabilized."
    physical_spike_calibration_steps: int = 0
    "Initially train only sensor response parameters while detaching rendered intensity."
    physical_spike_timing_weight: float = 0.0
    "Weight of phase-aware integrate-and-fire timing supervision inside the physical loss."
    physical_spike_domain: Literal["intensity", "rate"] = "intensity"
    "Interpret rendered luminance as irradiance or as the TFP firing-rate domain."
    physical_spike_sensor_weights: Tuple[float, ...] = (0.299, 0.587, 0.114)
    "RGB response weights; real Bayer 2x2 pooling uses (0.25, 0.5, 0.25)."
    physical_spike_calibrate_rate_gain: bool = False
    "Learn the unknown exposure/threshold scale for real firing-rate supervision."
    physical_spike_overlapping_counts: bool = False
    "Use every sliding temporal count window instead of disjoint windows."
    physical_spike_cumulative_weight: float = 0.0
    "Weight of the phase-invariant cumulative exposure residual."
    spike_densify_mode: Literal["joint", "curriculum"] = "joint"
    "Use summed or task-separated screen gradients for Gaussian densification."
    spike_densify_stop_step: int = 1000
    "Step after which curriculum densification excludes physical screen gradients."


class BadGaussiansModel(SplatfactoModel):
    """BAD-Gaussians Model

    Args:
        config: configuration to instantiate model
    """

    config: BadGaussiansModelConfig
    camera_optimizer: BadCameraOptimizer

    def __init__(self, config: BadGaussiansModelConfig, **kwargs) -> None:
        if config.spike_densify_mode not in ("joint", "curriculum"):
            raise ValueError("spike_densify_mode must be joint or curriculum")
        super().__init__(config=config, **kwargs)
        if self.config.use_physical_spike_loss and (not self.config.use_3dgs):
            raise ValueError("physical spike supervision requires use_3dgs=True")
        if self.config.physical_spike_start_step < 0:
            raise ValueError("physical_spike_start_step must be non-negative")
        if self.config.physical_spike_calibration_steps < 0:
            raise ValueError("physical_spike_calibration_steps must be non-negative")
        if self.config.physical_spike_cumulative_weight < 0.0:
            raise ValueError("physical_spike_cumulative_weight must be non-negative")
        if len(self.config.physical_spike_sensor_weights) != 3:
            raise ValueError("physical_spike_sensor_weights must contain three values")
        if any((weight < 0.0 for weight in self.config.physical_spike_sensor_weights)):
            raise ValueError("physical_spike_sensor_weights must be non-negative")
        if sum(self.config.physical_spike_sensor_weights) <= 0.0:
            raise ValueError("physical_spike_sensor_weights must not be all zero")
        if self.config.spike_densify_stop_step < 0:
            raise ValueError("spike_densify_stop_step must be non-negative")
        if self.config.spike_densify_mode != "joint" and (not self.config.use_physical_spike_loss):
            raise ValueError("spike-aware densification requires physical spike supervision")
        if self.config.spike_densify_mode != "joint" and self.config.enable_absgrad:
            raise ValueError("spike-aware densification is incompatible with absgrad")
        self.config.densify_grad_thresh /= self.config.camera_optimizer.num_virtual_views
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)
        self.spike_in_length = 41
        self.voxel_in_length = 34
        self.use_spike = self.config.use_spike
        self.use_3dgs = self.config.use_3dgs
        self.use_flip = self.config.use_flip
        self.use_multi_net = self.config.use_multi_net
        self.use_multi_reblur = self.config.use_multi_reblur
        if self.use_multi_net == False:
            self.spike_net = SpkRecon_Net(input_dim=self.spike_in_length)
        else:
            self.spike_net = Multi_SpkRecon_Net(
                input_dim=self.spike_in_length, voxel_dim=self.voxel_in_length
            )
        self.spike_net_parameter_count = sum(
            (parameter.numel() for parameter in self.spike_net.parameters())
        )
        self.weight_3dgs = self.config.weight_3dgs
        self.weight_spike = self.config.weight_spike
        self.weight_joint = self.config.weight_joint
        self.last_physical_spike_metrics = {}
        self.last_physical_xys_gradient = None
        self.virtual_xys = []
        self.virtual_radii = []
        self.spike_sensor = None
        if self.config.use_physical_spike_loss:
            self.spike_sensor = SpikeResponseModel(
                initial_gain=self.config.physical_spike_initial_gain,
                count_scales=self.config.physical_spike_scales,
                spatial_downsample=1,
                timing_weight=self.config.physical_spike_timing_weight,
                observation_domain=self.config.physical_spike_domain,
                overlapping_counts=self.config.physical_spike_overlapping_counts,
                cumulative_weight=self.config.physical_spike_cumulative_weight,
                calibrate_rate_gain=self.config.physical_spike_calibrate_rate_gain,
            )

    def populate_modules(self) -> None:
        super().populate_modules()
        self.camera_optimizer: BadCameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def forward(
        self,
        camera: Cameras,
        mode: TrajSamplingMode = "uniform",
        spike: torch.Tensor = torch.zeros(0),
    ) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera, mode=mode, spike=spike)

    def save_spike_net(self, path):
        torch.save(self.spike_net.state_dict(), path)

    def reconstruct_spike(self, voxel, spike, index):
        return self.spike_net(voxel, spike, index)

    def get_outputs(
        self,
        camera: Cameras,
        mode: TrajSamplingMode = "uniform",
        spike: torch.Tensor = torch.zeros(0),
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera. This camera should have all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        is_training = self.training and torch.is_grad_enabled()
        virtual_cameras = self.camera_optimizer.apply_to_camera(camera, mode)
        self.num_cam = len(virtual_cameras)
        if is_training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
        else:
            background = self.background_color.to(self.device)
        if self.crop_box is not None and (not is_training):
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {
                    "rgb": rgb,
                    "depth": depth,
                    "accumulation": accumulation,
                    "background": background,
                }
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        for cam in virtual_cameras:
            cam.rescale_output_resolution(1 / camera_downscale)
        virtual_views_img = []
        virtual_views_alpha = []
        virtual_views_recon = []
        virtual_views_tfp = []
        self.virtual_xys = []
        self.virtual_radii = []
        idx = 0
        if len(spike.shape) == 3:
            spike_voxel = spike[20:-20]
            spike_voxel = torch.cat(
                (
                    spike_voxel[: spike_voxel.shape[0] // 2],
                    spike_voxel[spike_voxel.shape[0] // 2 + 1 :],
                ),
                dim=0,
            )
            spike_voxel = torch.sum(
                spike_voxel[None].reshape(
                    -1, 4, self.voxel_in_length, spike.shape[-2], spike.shape[-1]
                ),
                dim=1,
            )
        for cam in virtual_cameras:
            if len(spike.shape) == 3:
                if mode == "start":
                    spike_idx = 40
                elif mode == "mid":
                    spike_idx = 88
                elif mode == "end":
                    spike_idx = 136
                elif mode == "uniform":
                    if len(virtual_cameras) == 1:
                        spike_idx = spike.shape[0] // 2
                    else:
                        spike_idx = 40 + idx * (96 // (len(virtual_cameras) - 1))
                spike_roi = spike[
                    spike_idx
                    - self.spike_in_length // 2 : spike_idx
                    + self.spike_in_length // 2
                    + 1
                ][None]
                if self.use_multi_net == False:
                    spike_recon = self.spike_net(spike_roi)[0, 0][..., None].repeat(1, 1, 3)
                else:
                    index = (spike_idx - 40) / 96
                    spike_recon = self.reconstruct_spike(spike_voxel, spike_roi, index)[0, 0][
                        ..., None
                    ].repeat(1, 1, 3)
                spike_tfp = torch.mean(
                    spike[spike_idx - 20 : spike_idx + 21 + 1], dim=0, keepdim=False
                )[..., None].repeat(1, 1, 3)
                virtual_views_recon.append(spike_recon)
                virtual_views_tfp.append(spike_tfp)
                idx += 1
            else:
                virtual_views_recon.append(spike)
                virtual_views_tfp.append(spike)
            R = cam.camera_to_worlds[0, :3, :3]
            T = cam.camera_to_worlds[0, :3, 3:4]
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            (W, H) = (int(cam.width.item()), int(cam.height.item()))
            self.last_size = (H, W)
            if crop_ids is not None:
                opacities_crop = self.opacities[crop_ids]
                means_crop = self.means[crop_ids]
                features_dc_crop = self.features_dc[crop_ids]
                features_rest_crop = self.features_rest[crop_ids]
                scales_crop = self.scales[crop_ids]
                quats_crop = self.quats[crop_ids]
            else:
                opacities_crop = self.opacities
                means_crop = self.means
                features_dc_crop = self.features_dc
                features_rest_crop = self.features_rest
                scales_crop = self.scales
                quats_crop = self.quats
            colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
            if self.config.sh_degree > 0:
                viewdirs = means_crop.detach() - cam.camera_to_worlds.detach()[..., :3, 3]
                viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                rgbs = spherical_harmonics(n, viewdirs, colors_crop)
                rgbs = torch.clamp(rgbs + 0.5, min=0.0)
            else:
                rgbs = torch.sigmoid(colors_crop[:, 0, :])
            BLOCK_WIDTH = 16
            (self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d) = project_gaussians(
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                cam.fx.item(),
                cam.fy.item(),
                cam.cx.item(),
                cam.cy.item(),
                H,
                W,
                BLOCK_WIDTH,
            )
            self.virtual_xys.append(self.xys)
            self.virtual_radii.append(self.radii)
            cam.rescale_output_resolution(camera_downscale)
            if self.radii.sum() == 0:
                rgb = background.repeat(H, W, 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {
                    "rgb": rgb,
                    "depth": depth,
                    "accumulation": accumulation,
                    "background": background,
                }
            if is_training:
                self.xys.retain_grad()
            assert (num_tiles_hit > 0).any()
            if self.config.rasterize_mode == "antialiased":
                alphas = torch.sigmoid(opacities_crop) * comp[:, None]
            elif self.config.rasterize_mode == "classic":
                alphas = torch.sigmoid(opacities_crop)
            (rgb, alpha) = rasterize_gaussians(
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                rgbs,
                alphas,
                H,
                W,
                BLOCK_WIDTH,
                background=background,
                return_alpha=True,
            )
            alpha = alpha[..., None]
            rgb = torch.clamp(rgb, max=1.0)
            virtual_views_img.append(rgb)
            virtual_views_alpha.append(alpha)
        depth_im = None
        virtual_views_recon = torch.stack(virtual_views_recon, dim=0)
        spike_blur = virtual_views_recon.mean(dim=0)
        virtual_views_tfp = torch.stack(virtual_views_tfp, dim=0)
        virtual_views_img = torch.stack(virtual_views_img, dim=0)
        rgb = virtual_views_img.mean(dim=0)
        physical_spike_intensity = None
        physical_spike_active = (
            self.config.use_physical_spike_loss
            and self.step >= self.config.physical_spike_start_step
        )
        if physical_spike_active:
            physical_signal = rgb_to_sensor_signal(
                virtual_views_img, self.config.physical_spike_sensor_weights
            )
            physical_signal = physical_signal.permute(0, 3, 1, 2)
            downsample = self.config.physical_spike_downsample
            if downsample > 1:
                physical_signal = F.avg_pool2d(physical_signal, downsample, downsample)
            physical_signal = physical_signal.unsqueeze(0)
            physical_spike_intensity = interpolate_temporal_sequence(
                physical_signal, target_steps=97
            )
        virtual_views_alpha = torch.stack(virtual_views_alpha, dim=0)
        alpha = virtual_views_alpha.mean(dim=0)
        if not is_training:
            depth_im = rasterize_gaussians(
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        outputs = {
            "rgb": rgb,
            "virtual_views_img": virtual_views_img,
            "spike_blur": spike_blur,
            "virtual_views_recon": virtual_views_recon,
            "virtual_views_tfp": virtual_views_tfp,
            "virtual_views_alpha": virtual_views_alpha,
            "depth": depth_im,
            "accumulation": alpha,
            "background": background,
        }
        if physical_spike_intensity is not None:
            outputs["physical_spike_intensity"] = physical_spike_intensity
        return outputs

    def after_train(self, step: int):
        assert step == self.step
        if self.use_3dgs == False and self.use_spike == True:
            self.config.stop_split_at = -1
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            radii_for_stats = self.radii
            if self.config.spike_densify_mode != "joint":
                visible_mask = (self.radii > 0).flatten()
                assert self.xys.grad is not None
                if self.last_physical_xys_gradient is None:
                    raise RuntimeError(
                        "physical screen gradient was not captured for spike-aware densification"
                    )
                if self.config.spike_densify_mode == "curriculum":
                    grads = curriculum_densification_gradients(
                        self.xys.grad.detach(),
                        self.last_physical_xys_gradient,
                        self.step,
                        self.config.spike_densify_stop_step,
                    )
            elif self.config.enable_absgrad:
                visible_mask = (self.radii > 0).flatten()
                assert self.xys.absgrad is not None
                grads = self.xys.absgrad.detach().norm(dim=-1)
            else:
                visible_mask = (self.radii > 0).flatten()
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            self.last_physical_xys_gradient = None
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = (
                    grads[visible_mask] + self.xys_grad_norm[visible_mask]
                )
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = radii_for_stats.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def refinement_after(self, optimizers, step: int):
        """Apply the inherited Gaussian splitting, cloning and culling schedule."""
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                assert self.xys_grad_norm is not None
                assert self.vis_counts is not None
                assert self.max_2Dsize is not None
                avg_grad_norm = (
                    self.xys_grad_norm
                    / self.vis_counts
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (
                    self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                dups = (
                    self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )
                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)
                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(), device=self.device, dtype=torch.bool
                        ),
                    )
                )
                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()
            else:
                deleted_mask = None
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)
            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    @torch.no_grad()
    def get_outputs_for_camera(
        self,
        camera: Cameras,
        obb_box: Optional[OrientedBox] = None,
        mode: TrajSamplingMode = "mid",
        spike: torch.Tensor = torch.zeros(0),
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        metadata = camera.metadata
        camera = camera.to(self.device)
        camera.metadata = metadata
        outs = self.get_outputs(camera, mode=mode, spike=spike)
        return outs

    def get_loss_l1_ssim(self, img1, img2):
        Ll1 = torch.abs(img1 - img2).mean()
        simloss = 1 - self.ssim(img1.permute(2, 0, 1)[None, ...], img2.permute(2, 0, 1)[None, ...])
        return (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

    def get_loss_mse(self, img1, img2):
        return torch.abs(img1 - img2).mean()

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=0):
        """Add loss from the spike-net

        Args:
            batch['image']: blurry input. H * W * 1
            outputs['rgb']: reblur result from the 3DGS. H * W * 1
            outputs['spike_blur']: reblur result from the spike-net. H * W * 1
            outputs['virtual_views_img'] : sequence from the 3DGS. 10 * H * W * 1
            outputs['virtual_views_recon'] : sequence from the spike-net. 10 * H * W * 1
            outputs['virtual_views_recon_flip'] : sequence from the spike-net with flipped spike input. 10 * H * W * 1
        Returns:
            _type_: _description_
        """
        loss_dict = {}
        if self.use_3dgs == True:
            if self.config.use_scale_regularization and self.step % 10 == 0:
                scale_exp = torch.exp(self.scales)
                scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                        torch.tensor(self.config.max_gauss_ratio),
                    )
                    - self.config.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
            else:
                scale_reg = torch.tensor(0.0).to(self.device)
            loss_dict["scale_reg_loss"] = scale_reg
            loss_dict["3dgs_reblur_loss"] = self.weight_3dgs * self.get_loss_l1_ssim(
                outputs["rgb"], batch["image"]
            )
        if self.use_spike == True and self.use_multi_reblur == False:
            loss_dict["spike_reblur_loss"] = self.weight_spike * self.get_loss_l1_ssim(
                outputs["spike_blur"], batch["image"]
            )
        elif self.use_spike == True and self.use_multi_reblur == True:
            spike = batch["spike"][..., None]
            recon_sequence = outputs["virtual_views_recon"]
            if self.num_cam == 1:
                center = spike.shape[0] // 2
                spike_blur = spike[max(center - 20, 0) : center + 22].mean(dim=0)
                loss_dict["spike_reblur_loss"] = self.weight_spike * self.get_loss_mse(
                    spike_blur, recon_sequence[0]
                )
            else:
                spike_cumsum = torch.cumsum(spike, dim=0)
                recon_cumsum = torch.cumsum(recon_sequence, dim=0)
                temp_loss = 0
                iter_idx = 0
                mid_cam = self.num_cam // 2
                for idx in range(self.num_cam // 4, self.num_cam // 2 + 1):
                    start_idx = mid_cam - idx
                    end_idx = mid_cam + idx
                    start_spike = 40 + start_idx * (96 // (self.num_cam - 1))
                    end_spike = 40 + end_idx * (96 // (self.num_cam - 1))
                    spike_blur = (spike_cumsum[end_spike] - spike_cumsum[start_spike - 1]) / (
                        end_spike - start_spike + 1
                    )
                    if start_idx == 0:
                        spike_reblur = recon_cumsum[end_idx] / (end_idx + 1)
                    else:
                        spike_reblur = (recon_cumsum[end_idx] - recon_cumsum[start_idx - 1]) / (
                            end_idx - start_idx + 1
                        )
                    temp_loss += self.get_loss_mse(spike_blur, spike_reblur)
                    iter_idx += 1
                loss_dict["spike_reblur_loss"] = self.weight_spike * temp_loss / iter_idx
        if self.use_3dgs == True and self.use_spike == True:
            if self.use_flip:
                loss_dict["3dgs_spike_loss"] = self.weight_joint * min(
                    self.get_loss_mse(outputs["virtual_views_img"], outputs["virtual_views_recon"]),
                    self.get_loss_mse(
                        torch.flip(outputs["virtual_views_img"], [0]),
                        outputs["virtual_views_recon"],
                    ),
                )
            else:
                loss_dict["3dgs_spike_loss"] = self.weight_joint * self.get_loss_mse(
                    outputs["virtual_views_img"], outputs["virtual_views_recon"]
                )
        if (
            self.training
            and self.config.use_physical_spike_loss
            and (step >= self.config.physical_spike_start_step)
        ):
            observed_spikes = batch["spike"][40:137][None, :, None]
            downsample = self.config.physical_spike_downsample
            if downsample > 1:
                observed_spikes = observed_spikes.permute(0, 2, 1, 3, 4)
                observed_spikes = F.avg_pool3d(
                    observed_spikes,
                    kernel_size=(1, downsample, downsample),
                    stride=(1, downsample, downsample),
                )
                observed_spikes = observed_spikes.permute(0, 2, 1, 3, 4)
            physical_intensity = outputs["physical_spike_intensity"]
            if step < self.config.physical_spike_calibration_steps:
                physical_intensity = physical_intensity.detach()
            (physical_loss, physical_metrics) = self.spike_sensor(
                physical_intensity, observed_spikes
            )
            loss_dict["physical_spike_loss"] = self.config.physical_spike_weight * physical_loss
            self.last_physical_spike_metrics = physical_metrics
        if self.use_3dgs:
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def normal(self, img1, img2, img3):
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        return (img1, img2, img3)

    def get_metrics_rgb_spike(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics = {}
        rgb = outputs["rgb"]
        spike_recon = outputs["spike_blur"]
        gt = batch["image"]
        rgb = torch.permute(rgb, (2, 0, 1))[None].clip(0, 1)
        gt = torch.permute(gt, (2, 0, 1))[None].clip(0, 1)
        spike_recon = torch.permute(spike_recon, (2, 0, 1))[None].clip(0, 1)
        normal_type = "double"
        if normal_type == "normal":
            (rgb, gt, spike_recon) = self.normal(rgb, gt, spike_recon)
        elif normal_type == "double":
            rgb = (rgb * 2).clip(0, 1)
            spike_recon = (spike_recon * 2).clip(0, 1)
        metrics["rgb_psnr"] = self.psnr(rgb, gt)
        metrics["rgb_ssim"] = self.ssim(rgb, gt)
        metrics["rgb_lpips"] = self.lpips(rgb, gt)
        metrics["spike_psnr"] = self.psnr(spike_recon, gt)
        metrics["spike_ssim"] = self.ssim(spike_recon, gt)
        metrics["spike_lpips"] = self.lpips(spike_recon, gt)
        metrics["spike_net_params"] = gt.new_tensor(float(self.spike_net_parameter_count))
        metrics["num_primitives"] = gt.new_tensor(float(self.means.shape[0]))
        metrics["peak_vram_gb"] = gt.new_tensor(torch.cuda.max_memory_allocated() / 1024**3)
        firing_rate = getattr(self.spike_net, "last_firing_rate", None)
        if firing_rate is not None:
            metrics["recon_firing_rate"] = firing_rate
        for key, value in self.last_physical_spike_metrics.items():
            metrics[f"physical_spike_{key}"] = value
        return metrics

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["spike_net_params"] = self.means.new_tensor(
            float(self.spike_net_parameter_count)
        )
        metrics_dict["num_primitives"] = self.means.new_tensor(float(self.means.shape[0]))
        metrics_dict["peak_vram_gb"] = self.means.new_tensor(
            torch.cuda.max_memory_allocated() / 1024**3
        )
        firing_rate = getattr(self.spike_net, "last_firing_rate", None)
        if firing_rate is not None:
            metrics_dict["recon_firing_rate"] = firing_rate
        for key, value in self.last_physical_spike_metrics.items():
            metrics_dict[f"physical_spike_{key}"] = value
        return metrics_dict

    def get_gaussian_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = super().get_gaussian_param_groups()
        return param_groups

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        if self.use_3dgs == True:
            param_groups = super().get_param_groups()
            self.camera_optimizer.get_param_groups(param_groups=param_groups)
        if self.use_spike == True:
            if self.use_3dgs == False:
                param_groups = {}
            param_groups["spike_net"] = list(self.spike_net.parameters())
        if self.config.use_physical_spike_loss:
            param_groups.setdefault("spike_net", [])
            param_groups["spike_net"].extend(self.spike_sensor.parameters())
        return param_groups
