"""
Image restoration trainer.
"""

from __future__ import annotations
import dataclasses
import functools
import json
from pathlib import Path
from typing import Literal, Type, cast
from typing_extensions import assert_never
import torch
from dataclasses import dataclass, field
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.trainer import TRAIN_INTERATION_OUTPUT
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from bad_gaussians.image_restoration_pipeline import (
    ImageRestorationPipeline,
    ImageRestorationPipelineConfig,
)
from bad_gaussians.bad_viewer import BadViewer
from bad_gaussians.spike_utils import setup_logging
from skipgs import SkipController
from bad_gaussians.checkpoints import latest_checkpoint_step


@dataclass
class ImageRestorationTrainerConfig(TrainerConfig):
    """Configuration for image restoration training"""

    _target: Type = field(default_factory=lambda: ImageRestorationTrainer)
    "The target class to be instantiated."
    pipeline: ImageRestorationPipelineConfig = field(default_factory=ImageRestorationPipelineConfig)
    "Image restoration pipeline configuration"
    use_skipgs: bool = False
    "Skip low-utility backward passes after Gaussian densification ends."
    skipgs_start_iter: int = 15000
    "First iteration eligible for SkipGS; matches the 3DGS densification stop."
    skipgs_threshold: float = 0.0
    "Skip when the current per-view loss ratio is at most one plus this value."
    skipgs_ema_decay: float = 0.95
    "Decay of the per-view loss exponential moving average."
    skipgs_warmup: int = 500
    "Number of post-densification iterations used to calibrate the backward budget."
    skipgs_min_backward_ratio: str = "auto"
    "Minimum backward ratio, represented as 'auto' or a numeric string."


class ImageRestorationTrainer(Trainer):
    """Image restoration Trainer class"""

    config: ImageRestorationTrainerConfig
    pipeline: ImageRestorationPipeline

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        needs_zero = [
            group
            for group in self.optimizers.parameters
            if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        device_type = self.device.split(":")[0]
        device_type = "cpu" if device_type == "mps" else device_type
        with torch.autocast(device_type=device_type, enabled=self.mixed_precision):
            (_, loss_dict, metrics_dict) = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        skip_backward = False
        if self.skip_controller is not None:
            view_id = self.pipeline.last_train_view_id
            loss_value = float(loss.detach().item())
            skip_backward = self.skip_controller.should_skip(view_id, loss_value, step)
            force_accumulation_commit = any(
                (
                    interval > 1 and step % interval == interval - 1
                    for interval in self.gradient_accumulation_steps.values()
                )
            )
            if skip_backward and force_accumulation_commit:
                skip_backward = False
            self.skip_controller.record(view_id, loss_value, skip_backward, step)
            summary = self.skip_controller.summary()
            metrics_dict["skipgs_skipped"] = loss.new_tensor(float(skip_backward))
            metrics_dict["skipgs_backward_ratio"] = loss.new_tensor(summary["bwd_ratio"])
            metrics_dict["skipgs_min_backward_ratio"] = loss.new_tensor(
                summary["min_bwd_ratio_final"]
            )
            if skip_backward:
                self.optimizers.scheduler_step_all(step)
                return (loss, loss_dict, metrics_dict)
        physical_xys_gradient = None
        model = self.pipeline.model
        if model.config.spike_densify_mode != "joint" and "physical_spike_loss" in loss_dict:
            physical_xys_gradient = torch.autograd.grad(
                loss_dict["physical_spike_loss"], model.xys, retain_graph=True, allow_unused=True
            )[0]
        gradient_scale = self.grad_scaler.get_scale()
        self.grad_scaler.scale(loss).backward()
        if physical_xys_gradient is not None:
            model.last_physical_xys_gradient = physical_xys_gradient.detach() * gradient_scale
        elif step < model.config.physical_spike_calibration_steps:
            model.last_physical_xys_gradient = torch.zeros_like(model.xys)
        needs_step = [
            group
            for group in self.optimizers.parameters
            if step % self.gradient_accumulation_steps[group]
            == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)
        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad
            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)
        return (loss, loss_dict, metrics_dict)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "test") -> None:
        """Set up the trainer.

        Args:
            test_mode: The test mode to use.
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()
        self.logger = setup_logging(self.base_dir / "result.txt")
        self.skip_controller = None
        if self.config.use_skipgs:
            densification_stop = self.pipeline.model.config.stop_split_at
            if self.config.skipgs_start_iter < densification_stop:
                raise ValueError(
                    f"SkipGS must start at or after densification stops: {self.config.skipgs_start_iter} < {densification_stop}"
                )
            min_backward_ratio = self.config.skipgs_min_backward_ratio
            if min_backward_ratio != "auto":
                min_backward_ratio = float(min_backward_ratio)
            self.skip_controller = SkipController(
                start_iter=self.config.skipgs_start_iter,
                threshold=self.config.skipgs_threshold,
                ema_decay=self.config.skipgs_ema_decay,
                warmup=self.config.skipgs_warmup,
                min_bwd_ratio=min_backward_ratio,
            )
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        (self.viewer_state, banner_messages) = (None, None)
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            assert_never(self.config.vis)
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = BadViewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()
        self._load_checkpoint()
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,
            )
        )
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging,
            max_iter=self.config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)
        if self.pipeline.datamanager.eval_dataset.cameras is None:
            self.config.steps_per_eval_all_images = int(9000000000.0)
            self.config.steps_per_eval_batch = int(9000000000.0)
            self.config.steps_per_eval_image = int(9000000000.0)

    def _load_checkpoint(self) -> None:
        if self.config.load_dir is not None and self.config.load_step is None:
            self.config.load_step = latest_checkpoint_step(self.config.load_dir)
        super()._load_checkpoint()
        if self.skip_controller is None or self._start_step == 0:
            return
        load_dir = self.config.load_dir
        if self.config.load_checkpoint is not None:
            load_dir = self.config.load_checkpoint.parent
        if load_dir is None:
            return
        state_path = Path(load_dir) / f"skipgs-{self._start_step - 1:09d}.pt"
        if state_path.exists():
            self.skip_controller.load_state_dict(torch.load(state_path, map_location="cpu"))
        else:
            print(f"SkipGS state not found at {state_path}; rebuilding EMA state after resume.")

    def save_checkpoint(self, step: int) -> None:
        super().save_checkpoint(step)
        if self.skip_controller is None:
            return
        state_path = self.checkpoint_dir / f"skipgs-{step:09d}.pt"
        torch.save(self.skip_controller.state_dict(), state_path)
        summary_path = self.checkpoint_dir / "skipgs_stats.json"
        summary_path.write_text(json.dumps(self.skip_controller.summary(), indent=2) + "\n")

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.
        Args:
            step: Current training step.
        """
        if step_check(step, self.config.steps_per_eval_batch):
            (_, eval_loss_dict, eval_metrics_dict) = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                (metrics_dict, images_dict) = self.pipeline.get_eval_image_metrics_and_images(
                    step=step
                )
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(
                step=step, output_path=self.base_dir
            )
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step
            )
            re = " ".join([f"{key}: {val}" for (key, val) in metrics_dict.items()])
            self.logger.info(str(step) + "---" + re)
