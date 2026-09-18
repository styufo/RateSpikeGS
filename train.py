# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""
from __future__ import annotations
import os
import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)
torch.backends.cudnn.benchmark = True


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    global opt
    config.machine.seed = config.machine.seed + global_rank + opt.seed_set
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    if opt.reset_seed_after_setup:
        _set_random_seed(config.machine.seed)
    if trainer._start_step:
        # Nerfstudio 1.0.3 treats this field as the number of additional updates.
        remaining = opt.max_num_iterations - trainer._start_step
        if remaining <= 0:
            raise ValueError("Checkpoint has already reached the requested final step")
        trainer.config.max_num_iterations = remaining
    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg
    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )
        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(
                main_func,
                world_size,
                num_devices_per_machine,
                machine_rank,
                dist_url,
                config,
                timeout,
                device_type,
            ),
        )
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data
    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt
    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()
    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(AnnotatedBaseConfigUnion, description=convert_markup_to_ansi(__doc__)))


from pathlib import Path
import argparse
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from bad_gaussians.bad_camera_optimizer import BadCameraOptimizerConfig
from bad_gaussians.bad_gaussians import BadGaussiansModelConfig
from bad_gaussians.image_restoration_full_image_datamanager import (
    ImageRestorationFullImageDataManagerConfig,
)
from bad_gaussians.image_restoration_pipeline import ImageRestorationPipelineConfig
from bad_gaussians.image_restoration_trainer import ImageRestorationTrainerConfig
from bad_gaussians.nerf_studio_dataparser import NerfstudioDataParserConfig
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a radiance field with nerfstudio.")
    parser.add_argument(
        "--seed_set", type=int, default=425, help="Seed for random number generation."
    )
    parser.add_argument(
        "--reset_seed_after_setup", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--use_3dgs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_spike", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multi_net", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multi_reblur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_real", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data_path", type=str, default="data/synthetic/wine")
    parser.add_argument("--data_name", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--camera_res_scale_factor", type=float, default=1.0)
    parser.add_argument(
        "--eval_mode", choices=("fraction", "filename", "interval", "all"), default="fraction"
    )
    parser.add_argument("--train_split_fraction", type=float, default=0.8)
    parser.add_argument("--eval_interval", type=int, default=8)
    parser.add_argument("--weight_3dgs", type=float, default=1)
    parser.add_argument("--weight_spike", type=float, default=1)
    parser.add_argument("--weight_joint", type=float, default=1)
    parser.add_argument("--net_lr", type=float, default=0.001)
    parser.add_argument("--num_cam", type=int, default=13)
    parser.add_argument(
        "--trajectory_mode", choices=("linear", "cubic", "bezier"), default="linear"
    )
    parser.add_argument("--trajectory_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--trajectory_trans_l2_penalty", type=float, default=0.0)
    parser.add_argument("--trajectory_rot_l2_penalty", type=float, default=0.0)
    parser.add_argument(
        "--use_physical_spike_loss", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--physical_spike_weight", type=float, default=1.0)
    parser.add_argument("--physical_spike_initial_gain", type=float, default=0.5)
    parser.add_argument("--physical_spike_scales", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--physical_spike_downsample", type=int, default=4)
    parser.add_argument("--physical_spike_start_step", type=int, default=0)
    parser.add_argument("--physical_spike_calibration_steps", type=int, default=0)
    parser.add_argument("--physical_spike_timing_weight", type=float, default=0.0)
    parser.add_argument(
        "--physical_spike_domain", choices=("intensity", "rate"), default="intensity"
    )
    parser.add_argument(
        "--physical_spike_sensor_weights", type=float, nargs=3, default=(0.299, 0.587, 0.114)
    )
    parser.add_argument(
        "--physical_spike_calibrate_rate_gain", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--physical_spike_overlapping_counts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--physical_spike_cumulative_weight", type=float, default=0.0)
    parser.add_argument("--spike_densify_mode", choices=("joint", "curriculum"), default="joint")
    parser.add_argument("--spike_densify_stop_step", type=int, default=1000)
    parser.add_argument("--stop_split_at", type=int, default=15000)
    parser.add_argument("--use_skipgs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skipgs_start_iter", type=int, default=15000)
    parser.add_argument("--skipgs_threshold", type=float, default=0.0)
    parser.add_argument("--skipgs_ema_decay", type=float, default=0.95)
    parser.add_argument("--skipgs_warmup", type=int, default=500)
    parser.add_argument("--skipgs_min_backward_ratio", type=str, default="auto")
    parser.add_argument("--max_num_iterations", type=int, default=30001)
    parser.add_argument("--steps_per_eval_image", type=int, default=3000)
    parser.add_argument("--steps_per_eval_batch", type=int, default=3000)
    parser.add_argument("--steps_per_eval_all_images", type=int, default=1000)
    parser.add_argument("--steps_per_save", type=int, default=2000)
    parser.add_argument("--scheduler_max_steps", type=int, default=None)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_step", type=int, default=None)
    parser.add_argument(
        "--vis",
        choices=("viewer", "tensorboard", "wandb", "viewer+tensorboard", "viewer+wandb"),
        default="viewer+tensorboard",
    )
    opt = parser.parse_args()
    scheduler_max_steps = opt.scheduler_max_steps
    if scheduler_max_steps is None:
        scheduler_max_steps = max(opt.max_num_iterations - 1, 1)
    method_config = ImageRestorationTrainerConfig(
        method_name="bad-gaussians",
        output_dir=Path(opt.output_dir) / opt.data_name,
        experiment_name=opt.exp_name,
        steps_per_eval_image=opt.steps_per_eval_image,
        steps_per_eval_batch=opt.steps_per_eval_batch,
        steps_per_save=opt.steps_per_save,
        steps_per_eval_all_images=opt.steps_per_eval_all_images,
        max_num_iterations=opt.max_num_iterations,
        use_skipgs=opt.use_skipgs,
        skipgs_start_iter=opt.skipgs_start_iter,
        skipgs_threshold=opt.skipgs_threshold,
        skipgs_ema_decay=opt.skipgs_ema_decay,
        skipgs_warmup=opt.skipgs_warmup,
        skipgs_min_backward_ratio=opt.skipgs_min_backward_ratio,
        load_dir=None if opt.load_dir is None else Path(opt.load_dir),
        load_step=opt.load_step,
        mixed_precision=False,
        use_grad_scaler=False,
        gradient_accumulation_steps={"camera_opt": 25},
        pipeline=ImageRestorationPipelineConfig(
            eval_render_start_end=True,
            eval_render_estimated=True,
            datamanager=ImageRestorationFullImageDataManagerConfig(
                cache_images="gpu",
                camera_res_scale_factor=opt.camera_res_scale_factor,
                dataparser=NerfstudioDataParserConfig(
                    data=Path(opt.data_path),
                    load_3D_points=True,
                    eval_mode=opt.eval_mode,
                    train_split_fraction=opt.train_split_fraction,
                    eval_interval=opt.eval_interval,
                ),
                use_real=opt.use_real,
            ),
            model=BadGaussiansModelConfig(
                camera_optimizer=BadCameraOptimizerConfig(
                    mode=opt.trajectory_mode,
                    num_virtual_views=opt.num_cam,
                    trajectory_smoothness_weight=opt.trajectory_smoothness_weight,
                    trans_l2_penalty=opt.trajectory_trans_l2_penalty,
                    rot_l2_penalty=opt.trajectory_rot_l2_penalty,
                ),
                use_scale_regularization=True,
                continue_cull_post_densification=False,
                cull_alpha_thresh=0.005,
                densify_grad_thresh=0.0004,
                num_downscales=0,
                resolution_schedule=250,
                tv_loss_lambda=None,
                use_spike=opt.use_spike,
                use_3dgs=opt.use_3dgs,
                use_flip=opt.use_flip,
                use_multi_net=opt.use_multi_net,
                use_multi_reblur=opt.use_multi_reblur,
                weight_3dgs=opt.weight_3dgs,
                weight_spike=opt.weight_spike,
                weight_joint=opt.weight_joint,
                use_physical_spike_loss=opt.use_physical_spike_loss,
                physical_spike_weight=opt.physical_spike_weight,
                physical_spike_initial_gain=opt.physical_spike_initial_gain,
                physical_spike_scales=tuple(opt.physical_spike_scales),
                physical_spike_downsample=opt.physical_spike_downsample,
                physical_spike_start_step=opt.physical_spike_start_step,
                physical_spike_calibration_steps=opt.physical_spike_calibration_steps,
                physical_spike_timing_weight=opt.physical_spike_timing_weight,
                physical_spike_domain=opt.physical_spike_domain,
                physical_spike_sensor_weights=tuple(opt.physical_spike_sensor_weights),
                physical_spike_calibrate_rate_gain=opt.physical_spike_calibrate_rate_gain,
                physical_spike_overlapping_counts=opt.physical_spike_overlapping_counts,
                physical_spike_cumulative_weight=opt.physical_spike_cumulative_weight,
                spike_densify_mode=opt.spike_densify_mode,
                spike_densify_stop_step=opt.spike_densify_stop_step,
                stop_split_at=opt.stop_split_at,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=0.00016, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-06, max_steps=scheduler_max_steps
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-05, max_steps=scheduler_max_steps
                ),
            },
            "spike_net": {
                "optimizer": AdamOptimizerConfig(lr=opt.net_lr, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-05, max_steps=scheduler_max_steps
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis=opt.vis,
    )
    main(method_config)
