"""
Full image datamanager for image restoration.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Type, Union, cast, Tuple, Dict
from functools import cached_property

import numpy as np
from PIL import Image
from nerfstudio.data.datasets.base_dataset import InputDataset

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import variable_res_collate, TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.rich_utils import CONSOLE
from bad_gaussians.spike_utils import load_vidar_dat
from bad_gaussians.spike_stream import PackedSpikeStream, PackedSpikeStreamSpec
from bad_gaussians.image_restoration_dataloader import (
    ImageRestorationFixedIndicesEvalDataloader,
    ImageRestorationRandIndicesEvalDataloader,
    set_camera_index,
)
from bad_gaussians.nerf_studio_dataparser import SpikeDataparserOutputs


class SpikeInputDataset(InputDataset):
    def __init__(
        self,
        dataparser_outputs: SpikeDataparserOutputs,
        scale_factor: float = 1.0,
        split="train",
        use_real=False,
    ):
        self.split = split
        self.use_real = use_real
        self._packed_stream = None
        if dataparser_outputs.spike_stream is not None:
            spec = PackedSpikeStreamSpec.from_manifest(
                dataparser_outputs.spike_stream,
                Path(dataparser_outputs.spike_stream.get("_base_dir", ".")),
            )
            self._packed_stream = PackedSpikeStream(spec)
        super().__init__(dataparser_outputs, scale_factor)

    def load_spike(self, image_idx: int) -> torch.Tensor:
        if self._packed_stream is not None:
            center = self._dataparser_outputs.spike_frame_indices[image_idx]
            return torch.from_numpy(self._packed_stream.read_window(center))
        spike_filenames = self._dataparser_outputs.spike_filenames
        if self.use_real:
            spike = load_vidar_dat(spike_filenames[image_idx], width=400, height=250)
            spike = spike[spike.shape[0] // 2 - 88 : spike.shape[0] // 2 + 89]
        else:
            spike = load_vidar_dat(spike_filenames[image_idx], width=600, height=400)
        return torch.from_numpy(spike)

    def _load_external_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.scale_factor != 1.0:
            width, height = image.size
            image = image.resize(
                (int(width * self.scale_factor), int(height * self.scale_factor)),
                Image.Resampling.BILINEAR,
            )
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        data = {"image_idx": image_idx}
        # spike reading
        image_idx = data["image_idx"]
        stream_mode = self._packed_stream is not None
        if stream_mode:
            data.update({"spike_index": image_idx})
        else:
            spike = self.load_spike(image_idx)
            data.update({"spike": spike})
        input_type = "tfp"
        # update sharp images for test dataset
        if self.split == "train":
            # blurry input from the long tfp
            if stream_mode and self._dataparser_outputs.tfp_filenames:
                data.update(
                    {
                        "image": self._load_external_image(
                            self._dataparser_outputs.tfp_filenames[image_idx]
                        )
                    }
                )
            elif input_type == "tfp":
                tfp_long = torch.mean(data["spike"][40:-40].float(), dim=0, keepdim=False)[
                    ..., None
                ].repeat(1, 1, 3)
                data.update({"image": tfp_long})
            # blurry input from the rgb image
            elif input_type == "blur":
                image = self.get_image_float32(image_idx)
                weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 3)
                image = torch.sum(image * weights, dim=-1, keepdim=True).repeat(1, 1, 3)
                data.update({"image": image})
        elif self.split == "test":
            image = self.get_image_float32(image_idx)
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 3)
            image = torch.sum(image * weights, dim=-1, keepdim=True).repeat(1, 1, 3)
            data.update({"image": image})
        return data


@dataclass
class ImageRestorationFullImageDataManagerConfig(FullImageDatamanagerConfig):
    """Datamanager for image restoration"""

    _target: Type = field(default_factory=lambda: ImageRestorationFullImageDataManager)
    """Target class to instantiate."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    use_real: bool = False
    """Input is real-world dataset"""


class ImageRestorationFullImageDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Data manager implementation for image restoration
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ImageRestorationFullImageDataManagerConfig

    def __init__(
        self,
        config: ImageRestorationFullImageDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

        self.eval_dataparser_outputs.metadata["optimizer_train_global_indices"] = (
            self.train_dataparser_outputs.metadata["global_image_indices"]
        )

        self.degraded_eval_dataset = self.dataset_type(
            dataparser_outputs=self.eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split="train",
            use_real=self.config.use_real,
        )

        self._fixed_indices_eval_dataloader = ImageRestorationFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.degraded_eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = ImageRestorationRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.degraded_eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    @property
    def fixed_indices_eval_dataloader(self):
        """Returns the fixed indices eval dataloader"""
        return self._fixed_indices_eval_dataloader

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch with the spike

        Returns a Camera instead of raybundle"""
        camera, data = super().next_train(step)
        if "spike_index" in data:
            data["spike"] = self.train_dataset.load_spike(int(data.pop("spike_index")))
        # load the spike to float and cuda
        data["spike"] = data["spike"].float().to(self.device)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch. Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        )
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        if "spike_index" in data:
            data["spike"] = self.eval_dataset.load_spike(int(data.pop("spike_index")))
        data["image"] = data["image"].to(self.device)
        data["spike"] = data["spike"].float().to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        set_camera_index(camera, self.eval_dataset, image_idx)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        return SpikeInputDataset

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split="train",
            use_real=self.config.use_real,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        return self.dataset_type(
            dataparser_outputs=self.eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            split="test",
            use_real=self.config.use_real,
        )
