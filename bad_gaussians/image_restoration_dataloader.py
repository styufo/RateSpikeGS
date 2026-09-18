"""
Image Restoration Dataloaders.
"""

from typing import Dict, Optional, Tuple, Union

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader, FixedIndicesEvalDataloader
from nerfstudio.utils.misc import get_dict_to_torch


def set_camera_index(camera: Cameras, dataset: InputDataset, image_idx: int) -> None:
    """Map dataset IDs to train-local pose parameters; held-out poses stay fixed."""
    camera_idx = image_idx
    dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
    if dataparser_outputs is not None:
        global_indices = dataparser_outputs.metadata.get("global_image_indices")
        if global_indices is not None:
            camera_idx = int(global_indices[image_idx])
            train_indices = dataparser_outputs.metadata.get("optimizer_train_global_indices")
            if train_indices is None:
                raise ValueError("Evaluation requires the training camera index mapping")
            if camera_idx not in train_indices:
                if camera.metadata is not None:
                    camera.metadata.pop("cam_idx", None)
                return
            camera_idx = list(train_indices).index(camera_idx)
    if camera.metadata is None:
        camera.metadata = {}
    camera.metadata["cam_idx"] = camera_idx


def _load_degraded_sample(
    dataset: InputDataset, image_idx: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the TFP image and either an eager or lazy spike sample."""
    degraded_batch = dataset[image_idx]
    if "spike_index" in degraded_batch:
        load_spike = getattr(dataset, "load_spike", None)
        if load_spike is None:
            raise TypeError("a dataset returning spike_index must implement load_spike")
        spike = load_spike(int(degraded_batch["spike_index"]))
    else:
        spike = degraded_batch["spike"]
    return degraded_batch["image"], spike.float().to(device)


class ImageRestorationRandIndicesEvalDataloader(RandIndicesEvalDataloader):
    """eval_dataloader that returns random images.

    Args:
        input_dataset: Ground-truth images for evaluation
        degraded_dataset: Corresponding training images with degradation.
        device: Device to load data to.
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        degraded_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        self.degraded_dataset = degraded_dataset

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        batch["degraded"], batch["spike"] = _load_degraded_sample(
            self.degraded_dataset, image_idx, self.device
        )
        batch["image"] = batch["image"].to(self.device)
        assert isinstance(batch, dict)
        set_camera_index(camera, self.input_dataset, image_idx)
        return camera, batch


class ImageRestorationFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):
    """fixed_indices_eval_dataloader that returns a fixed set of indices.

    Args:
        input_dataset: Ground-truth images for evaluation
        degraded_dataset: Corresponding training images with degradation.
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        degraded_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, image_indices, device, **kwargs)
        self.degraded_dataset = degraded_dataset

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        batch["degraded"], batch["spike"] = _load_degraded_sample(
            self.degraded_dataset, image_idx, self.device
        )
        batch["image"] = batch["image"].to(self.device)
        assert isinstance(batch, dict)
        set_camera_index(camera, self.input_dataset, image_idx)
        return camera, batch
