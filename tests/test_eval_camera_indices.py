from types import SimpleNamespace

from bad_gaussians.image_restoration_dataloader import set_camera_index


class DummyCamera:
    def __init__(self):
        self.metadata = None


def test_set_camera_index_uses_local_index_without_mapping():
    camera = DummyCamera()
    dataset = SimpleNamespace(_dataparser_outputs=SimpleNamespace(metadata={}))

    set_camera_index(camera, dataset, 3)

    assert camera.metadata["cam_idx"] == 3


def test_set_camera_index_uses_train_local_mapping():
    camera = DummyCamera()
    dataset = SimpleNamespace(
        _dataparser_outputs=SimpleNamespace(
            metadata={
                "global_image_indices": [1, 4, 7],
                "optimizer_train_global_indices": [1, 4, 7],
            }
        )
    )

    set_camera_index(camera, dataset, 2)

    assert camera.metadata["cam_idx"] == 2


def test_held_out_camera_does_not_use_a_training_pose():
    camera = DummyCamera()
    camera.metadata = {"cam_idx": 123}
    dataset = SimpleNamespace(
        _dataparser_outputs=SimpleNamespace(
            metadata={
                "global_image_indices": [4],
                "optimizer_train_global_indices": [1, 7],
            }
        )
    )
    set_camera_index(camera, dataset, 0)
    assert "cam_idx" not in camera.metadata
