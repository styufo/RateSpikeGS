import unittest

import torch

from bad_gaussians.image_restoration_dataloader import _load_degraded_sample


class _EagerDataset:
    def __getitem__(self, index):
        return {"image": torch.tensor([float(index)]), "spike": torch.ones(2, 3, 4)}


class _LazyDataset:
    def __init__(self):
        self.loaded = []

    def __getitem__(self, index):
        return {"image": torch.tensor([float(index)]), "spike_index": index + 7}

    def load_spike(self, index):
        self.loaded.append(index)
        return torch.full((2, 3, 4), float(index))


class ImageRestorationDataloaderTest(unittest.TestCase):
    def test_eager_spike_sample(self):
        image, spike = _load_degraded_sample(_EagerDataset(), 2, "cpu")
        torch.testing.assert_close(image, torch.tensor([2.0]))
        torch.testing.assert_close(spike, torch.ones(2, 3, 4))

    def test_lazy_spike_sample(self):
        dataset = _LazyDataset()
        image, spike = _load_degraded_sample(dataset, 3, "cpu")
        self.assertEqual(dataset.loaded, [10])
        torch.testing.assert_close(image, torch.tensor([3.0]))
        torch.testing.assert_close(spike, torch.full((2, 3, 4), 10.0))


if __name__ == "__main__":
    unittest.main()
