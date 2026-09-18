import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bad_gaussians.spike_stream import PackedSpikeStream, PackedSpikeStreamSpec
from bad_gaussians.spike_sensor import rgb_to_sensor_signal


class PackedSpikeStreamTest(unittest.TestCase):
    def _make_stream(self, directory: Path) -> Path:
        frames = []
        for index in range(5):
            frame = np.zeros((4, 8), dtype=np.uint8)
            frame[index % 4, :] = 1
            frames.append(frame)
        packed = np.packbits(np.stack(frames).reshape(5, -1), axis=1, bitorder="little")
        path = directory / "stream.dat"
        path.write_bytes(packed.tobytes())
        return path

    def test_reads_centered_window_and_preserves_bit_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_stream(Path(tmp))
            reader = PackedSpikeStream(
                PackedSpikeStreamSpec(path=path, width=8, height=4, window_frames=3)
            )
            window = reader.read_window(2)
            self.assertEqual(window.shape, (3, 4, 8))
            np.testing.assert_array_equal(window[0, 1], np.ones(8))
            np.testing.assert_array_equal(window[1, 2], np.ones(8))
            np.testing.assert_array_equal(window[2, 3], np.ones(8))

    def test_edge_padding_and_bayer_block_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_stream(Path(tmp))
            reader = PackedSpikeStream(
                PackedSpikeStreamSpec(
                    path=path,
                    width=8,
                    height=4,
                    window_frames=3,
                    spatial_reduce=2,
                    value_scale=4.0,
                )
            )
            window = reader.read_window(0)
            self.assertEqual(window.shape, (3, 2, 4))
            self.assertTrue(np.isfinite(window).all())
            # Frame zero is repeated at the left boundary and its 2x2 block
            # sum is divided by the four Bayer sites.
            np.testing.assert_allclose(window[0], window[1])
            self.assertAlmostEqual(float(window[1, 0, 0]), 0.5)

    def test_manifest_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_stream(Path(tmp))
            spec = PackedSpikeStreamSpec.from_manifest(
                {"path": "stream.dat", "width": 8, "height": 4}, Path(tmp)
            )
            self.assertEqual(spec.path, path.resolve())

    def test_sensor_weights_match_bayer_2x2_average(self):
        rgb = torch.tensor([[[[0.2, 0.4, 0.8]]]])
        signal = rgb_to_sensor_signal(rgb, weights=(0.25, 0.5, 0.25))
        torch.testing.assert_close(signal.flatten(), torch.tensor([0.45]))

    def test_sensor_weights_reject_zero_response(self):
        with self.assertRaises(ValueError):
            rgb_to_sensor_signal(torch.ones(1, 1, 1, 3), weights=(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
