"""Lazy readers for packed binary spike-camera streams.

SpikeGS stores one continuous little-endian bit-packed stream per scene.  The
reader deliberately returns a short temporal window instead of materialising
one file per camera view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PackedSpikeStreamSpec:
    path: Path
    width: int
    height: int
    window_frames: int = 177
    spatial_reduce: int = 1
    reverse_vertical: bool = False
    bitorder: str = "little"
    value_scale: float = 1.0
    boundary_mode: str = "edge"

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any], base_dir: Path) -> "PackedSpikeStreamSpec":
        path = Path(manifest["path"])
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return cls(
            path=path,
            width=int(manifest["width"]),
            height=int(manifest["height"]),
            window_frames=int(manifest.get("window_frames", 177)),
            spatial_reduce=int(manifest.get("spatial_reduce", 1)),
            reverse_vertical=bool(manifest.get("reverse_vertical", False)),
            bitorder=str(manifest.get("bitorder", "little")),
            value_scale=float(manifest.get("value_scale", 1.0)),
            boundary_mode=str(manifest.get("boundary_mode", "edge")),
        )

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("spike stream dimensions must be positive")
        if self.width % 8 != 0:
            raise ValueError("packed spike width must be divisible by 8")
        if self.window_frames <= 0 or self.window_frames % 2 == 0:
            raise ValueError("window_frames must be a positive odd number")
        if self.spatial_reduce <= 0:
            raise ValueError("spatial_reduce must be positive")
        if self.height % self.spatial_reduce or self.width % self.spatial_reduce:
            raise ValueError("spatial_reduce must divide both dimensions")
        if self.bitorder not in ("little", "big"):
            raise ValueError("bitorder must be little or big")
        if self.boundary_mode not in ("edge", "zero"):
            raise ValueError("boundary_mode must be edge or zero")
        if self.value_scale <= 0:
            raise ValueError("value_scale must be positive")


class PackedSpikeStream:
    """Read bit-packed spikes from a continuous stream on demand."""

    def __init__(self, spec: PackedSpikeStreamSpec):
        spec.validate()
        self.spec = spec
        self.bytes_per_frame = spec.height * spec.width // 8
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        size = spec.path.stat().st_size
        if size % self.bytes_per_frame:
            raise ValueError(
                f"stream size {size} is not divisible by packed frame size {self.bytes_per_frame}"
            )
        self.frame_count = size // self.bytes_per_frame
        self._packed = np.memmap(spec.path, dtype=np.uint8, mode="r").reshape(
            self.frame_count, self.bytes_per_frame
        )

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        reduce = self.spec.spatial_reduce
        return (
            self.spec.window_frames,
            self.spec.height // reduce,
            self.spec.width // reduce,
        )

    def _decode(self, start: int, end: int) -> np.ndarray:
        packed = np.asarray(self._packed[start:end])
        decoded = np.unpackbits(packed, axis=1, bitorder=self.spec.bitorder)
        decoded = decoded.reshape(end - start, self.spec.height, self.spec.width)
        if self.spec.reverse_vertical:
            decoded = decoded[:, ::-1]
        return decoded

    def read_window(self, center_frame: int) -> np.ndarray:
        """Return a normalized [T,H,W] float32 window around ``center_frame``."""
        half = self.spec.window_frames // 2
        requested = np.arange(center_frame - half, center_frame + half + 1)
        valid = (requested >= 0) & (requested < self.frame_count)
        if self.spec.boundary_mode == "zero":
            decoded = np.zeros(
                (self.spec.window_frames, self.spec.height, self.spec.width), dtype=np.uint8
            )
            if valid.any():
                valid_positions = np.flatnonzero(valid)
                decoded[valid_positions] = self._decode(
                    int(requested[valid].min()), int(requested[valid].max()) + 1
                )[valid_positions - valid_positions[0]]
        else:
            clipped = np.clip(requested, 0, self.frame_count - 1)
            start, end = int(clipped.min()), int(clipped.max()) + 1
            decoded = self._decode(start, end)
            decoded = decoded[clipped - start]

        reduce = self.spec.spatial_reduce
        if reduce > 1:
            # The 2x2 Bayer reduction is intentionally performed before any
            # temporal averaging, preserving the sensor's firing statistics.
            decoded = decoded.reshape(
                decoded.shape[0],
                self.spec.height // reduce,
                reduce,
                self.spec.width // reduce,
                reduce,
            ).sum(axis=(2, 4), dtype=np.uint16)
            decoded = decoded.astype(np.float32) / self.spec.value_scale
        else:
            decoded = decoded.astype(np.float32) / self.spec.value_scale
        return decoded
