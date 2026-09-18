# RateSpikeGS

**Rate-Guided Gaussian Splatting for Spike Stream Reconstruction**

RateSpikeGS builds on [USP-Gaussian](https://github.com/chenkang455/USP-Gaussian)
to jointly optimize a Gaussian scene, camera trajectories, and a spike image
reconstruction network. This repository contains training and rendering code,
synthetic experiment presets, and focused tests. Datasets and checkpoints are not included.

## Method

- **Multi-scale firing-rate consistency (MFRC):** compares observed and rendered
  firing statistics over overlapping temporal windows, with a cumulative residual term.
- **Densification curriculum (DC):** includes MFRC screen-space gradients in early
  densification statistics and removes their contribution after step 1,000.
  MFRC remains active for continuous parameter updates.
- **Gaussian skipping:** integrates the controller from
  [SkipGS](https://github.com/ASU-ESIC-FAN-Lab/SkipGS), starting after densification
  stops at step 15,000. It skips selected backward updates, not forward rendering.

Recon-Net is inherited from USP-Gaussian. The internal `bad_gaussians` namespace
is retained for compatibility with inherited code and checkpoint configuration objects.

## Installation

The development environment uses Linux, Python 3.9, PyTorch 2.1.2, CUDA 11.8,
Nerfstudio 1.0.3, gsplat 0.1.11, and NVIDIA V100 GPUs. A CUDA toolkit and a
compatible C++ compiler are needed to build extensions. Other GPU architectures
require compatible CUDA extension builds.

```bash
git clone https://github.com/styufo/RateSpikeGS.git
cd RateSpikeGS
conda create -n ratespikegs python=3.9 -y
conda activate ratespikegs
pip install --upgrade pip setuptools wheel
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip install ninja
pip install 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'
pip install -r requirements.txt
pip install --no-deps -e .
```

The tiny-cuda-nn installation follows the upstream USP-Gaussian instructions;
it belongs to the inherited Nerfstudio environment rather than the Gaussian
representation itself. The release is tested in the existing research environment,
not yet in a newly provisioned environment from these commands.

## Data

Obtain the synthetic dataset through the
[USP-Gaussian repository](https://github.com/chenkang455/USP-Gaussian) and keep its
directory layout and camera metadata intact. Each scene includes `transforms.json`,
the point cloud referenced by the metadata, `spike_data`, and `sharp_data`.
The four scenes are Wine, Tanabata, Factory, and Outdoorpool.
Dataset licensing and access remain with the original providers.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_scene.py \
    --data /path/to/factory --output outputs --preset full
```

The default final step is 30,000 (steps 0 through 30,000, following upstream
indexing). A single GPU is used. No development-server-specific paths are needed.

| Preset | MFRC | DC | Gaussian skipping |
| --- | --- | --- | --- |
| `baseline` | No | No | No |
| `mfrc` | Yes | No | No |
| `mfrc_dc` | Yes | Yes | No |
| `full` | Yes | Yes | Yes |

Without DC, inherited Gaussian splitting and cloning remain enabled and use
joint loss gradients. `configs/synthetic.json` records shared hyperparameters.
Use `--config other.json` for an alternative configuration or `--dry-run`
to inspect the generated command. `python train.py --help` lists low-level options.

Resume with an **absolute final step**, not a number of additional steps:

```bash
python scripts/train_scene.py --data /path/to/factory --preset full \
    --resume outputs/factory/full/bad-gaussians/TIMESTAMP/nerfstudio_models \
    --steps 30000
```

Keep the original preset and configuration when resuming. Checkpoints include
optimizer state; the SkipGS controller is saved alongside them. Exact random
number generator state is not restored, so resume is not bitwise equivalent to
an uninterrupted run.

## Rendering and Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
    --config outputs/factory/full/bad-gaussians/TIMESTAMP/config.yml \
    --output renders/factory --indices 26 27
```

Omit `--indices` to evaluate the complete configured split. Each view contains
`gs.png`, `recon.png`, `input.png`, and `gt.png`; `metrics.json` contains per-view
and mean metrics. `--data` and `--checkpoint-dir` relocate downloaded experiments.
Only load configurations and checkpoints from trusted sources.

The synthetic preset uses **all views for training and evaluation**, matching
the reconstruction protocol used in these experiments. This is not a held-out
novel-view benchmark. GS and Recon-Net predictions both use the inherited fixed
`clamp(2 * prediction, 0, 1)` exposure convention, not per-image tone matching.
`rgb_*` metrics refer to GS renders; `spike_*` metrics refer to Recon-Net outputs.

## Real Streams (Experimental)

`scripts/prepare_spikegs_real.py` converts the supported SpikeGS real sequence
layout, COLMAP cameras, TFP images, and packed raw stream into a lazy-loading
manifest. See `python scripts/prepare_spikegs_real.py --help`.
This adapter is experimental: verify raw dimensions, timestamps, Bayer layout,
and calibration for each sequence before training. It is not a universal raw
spike decoder. Real data without aligned sharp ground truth cannot support
GT-based PSNR/SSIM/LPIPS claims. The renderer labels the reference as `proxy.png`
and omits reference-based metrics for real-data runs.

## Tests

```bash
python -m pytest -q
```

See [reproducibility notes](docs/REPRODUCIBILITY.md) for configuration details and
release validation scope.

## Acknowledgments and Licensing

This work builds on USP-Gaussian, BAD-Gaussians, Nerfstudio, gsplat, and SkipGS.
Gaussian skipping integrates SkipGS rather than introducing an independent
skipping algorithm. Please cite the respective works when using their contributions;
the original project pages provide bibliographic entries.
Licensing scope and upstream provenance are recorded in
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
