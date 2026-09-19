# RateSpikeGS

**Rate-Guided Gaussian Splatting for Spike Stream Reconstruction**

## Installation

The development environment uses NVIDIA V100 GPUs. Other GPU architectures
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

The tiny-cuda-nn installation follows the upstream USP-Gaussian instructions.

## Data

Obtain the synthetic dataset through the
[USP-Gaussian repository](https://github.com/chenkang455/USP-Gaussian) and keep its
directory layout and camera metadata intact. 

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_scene.py \
    --data /path/to/factory --output outputs --preset full
```

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
