# Third-party notices

## USP-Gaussian

Source: https://github.com/chenkang455/USP-Gaussian

Upstream base commit: `359753703314257c453f561138ebe2d2e4caafad`.

The training framework, Recon-Net, camera trajectory modeling, data handling,
and baseline losses derive from USP-Gaussian. Its README displays an MIT license
badge; that revision does not contain a standalone LICENSE file. This release
records that declaration without replacing inherited licensing terms or claiming
ownership of upstream contributions. Consult the upstream authors if your use
requires a consolidated license grant.

Modified inherited files include the training entry point and the model, trainer,
pipeline, data parser, and data manager modules. Changes introduce MFRC, gradient
selection for densification, SkipGS integration, continuous-stream support,
evaluation-camera mapping, and packaging. Recon-Net retains the upstream architecture.

## BAD-Gaussians and Nerfstudio

- https://github.com/WU-CVGL/BAD-Gaussians
- https://github.com/nerfstudio-project/nerfstudio

USP-Gaussian builds on these projects. Existing Apache-2.0 notices are preserved
in inherited source files. The Apache-2.0 license text accompanies this release
in `licenses/Apache-2.0.txt`. Upstream authors retain their copyrights.

## SkipGS

Source: https://github.com/ASU-ESIC-FAN-Lab/SkipGS

Pinned dependency: `0d595f225eb99bd279ae074dd662e01a5ad0fae8`.
License: MIT, Copyright (c) 2026 ESIC-FAN-Lab.

The controller is installed as an external dependency, not vendored. RateSpikeGS
adapts its training integration; the skipping algorithm is credited to SkipGS.

## Other dependencies and datasets

PyTorch, gsplat, tiny-cuda-nn, PyPose, and the other installed dependencies retain
their own licenses. Datasets, model weights, and third-party illustrations are
not redistributed in this repository.
