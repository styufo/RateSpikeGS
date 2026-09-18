# Initial Release Validation

Validated on 2026-09-18 in the existing CUDA 11.8 / PyTorch 2.1.2 environment
with NVIDIA V100 GPUs. No datasets or test outputs are included in the repository.

| Check | Result |
| --- | --- |
| Unit tests | 23 passed |
| Python package wheel build | Passed |
| Pinned SkipGS installation from GitHub | Passed |
| Baseline preset launcher, Factory steps 0-2 | Passed |
| Full-method integration, Factory steps 0-7 | Passed |
| Resume from step 7 through absolute step 80 | Passed; final checkpoint is step 80 |
| SkipGS execution in accelerated smoke schedule | 42 skipped updates recorded |
| Render selected Factory views 26 and 27 | GS, Recon-Net, input, reference and metrics exported |
| MFRC implementation versus frozen source | Identical Python AST |
| Recon-Net versus upstream source | Identical Python AST |
| Staged-file whitespace and development-path scan | Passed |

The integration smoke schedule advances the curriculum cutoff to step 2,
densification stop and SkipGS start to step 4, and uses a deliberately permissive
skipping threshold. These settings exercise code paths quickly; they are **not**
the published synthetic preset and establish no quality or speedup claim.

Release integration fixes cover checkpoint selection in directories containing
SkipGS sidecars and absolute-step resume semantics. GPU smoke tests retain
upstream PyPose warnings and the scheduler warning associated with accumulated
updates. Neither caused the checked runs to fail.

A clean-machine installation and full four-scene 30k release reproduction have
not been run as part of this packaging validation.
