# Reproducibility

## Shared synthetic preset

The release keeps the full USP image reconstruction network, temporal reversal,
multi-window reblur supervision, and 13 virtual camera samples. Linear camera
trajectory interpolation and the inherited optimizer schedules are retained.
The seed is 425; it is **not reset after model and dataset setup**. This matters
because initialization and data loading consume random numbers. CUDA operations
and hardware differences can still produce different optimization trajectories.

The default all-view protocol includes every supplied view in both training and
evaluation. Do not describe these results as held-out novel-view scores. For
other protocols, set `eval_mode` explicitly and report the split.

## Components

MFRC uses temporal window lengths 4, 8, 16, and 32 in the 97-sample comparison
interval, spatial reduction by 4, overlapping windows, and cumulative weight 0.1.
The overall MFRC loss weight is 0.1. Predicted grayscale rates use RGB weights
0.299, 0.587, and 0.114. Temporal interpolation aligns rendered virtual views with
the observation interval; it does not create additional observed spikes.

The curriculum uses the norm of the total screen-space gradient before step
1,000 and the norm of total minus weighted MFRC gradient thereafter. This changes
densification statistics only. It does not turn off MFRC in the optimization loss.
The default densification stop is step 15,000.

SkipGS starts at step 15,000, uses threshold 0, EMA decay 0.95, warmup 500, and
the controller's automatic minimum backward ratio. Camera gradient accumulation
commit steps are forced to run backward. Skipped iterations still run the forward
pass and advance learning-rate schedules. SkipGS does not prune Gaussian primitives.

## Checkpoints and evaluation

The launcher interprets `--steps` as the absolute final step. Resume accounts for
Nerfstudio 1.0.3's additional-iteration semantics. Use matching configuration
values when resuming. Random-number states are not checkpointed.

GS renders and Recon-Net outputs are different predictions; report them separately.
Evaluation uses the original fixed twofold intensity scaling and clamping for
both. The renderer uses the same train-local camera index mapping as evaluation;
held-out cameras have no fitted training-camera offset.

The training logger inherits proxy-reference diagnostics for real streams. These
are not sharp-GT quality scores. The public renderer deliberately omits those
scores for `use_real` runs.

## Release scope

This is a curated source release, not an export of the research working tree.
It excludes exploratory SNN/2DGS/topology variants, datasets, checkpoints,
private experiment scripts, visualization tone adjustments, cloud tooling,
and machine-specific paths. The core synthetic method was extracted from the
frozen full-method implementation; the data adapter includes the later
train-local evaluation-camera mapping fix.

Focused tests cover spike decoding, data splits, camera indices, MFRC gradients,
and curriculum selection. Short GPU integration tests check execution paths;
they do not substitute for a new four-scene 30k reproduction of the release.
