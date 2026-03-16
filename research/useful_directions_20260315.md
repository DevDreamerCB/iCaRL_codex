# Useful Directions 2026-03-15

## Current Strong Line

Current best full confirm in this repo:

- `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm`
- `task1=84.18`
- `task2=60.80`
- `task3=48.91`
- `score=59.53`

Current strongest newer short signal under active validation:

- `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_short`
- `task1=84.72`
- `task2=61.50`
- `task3=48.80`
- `score=59.58`

## Directions Already Proven Useful

These are the directions that repeatedly helped and are worth keeping:

- remove supervised contrastive loss
- normalized NME
- stage3-only hybrid NME-logits calibration
- task adapter (`dim=16`)
- task affine from stage3
- stage3-only feature distillation
- tuned old-class weighting
- tuned LwF
- fixed memory budget (`mem36`) with replaymix2

## Directions With Weak Or Negative Evidence

These were tested and are not worth prioritizing now:

- prototype alignment
- embedding-only tuning
- task BN
- most prompt / LoRA variants
- replay left-right flip augmentation
- subject-kmeans / subject-herding / subject-diverse exemplar selection
- PCA herding
- subject-class alignment loss
- full-run prototype blend on non-hybrid line
- subject-averaged prototype blend

## Literature-Informed Directions That Match This Project

### 1. Bias correction / calibration

Why it matches:

- this project already shows a mismatch between training-time logits and final NME-based classification
- later tasks often improve `task3` at the cost of `task2`, which is exactly the kind of bias-tradeoff calibration methods target

Evidence:

- BiC explicitly argues that the last FC layer is biased toward new classes and corrects this with a simple linear calibration model
- T-CIL shows that post-hoc calibration in class-incremental learning can be improved even under limited memory

Sources:

- BiC / Large Scale Incremental Learning:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html
- T-CIL:
  https://arxiv.org/abs/2503.22163

Practical implication here:

- keep preferring lightweight post-hoc or stage3-only calibration over heavy architectural changes

### 2. Lightweight plug-in adaptation on top of a strong frozen/pretrained backbone

Why it matches:

- your pretrained EEG backbone is already strong
- the repo consistently benefits from small task-specific modules instead of larger rewrites

Evidence:

- AANets improves several class-incremental baselines as a plug-in architecture with relatively small parameter overhead and works well under strict memory control

Source:

- Adaptive Aggregation Networks for CIL:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Adaptive_Aggregation_Networks_for_Class-Incremental_Learning_CVPR_2021_paper.pdf

Practical implication here:

- continue favoring small stage-specific modules, calibrated fusion, or lightweight prototype refinement

### 3. Prototype calibration is promising, but must be constrained

Why it matches:

- current experiments show prototype blend can help, but naive global blending hurts or is unstable
- the better variants so far are the more constrained ones: stage3-only, and now `new_only`

Evidence:

- ConCM highlights prototype bias as a key failure mode and proposes memory-aware prototype calibration
- ProNECL, although a non-exemplar EEG method, also centers the solution around cross-subject prototype memory and alignment

Sources:

- ConCM:
  https://arxiv.org/abs/2506.19558
- ProNECL:
  https://arxiv.org/abs/2511.20696

Practical implication here:

- prototype refinement should remain local and targeted:
  - stage3-only
  - new-only
  - memory-aware
  - not full global replacement

## Current Working Hypothesis

The most promising unresolved line is:

- `stage3-only prototype refinement`
- applied only to the genuinely new class (`new_only`)
- combined with calibrated hybrid NME-logits fusion
- combined with lower oldweight (`3.0`) to keep `task2`
- paired with sharper LwF temperature (`T=1.5`)

Reason:

- this is the first prototype-based variant that improved `task2` strongly without collapsing `task3`
- it is also easier to explain in a thesis than a large architectural rewrite

## Immediate Next Steps

1. Keep `new_only stage3 ptblend + T=1.5` as the new primary line.

2. Search only in a very small neighborhood around this line:
- oldweight
- stage3 feature distillation strength
- hybrid calibration weight

3. If close-range tuning stalls:
- stop spending more time on prototype blend variants
- switch back to calibration-only directions, especially lightweight stage3 bias correction

## New Calibration Family Under Test

### Weight Alignment (WA)

Why it matches:

- the current repo already shows that classifier calibration matters
- `group-bias` helps task2, but it is a broad score-level correction; WA is a cleaner classifier-head correction
- WA is a standard class-incremental technique for old/new classifier norm imbalance, and it can be inserted here without changing the backbone or replay logic

Implementation choice in this repo:

- apply WA only to `fc` scores at evaluation time
- start from `stage3` only
- keep it compatible with the existing `hybrid NME-logits` pipeline

Practical implication here:

- if WA helps on top of the current best line, it becomes the first genuinely new post-hoc calibration family after the recent `ptblend alpha` gains
- if it fails, the repo can reject it cleanly without touching the training loop

## Evening Update

- current best full is still:
  - `84.18 / 60.80 / 48.91`
  - `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm`
- `group-bias` has become the strongest balance-oriented backup family:
  - `old_weight=0.5` full: `84.18 / 61.01 / 48.63`
  - `old_weight=0.6` full: `84.18 / 61.01 / 48.59`
- rejected on top of the current best line:
  - prototype neighbor calibration
  - subject-averaged `new_only` prototype blend
  - diverse herding as a primary performance line
- working implication:
  - keep `global + new_only + stage3-only ptblend` as the main prototype refinement
  - treat `group-bias` as the best current task2-oriented balancing knob
  - next high-value direction is to combine the best calibration ideas with the strongest non-hybrid line, instead of continuing to stack more local prototype refinements on the hybrid line

## Decision Rule

Keep a new direction only if it improves at least one of:

- stage 2 total accuracy
- stage 3 total accuracy
- stage 3 old-class stability

without an unacceptable drop on the other main metrics.
