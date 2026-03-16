# Experiment Flow Summary

Date: `2026-03-15`

## Goal

Improve cross-subject class-incremental learning performance for:

- `task2` total accuracy
- `task3` total accuracy
- especially `task3` old-class retention

Main scenario:

- stage 1 train: subjects `1,2,3`, classes `A,B`
- stage 2 train: subjects `4,5,6`, classes `B,C`
- stage 3 train: subjects `7,8,9`, classes `C,D`

Class map:

- `A`: left hand
- `B`: right hand
- `C`: both feet
- `D`: tongue

## Current Best Results

### Best Full Confirm So Far

Run:

- `lwf0175T1375_normnmehybrid_s3old04_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm`

Metrics:

- `task1 = 85.72`
- `task2 = 59.65`
- `task3 = 48.70`
- `score = 59.39`

Interpretation:

- this is the current best overall full confirm
- it improves `task3` beyond the earlier non-hybrid lines
- `task2` is still below the strongest pre-hybrid line

### Strongest Pre-Hybrid Full Reference

Run:

- `lwf0175T15_normnme_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm`

Metrics:

- `task1 = 84.18`
- `task2 = 60.75`
- `task3 = 48.38`
- `score = 59.25`

Interpretation:

- stronger `task2`
- slightly weaker `task3`
- important as the cleanest non-hybrid baseline for the late-stage recipe

### Best Low-Memory Full Reference

Run:

- `lwf0175T15_normnme_adapter16_mem30_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm`

Metrics:

- `task1 = 84.18`
- `task2 = 60.80`
- `task3 = 47.88`
- `score = 59.02`

Interpretation:

- useful storage/performance tradeoff
- lower replay memory hurts `task3`, but not catastrophically

## Experiment Timeline

### 1. Baseline Understanding

Confirmed:

- training-time validation used `fc logits`
- final report used exemplar-mean / `NME`
- this mismatch later became one of the key research directions

### 2. Early Fast Ablations

Tried:

- remove contrastive loss
- embedding-only finetuning
- prototype alignment
- prompt / LoRA / BN / affine / adapter variants

Conclusions:

- removing contrastive loss was clearly helpful
- most early architectural variants produced short-run signals but failed in full confirm

### 3. Corrected Pipeline Phase

Important event:

- fixed experiment-flow / loader issues so later results are more trustworthy than early ones

Main gains started to come from:

- `LwF`
- old-class reweighting
- `replaymix2`
- `adapter16`
- `normalized NME`

### 4. Strong Mainline Formation

The most effective recipe before hybrid fusion became:

- `adapter16`
- `task-affine from stage3`
- `oldweight` tuned upward
- `stage3-only feature distillation`
- `stage_epochs = 10,12,12`
- `normalized NME`

This pushed `task3` from the earlier `37-42` range up to `46+` and then `48+`.

### 5. Exemplar / Replay Variants

Tried and rejected:

- replay left-right flip with label swap
- subject-balanced k-means exemplar selection
- subject-balanced herding
- subject-diverse herding
- PCA herding
- age-memory / age-replay variants as mainline components

Typical pattern:

- some short improvements
- full confirm usually failed, especially on `task2`

### 6. Prototype / Calibration Variants

Tried and mostly rejected:

- age-NME
- radius-NME
- group-bias calibration alone

Useful outcome:

- `normalized NME` remained clearly valuable
- heavier prototype calibration variants did not beat the simpler strong line

### 7. Hybrid NME-Logits Phase

Reason:

- training optimizes `fc logits`
- final evaluation uses `NME`
- this mismatch looked like a real bottleneck

Implemented:

- calibrated `NME + logits` hybrid scoring
- later restricted to `stage3-only hybrid`

Key results:

- full hybrid: `85.72 / 59.31 / 48.59`
- stage3-only hybrid: `85.72 / 59.62 / 48.59`
- stage3-only hybrid + lighter calibration objective + tuned temperature:
  - `85.72 / 59.65 / 48.70`

Conclusion:

- hybrid scoring is the first new late-stage direction that clearly improved `task3` in full confirm
- but the `task2` tradeoff remains the main unresolved issue

### 8. Schedule / Cost / Fine-Tuning Around Hybrid

Tried:

- `memory = 30`, `32`
- lower / higher `oldweight`
- lower `feature distill`
- `T = 1.375`
- `10,13,11` schedule

Conclusions:

- `memory=30/32` is useful as cost/performance reference, but not the top line
- `T=1.375` was helpful inside the hybrid line
- `10,13,11` recovers `task2` somewhat, but loses too much `task3`

### 9. Subject-Class Alignment

New lightweight direction:

- same-class cross-subject feature alignment during training

Short results:

- `lambda=0.1`: `85.42 / 60.19 / 47.72`
- `lambda=0.05`: `85.19 / 60.03 / 46.30`

Conclusion:

- it helps `task2`
- but hurts `task3`
- currently rejected as a mainline method

## Effective Components

These are the components that repeatedly survived confirm:

- remove contrastive loss
- `LwF`
- stronger old-class BCE weighting
- `replaymix2`
- `adapter16`
- `task-affine` from `stage3`
- `normalized NME`
- `stage3-only feature distillation`
- `stage_epochs = 10,12,12`
- `stage3-only hybrid NME-logits`
- `T = 1.375` on the best hybrid line

## Rejected Or Low-Value Directions

- original contrastive loss
- embedding-only finetuning
- prototype alignment
- prompt / LoRA / BN variants tried so far
- replay left-right flip
- subject-balanced replay selection variants
- PCA herding
- age-NME / radius-NME
- current subject-class alignment implementation
- pushing memory size as the main source of gain

## Current Open Problem

The current strongest line improves `task3`, but still leaves a tradeoff:

- best hybrid line: better `task3`
- best non-hybrid line: better `task2`

So the main unresolved question is:

- how to keep the hybrid line's `task3` gain while recovering more `task2`

## Recommended Next Steps

1. Finish the current hybrid-neighborhood search only if it directly targets the `task2` drop.
2. Stop spending more budget on replay-sample selection variants unless a new paper gives a much cleaner idea.
3. Move to lightweight prototype refinement that does not touch the backbone.
4. Keep using `screen -> confirm`, because many short gains have failed in full confirm.

## File References

- current best metrics: [latest.md](/data1/bochen/cbcontinual/iCaRL_codex/metrics/latest.md)
- full experiment history: [experiments.csv](/data1/bochen/cbcontinual/iCaRL_codex/metrics/experiments.csv)
- concise run history: [results.tsv](/data1/bochen/cbcontinual/iCaRL_codex/research/results.tsv)
- running notes: [notes.md](/data1/bochen/cbcontinual/iCaRL_codex/research/notes.md)
