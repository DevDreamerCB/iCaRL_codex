# Replay Directions 2026-03-16

## Goal

Identify replay-related continual learning methods that are most likely to improve this repo under the current constraints:

- keep the pretrained backbone intact
- prefer small changes inside `iCaRL_codex`
- prioritize `task3`, then `task2`
- keep memory budget fixed unless explicitly studying cost-performance tradeoffs

## Current Replay Baseline In This Repo

Current strongest line already includes:

- fixed memory budget (`mem36`)
- herding-style exemplar selection
- `replay_batch_size=2`
- stage3-only prototype refinement (`new_only`)
- hybrid `NME + logits`
- old/new balancing through `oldweight`, `LwF`, and stage3-only feature distillation

This means new replay work should preferably improve one of:

- which samples are replayed
- how replay samples are used during training
- how replay interacts with current-task samples

not simply add another global bias knob.

## Most Relevant Replay Papers

### 1. ReCIL

Why it matters:

- it is directly about cross-subject class-incremental motor imagery
- its replay design is explicitly split into `global` and `local` replay

Most transferable ideas:

- use replay not only for local exemplars, but also for subject-level alignment information
- replay sample selection should consider both representativeness and diversity
- dimensionality reduction can help replay selection when feature-space distances are noisy

Source:

- Yang et al., "ReCIL: Rehearsal-Based Class Incremental Learning for Cross-Subject Motor Imagery Classification", IEEE TBME 2025  
  PDF: [/data1/bochen/cbcontinual/iCaRL_codex/2025_CS-CIL_TBME_Yang.pdf](/data1/bochen/cbcontinual/iCaRL_codex/2025_CS-CIL_TBME_Yang.pdf)

Repo verdict:

- replay selection ideas from this paper were partly tested already
- `subject_kmeans`, `subject_herding`, `subject_diverse_herding` were weak in this codebase
- the remaining promising part is `global replay of alignment information`, not the local replay heuristic itself

### 2. MIR: Maximally Interfered Retrieval

Core idea:

- replay the memory samples whose predictions would be harmed the most by the next update

Source:

- Aljundi et al., "Online Continual Learning with Maximal Interfered Retrieval", NeurIPS 2019  
  https://papers.nips.cc/paper_files/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html

Why it may fit here:

- this repo already uses a replay buffer; MIR changes retrieval, not the backbone
- it matches the user goal of protecting old classes in `task3`

Cost/risk:

- medium
- requires scoring replay candidates with a look-ahead criterion
- more invasive than current prototype calibration

Repo verdict:

- promising, but not the next easiest step
- worth trying after lighter replay-usage methods are exhausted

### 3. DER / DER++

Core idea:

- store old examples together with historical logits and distill them during replay
- DER++ adds a replay label loss on top of logit replay

Source:

- Buzzega et al., "Dark Experience for General Continual Learning", NeurIPS 2020  
  https://papers.nips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf

Why it may fit here:

- this repo already has `prev_model` and LwF machinery
- a DER-like replay target is easier to add than MIR or ASER
- it directly addresses stability of replayed examples

Cost/risk:

- low to medium
- requires storing replay logits or computing them before memory revision

Repo verdict:

- very relevant
- likely the most promising next replay family after current prototype/hybrid tuning

### 4. X-DER

Core idea:

- extend DER with memory revision and future-aware class handling

Source:

- Boschini et al., "Class-Incremental Continual Learning into the eXtended DER-verse", TPAMI / arXiv 2022  
  https://arxiv.org/abs/2201.00766

Why it may fit here:

- this repo is already beyond vanilla replay and already uses post-hoc calibration
- X-DER is directly class-incremental, not only online continual learning

Cost/risk:

- medium to high
- more moving parts than DER++

Repo verdict:

- conceptually strong
- too large as the next immediate patch, but useful as a design reference

### 5. ASER

Core idea:

- score memory samples using adversarial Shapley value, favoring points that preserve old boundaries while challenging the current update

Source:

- Shim et al., "Online Class-Incremental Continual Learning with Adversarial Shapley Value", AAAI 2021  
  https://arxiv.org/abs/2009.00093

Why it may fit here:

- it is a replay selection method, so it respects the pretrained backbone constraint

Cost/risk:

- high
- selection is much heavier than herding and much harder to explain/maintain

Repo verdict:

- not a good next implementation for this repo
- too expensive and too far from the current lightweight strategy

### 6. Gradient-Matching Coresets

Core idea:

- choose replay memory so its gradients match the full dataset gradients

Source:

- Balles et al., "Gradient-Matching Coresets for Rehearsal-Based Continual Learning", arXiv 2022  
  https://arxiv.org/abs/2203.14544

Why it may fit here:

- it targets replay sample quality directly

Cost/risk:

- high
- complicated to plug into this repo and not very thesis-friendly compared with simpler replay heuristics

Repo verdict:

- interesting academically
- not a priority for this project

### 7. RAR: Repeated Augmented Rehearsal

Core idea:

- strengthen rehearsal by repeating and augmenting replay

Source:

- Zhang et al., "A simple but strong baseline for online continual learning: Repeated Augmented Rehearsal", NeurIPS 2022  
  https://arxiv.org/abs/2209.13917

Why it may fit here:

- the current repo already shows that replay usage matters as much as replay selection
- repeated/augmented rehearsal is easy to adapt to EEG if augmentation is class-safe

Cost/risk:

- low to medium
- safer than MIR/ASER

Repo verdict:

- useful as inspiration
- better to adapt the idea as `replay-aware augmentation` instead of copying the exact online OCL procedure

### 8. RehearMixup

Core idea:

- improve rehearsal by doing mixup between current and replay data, or within current/replay subsets

Source:

- Zhang et al., "RehearMixup: Improving rehearsal-based continual learning", Neural Networks 2025  
  https://www.sciencedirect.com/science/article/abs/pii/S0925231225020764

Why it may fit here:

- very easy to plug into the current repo
- acts on replay usage, not replay storage
- matches the current need for lightweight, reversible changes

Cost/risk:

- low
- but class semantics matter for EEG MI data, so mixup can easily blur discriminative motor patterns

Repo verdict:

- worth trying once
- currently under test in this repo as a lightweight replay-aware mixup branch

## Priority Ranking For This Repo

### Highest Priority

1. `DER-lite / DER++-style replay logits`
2. `incremental EA / alignment-memory replay`
3. `replay-aware augmentation or mixup`, but only if class-safe and lightweight

### Medium Priority

1. `MIR` retrieval
2. `RAR`-style repeated augmented rehearsal

### Low Priority

1. `ASER`
2. `gradient-matching coreset replay`
3. full `X-DER` reimplementation

## Current Repo Status

Already tested and weak:

- subject-balanced replay variants
- diverse herding variants
- replay left-right flip augmentation

Currently under test:

- replay-aware mixup on top of the current best `alpha=0.2` hybrid/prototype line

## Recommended Next Replay Experiments

### A. DER-lite for replayed samples

Minimal version:

- store old replay logits at exemplar construction time
- add replay-logit consistency on replayed samples only
- keep current BCE/LwF losses unchanged

Why this is the best next replay family:

- closest to current code
- easiest to explain in a thesis
- directly targets old-class stability on replay data

### B. Alignment-memory replay

Minimal version:

- keep a stage-wise running EA reference per subject group or per stage
- use it only when replayed samples are mixed with current samples

Why this is attractive:

- directly derived from ReCIL
- genuinely cross-subject, not generic image CL

### C. Safer replay mixup variants

If the current mixup branch is weak:

- try `intra-current` only, not `current-replay`
- or mix only within current stage data, while still replaying memory normally

Why:

- RehearMixup reports that not all mixup strategies behave equally well
- EEG MI semantics may dislike cross-memory interpolation
