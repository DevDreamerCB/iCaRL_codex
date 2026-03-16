# Overnight Notes

- Starting point: `oldweight2_confirm = 85.73 / 58.74 / 42.53`
- Combined screen best so far: `oldweight2 + age-memory(0.5) = 85.65 / 59.26 / 46.45`
- Overly strong age-memory (`1.0`) already looks worse than `0.5`

## 2026-03-13T23:34:50 `lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.42 / 59.57 / 46.26
- score: 58.08
- why: a milder age-aware exemplar budget may keep most of the task3 gain from the combined method while recovering more task2 than power 0.5
- change: true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.0 plus age-aware exemplar memory budgets power 0.25

## 2026-03-13T23:35:27 `lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.42 / 59.57 / 46.26
- score: 58.08
- why: a milder age-aware exemplar budget may keep most of the task3 gain from the combined method while recovering more task2 than power 0.5
- change: true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.0 plus age-aware exemplar memory budgets power 0.25

## 2026-03-14T00:25:52 `lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.42 / 59.57 / 46.26
- score: 58.08
- why: a milder age-aware exemplar budget may keep most of the task3 gain from the combined method while recovering more task2 than power 0.5
- change: true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.0 plus age-aware exemplar memory budgets power 0.25

## 2026-03-14T00:26:26 `lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.42 / 59.57 / 46.26
- score: 58.08
- why: a milder age-aware exemplar budget may keep most of the task3 gain from the combined method while recovering more task2 than power 0.5
- change: true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.0 plus age-aware exemplar memory budgets power 0.25

## 2026-03-14T00:36:21 `lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_confirm`
- verdict: confirm_completed
- mode: full
- task1/task2/task3: 85.65 / 58.46 / 42.4
- score: 55.87
- why: the mild age-memory combination may be the most stable way to add exemplar bias on top of oldweight2 without over-hurting task2
- change: promote combined oldweight2+age-memory(0.25) candidate to 3-seed confirm

## 2026-03-14T00:38:07 `lwf015_replaymix2_adapter16_mem36_oldweight25_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.88 / 60.26 / 46.53
- score: 58.52
- why: if power 2.0 still under-regularizes the oldest class, a slightly stronger old-class weight of 2.5 may push task3 further without the instability of changing replay or memory budgets
- change: true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.5

## 2026-03-14T00:48:11 `lwf015_replaymix2_adapter16_mem36_oldweight25_confirm`
- verdict: confirm_completed
- mode: full
- task1/task2/task3: 85.73 / 58.49 / 42.72
- score: 56.05
- why: oldweight 2.5 may improve task3 retention beyond oldweight2 if the gain is coming from stronger oldest-class protection rather than overfitting
- change: promote old-class BCE power 2.5 candidate to 3-seed confirm

## 2026-03-14T00:49:52 `lwf015_replaymix2_adapter16_mem42_oldweight2_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.19 / 59.72 / 45.72
- score: 57.81
- why: once oldweight2 stabilizes forgetting, a modestly larger memory budget of 42 may improve task2/task3 without the instability seen in earlier sampler variants
- change: true LwF on replaymix2+adapter16 with memory size 42 and old-class BCE power 2.0

## 2026-03-14T00:59:49 `lwf015_replaymix2_adapter16_mem42_oldweight2_confirm`
- verdict: confirm_completed
- mode: full
- task1/task2/task3: 85.73 / 58.36 / 42.95
- score: 56.13
- why: the corrected pipeline may finally benefit from a slightly larger memory when paired with oldweight2 rather than sampler changes
- change: promote memory42+oldweight2 candidate to 3-seed confirm

## 2026-03-14T01:01:31 `lwf015_replaymix2_adapter16_mem42_oldweight2_agemem025_short`
- verdict: promoted_to_confirm
- mode: screen
- task1/task2/task3: 85.19 / 59.95 / 45.72
- score: 57.88
- why: memory42 plus a mild age-aware memory bias may create a better task2/task3 tradeoff than either one alone under the oldweight2 training regime
- change: true LwF on replaymix2+adapter16 with memory size 42, old-class BCE power 2.0, and age-aware exemplar memory power 0.25

## 2026-03-14T01:11:24 `lwf015_replaymix2_adapter16_mem42_oldweight2_agemem025_confirm`
- verdict: confirm_completed
- mode: full
- task1/task2/task3: 85.73 / 58.41 / 42.22
- score: 55.78
- why: a slightly larger memory with mild age bias may be the most thesis-friendly extension of the current oldweight2 line if the short-run gain holds
- change: promote memory42+oldweight2+age-memory(0.25) candidate to 3-seed confirm

## 2026-03-14T10:25:42 `lwf015_normnme_adapter16_mem36_oldweight225_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.57 / 46.86
- score: 58.44
- note: lwf015_normnme_adapter16_mem36_oldweight225_stage101214_confirm

## 2026-03-14T10:30:52 `lwf015_normnme_agenme01_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.64 / 59.85 / 46.97
- score: 58.37
- note: lwf015_normnme_agenme01_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T10:37:50 `lwf015_normnme_agenme02_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 85.03 / 59.21 / 46.64
- score: 58.09
- note: lwf015_normnme_agenme02_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T10:37:50 `lwf015_normnme_radius05_adapter16_mem36_oldweight25_stage101214_short`
- verdict: screen_rejected
- task1/task2/task3: 84.49 / 61.81 / 46.53
- score: 58.71
- note: lwf015_normnme_radius05_adapter16_mem36_oldweight25_stage101214_short

## 2026-03-14T10:37:50 `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.62 / 46.99
- score: 58.52
- note: lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:00:04 `lwf015_normnme_adapter16_mem36_oldweight275_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.65 / 47.19
- score: 58.63
- note: lwf015_normnme_adapter16_mem36_oldweight275_stage101214_confirm

## 2026-03-14T11:00:04 `event_bootstrap`
- bootstrapped family stats from 3 historical events without replaying old notes

## 2026-03-14T11:00:04 `best_update`
- new best full confirm `lwf015_normnme_adapter16_mem36_oldweight275_stage101214_confirm` -> 84.18 / 60.65 / 47.19

## 2026-03-14T11:06:49 `lwf015_normnme_adapter16_mem36_oldweight25_stage101213_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.62 / 47.21
- score: 58.63
- note: lwf015_normnme_adapter16_mem36_oldweight25_stage101213_confirm

## 2026-03-14T11:06:49 `event_run_completed`
- `lwf015_normnme_adapter16_mem36_oldweight25_stage101213_short` (schedule_tune) -> 84.72 / 61.5 / 48.5, score 59.64

## 2026-03-14T11:06:49 `event_run_completed`
- `lwf015_normnme_adapter16_mem36_oldweight25_stage101213_confirm` (schedule_tune) -> 84.18 / 60.62 / 47.21, score 58.63

## 2026-03-14T11:06:51 `lwf01_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.88 / 46.77
- score: 58.48
- note: lwf01_normnme_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:06:51 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:13:49 `lwf015_normnme_agenme005_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.37 / 47.0
- score: 58.45
- note: lwf015_normnme_agenme005_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:13:50 `event_run_completed`
- `lwf015_normnme_agenme005_adapter16_mem36_oldweight25_stage101214_short` (age_nme) -> 84.49 / 61.5 / 48.38, score 59.54

## 2026-03-14T11:13:50 `event_run_completed`
- `lwf015_normnme_agenme005_adapter16_mem36_oldweight25_stage101214_confirm` (age_nme) -> 84.18 / 60.37 / 47.0, score 58.45

## 2026-03-14T11:20:48 `lwf015_normnme_radius025_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 61.21 / 46.5
- score: 58.45
- note: lwf015_normnme_radius025_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:20:48 `event_run_completed`
- `lwf015_normnme_radius025_adapter16_mem36_oldweight25_stage101214_short` (radius_nme) -> 84.49 / 61.73 / 47.49, score 59.16

## 2026-03-14T11:20:48 `event_run_completed`
- `lwf015_normnme_radius025_adapter16_mem36_oldweight25_stage101214_confirm` (radius_nme) -> 84.18 / 61.21 / 46.5, score 58.45

## 2026-03-14T11:20:50 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:27:38 `lwf02_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.36 / 47.26
- score: 58.57
- note: lwf02_normnme_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:27:38 `event_run_completed`
- `lwf02_normnme_adapter16_mem36_oldweight25_stage101214_short` (lwf_tune) -> 84.72 / 61.65 / 48.42, score 59.65

## 2026-03-14T11:27:38 `event_run_completed`
- `lwf02_normnme_adapter16_mem36_oldweight25_stage101214_confirm` (lwf_tune) -> 84.18 / 60.36 / 47.26, score 58.57

## 2026-03-14T11:27:40 `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.62 / 46.99
- score: 58.52
- note: lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:27:40 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:27:51 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:28:01 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:28:04 `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- verdict: confirm_rejected
- task1/task2/task3: 84.18 / 60.62 / 46.99
- score: 58.52
- note: lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm

## 2026-03-14T11:28:08 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:28:18 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:28:29 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:28:39 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:28:50 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:29:01 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:29:11 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:29:22 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:29:32 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:29:43 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:29:53 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:30:04 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:30:14 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:30:25 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:30:35 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:30:46 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:30:57 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:31:07 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:31:18 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:31:28 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:31:39 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:31:49 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:32:00 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-14T11:32:10 `direction_switch`
- switch to `replay_loss_line` with candidate families `['replay_tune', 'lwf_tune', 'balance_tune']` after stale rounds

## 2026-03-14T11:32:21 `direction_switch`
- switch to `normalized_nme_schedule_line` with candidate families `['oldweight_tune', 'balance_tune', 'schedule_tune', 'lwf_tune']` after stale rounds

## 2026-03-14T11:32:32 `direction_switch`
- switch to `prototype_calibration_line` with candidate families `['age_nme', 'radius_nme']` after stale rounds

## 2026-03-15T23:10:00 `screen_confirm_summary`
- current best full remains `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm`
- best full metrics remain `task1=84.18, task2=60.80, task3=48.91, score=59.53`
- `group_bias old_weight=0.5` is a useful balance line, not a new best: full `84.18 / 61.01 / 48.63`
- `group_bias old_weight=0.6` is slightly worse than `0.5`: full `84.18 / 61.01 / 48.59`
- `group_bias old_weight=0.4` has the same short profile as `0.5` (`84.72 / 61.65 / 48.50`), so the group-bias family appears flat below `0.6`
- `prototype neighbor calibration` does not help on top of the current best line:
  - `beta=0.05`: `84.72 / 61.50 / 48.23`
  - `beta=0.10`: `84.72 / 61.50 / 48.19`
  - `beta=0.20`: `84.72 / 61.50 / 48.15`
- `diverse_herding` is useful mainly as a task2-oriented replay variant, but hurts task3 too much under the same memory budget:
  - `lambda=0.05`: `84.72 / 62.27 / 47.69`
  - `lambda=0.02`: `84.72 / 61.73 / 47.88`
- `subject_avg + new_only ptblend` is weaker than the current `global + new_only ptblend`: `84.72 / 61.50 / 48.19`
- `non-hybrid + new_only ptblend` also fails to beat the current hybrid best: `84.72 / 61.50 / 48.26`
- current practical conclusion:
  - keep `global new_only stage3 ptblend` as the main prototype refinement
  - keep `gbias=0.5` as a balanced backup line if task2 matters more
  - do not continue spending compute on neighbor-calibration or subject-averaged ptblend on this branch

## 2026-03-16T09:00:00 `new_families_summary`
- added `overlap_only` scope for current prototype blend to test overlap-class-only prototype refinement
- `overlap_only` on the current best line is weak: `84.72 / 61.50 / 47.80`
- added `exemplar_mode_start_task` to allow stage-specific replay selection experiments
- `stage3-only diverse_herding` is weak: `84.72 / 61.50 / 48.07`
- `stage3-only subject_diverse_herding` is also weak: `84.72 / 61.50 / 48.11`
- moving hybrid NME-logits calibration earlier to `stage2` hurts task2 too much: `84.72 / 59.72 / 48.80`
- current implication:
  - the strongest prototype refinement still targets only the genuinely new class (`new_only`)
  - stage-specific replay diversity does not rescue task3 on top of the current best line
  - keep `hybrid_start_task=3`; do not move hybrid calibration earlier on this branch

## 2026-03-16T09:50:00 `affine2_line_summary`
- moving `task-affine` start from stage3 to stage2 is the first clear new gain after the previous plateau:
  - full `84.18 / 60.80 / 49.04`
- adding `group-bias 0.5` on top of the new `affine2` line improves task2 but does not beat the new best:
  - full `84.18 / 61.01 / 48.85`
- increasing `oldweight` from `3.0` to `3.125` on top of the `affine2` line gives the current best full:
  - full `84.18 / 60.75 / 49.15`
- current implication:
  - keep `task-affine start_task=2` as part of the main line
  - continue searching near the `affine2 + oldweight 3.125` line before switching research families again

## 2026-03-16T12:10:00 `affine2_refine_summary`
- `oldweight=3.25` on the affine2 line is worse than `3.125`: `84.72 / 61.42 / 49.07`
- `stagefd=0.02` is also weaker than the current best: `84.72 / 61.50 / 49.00`
- slightly stronger stage3-only feature distillation helps on the affine2 line:
  - short `84.72 / 61.50 / 49.46`
  - full `84.18 / 60.75 / 49.22`
- current implication:
  - the affine2 line now prefers `oldweight=3.125`
  - the current best stage3 feature distillation is `0.03`, not `0.025`
  - continue near `affine2 + oldweight 3.125 + stagefd 0.03`

## 2026-03-16T12:55:00 `new_best_update`
- FeCAM-style diagonal covariance NME is clearly negative on this EEG feature space: `84.49 / 60.42 / 47.22`
- `hybrid_old_weight=0.5` on the current affine2 best line only ties the previous best and does not improve it
- reducing the `new_only` stage3 prototype-blend alpha from `0.7` to `0.6` improves the current best line:
  - short `84.72 / 61.50 / 49.50`
  - full `84.18 / 60.75 / 49.27`
- current implication:
  - keep the new main line as `affine2 + oldweight 3.125 + stagefd 0.03 + ptblend alpha 0.6`
  - stop spending more time on FeCAM-style covariance NME for this repo

## 2026-03-16T13:35:00 `ptblend_alpha_line_summary`
- the `new_only stage3 prototype-blend alpha` line continues to improve as the blend gets lighter:
  - `alpha=0.5` full: `84.18 / 60.75 / 49.28`
  - `alpha=0.4` full: `84.18 / 60.75 / 49.29`
- the corresponding screens remain strong and do not collapse as alpha decreases:
  - `alpha=0.4` short: `84.72 / 61.50 / 49.54`
  - `alpha=0.3` short: `84.72 / 61.50 / 49.54`
  - `alpha=0.2` short: `84.72 / 61.50 / 49.58`
- current implication:
  - the main line still prefers lighter `new_only` prototype refinement at stage3
  - this is now the highest-value close-range search axis, so keep pushing `alpha` downward before switching to another family
  - `alpha=0.3` and `alpha=0.2` both deserve full confirm

## 2026-03-16T13:45:00 `ptblend_alpha_peak_summary`
- the `new_only stage3 prototype-blend alpha` peak is now localized around `0.15~0.2`
- completed full confirms:
  - `alpha=0.3` full: `84.18 / 60.75 / 49.34`
  - `alpha=0.2` full: `84.18 / 60.75 / 49.36`
- additional screens:
  - `alpha=0.15` short: `84.72 / 61.50 / 49.58`
  - `alpha=0.10` short: `84.72 / 61.50 / 49.50`
- current implication:
  - `alpha=0.2` is the current best full point on this axis
  - `alpha=0.1` already drops, so the repo should stop pushing this parameter downward blindly
  - the next useful action is to validate `alpha=0.15` only if needed, and otherwise switch compute to a genuinely new family

## 2026-03-16T13:55:00 `replay_family_update`
- `weight alignment (WA)` was added as a new calibration family inspired by class-incremental replay literature
- first screen result on the current best line:
  - `84.72 / 61.50 / 49.42`
- verdict:
  - weaker than the current best `ptblend alpha` line
  - do not promote WA to full on this branch
- `replay-aware mixup` was then added as a new replay-usage family
- first screen on the current best `alpha=0.2` line finished weak:
  - `84.72 / 61.50 / 49.11`
- verdict:
  - current replay-aware mixup implementation is negative; do not promote
- supporting literature review for replay methods is now summarized in `research/replay_directions_20260316.md`

## 2026-03-16T14:05:00 `replay_priority_update`
- after rejecting `WA` and the first `replay-aware mixup` branch, the next replay family to prioritize is `DER-lite / replay logits`
- reason:
  - it is the closest replay-family method to the current codebase
  - it directly targets replay sample stability, not only score calibration
  - it is easier to explain than MIR / ASER / full X-DER
- first `DER-lite` short on top of the current best `alpha=0.2` line is now running

## 2026-03-16T14:15:00 `derlite_debug_note`
- the first `DER-lite` short crashed after stage3 training, not during optimization
- root cause:
  - replay exemplars now carry extra stored-logit tensors
  - `_build_bias_calibration_set()` still unpacked replay tensors as if there were only three fields
- fix:
  - make bias-calibration set construction consume only the first three tensors `(x, y, subject_id)`
- action:
  - rerun the same `DER-lite` short after the compatibility fix

## 2026-03-16T14:30:00 `derlite_screen_summary`
- `DER-lite` is not crashing anymore, but the family is currently weak on this repo:
  - original screen (`start_task=2`, `lambda=0.5`): `84.72 / 59.26 / 48.92`
  - stage3-only screen (`lambda=0.5`): `84.72 / 61.50 / 49.00`
  - stage3-only screen (`lambda=0.2`): `84.72 / 61.50 / 49.27`
  - stage3-only screen (`lambda=0.1`): `84.72 / 61.50 / 49.07`
- verdict:
  - stage3-only is clearly better than starting DER-lite at stage2
  - but even the best stage3-only setting is still below the current best short line (`49.58`)
  - do not promote DER-lite to full on this branch

## 2026-03-16T14:35:00 `alignment_memory_replay_start`
- after the weak DER-lite screens, the next replay family is now `alignment-memory replay`
- current implementation strategy:
  - store raw exemplars in parallel with aligned exemplars
  - when enabled from stage3 onward, re-align raw replay exemplars using the current stage's global EA reference before replay
- first screen on top of the current best `alpha=0.2` line is now running

## 2026-03-16T14:55:00 `alignment_memory_replay_summary`
- `alignment-memory replay` has mixed but insufficient evidence on this repo:
  - `stage3-start` screen on the current best line: `84.72 / 61.50 / 49.54`
  - same family combined with `ptblend alpha=0.15`: `84.72 / 61.50 / 49.54`
  - `stage2-start` screen collapses badly: `84.72 / 59.57 / 39.89`
- verdict:
  - the family has some signal when restricted to stage3
  - but it still does not beat the strongest current short point (`49.58`)
  - moving it earlier to stage2 is clearly harmful

## 2026-03-16T15:00:00 `global_kmeans_replay_start`
- after the weak alignment-memory replay results, the next replay-selection family is `global_kmeans`
- current implementation strategy:
  - replace herding only for newly added classes from stage3 onward
  - choose replay exemplars by global feature-space k-means centers within each class
- reason:
  - this is the cleanest way to test the ReCIL-style "representative + diverse" replay idea without forcing subject balancing
- first screen on top of the current best `alpha=0.2` line is now running

## 2026-03-16T15:10:00 `global_kmeans_replay_summary`
- `global_kmeans` replay selection is also competitive but not enough:
  - `84.72 / 61.50 / 49.50`
- verdict:
  - stronger than many rejected replay families
  - still below the current strongest short point (`49.58`)
- next action:
  - test `pca_kmeans` as the closest replay-selection variant to ReCIL's dimensionality-reduced local replay idea

## 2026-03-16T15:20:00 `pca_kmeans_replay_summary`
- `pca_kmeans` ends up essentially tied with `global_kmeans`:
  - `84.72 / 61.50 / 49.50`
- verdict:
  - dimensionality reduction does not rescue the replay-selection family in this repo
  - the ReCIL-style replay-selection branch is now close to exhausted under fixed memory and the current backbone
- next action:
  - stop spending compute on more replay-selection variants
  - move to replay-usage families instead

## 2026-03-16T15:35:00 `hard_replay_sampling_summary`
- implemented a lightweight MIR-style replay family:
  - store replay logits
  - estimate replay hardness by current-model mismatch against stored logits
  - sample replay exemplars with higher probability when their mismatch is larger
- first screens:
  - `hardreplay power=0.5, stage3-only`: `84.72 / 61.50 / 49.58`
  - `hardreplay power=1.0, stage3-only`: `84.72 / 61.50 / 49.58`
- verdict:
  - the family is stable after the replay-logit plumbing fix
  - it is now competitive with the strongest current replay families
  - it ties the best current prototype-blend neighborhood on `task3` while keeping `task2` at `61.50`
- next action:
  - promote the simpler `power=1.0` version to full confirm
  - in parallel, test `RAR-lite` style repeated rehearsal from stage3 onward as a separate replay-usage family

## 2026-03-16T14:45:00 `alignment_memory_replay_first_signal`
- first `alignment-memory replay` screen on top of the current best `alpha=0.2` line:
  - `84.72 / 61.50 / 49.54`
- verdict:
  - this is stronger than the failed DER-lite family and competitive with the best prototype-blend neighborhood
  - it still does not beat the current strongest short point (`49.58`), so it is not enough by itself
- next action:
  - test whether `alignment-memory replay` synergizes with the stronger `ptblend alpha=0.15` neighborhood instead of the `alpha=0.2` baseline
