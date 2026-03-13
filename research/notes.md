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
