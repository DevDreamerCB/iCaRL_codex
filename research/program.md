# Overnight Program

## Goal
- Prioritize `task3` total accuracy and old-class retention.
- Preserve `task2` as much as possible.

## Current Best Lines
- `oldweight2`: strong task3 confirm signal
- `oldweight2 + mild age-memory`: strongest task3 screen signal among combined methods

## Night Loop
1. Run one short screen candidate.
2. If `task3` or score is promising, promote to `3 seeds + 30 epochs`.
3. Append results to `metrics/experiments.csv` and `research/results.tsv`.
4. Record keep/reject notes in `research/notes.md`.
5. Commit and push after completed promotions.

## Candidate Priority
1. `oldweight2 + age-memory(0.25)`
2. `oldweight2.5`
3. `memory42 + oldweight2`
4. `memory42 + oldweight2 + age-memory(0.25)`
