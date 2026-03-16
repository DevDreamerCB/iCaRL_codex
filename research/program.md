# AutoResearch Program

## Goal

Continuously improve the BCI cross-subject class-incremental setting in `iCaRL_codex`, with priority:

1. `task3` total accuracy
2. `task2` total accuracy
3. `task3` old-class retention, especially `left hand` and `right hand`

Secondary goals:

- reduce seed variance
- keep the method lightweight and thesis-friendly
- keep memory roughly fixed unless memory itself is the variable under study

## Hard Constraints

- Only modify files inside `iCaRL_codex`
- Do not break the pretrained backbone structure
- Prefer lightweight changes: adapter, prototype inference, loss balancing, replay policy, training schedule
- Do not rely on increasing `memory_size` as the main story

## Main Research Loop

1. Read current `research/state.json`
2. Read `metrics/experiments.csv`
3. Identify current best confirmed line
4. Generate 2-4 nearby candidates
5. Run `screen`: `1 seed + 10 epoch`
6. Promote only promising candidates to `confirm`: `3 seeds + 30 epoch`
7. Update:
   - `metrics/experiments.csv`
   - `metrics/latest.md`
   - `research/results.tsv`
   - `research/notes.md`
   - `research/state.json`
8. If confirm improves best full result, adopt it as the new active line
9. If several nearby candidates fail, change direction but stay close to the current best explanation

## Keep / Discard Rules

Promising screen candidate:

- `task3 >= current_best_full_task3 + 0.5`, or
- `score >= current_best_full_score + 0.25`, or
- `task3 >= current_best_screen_task3 - 0.1` and `task2 >= current_best_screen_task2 - 0.5`

Confirmed improvement:

- improves `task3`, or
- improves score,

without unacceptable collapse in the other main metrics.

Default collapse rule:

- reject if `task2` drops by more than `1.5` and `task3` does not clearly improve

## Current Active Line

Current best confirmed line at the time of writing:

- `LwF=0.15`
- `normalized NME`
- `adapter16`
- `replaymix2`
- `oldweight=2.5`
- `stage_epochs=[10,12,14]`
- `memory_size=36`

## Candidate Priority Near Current Line

1. small prototype-side bias correction
2. small old-class weight adjustments
3. small stage schedule adjustments
4. lightweight NME calibration
5. only then consider a new family of methods

## Direction Change Rule

If `5` nearby candidates fail to beat the current best full confirm, move to a new family:

- prototype-side calibration
- bias correction
- domain-specific normalization
- stronger but still lightweight distillation
