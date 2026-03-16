# Research Memory

## Project Goal

Improve cross-subject class-incremental learning for the BCI graduation project.

Priority:

1. `task3`
2. `task2`
3. `task3` old-class retention

## Stable Scenario

- Stage 1 train: subjects `1,2,3`, classes `A,B`
- Stage 2 train: subjects `4,5,6`, classes `B,C`
- Stage 3 train: subjects `7,8,9`, classes `C,D`

Class map:

- `A`: left hand
- `B`: right hand
- `C`: both feet
- `D`: tongue

## Current Best Confirmed Line

- `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm`
- task1/task2/task3: `84.18 / 60.62 / 46.99`

Key ingredients:

- no contrastive loss
- `LwF=0.15`
- `normalized NME`
- `adapter16`
- `replaymix2`
- `oldweight=2.5`
- `stage_epochs=[10,12,14]`
- `memory_size=36`

## What Has Worked

- remove current contrastive loss
- normalized NME
- lightweight adapter
- old-class reweighting
- explicit LwF
- slightly longer later-stage schedule

## What Has Not Worked Well

- current contrastive implementation
- prototype alignment
- embedding-only training
- replay batch too large
- aggressive longer schedules
- simple feature distillation

## Current Open Hypotheses

- prototype-side calibration can still improve task3 without increasing memory
- very small inference bias may be better than larger training changes
- task3 gain is currently more geometry-limited than capacity-limited

## Direction Policy

- start near the current best line
- if 5 nearby candidates fail, switch research family
- prefer prototype-side methods before new PEFT structures

- current best full confirm: `lwf015_normnme_adapter16_mem36_oldweight275_stage101214_confirm` -> 84.18 / 60.65 / 47.19
