# iCaRL Codex Research Rules

## Scope
All research work for this project should prefer edits inside this directory only:
- `/data1/bochen/cbcontinual/iCaRL_codex`

Do not modify sibling folders such as:
- `iCaRL_cb`
- `iCaRL_new`
- `DoRA`
- `VPT`
- other experiment folders

## Objective
Continuously improve accuracy for:
- `task1`
- `task2`
- `task3`

Priority order:
1. `task3`
2. `task2`
3. `task1`

If a change improves `task1` but hurts `task2` or `task3`, it is usually not a good change.

## Model Constraints
- keep the pretrained backbone structure intact as much as possible
- do not redesign the original pretrained model wholesale
- prefer parameter-efficient tuning on top of the pretrained model:
  - adapter
  - LoRA
  - prompt tuning
  - task-specific lightweight modules

## Preferred Directions
- prioritize tuning around the embedding layer, because this pretrained model already works well when mainly fine-tuning embedding-related parameters
- task-specific embedding modules are allowed, including ideas such as per-task embeddings or lightweight MoE-style routing around the embedding stage
- for EEG subject shift and noise, consider transfer learning, feature alignment, and robust continual learning methods
- if contrastive learning is used, implement and validate it carefully instead of assuming it helps
- literature search is encouraged, especially for EEG continual learning, domain adaptation, subject alignment, replay, distillation, and parameter-efficient transfer

## Experiment Policy
- accepted strategy:
  - fast screening when needed
  - stronger verification with `3 seeds + full epochs`
- when GPU usage is light, parallel experiments are allowed
- when GPU usage is heavy, run experiments one at a time
- every confirmed improvement should record what exact structure or method caused the gain

## GitHub Visibility
Research outputs should be kept in tracked text files so they can be pushed to GitHub and viewed there:
- `metrics/experiments.csv`
- `metrics/latest.md`

After meaningful runs, update these metric files.

## Experiment Style
- prefer small, testable ablations
- prefer changes that are easy to explain in a thesis
- do not change the data split protocol unless explicitly requested
- preserve the current scenario definition for subjects and classes

## Suggested Workflow
1. run one experiment
2. parse the produced log
3. update `metrics/experiments.csv`
4. refresh `metrics/latest.md`
5. commit and push to GitHub
