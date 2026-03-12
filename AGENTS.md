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
