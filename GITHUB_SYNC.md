# GitHub Sync

## Goal
Push code changes and tracked metric summaries from `iCaRL_codex` to a GitHub repository so metric changes are visible there.

Tracked metric files:
- `metrics/experiments.csv`
- `metrics/latest.md`

## Current Limitation
This local directory is not currently a git repository, so nothing can be pushed yet.

## Recommended Repo Choice
From your GitHub account, the most reasonable target seems to be:
- `cb_projects`

If you want cleaner history, create a dedicated new repository such as:
- `cbcontinual`
- `bci-continual-learning`
- `graduation-project-bci`

## Minimal Setup
After you decide the target repository, run something like:

```bash
cd /data1/bochen/cbcontinual
git init
git add iCaRL_codex
git commit -m "Initialize iCaRL_codex research workspace"
git branch -M main
git remote add origin git@github.com:DevDreamerCB/<your-repo-name>.git
git push -u origin main
```

## Daily Workflow
```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./run_with_metrics.sh
cd /data1/bochen/cbcontinual
git add iCaRL_codex
git commit -m "Update experiment and metrics"
git push
```

## What GitHub Will Show Clearly
- latest run summary in `metrics/latest.md`
- run history in `metrics/experiments.csv`
- code diffs for each experimental change

## Better Visibility Option
If you want, later we can also generate:
- `metrics/best_results.md`
- `metrics/plots/*.png`
- a GitHub Actions workflow to rebuild summaries automatically after each push
