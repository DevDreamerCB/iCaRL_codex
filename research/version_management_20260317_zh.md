# 版本管理与方法回溯说明

日期：`2026-03-17`

## 1. 目的

本文件用于回答两个问题：

1. 现在怎样从仓库中找到“表现较好的方法”对应的代码与参数
2. 从当前开始，怎样保证后续每个重要方法都能被稳定回溯

## 2. 当前回溯机制

目前仓库中的回溯信息分为四层：

### 2.1 指标层

所有实验结果统一记录在：

- [metrics/experiments.csv](/data1/bochen/cbcontinual/iCaRL_codex/metrics/experiments.csv)
- [metrics/latest.md](/data1/bochen/cbcontinual/iCaRL_codex/metrics/latest.md)

这里可以查到：

- `run_tag`
- `task1 / task2 / task3`
- `score`
- `memory_size`
- `stage_epochs`
- 关键方法名
- 对应日志目录

### 2.2 方法说明层

主要文档：

- [research/full_results_progression_20260316_zh.md](/data1/bochen/cbcontinual/iCaRL_codex/research/full_results_progression_20260316_zh.md)
- [research/method_code_index_20260316_zh.md](/data1/bochen/cbcontinual/iCaRL_codex/research/method_code_index_20260316_zh.md)
- [research/method_faq_20260316_zh.md](/data1/bochen/cbcontinual/iCaRL_codex/research/method_faq_20260316_zh.md)

这里可以查到：

- 方法做了什么
- 代码主要落在哪些文件
- 为什么保留或放弃

### 2.3 日志层

`logs/` 现在只保留了主线递进的关键 `full` 日志目录。

用途：

- 查看当时的训练日志
- 查看每个 seed 的详细输出
- 回看该方法对应的正式结果

### 2.4 Git 层

当前开始，重要状态通过：

- commit
- git tag

进行固定。

## 3. 当前已有的重要 tag

### 3.1 `research-snapshot-20260316`

用途：

- 固定早期研究整理快照
- 主要对应“研究故事、方法索引、结果整理”第一次系统成型的版本

### 3.2 `clean-snapshot-20260317`

用途：

- 固定“清理后仓库结构”
- 删除冗余文档、只保留关键 full 日志后的版本

### 3.3 `best-full-20260317`

用途：

- 固定当前主线最好方法所在的代码快照
- 方便后续直接回到“当前 best full 对应的代码状态”

## 4. 当前最佳方法如何回溯

当前 best full：

- 运行名：`lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm`
- 指标：`84.18 / 60.75 / 49.36`

回溯方式：

1. 先看 [metrics/latest.md](/data1/bochen/cbcontinual/iCaRL_codex/metrics/latest.md)
2. 再看对应日志目录：
   - [logs/20260316_132136_20260316_132132_lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm](/data1/bochen/cbcontinual/iCaRL_codex/logs/20260316_132136_20260316_132132_lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm)
3. 若要回到当前代码快照，直接 checkout：
   - `git checkout best-full-20260317`

## 5. 一个必须说明的现实限制

早期有一段探索过程没有做到“每个方法一个 commit / tag”。

因此：

- 这些方法的实验结果和参数是可以回溯的
- 但不是每一条都能做到“精确 checkout 回当时代码”

也就是说：

- **结果和方法说明是完整的**
- **逐条代码快照不是对所有历史方法都完美存在**

从当前开始，这个问题会通过下面的规则避免。

## 6. 从现在开始的版本管理规则

后续一律按下面规则执行：

1. 每个新方法家族单独一个 commit  
例如：
- `add shared-task adapter family`
- `add replay retrieval family`

2. 每个“当前最好正式结果”打一个 tag  
例如：
- `best-full-20260317`
- `best-full-20260320`

3. 每次方法变更都要能在文档里对应到：
- 方法名
- 关键超参数
- `run_tag`
- 日志目录
- commit / tag

4. 不再允许长时间积累大量未提交改动

## 7. 建议的日常使用方式

### 查看当前最好代码

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
git checkout best-full-20260317
```

### 回到最新主线

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
git checkout main
git pull
```

### 给一个新方法打快照 tag

使用：

- [snapshot_method.sh](/data1/bochen/cbcontinual/iCaRL_codex/snapshot_method.sh)

例如：

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./snapshot_method.sh best-full-20260320 "new best full after shared adapter tuning"
```

## 8. 当前建议

如果你后面还会继续长期做实验，最稳的工作流是：

1. 先在 `main` 上改代码
2. 跑实验
3. 结果成立后立刻：
   - commit
   - 打 tag
   - 更新 `metrics/latest.md`
   - 更新研究文档

这样你就能始终做到：

- 找到“当前最好方法”的代码
- 找到“过去某个阶段最好方法”的代码
- 找到“对应的参数和日志”
