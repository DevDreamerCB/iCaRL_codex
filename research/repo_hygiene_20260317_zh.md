# 仓库整洁与后续实验规则

日期：`2026-03-17`

## 1. 目的

本文件用于约束后续实验，避免出现以下问题：

- 代码分支越来越多，主路径难以阅读
- 日志目录和运行名越来越冗长
- 已经证伪的方法还长期残留在训练代码里
- 很难从仓库中找到“当前最好方法”的代码、参数和日志

## 2. 代码层面的规则

后续代码只保留三类内容：

### 2.1 `stable`

当前正式主线里已经证明有效的方法。

例如：

- `normalized NME`
- `hybrid NME + logits`
- `adapter / shared adapter`
- `task affine / task BN`
- `LwF`
- `oldweight`
- `feature distill`
- `prototype blend`

### 2.2 `candidate`

当前仍值得继续探索，但还没有完全定论的方法家族。

要求：

- 同一时间最多保留 `1~2` 个候选家族
- 候选必须有清楚的理论动机
- 候选若连续失败，应尽快删除

### 2.3 `rejected`

已经明确判弱的方法，不再保留在主训练路径里。

这些方法的历史只保留在：

- `metrics/experiments.csv`
- `research/results.tsv`
- `research/failed_methods_20260317_zh.md`

而不继续留在 `main.py / iCaRL.py / mlm.py` 的核心路径中。

## 3. 日志与运行名规则

### 3.1 `logs/` 只保留关键 `full`

当前仓库只保留主线递进中得分逐步升高的 `full` 日志。

后续规则：

- `screen` 日志默认不长期保留
- 普通失败 `full` 也不长期保留
- `logs/` 只保留：
  - 基线
  - 关键里程碑
  - 当前 best
  - 极少数有代表性的平行对照

### 3.2 运行名不再承担全部参数记录

后续运行名原则：

- 只保留核心方法缩写
- 不再把所有参数都塞进目录名

详细参数应以：

- `metrics/experiments.csv`
- `research/results.tsv`

为准。

## 4. 文档层面的规则

后续研究文档只保留这些主文件：

- `story_20260316_zh.md`
- `full_results_progression_20260316_zh.md`
- `method_code_index_20260316_zh.md`
- `method_faq_20260316_zh.md`
- `stage_parameter_flow_20260317_zh.md`
- `replay_directions_20260316.md`
- `failed_methods_20260317_zh.md`
- `version_management_20260317_zh.md`
- `notes.md`
- `results.tsv`

不再重复生成同义但信息高度重叠的中间总结文件。

## 5. 版本管理规则

后续必须遵守：

1. 每个新方法家族第一次进入仓库时单独 commit
2. 每个当前最好正式结果都必须打 tag
3. 代码与文档同步更新
4. 不再长期积累大量未提交改动

## 6. 当前决定删除的旧机制

以下自动研究脚本已从仓库移除：

- `agent_loop.py`
- `agentctl.sh`
- `autoresearch_icarl.py`
- `overnight_research.py`
- `overnight_research.sh`
- `run_agent_loop.sh`
- `run_autoresearch.sh`
- `run_with_metrics.sh`

原因：

- 它们属于早期自动研究阶段的临时设施
- 当前仓库重点已经转向“主线方法清理 + 可解释实验”
- 继续保留只会增加代码和入口冗余

## 7. 下一阶段建议

后续实验建议优先聚焦在：

- 更通用的 `shared + task-specific` 轻量参数化
- 更少蒸馏项的简洁版本
- 能同时兼顾 `task2` 与 `task3` 的方法

而不是继续做明显偏 `task3 specialist` 的特判方法。
