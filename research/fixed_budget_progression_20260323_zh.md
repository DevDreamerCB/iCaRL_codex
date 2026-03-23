# 固定预算标准下的实验递进整理

日期：`2026-03-23`

## 1. 评价标准说明

本文件开始采用新的主评价标准：

- 主标准：`stage_epochs = [10, 12, 12]`
- `screen`：`1 seed`
- `full`：`3 seeds`

这样做的原因是：

- 在当前仓库里，较短的分阶段训练预算更符合“连续学习固定增量更新预算”的习惯。
- 最近一批 `30/30/30` 的 full 结果说明，较长训练预算会把“方法增益”和“训练时长增益”混在一起。
- 对当前主线来说，`10,12,12` 的口径更能体现旧任务保持与新任务学习之间的真实平衡。

本文优先展示：

- 在这一固定预算标准下，哪些方法真正有效
- 哪些方法 short 有信号但 full 站不住
- 哪些方向已经可以判弱

指标顺序统一为：

- `task1 / task2 / task3`

## 2. 当前最可信的主线

目前在固定预算标准下，当前最可信的 full 是：

- `s2ace00_ptblendnew02_stage101212_confirm`
- `84.18 / 60.70 / 49.40`
- `score = 59.75`

这条方法的组成是：

- `stage2 asymmetric BCE (s2ace00)`
- `normalized NME`
- `hybrid NME + logits`
- `new_only prototype blend @ stage3`
- `task adapter dim=16`
- `task affine`
- `LwF`
- `stage3-only feature distill`
- `oldweight = 3.125`
- `stage_epochs = [10, 12, 12]`

这条线之所以可信，不是因为它绝对分数最高，而是因为：

- 它是在统一固定预算下得到的
- 它同时兼顾了 `task2` 和 `task3`
- 它的方法逻辑比后面那些“只修最后阶段”的技巧更清楚

## 3. 固定预算下最重要的主线递进

下面这条链最适合写进论文正文。

| 阶段 | 运行名 | 主要变化 | task1 | task2 | task3 | 说明 |
|---|---|---|---:|---:|---:|---|
| 0 | `baseline_confirm` | 原始基线 | 85.80 | 59.85 | 37.23 | 起点，未按固定预算重新跑，仅作原始参考 |
| 1 | `no_contrastive_confirm` | 去掉 contrastive | 85.80 | 60.78 | 40.24 | 第一个明确有效的改进 |
| 2 | `no_contrastive_adapter16_s2_confirm` | 加 `adapter16`，从 stage2 开始 | 85.34 | 61.52 | 41.34 | 轻量 PEFT 有效 |
| 3 | `lwf015_replaymix2_adapter16_mem36_oldweight25_confirm` | 加 `LwF + replaymix2 + oldweight` | 85.73 | 58.49 | 42.72 | 旧类保持开始成型 |
| 4 | `lwf015_normnme_adapter16_mem36_oldweight25_confirm` | 加 `normalized NME` | 86.11 | 59.46 | 44.85 | prototype 几何修正是关键拐点 |
| 5 | `lwf015_normnme_adapter16_mem36_oldweight30_stage101212_confirm` | 用 `10,12,12` 和更强 oldweight | 84.18 | 60.54 | 47.79 | 非 hybrid 主线成型 |
| 6 | `lwf0175T15_normnme_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm` | `LwF T=1.5 + stage3 feature distill + task-affine` | 84.18 | 60.86 | 48.29 | 非 hybrid 成熟强线 |
| 7 | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | hybrid + new_only ptblend + affine2 | 84.18 | 60.75 | 49.36 | 旧的固定预算 best |
| 8 | `s2ace00_ptblendnew02_stage101212_confirm` | 在 stage2 引入 asymmetric BCE，保留简化后的 stage3 主线 | 84.18 | 60.70 | 49.40 | 当前固定预算主线 |

## 4. 为什么 `s2ace00` 是当前最值得保留的新点

最近这批实验的真正核心发现不是某个 `stage3` 小技巧，而是：

- `task2` 的主要瓶颈来自 `A=Left Hand` 在 `subjects 4/5/6` 上的迁移失败
- `4/5/6 joint` 上界并不低，说明这不是数据本身完全学不会
- 更合理的做法不是继续堆 `stage2` 的 replay trick，而是避免当前样本对缺失类 `A` 形成错误负压力

因此，`s2ace00` 的意义是：

- 它直接针对“当前 task 没有的类别怎么处理”这个问题
- 它是经典 continual learning 文献里可解释的方向
- 它比最近那批 `stage2 replay / stage2 distill / balanced finetune` 更对位

## 5. 固定预算下重新验证过、但没有成立的新方向

### 5.1 `stage2 current-class CE`

文献启发：

- `ER-ACE`
- `Separated Softmax`

实验：

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_s2curce010_short` | `84.72 / 63.50 / 49.58` | short 很强 |
| `s2ace00_ptblendnew02_s2curce010_confirm` | `86.11 / 59.62 / 47.53` | full 明显崩掉，`0.1` 太强 |
| `s2ace00_ptblendnew02_s2curce005_short` | `84.72 / 62.19 / 49.73` | 仍有信号 |
| `s2ace00_ptblendnew02_s2curce005_confirm` | `86.11 / 59.77 / 47.34` | 仍然不稳 |
| `s2ace00_ptblendnew02_s2curce0025_stage101212_short` | `84.72 / 60.80 / 50.15` | 在正确口径下不如基线 |

结论：

- 这条 family 方向有启发，但在当前 repo 上非常不稳
- 在 `10,12,12` 正确口径下没有打过当前主线
- 因此目前先判为“不保留到主线”

### 5.2 只增加 `stage2` 的训练预算

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_stage101412_short` | `84.72 / 59.65 / 49.23` | `stage2` 不是单纯训得不够久 |

### 5.3 `EEIL-style balanced finetune`

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_s2bft2_short` | `84.72 / 60.57 / 49.31` | 全模型版不够稳 |
| `s2ace00_ptblendnew02_s2bft2head_short` | `84.72 / 61.57 / 48.92` | 只训头更合理，但仍不如主线 |

### 5.4 把 `feature distill` 提前到 `stage2+3`

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_fd0010s23_short` | `84.72 / 62.11 / 49.31` | 没有让方法更通用，反而更像过约束 |

### 5.5 把 asymmetric BCE 从 `stage2` 推广到 `stage3`

这个方向是最近新增的重要检验，因为它直接回答了一个更通用的方法学问题：

- `s2ace00` 是否只是一个 `stage2` 特判技巧
- 还是它其实可以推广成“所有阶段都适用的缺失类不当负类压制”机制

实验：

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_s3ace25_ptblendnew02_stage101212_short` | `84.72 / 60.96 / 49.61` | 比同口径主线弱 |
| `s2ace00_s3ace00_ptblendnew02_stage101212_short` | `84.72 / 60.96 / 49.69` | 仍不如同口径主线 |

对比同口径主线：

| 运行名 | 结果 |
|---|---:|
| `s2ace00_ptblendnew02_stage101212_short` | `84.72 / 60.96 / 50.54` |

结论：

- asymmetric BCE 作为“缺失旧类不应被当前样本压制”的思想，本身是合理的。
- 但在当前场景里，它**最清晰、最有效的作用阶段仍然是 `stage2`**。
- 一旦直接推广到 `stage3`，`task2` 不涨，`task3` 也没有超过同口径主线。
- 这说明：
  - `stage2` 的“缺失类 A 在新域 4/5/6 中被错误当负类压制”是一个非常特殊、非常强的结构性问题；
  - `stage3` 的 old/new/overlap 关系更复杂，不能简单复用同一个 asymmetric BCE 规则。

## 6. 固定预算下的主线简化判断

### 6.1 `task affine`

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_noaffine_confirm` | `84.18 / 61.57 / 49.58` | `task affine` 有帮助，但不是主驱动 |

### 6.2 `task adapter`

| 运行名 | 结果 | 结论 |
|---|---:|---|
| `s2ace00_ptblendnew02_noadapter_short` | `85.42 / 62.35 / 48.84` | 删掉 adapter 后 `task2` 受损更明显 |
| `s2ace00_ptblendnew02_adapter8_confirm` | `85.96 / 61.39 / 49.11` | 缩小到 8 维后更轻，但仍不如当前主线 |
| `s2ace00_ptblendnew02_adapter12_short` | `82.64 / 61.11 / 48.92` | `12` 明显不如 `16` |

结论：

- `task adapter` 目前仍然比 `task affine` 更关键
- `adapter8` 是可以作为“更轻版本”保留的候选，但不是主结果

## 7. 当前对“多少 epoch 更合适”的判断

最近重新校正后，结论更明确：

- 不应该再混用默认 `30/30/30` 和较短分阶段预算
- 对当前主线来说，`10,12,12` 仍然是更合理的主比较标准

原因：

- `s2ace00_ptblendnew02_stage101212_short`
  - `84.72 / 60.96 / 50.54`
- `s2ace00_ptblendnew02_s2curce0025_stage101212_short`
  - `84.72 / 60.80 / 50.15`

说明：

- 正确口径下，当前主线本身已经比最近那批 `30 epochs` 结果更可信
- 某些看似“有提升”的新方法，其实是被训练时长掩盖了真实效果

因此后续推荐：

- 主比较统一用 `stage_epochs = [10,12,12]`
- `30/30/30` 只作为补充收敛性检查，不作为论文主表

## 8. 当前最值得写进论文的固定预算对照

如果只保留少量关键结果，建议用下面这几条：

| 类型 | 运行名 | task1 | task2 | task3 | 作用 |
|---|---|---:|---:|---:|---|
| 原始参考 | `baseline_confirm` | 85.80 | 59.85 | 37.23 | 原始起点 |
| 去掉 contrastive | `no_contrastive_confirm` | 85.80 | 60.78 | 40.24 | 证明 contrastive 是负收益 |
| 加 adapter | `no_contrastive_adapter16_s2_confirm` | 85.34 | 61.52 | 41.34 | 证明轻量 PEFT 有效 |
| normNME 拐点 | `lwf015_normnme_adapter16_mem36_oldweight25_confirm` | 86.11 | 59.46 | 44.85 | prototype 几何修正的重要性 |
| 旧 fixed-budget best | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | 84.18 | 60.75 | 49.36 | 历史最强 stage101212 主线 |
| 当前 fixed-budget 主线 | `s2ace00_ptblendnew02_stage101212_confirm` | 84.18 | 60.70 | 49.40 | 当前更通用、更可解释的主线 |

## 9. 当前最重要的结论

1. 主比较标准已经重新固定为：
   - `stage_epochs = [10,12,12]`

2. 最近那批 `30 epochs` full 结果只适合作为补充，不应再和主表混用。

3. 目前最好的“更通用、也更容易讲”的方向是：
   - `s2ace00 + new_only ptblend`
   - 在固定预算 `10,12,12` 下使用

4. `stage2 current-class CE`、balanced finetune、单纯拉长 stage2、把 feature distill 前移、以及把 asymmetric BCE 直接推广到 `stage3`，这些在正确口径下都没有打过当前主线。

5. 后续若继续做新实验，应当在这个固定预算标准下展开，而不是再回到默认 `30/30/30`。
