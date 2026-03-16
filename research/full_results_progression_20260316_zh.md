# Full 实验递进式整理

日期：`2026-03-16`

说明：

- 本文只整理 `3 seeds + 30 epochs` 的 `full` 实验
- 目标是帮助把当前工作写进论文或实验记录中，形成“逐步叠加改进”的比较链条
- 指标顺序统一为：`task1 / task2 / task3`
- 这里优先展示“主线形成过程”和“典型无效举措”

## 1. 一条最清楚的主线

下面这条链最适合写成论文里的主实验演化过程。

| 阶段 | 运行名 | 主要变化 | task1 | task2 | task3 | 结论 |
|---|---|---|---:|---:|---:|---|
| 0 | `baseline_confirm` | 原始基线 | 85.80 | 59.85 | 37.23 | 起点 |
| 1 | `no_contrastive_confirm` | 去掉对比损失 | 85.80 | 60.78 | 40.24 | 第一个明确有效改进 |
| 2 | `no_contrastive_adapter16_s2_confirm` | 加 `adapter16`，从 stage2 开始 | 85.34 | 61.52 | 41.34 | 轻量 PEFT 有效 |
| 3 | `lwf015_replaymix2_adapter16_confirm` | 引入 `LwF` | 85.73 | 58.90 | 41.56 | 对 task3 有帮助，但 task2 先掉 |
| 4 | `lwf015_replaymix2_adapter16_mem36_oldweight25_confirm` | 强化旧类权重 | 85.73 | 58.49 | 42.72 | 旧类保持继续增强 |
| 5 | `lwf015_normnme_adapter16_mem36_oldweight25_confirm` | 引入 `normalized NME` | 86.11 | 59.46 | 44.85 | 一个大拐点 |
| 6 | `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm` | 更合理的分阶段 epoch | 84.18 | 60.62 | 46.99 | 后期训练策略明显有效 |
| 7 | `lwf015_normnme_adapter16_mem36_oldweight30_stage101212_confirm` | oldweight 进一步调优 | 84.18 | 60.54 | 47.79 | 非 hybrid 主线基本成型 |
| 8 | `lwf0175T15_normnme_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm` | `LwF T=1.5` + stage3 feature distill + task-affine | 84.18 | 60.86 | 48.29 | 非 hybrid 主线成熟 |
| 9 | `lwf0175T15_normnmehybrid_s3old04_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm` | 引入 `hybrid NME+logits` | 85.72 | 59.62 | 48.52 | 解决 logits/NME 错配开始见效 |
| 10 | `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm` | `stage3-only + new_only` prototype blend | 84.18 | 60.80 | 48.91 | prototype 线第一次真正站住 |
| 11 | `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_affine2_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm` | task-affine 提前到 stage2 | 84.18 | 60.80 | 49.04 | `affine2` 有效 |
| 12 | `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | `stagefd=0.03`，oldweight 精修 | 84.18 | 60.75 | 49.22 | 进入 49+ 区间 |
| 13 | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | 将 `new_only ptblend alpha` 压到 `0.2` | 84.18 | 60.75 | 49.36 | 当前最好 full |

## 2. 为什么这条主线好讲

这条链的逻辑很顺：

1. 先证明当前对比损失不适合本任务
2. 再证明轻量 adapter 对增量场景有帮助
3. 然后用 `LwF + oldweight` 强化旧类保持
4. 再用 `normalized NME` 修正原型几何
5. 再用 `stage schedule + stage3 distill + task-affine` 稳定后期学习
6. 最后处理 `logits` 与 `NME` 的决策错配，并只对新增类做 prototype refinement

这比“同时堆很多模块”更适合论文叙事。

## 3. 当前最值得放进论文正文的关键 full 结果

如果只保留少量表格行，我建议优先保留下面这些：

| 类型 | 运行名 | task1 | task2 | task3 | 用途 |
|---|---|---:|---:|---:|---|
| 原始基线 | `baseline_confirm` | 85.80 | 59.85 | 37.23 | 原始 iCaRL 参考 |
| 去掉对比损失 | `no_contrastive_confirm` | 85.80 | 60.78 | 40.24 | 证明当前 contrastive 是负收益 |
| 轻量适配 | `no_contrastive_adapter16_s2_confirm` | 85.34 | 61.52 | 41.34 | 证明 adapter 有用 |
| corrected pipeline 早期强线 | `lwf015_replaymix2_adapter16_mem36_oldweight25_confirm` | 85.73 | 58.49 | 42.72 | 过渡里程碑 |
| normNME 拐点 | `lwf015_normnme_adapter16_mem36_oldweight25_confirm` | 86.11 | 59.46 | 44.85 | 证明 NME 归一化重要 |
| 非 hybrid 成熟线 | `lwf0175T15_normnme_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm` | 84.18 | 60.86 | 48.29 | 最强非 hybrid 参考 |
| 当前最佳 | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | 84.18 | 60.75 | 49.36 | 当前 best full |

## 4. 典型无效或负收益 full 结果

这些结果很有用，因为它们能帮你解释“不是所有看起来合理的方法都有效”。

| 方法 | 运行名 | task1 | task2 | task3 | 结论 |
|---|---|---:|---:|---:|---|
| 只训 embedding | `embedding_only_confirm` | 83.33 | 51.08 | 37.65 | 明显不如基线 |
| prototype alignment | `proto_align_confirm` | 85.80 | 59.59 | 35.65 | task3 更差 |
| no-balance | `no_contrastive_nobalance_confirm` | 85.57 | 60.73 | 40.99 | 不能稳定超过主线 |
| replaymix2 早期版本 | `no_contrastive_adapter16_s2_lr15_replaymix2_confirm` | 85.73 | 59.85 | 40.57 | 不如更成熟主线 |
| task BN | `lwf015_normnme_adapter16_mem36_oldweight30_stage101212_taskbn_confirm` | 84.18 | 60.85 | 47.22 | task3 不够强 |
| task affine 早期版本 | `lwf015_normnme_adapter16_mem36_oldweight30_stage101212_taskaffine_confirm` | 84.18 | 60.39 | 47.49 | 不如后期 `affine2` |
| group-bias 作为主线 | `lwf0175T15_normnmehybrid_gbias05_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm` | 84.18 | 61.01 | 48.63 | 更平衡，但不是最强主线 |

## 5. replay 相关 full 结果里最值得保留的几条

目前 replay family 的 full 结果并不多真正超过主线，但下面这些值得记录：

| 方法 | 运行名 | task1 | task2 | task3 | 结论 |
|---|---|---:|---:|---:|---|
| DER-lite 风格启发前的强线 | `lwf015_replaymix2_adapter16_mem42_oldweight2_confirm` | 85.73 | 58.36 | 42.95 | 早期 replay 改进节点 |
| alignment-memory replay | 目前主要停留在 short，full 没形成更强主线 | - | - | - | 有信号，但未形成正式突破 |
| hard replay / MIR-lite | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_hardreplay10s3_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` | 84.18 | 60.75 | 49.36 | 与当前 best full 持平，没有超过 |

## 6. 哪些 full 结果更适合写“方法递进”

如果你想在论文里做“递进式比较”，我建议按下面这组写：

| 顺序 | 方法摘要 | 对应 full 结果 |
|---|---|---|
| 1 | 原始 iCaRL | `baseline_confirm` |
| 2 | 去掉 contrastive | `no_contrastive_confirm` |
| 3 | 加轻量 adapter | `no_contrastive_adapter16_s2_confirm` |
| 4 | 加 LwF + replaymix2 + oldweight | `lwf015_replaymix2_adapter16_mem36_oldweight25_confirm` |
| 5 | 加 normalized NME | `lwf015_normnme_adapter16_mem36_oldweight25_confirm` |
| 6 | 调整 stage schedule | `lwf015_normnme_adapter16_mem36_oldweight25_stage101214_confirm` |
| 7 | 非 hybrid 最强成熟线 | `lwf0175T15_normnme_adapter16_mem36_oldweight325_stage101212_stagefd0025_taskaffine_s3_confirm` |
| 8 | 加 hybrid | `lwf0175T15_normnmehybrid_s3old04_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm` |
| 9 | 加 new_only prototype blend | `lwf0175T15_normnmehybrid_s3old04_ptblend07s3new_adapter16_mem36_oldweight30_stage101212_stagefd0025_taskaffine_s3_confirm` |
| 10 | 加 affine2 + alpha 精修 | `lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm` |

这组对照最像“一个研究故事”，而不是“很多散实验”。

## 7. 当前缺少但值得补的 full 对照

严格来说，下面几条如果补出来，论文会更完整：

1. **纯 LwF-only full**
- 现在最接近的是 `lwf_replaymix2_adapter16_confirm`
- 但它还带了 adapter / replaymix2
- 如果你想单独证明 “LwF 本身的增益”，最好补一条更纯的 full

2. **纯 normNME-only full**
- 现在最接近的是 `lwf015_normnme_adapter16_mem36_oldweight25_confirm`
- 但它已经叠加了 `LwF + adapter + oldweight`
- 如果想把贡献拆得更细，可以补一条更纯的 `normNME` 对照

3. **pure hybrid without ptblend/full affine2**
- 现在已有 hybrid full，但最终最优线里又叠了 ptblend 和 affine2
- 如果论文要更强调模块贡献，最好再单独放一条“只加 hybrid”的 full

## 8. 当前我的判断

如果你的目标是：

- 让论文叙事清楚
- 让比较链条尽量干净

那么最值得采用的写法不是把 99 条 full 全都塞进去，而是：

1. 选出上面第 6 节那 10 条主线
2. 再从第 4 节选 3 到 5 条负对照
3. 最后把第 7 节列成“后续可补实验”

这样会比“把所有 full 一股脑堆表格”更有说服力。
