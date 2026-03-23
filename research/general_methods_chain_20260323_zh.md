# 通用连续学习方法链复验总结（更合理底座）

## 1. 目的

这份文档用来纠正上一版“通用方法链”的起点偏弱问题。

上一版链条错误地从过弱底座出发：

- `memory=24`
- `replay_batch_size=0`

这会把很多本来依赖更合理 replay 底座的方法误判为“完全不成立”。

因此，这一版重新定义一个更合理、也更接近你前面强结果记忆的通用底座：

- 去掉 contrastive loss
- `memory=36`
- `replaymix2`，即 `replay_batch_size=2`

在这个更强底座上，再依次尝试你指定的通用组件，并且统一使用：

- `screen = 1 seed`
- `stage_epochs = 10,12,12` 只在明确加入这一步之后才开启

注意：

- 这里的目的仍然不是追求绝对最高分。
- 这里要回答的是：**从一个更合理的 replay 基线出发，哪些方法能被写成“适用于所有 task 的通用 continual learning 组件”。**

## 2. 方法链设计

本轮按如下顺序逐步加方法：

1. `ugen00`：更合理底座  
   去掉 contrastive + `memory=36` + `replaymix2`
2. `ugen01`：`task adapter dim=16`，从 `stage1` 开始
3. `ugen02`：`stage_epochs = [10,12,12]`
4. `ugen03`：`oldweight = 3.125`
5. `ugen04`：`normalized NME`
6. `ugen05`：`LwF (lambda=0.175, T=1.5)`
7. `ugen06`：`task affine`，从 `stage1` 开始
8. `ugen07`：`hybrid NME + logits`，从最早可用阶段统一启用

说明：

- 这条链是“在同一条更合理底座上逐步叠加”，不是给每个组件单独找最佳上下文。
- 因此某一步如果明显变差，说明它**不能简单地作为全 task 通用组件加入这条链**。

## 3. 结果总表

| 步骤 | 运行名 | 改动 | task1 | task2 | task3 | score | 判断 |
|---|---|---|---:|---:|---:|---:|---|
| 0 | `ugen00_base_mem36_rp2_short` | 去掉 contrastive + `mem36 + replaymix2` | 84.95 | 57.56 | 42.71 | 55.61 | 合理底座 |
| 1 | `ugen01_adapter16all_mem36_rp2_short` | `task adapter dim=16`，从 stage1 开始 | 85.42 | 57.48 | 42.59 | 55.62 | 基本无增益 |
| 2 | `ugen02_stage101212_mem36_rp2_short` | 改为 `stage_epochs=10,12,12` | 85.42 | 60.19 | 41.47 | 55.88 | 明显提升 task2 |
| 3 | `ugen03_oldweight3125_mem36_rp2_short` | 再加 `oldweight=3.125` | 85.42 | 52.39 | 39.81 | 52.71 | 失败，过强且全局不稳 |
| 4 | `ugen04_normnme_mem36_rp2_short` | 再加 `normalized NME` | 85.42 | 56.64 | 44.95 | 56.55 | 明显修复 task3 |
| 5 | `ugen05_lwf_mem36_rp2_short` | 再加 `LwF` | 85.42 | 55.86 | 46.22 | 56.95 | task3 继续提升 |
| 6 | `ugen06_affineall_mem36_rp2_short` | 再加 `task affine`，从 stage1 开始 | 84.26 | 55.94 | 45.06 | 56.16 | 失败，收益回退 |
| 7 | `ugen07_hybridall_mem36_rp2_short` | 再加 `hybrid NME+logits` 统一启用 | 84.26 | 55.17 | 45.99 | 56.40 | 失败，没优于 `ugen05` |

## 4. 每一步的解释

### 4.1 `ugen00_base_mem36_rp2_short`

结果：

- `84.95 / 57.56 / 42.71`

解释：

- 这是当前更合理的“通用链起点”。
- 它不包含 stage-specific trick，只是把 replay 基线恢复到：
  - `memory=36`
  - `replay_batch=2`
- 这个起点已经明显比上一版错误链更合理，也更接近你记忆中的强 baseline。

结论：

- 后续通用方法应该从这个底座出发，而不是从 `mem24 + no replaymix` 出发。

### 4.2 `ugen01_adapter16all_mem36_rp2_short`

结果：

- `85.42 / 57.48 / 42.59`

解释：

- 这一步测试：`task adapter dim=16` 如果从 `stage1` 就作为统一组件启用，能否自然带来增益。
- 结果几乎没有真实收益：
  - `task1` 略升
  - `task2/task3` 没有改善

结论：

- `adapter16` 在这个口径下不是一个“自然成立的全 task 通用组件”。
- 它更像条件性有效的方法，需要更合适的上下文。

### 4.3 `ugen02_stage101212_mem36_rp2_short`

结果：

- `85.42 / 60.19 / 41.47`

解释：

- 这一步说明：在更合理 replay 底座上，`10,12,12` 作为训练预算设置，最先作用的是 `task2`。
- `task2` 从 `57.48` 拉到 `60.19`，说明：
  - stage2 的预算分配确实重要
  - `30/30/30` 不是默认最优口径

结论：

- `10,12,12` 更适合被当成主比较标准，而不是简单的“辅助设置”。

### 4.4 `ugen03_oldweight3125_mem36_rp2_short`

结果：

- `85.42 / 52.39 / 39.81`

解释：

- 这一步很重要，因为它说明：
  - `oldweight=3.125` 不是一个可以机械地全局加进通用链的组件。
- 在这个上下文里，它把 `task2/task3` 一起拉坏了。

结论：

- `oldweight` 只有在更具体、条件性更强的主线里才成立。
- 它不适合被表述成“全 task 通用组件”。

### 4.5 `ugen04_normnme_mem36_rp2_short`

结果：

- `85.42 / 56.64 / 44.95`

解释：

- 在被 `oldweight` 拉坏之后，`normalized NME` 把结果明显拉回来了，尤其是 `task3`。
- 这符合之前很多实验的经验：
  - `normNME` 更像一个统一的 classifier geometry 修正
  - 不依赖某个特定 task 才有效

结论：

- `normalized NME` 在更合理底座上仍然成立，而且更适合被保留成“通用 continual classifier improvement”。

### 4.6 `ugen05_lwf_mem36_rp2_short`

结果：

- `85.42 / 55.86 / 46.22`

解释：

- `LwF` 在这条链里没有把 `task2` 再往上拉，但把 `task3` 从 `44.95` 提到了 `46.22`。
- 这说明它在这条通用链中的作用更偏：
  - 旧知识保持
  - 而不是普遍抬高所有 task

结论：

- `LwF` 在这条更合理底座的通用链里仍然是成立的。

### 4.7 `ugen06_affineall_mem36_rp2_short`

结果：

- `84.26 / 55.94 / 45.06`

解释：

- 把 `task affine` 从 `stage1` 开始全阶段统一启用，并没有在这条链里形成正增益。
- 它会把已经由 `normNME + LwF` 恢复出来的收益部分打掉。

结论：

- `task affine` 不适合在当前口径下被写成“全 task 通用组件”。

### 4.8 `ugen07_hybridall_mem36_rp2_short`

结果：

- `84.26 / 55.17 / 45.99`

解释：

- `hybrid NME+logits` 在别的主线里有效，但这里的关键问题是：
  - 如果把它当成“统一早期开启的通用组件”，它并没有改善整体链条。
- `task3` 比 `ugen06` 略回一点，但总体不如 `ugen05`。

结论：

- `hybrid` 更像条件性决策层修正，不适合作为这条通用链里的默认组件。

## 5. 当前结论

在“更合理 replay 底座 + 全 task 通用组件”的标准下，本轮结论是：

### 明确成立的通用点

1. 去掉 contrastive loss
2. 更合理的 replay 基线：`memory=36 + replaymix2`
3. `stage_epochs = 10,12,12`
4. `normalized NME`
5. `LwF`

### 不适合直接作为通用组件加入的点

1. `adapter16` 从 `stage1` 开始全开
2. 全局 `oldweight=3.125`
3. `task affine` 从 `stage1` 开始全开
4. `hybrid NME+logits` 作为统一早期开启组件

也就是说，如果要写一条“更通用、对所有 task 都适用”的 pipeline，这一轮更可信的表述应该是：

- 强 replay 底座
- 更合理的训练预算
- `normalized NME`
- `LwF`

而不是把前面你记忆中强线里的所有条件性组件都塞进“通用方法链”里。

## 6. 和当前更强主线的关系

当前更强、也更贴近真实问题的 fixed-budget 主线仍然是：

- `s2ace00_ptblendnew02_stage101212_confirm`
- `84.18 / 60.70 / 49.40`

它更强，是因为它额外利用了：

- `stage2 asymmetric BCE`
- `stage3 new_only ptblend`

这两者都带有更明显的阶段性，所以不应和这份“通用方法链”混为一谈。

更准确的论文叙事应该分层：

1. 先给出“通用 continual 方法链”
2. 再给出“针对当前场景瓶颈的阶段增强版主线”

这样方法故事会更清楚，也不会把条件性技巧误写成通用方法。
