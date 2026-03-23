# 通用连续学习方法链复验总结（固定预算口径）

## 1. 目的

这份文档专门回答一个更严格的问题：

> 如果只选择“尽量适用于所有 task”的通用连续学习方法，而不使用明显偏向某个阶段的处理，那么在当前 `iCaRL_codex` 中，哪些改动真的成立？

这里的比较标准和之前统一后的主标准一致：

- `stage_epochs = [10, 12, 12]`
- `screen = 1 seed`
- 从最原始的 `iCaRL` 出发
- 明确去掉 contrastive loss
- 不引入 `stage3-only` 的 prototype 修补、feature distill 或 asymmetric BCE

因此，这份文档的目标不是追求最高分，而是回答：

- 哪些方法可以被称为“通用 continual learning 方法”
- 哪些方法虽然在别的配置里有效，但不适合作为“全阶段通用组件”

## 2. 实验链设计

本轮按一条固定链条逐步增加方法：

1. 原始 `iCaRL`，只去掉 contrastive
2. `task adapter dim=16`，从 `stage1` 开始
3. `stage_epochs = [10,12,12]`
4. `oldweight = 3.125`
5. `normalized NME`
6. `LwF`
7. `task affine`，从 `stage1` 开始
8. `hybrid NME + logits`，从最早可用阶段开始

说明：

- 这条链的逻辑是“逐步加通用组件”，不是“给每个组件单独找它的最佳上下文”。
- 因此某一条如果表现变差，说明它**不能在这条通用主链中自然成立**。
- 这和之前那条更强的 `s2ace00 + new_only ptblend` 主线不同，后者虽然更强，但包含明显更有阶段性的设计。

## 3. 结果总表

| 步骤 | 运行名 | 改动 | task1 | task2 | task3 | score | 判断 |
|---|---|---|---:|---:|---:|---:|---|
| 0 | `gen00_nocontrastive_short` | 原始 iCaRL，仅去掉 contrastive | 84.72 | 56.48 | 43.29 | 55.53 | 通用起点 |
| 1 | `gen01_adapter16all_short` | `task adapter dim=16`，从 stage1 开始 | 85.42 | 57.64 | 39.89 | 54.32 | 失败，明显伤 task3 |
| 2 | `gen02_stage101212_short` | 在上一步基础上改 `10,12,12` | 85.42 | 57.79 | 39.20 | 54.02 | 失败，没救回 adapter 全开带来的问题 |
| 3 | `gen03_oldweight3125_short` | 再加 `oldweight=3.125` | 85.42 | 53.70 | 39.93 | 53.16 | 失败，全局 oldweight 过强 |
| 4 | `gen04_normnme_short` | 再加 `normalized NME` | 85.42 | 54.86 | 45.22 | 56.15 | 成立，第一个真正恢复主线的通用组件 |
| 5 | `gen05_lwf_short` | 再加 `LwF(0.175, T=1.5)` | 85.42 | 54.63 | 46.76 | 56.85 | 成立，继续提升旧知识保持 |
| 6 | `gen06_affineall_short` | 再加 `task affine`，从 stage1 开始 | 84.49 | 55.86 | 43.09 | 55.20 | 失败，作为 all-task 组件不成立 |
| 7 | `gen07_hybridall_short` | 再加 `hybrid NME+logits` 全阶段启用 | 84.49 | 56.40 | 43.33 | 55.48 | 失败，不能机械推广成通用组件 |

## 4. 每一步的解释

### 4.1 `gen00_nocontrastive_short`

结果：

- `84.72 / 56.48 / 43.29`

解释：

- 这是最原始、最干净的出发点。
- 只去掉 contrastive loss，不加任何后续 trick。
- 它确认了一个旧结论：当前 repo 里 contrastive 并不是正收益项，原始 iCaRL 去掉它后更稳。

结论：

- 可以作为“通用方法链”的 baseline。

### 4.2 `gen01_adapter16all_short`

结果：

- `85.42 / 57.64 / 39.89`

解释：

- `task adapter` 在别的上下文中有帮助，但这里是把它从 `stage1` 就全开。
- 结果说明：**它不是一个天然能套在所有阶段上的通用组件**。
- `task1` 提升了，`task2` 也略升，但 `task3` 明显崩掉。

结论：

- `task adapter` 更像“条件性有效的阶段参数化”，而不是当前意义下的“通用 continual 方法”。

### 4.3 `gen02_stage101212_short`

结果：

- `85.42 / 57.79 / 39.20`

解释：

- 这一步验证：更短、更合理的 stage 预算，是否能自然修复前一条通用链里的问题。
- 结果表明：不能。
- `10,12,12` 本身在别的主线里是合理的，但它不是一个能自动纠正“全阶段 adapter”副作用的万能设置。

结论：

- `10,12,12` 是一个重要训练预算口径，但不是独立方法点。

### 4.4 `gen03_oldweight3125_short`

结果：

- `85.42 / 53.70 / 39.93`

解释：

- 这一条验证：`oldweight=3.125` 作为“全局通用旧类保护”是否成立。
- 结果是显著负收益，尤其 `task2` 掉得很厉害。
- 说明旧类加权在当前项目里**不能简单地理解成通用组件**。
- 它只有在更合适的上下文里才成立。

结论：

- `oldweight` 不是当前口径下的通用主线组件。

### 4.5 `gen04_normnme_short`

结果：

- `85.42 / 54.86 / 45.22`

解释：

- 这是当前通用链里第一个真正把结果明显拉回来的组件。
- `normalized NME` 的作用不是改训练，而是稳定 prototype classifier 的几何。
- 这件事对三个阶段都成立，因此它更像真正的“通用 continual classifier improvement”。

结论：

- `normalized NME` 是这条通用链里最明确成立的方法点之一。

### 4.6 `gen05_lwf_short`

结果：

- `85.42 / 54.63 / 46.76`

解释：

- 在 `normNME` 之后再加 `LwF`，`task3` 继续恢复。
- `task2` 没有继续涨，但整体 `score` 更高。
- 说明在这条通用链里，`LwF` 仍然是经典而有效的旧知识保持机制。

结论：

- `LwF` 是第二个明确成立的通用 continual 组件。

### 4.7 `gen06_affineall_short`

结果：

- `84.49 / 55.86 / 43.09`

解释：

- `task affine` 如果从 `stage1` 开始全开，反而会把前面恢复出来的收益重新打掉。
- 这说明它和 `task adapter` 一样，更像条件性组件。

结论：

- `task affine` 不适合被写成“全阶段通用组件”。

### 4.8 `gen07_hybridall_short`

结果：

- `84.49 / 56.40 / 43.33`

解释：

- `hybrid NME+logits` 在别的主线中有效，但那通常依赖更特殊的上下文。
- 如果把它机械地从更早阶段就全开，并当作通用决策层修正，它并没有带来提升。

结论：

- `hybrid` 更像“条件性有效的决策层组件”，不是本轮定义下的通用方法。

## 5. 当前可得出的结论

在“只保留全阶段通用 continual 方法”的严格标准下，本轮链条的结论很明确：

### 成立的组件

1. `no contrastive`
2. `normalized NME`
3. `LwF`

这三项是当前最稳、最适合写成通用 continual learning 方法的部分。

### 不成立或不适合当前口径的组件

1. `task adapter dim=16 from stage1`
2. 全局 `oldweight=3.125`
3. `task affine` 从 stage1 全开
4. `hybrid NME+logits` 提前为全阶段统一组件

这些方法不是永远没用，而是：

- 在更特殊的上下文里可能有效
- 但不适合被表述成“适用于所有 task 的通用连续学习方法”

## 6. 与当前更强主线的关系

当前固定预算下更强的主线 full 仍然是：

- `s2ace00_ptblendnew02_stage101212_confirm`
- `84.18 / 60.70 / 49.40`

这条更强，是因为它额外加入了：

- `stage2 asymmetric BCE`
- `stage3 new_only ptblend`

也就是说，它更强，但也更具阶段性。

因此可以把两套结论分开：

### 通用方法链

- 更适合论文里“基础 pipeline 递进”
- 可解释性更强
- 最终保留下来的核心是：
  - `no contrastive`
  - `normNME`
  - `LwF`

### 更强但更有条件的方法链

- 更适合作为后续增强版方法
- 主线是：
  - `s2ace00 + new_only ptblend`

## 7. 推荐写法

如果要写论文或汇报，我建议这样组织：

### 第一层：通用 continual 方法链

从原始 iCaRL 出发，逐步说明：

1. 去掉 contrastive
2. 加 `normalized NME`
3. 加 `LwF`

然后明确说明：

- `task adapter / task affine / hybrid` 在“全阶段统一启用”的严格标准下没有成立
- 因此它们不作为通用方法保留

### 第二层：针对当前场景的增强版

再说明：

- 由于当前场景的核心瓶颈出现在 `stage2`
- 需要更有针对性的机制
- 因而最终形成了 `s2ace00 + new_only ptblend` 的更强版本

## 8. 一句话总结

本轮复验说明：

> 如果严格限定为“适用于所有 task 的通用 continual learning 方法”，  
> 当前真正能稳定保留下来的核心组件是：  
> **去掉 contrastive + normalized NME + LwF**。  
> 其他看起来有效的模块，大多仍然是条件性方法，而不是通用主线。
