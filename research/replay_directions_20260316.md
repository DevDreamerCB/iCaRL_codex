# 重放方向调研与当前判断

日期：`2026-03-16`

## 1. 目的

本文只整理“与 replay 相关、且对当前代码库仍有参考价值”的方法方向，目标是回答两个问题：

1. 在当前约束下，哪些 replay 方法最可能继续提升 `task3`
2. 哪些 replay 方法已经尝试过，可以暂时停止投入

当前约束为：

- 保持预训练 backbone 主体不变
- 尽量只在 `iCaRL_codex` 内做小改动
- 优先提升 `task3`，其次 `task2`
- 默认固定 memory 预算，除非专门研究成本与性能权衡

## 2. 当前 replay 基线是什么

当前最强主线里，replay 相关的基础组件已经包括：

- 固定 exemplar memory：`mem36`
- herding 风格 exemplar 选择
- `replay_batch_size=2`
- `stage3-only + new_only` prototype refinement
- `hybrid NME + logits`
- 通过 `LwF + oldweight + stage3 feature distill` 做 old/new 平衡

所以后续的 replay 研究，不应只是再加一个全局偏置旋钮，而应主要回答：

- replay 哪些样本更值得保留
- replay 样本在训练中应该怎样使用
- replay 如何更好地与当前阶段样本交互

## 3. 文献里最值得借鉴的 replay 方向

### 3.1 ReCIL

为什么重要：

- 它直接针对“跨被试类增量运动想象分类”
- replay 设计明确区分了 `global replay` 与 `local replay`

最值得借鉴的思想：

- replay 不仅可以存局部 exemplar，也可以存“跨被试对齐信息”
- exemplar 选择应同时考虑代表性和多样性
- 当特征空间距离噪声较大时，可考虑先降维再做 replay selection

来源：

- Yang et al., ReCIL, IEEE TBME 2025  
  本地 PDF：[/data1/bochen/cbcontinual/iCaRL_codex/2025_CS-CIL_TBME_Yang.pdf](/data1/bochen/cbcontinual/iCaRL_codex/2025_CS-CIL_TBME_Yang.pdf)

当前代码库判断：

- 论文里的 replay 选样思想已经部分试过
- `subject_kmeans`、`subject_herding`、`subject_diverse_herding` 在本仓库中都偏弱
- 真正还值得保留的是“对齐信息 replay”这部分启发，而不是继续抠局部选样细节

### 3.2 MIR

核心思想：

- 不是随机 replay，而是优先重放“最容易被当前更新破坏”的样本

来源：

- Aljundi et al., MIR, NeurIPS 2019  
  https://papers.nips.cc/paper_files/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html

为什么适合这里：

- 当前仓库已经有 replay buffer
- 改 retrieval，不改 backbone
- 非常贴合 `task3` 旧类保持这个目标

判断：

- 方向有价值
- 但比当前 `hard replay / MIR-lite` 要更复杂
- 如果后续继续做 replay family，这条仍然值得作为中期目标

### 3.3 DER / DER++

核心思想：

- replay 样本不仅存输入和标签，还存历史 logits
- 在 replay 时直接蒸馏历史输出

来源：

- Buzzega et al., DER / DER++, NeurIPS 2020  
  https://papers.nips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf

为什么适合这里：

- 当前代码已经有 `prev_model` 和 `LwF` 机制
- DER 风格比 MIR 更容易往现有框架里接

当前判断：

- 已做了 `DER-lite`
- short 最好约 `49.27`
- 比当前 strongest short 弱
- 目前先判为“合理但不够强”

### 3.4 X-DER

核心思想：

- 在 DER 基础上进一步做 memory revision 和 class-incremental 校正

来源：

- Boschini et al., X-DER, 2022  
  https://arxiv.org/abs/2201.00766

判断：

- 理论上很强
- 但实现复杂度明显更高
- 对当前毕业设计来说，不是优先补丁，而是后续设计参考

### 3.5 RAR

核心思想：

- 对 replay 样本做重复和增强，提升 rehearsal 强度

来源：

- Zhang et al., Repeated Augmented Rehearsal, NeurIPS 2022  
  https://arxiv.org/abs/2209.13917

判断：

- 思想上适合当前仓库
- 但当前实现出的 `RAR-lite / replay-aware augmentation` 不够强
- 更适合作为“启发”，不建议照搬

## 4. 当前仓库里已经试过的 replay 相关 family

### 4.1 已判弱

- 左右镜像 replay 增强
- `subject_kmeans`
- `subject_herding`
- `subject_diverse_herding`
- `global_kmeans`
- `pca_kmeans`
- `DER-lite`
- `RAR-lite`
- 当前这版 replay-aware mixup

这些方法的问题通常有两种：

1. short 有信号，但 full confirm 站不住
2. 能抬一点 `task2`，但会伤 `task3`

### 4.2 仍可视作“平局备选”的

- `hard replay / MIR-lite`

代表结果：

- `hardreplay10s3_confirm`
- `84.18 / 60.75 / 49.36`

判断：

- 与当前 best full 持平
- 说明“更有针对性的 replay retrieval”方向本身没有错
- 但目前这版实现还没带来额外突破

### 4.3 还有一点启发价值的

- `alignment-memory replay`

代表 short：

- `replaygea_s3`：`84.72 / 61.50 / 49.54`

判断：

- `stage3-only` 有一定价值
- 但没有继续超过主线
- 后续如果继续做 replay，更值得保留的是“对齐信息 replay”的思想，而不是当前具体实现

## 5. 当前最值得继续的 replay 方向

如果后续仍然想做 replay family，我建议按下面顺序：

1. 更正式的 MIR / retrieval-aware replay  
原因：
直接针对“哪些旧样本最容易在 `task3` 被破坏”

2. replay 与偏置校正联动  
原因：
当前实验说明，单独换 replay 不够，必须和 `hybrid / bias / prototype` 一起设计

3. replay 与跨被试对齐联动  
原因：
EEG 场景里真正困难的不是只有旧类保持，还有 subject shift

## 6. 当前总结

截至目前，对这个仓库最合理的结论是：

- replay 仍然重要
- 但“单靠换一种 exemplar 选样法”已经很难突破当前平台
- 真正值得继续的方向是：
  - retrieval-aware replay
  - replay 与分类器校准联动
  - replay 与跨被试对齐联动

也就是说，后续 replay 研究更应该是“联合设计”，而不是继续孤立地微调 exemplar selection。

- Zhang et al., "RehearMixup: Improving rehearsal-based continual learning", Neural Networks 2025  
  https://www.sciencedirect.com/science/article/abs/pii/S0925231225020764

Why it may fit here:

- very easy to plug into the current repo
- acts on replay usage, not replay storage
- matches the current need for lightweight, reversible changes

Cost/risk:

- low
- but class semantics matter for EEG MI data, so mixup can easily blur discriminative motor patterns

Repo verdict:

- worth trying once
- currently under test in this repo as a lightweight replay-aware mixup branch

## Priority Ranking For This Repo

### Highest Priority

1. `DER-lite / DER++-style replay logits`
2. `incremental EA / alignment-memory replay`
3. `replay-aware augmentation or mixup`, but only if class-safe and lightweight

### Medium Priority

1. `MIR` retrieval
2. `RAR`-style repeated augmented rehearsal

### Low Priority

1. `ASER`
2. `gradient-matching coreset replay`
3. full `X-DER` reimplementation

## Current Repo Status

Already tested and weak:

- subject-balanced replay variants
- diverse herding variants
- replay left-right flip augmentation

Currently under test:

- replay-aware mixup on top of the current best `alpha=0.2` hybrid/prototype line

## Recommended Next Replay Experiments

### A. DER-lite for replayed samples

Minimal version:

- store old replay logits at exemplar construction time
- add replay-logit consistency on replayed samples only
- keep current BCE/LwF losses unchanged

Why this is the best next replay family:

- closest to current code
- easiest to explain in a thesis
- directly targets old-class stability on replay data

### B. Alignment-memory replay

Minimal version:

- keep a stage-wise running EA reference per subject group or per stage
- use it only when replayed samples are mixed with current samples

Why this is attractive:

- directly derived from ReCIL
- genuinely cross-subject, not generic image CL

### C. Safer replay mixup variants

If the current mixup branch is weak:

- try `intra-current` only, not `current-replay`
- or mix only within current stage data, while still replaying memory normally

Why:

- RehearMixup reports that not all mixup strategies behave equally well
- EEG MI semantics may dislike cross-memory interpolation
