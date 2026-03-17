# 已验证不可行或性价比很低的方法

日期：`2026-03-17`

说明：

- 本文整理的是截至当前已经尝试、但没有进入主线的方法。
- 这里只保留对后续写论文或继续做实验真正有参考价值的负结果。
- 指标顺序统一为：`task1 / task2 / task3`
- 结果优先使用 `3 seeds + 30 epochs` 的 `full`；若该方向没有升到 `full`，则注明是 `short`

## 1. 结构层面无效或偏弱的方法

### 1.1 只训练 embedding

- 运行名：`embedding_only_confirm`
- 结果：`83.33 / 51.08 / 37.65`
- 判断：
  - 明显不如原始基线
  - 说明单纯冻结 transformer、只调 embedding 不能解决当前增量遗忘问题

### 1.2 Prototype alignment

- 运行名：`proto_align_confirm`
- 结果：`85.80 / 59.59 / 35.65`
- 判断：
  - `task3` 明显变差
  - 说明简单把 replay 特征拉向旧 prototype，会破坏后期判别性

### 1.3 早期 task-BN / task-affine / prompt / LoRA

代表结果：

- `lwf015_normnme_adapter16_mem36_oldweight30_stage101212_taskbn_s3_confirm`
  - `84.18 / 60.85 / 47.22`
- 早期 `task-affine` full
  - `84.18 / 60.39 / 47.49`

判断：

- 这些方法不是完全没用，但都不如后期的 `affine2 + hybrid + ptblend` 主线
- 因此不再单独作为研究重点

## 2. 重放相关但未形成正式突破的方法

### 2.1 no-balance / sampler 相关

- `no_contrastive_nobalance_confirm`
  - `85.57 / 60.73 / 40.99`

判断：

- 能带来一些局部提升，但 full 不稳定
- 后续 corrected pipeline 下没有成为长期主线

### 2.2 replay 左右镜像增强

- 只在 `short` 有少量信号，未超过更强主线
- 判断：
  - 想法合理，但在当前实现下收益不稳定
  - 不适合作为主方法

### 2.3 subject-balanced replay / diverse herding

代表结果：

- `subject_kmeans_short`：`86.34 / 61.81 / 47.76`
- `subject_herding_short`：`84.72 / 58.80 / 45.95`
- `subject_diverse_herding_short`：`84.72 / 59.72 / 46.80`

判断：

- 思路来自跨被试多样性，但在当前代码库里都不如现有主线
- 说明“被试多样性”不能只靠 exemplar 选样硬注入

### 2.4 PCA / global kmeans exemplar selection

- `globalkmeans_short`：`84.72 / 61.50 / 49.50`
- `pcakmeans_short`：`84.72 / 61.50 / 49.50`

判断：

- 很接近强 short，但没有形成比主线更强的 full
- 可作为“值得提到但未最终采用”的方法

### 2.5 DER-lite

代表 short：

- `derlite05s3`：`84.72 / 61.50 / 49.00`
- `derlite02s3`：`84.72 / 61.50 / 49.27`
- `derlite01s3`：`84.72 / 61.50 / 49.07`

判断：

- 这条 replay family 不是完全错误
- 但没有超过当前最强 short，也没有值得升 `full`

### 2.6 alignment-memory replay

代表 short：

- `replaygea_s3`：`84.72 / 61.50 / 49.54`
- `replaygea_s2`：`84.72 / 59.57 / 39.89`

判断：

- 只从 `stage3` 开始有一定信号
- 提前到 `stage2` 会明显伤整体平衡
- 最终没有超过主线

### 2.7 hard replay / MIR-lite

- `hardreplay10s3_confirm`
  - `84.18 / 60.75 / 49.36`

判断：

- 与当前 best full 持平
- 说明它是合理备选，但没有带来新增突破

### 2.8 replay-aware mixup / RAR-lite

- `mixup02l05_short`：`84.72 / 61.50 / 49.11`
- `rarlite05s3_short`：`84.72 / 61.50 / 49.07`

判断：

- 都弱于当前主线
- 当前这类 replay augmentation 方案不值得继续深挖

## 3. 原型分类器拓展中已判弱的方法

### 3.1 age-NME / radius-NME

代表结果：

- `agenme01_confirm`：`84.64 / 59.85 / 46.97`
- `radius025_confirm`：`84.18 / 61.21 / 46.50`

判断：

- 能改变 `task2/task3` 平衡
- 但都没有超过后来 `normalized NME + hybrid` 主线

### 3.2 FeCAM 风格 diag-cov NME

- `fecamdiag01_short`：`84.49 / 60.42 / 47.22`

判断：

- 在当前 EEG 特征空间上明显偏弱
- 说明协方差距离并不适合这套 backbone 和 exemplar 设定

## 4. 偏置校正中未成为主线的方法

### 4.1 group-bias

代表结果：

- `gbias05_confirm`：`84.18 / 61.01 / 48.63`
- `gbias06_confirm`：`84.18 / 61.01 / 48.59`

判断：

- 这是一个不错的“更平衡备选线”
- 它能把 `task2` 拉高到 `61+`
- 但 `task3` 比当前 best 主线弱，因此不作为最终主线

### 4.2 早期 hybrid 近邻过度微调

包括：

- 过大的 `hybrid_old_weight`
- 过强的 `alpha max`
- 提前到 `stage2` 的 hybrid

判断：

- 这些都会伤 `task2` 或破坏后期平衡
- 后来保留下来的版本是 `stage3-only hybrid`

## 5. 当前总体判断

到目前为止，真正“明显不可行”与“暂时不值得继续”的方法可以分成两类：

1. **明确负收益**
- embedding-only
- prototype alignment
- FeCAM diag-cov

2. **有一定信号，但不如当前主线**
- DER-lite
- alignment-memory replay
- hard replay / MIR-lite
- group-bias
- subject-balanced replay
- PCA/global kmeans exemplar 选择

如果后续还要继续做实验，建议优先投入在：

- 更强的 old/new 偏置校正
- 更稳的 prototype refinement
- 更贴 EEG 跨被试的 subject-invariant 约束

而不是继续在上述已判弱方法上重复投入算力。
