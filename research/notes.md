# 研究笔记

日期：`2026-03-17`

本文件只保留当前仍有参考价值的研究结论，不再保留早期自动研究器产生的大量事件流日志。

## 当前最佳正式结果

- 运行名：`lwf0175T15_normnmehybrid_s3old04_ptblend02s3new_affine2_adapter16_mem36_oldweight3125_stage101212_stagefd003_taskaffine_s3_confirm`
- 指标：`task1=84.18, task2=60.75, task3=49.36`
- 综合分：`59.74`

## 当前主线

当前最值得继续保留和解释的主线组合为：

- 去掉对比损失
- `LwF(lambda=0.175, T=1.5)`
- `old_class_weight_power=3.125`
- `normalized NME`
- `stage3-only hybrid NME+logits`
- `task adapter(dim=16)`
- `task affine` 从 `stage2` 开始
- `stage3-only feature distill=0.03`
- `memory=36`
- `stage_epochs=[10,12,12]`
- `stage3-only + new_only prototype blend alpha=0.2`

## 已明确有效的方向

- 去掉当前实现的对比损失
- 轻量 `adapter`
- `LwF + oldweight` 的组合
- `normalized NME`
- 更合理的分阶段 epoch
- `stage3-only feature distill`
- `task-affine`
- `stage3-only hybrid NME+logits`
- `new_only stage3 prototype blend`

## 已基本判弱的方向

- 只训 embedding
- prototype alignment
- 当前实现的 supervised contrastive loss
- 早期 prompt / LoRA / task-BN
- 左右镜像 replay 增强
- subject-balanced replay / subject-diverse herding
- PCA / global kmeans exemplar 选择
- WA
- 当前这版 replay-aware mixup
- FeCAM 风格的 diag-cov NME
- DER-lite
- alignment-memory replay
- RAR-lite

## 对后续实验的判断

- 单纯围绕 exemplar replay 再做小修小补，收益已经接近平台区。
- 如果继续提升，优先考虑：
  - 更强的 old/new 偏置校正
  - 更稳的 prototype refinement
  - 更贴 EEG 跨被试的 subject-invariant 约束
  - replay 与分类器校准的联合设计
2026-03-23 analysis: BiC-lite short (84.72/62.19/49.34) did not beat the new main line, so the post-hoc scale/shift idea is rejected for now. nofd full (84.18/61.57/49.43) shows stage3 feature distill is still useful and should stay in the main line. noaffine short (84.72/62.19/49.11) suggests task affine helps modestly but is not the primary driver; next step is to test noadapter to separate the two stage-conditioned parameter blocks.
2026-03-23 later analysis: after `s2ace00` fixed the main stage2 bottleneck, simply adding stage2 epochs (10,14,12) or EEIL-style stage2 balanced finetune did not beat the main line. The strongest new signal is `s2curce010`: a small stage2-only current-class CE over active classes `B/C` on current-domain samples. It reaches 84.72/63.50/49.58 in short and is currently the most promising literature-aligned stage2 extension on top of the `s2ace00 + new_only ptblend` line.
