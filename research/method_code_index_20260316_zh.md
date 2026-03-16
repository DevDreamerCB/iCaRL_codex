# 方法与代码索引

日期：`2026-03-16`

这份文件回答两个问题：

1. 某个方法在代码里是怎么打开的
2. 主要改动落在哪些文件

## 1. 主要代码入口

- 训练与环境变量入口：[main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py)
- 自动实验入口：[auto_experiment.py](/data1/bochen/cbcontinual/iCaRL_codex/auto_experiment.py)
- 增量学习主逻辑：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)
- 数据与 EA 对齐：[midata.py](/data1/bochen/cbcontinual/iCaRL_codex/midata.py)

## 2. 当前重要方法对应的开关

### 去掉对比损失

- 环境变量：`ICARL_USE_CONTRASTIVE=false`
- 代码位置：[main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py)、[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### LwF

- 环境变量：
  - `ICARL_USE_LWF=true`
  - `ICARL_LWF_LAMBDA`
  - `ICARL_LWF_T`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### old-class weighted BCE

- 环境变量：
  - `ICARL_OLD_CLASS_WEIGHT_POWER`
  - `ICARL_STAGE_OLD_CLASS_WEIGHT_POWERS`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### normalized NME

- 环境变量：`ICARL_USE_NORMALIZED_NME=true`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### hybrid NME + logits

- 环境变量：
  - `ICARL_USE_HYBRID_NME_LOGITS=true`
  - `ICARL_HYBRID_START_TASK`
  - `ICARL_HYBRID_OLD_WEIGHT`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### task adapter

- 环境变量：
  - `ICARL_USE_TASK_ADAPTER=true`
  - `ICARL_TASK_ADAPTER_DIM`
  - `ICARL_TASK_ADAPTER_START_TASK`
- 代码位置：
  - [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py)
  - backbone 初始化部分

### task affine

- 环境变量：
  - `ICARL_USE_TASK_AFFINE=true`
  - `ICARL_TASK_AFFINE_START_TASK`
- 代码位置：
  - [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py)
  - backbone 初始化部分

### stage-only feature distillation

- 环境变量：
  - `ICARL_USE_FEATURE_DISTILL=true`
  - `ICARL_STAGE_FEATURE_DISTILL_LAMBDAS`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### current prototype blend

- 环境变量：
  - `ICARL_USE_CURRENT_PROTOTYPE_BLEND=true`
  - `ICARL_CURRENT_PROTOTYPE_BLEND_ALPHA`
  - `ICARL_CURRENT_PROTOTYPE_BLEND_START_TASK`
  - `ICARL_CURRENT_PROTOTYPE_BLEND_SCOPE`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### replay logits distill / DER-lite

- 环境变量：
  - `ICARL_USE_REPLAY_LOGITS_DISTILL=true`
  - `ICARL_REPLAY_LOGITS_LAMBDA`
  - `ICARL_REPLAY_LOGITS_START_TASK`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### replay global EA / alignment-memory replay

- 环境变量：
  - `ICARL_USE_REPLAY_GLOBAL_EA=true`
  - `ICARL_REPLAY_GLOBAL_EA_START_TASK`
- 代码位置：
  - [midata.py](/data1/bochen/cbcontinual/iCaRL_codex/midata.py)
  - [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### hard replay / MIR-lite

- 环境变量：
  - `ICARL_USE_REPLAY_HARDNESS=true`
  - `ICARL_REPLAY_HARDNESS_POWER`
  - `ICARL_REPLAY_HARDNESS_START_TASK`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### RAR-lite / repeated rehearsal

- 环境变量：
  - `ICARL_USE_REPLAY_REPEAT=true`
  - `ICARL_REPLAY_REPEAT_LAMBDA`
  - `ICARL_REPLAY_REPEAT_START_TASK`
- 代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

### exemplar selection family

- 环境变量：
  - `ICARL_EXEMPLAR_MODE`
  - `ICARL_EXEMPLAR_MODE_START_TASK`
  - `ICARL_EXEMPLAR_DIVERSITY_LAMBDA`

常见取值：

- `legacy_herding`
- `global_kmeans`
- `pca_kmeans`
- `subject_kmeans`
- `subject_herding`
- `subject_diverse_herding`

代码位置：[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py)

## 3. 如何从实验名反推方法

实验名一般遵循这样的命名规则：

- `lwf0175`：`LwF lambda = 0.175`
- `T15`：`LwF temperature = 1.5`
- `normnme`：使用 normalized NME
- `hybrid`：使用 NME+logits 融合
- `s3old04`：hybrid old-weight 为 `0.4`
- `ptblend02s3new`：prototype blend `alpha=0.2`，从 `stage3` 开始，仅修正新增类
- `affine2`：task affine 从 `stage2` 开始
- `adapter16`：task adapter 维度为 `16`
- `mem36`：memory size `36`
- `oldweight3125`：old-class weight power `3.125`
- `stage101212`：各阶段 epoch 为 `10,12,12`
- `stagefd003`：`stage3 feature distill = 0.03`
- `hardreplay10s3`：困难重放采样，power `1.0`，从 `stage3` 开始

## 4. 现在怎么看“某个方法对应的代码”

最简单的方法是两步：

1. 先在 [metrics/experiments.csv](/data1/bochen/cbcontinual/iCaRL_codex/metrics/experiments.csv) 里找到实验名
2. 再按上面的命名规则和环境变量，到 [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py)、[auto_experiment.py](/data1/bochen/cbcontinual/iCaRL_codex/auto_experiment.py)、[iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py) 里查开关

后面如果继续严格做 git 提交和 tag，这份文件就能和 commit/tag 一起构成稳定的“方法索引”。
