# 方法 FAQ

日期：`2026-03-16`

这份文档是对当前主线方法的快速解释，重点回答：

- 这个方法具体加在代码哪里
- 它在训练时怎么生效
- 为什么会有效，或者为什么后来被放弃

## 1. adapter 是加在哪里的？哪些阶段训练，哪些阶段不训练？

`adapter` 加在 **embedding 输出之后、transformer 之前**。

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L14)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L365)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L409)

它本质上是一个残差瓶颈层：

- `LayerNorm`
- `Linear down`
- `GELU`
- `Dropout`
- `Linear up`
- 最后做 residual add

是否启用由 `start_task` 决定：

- `start_task=0`：三个阶段都启用
- `start_task=1`：从 `stage2` 开始启用
- `start_task=2`：从 `stage3` 开始启用

早期为了更稳，我先试的是 `adapter16_s2`，也就是从 `stage2` 开始启用。  
但后期最优主线里的 `adapter16` 没有 `s2` 后缀，意味着它其实是 **从 stage1 就启用** 的。

## 2. replaymix2 是什么？

`replaymix2` 不是 mixup，它指的是：

- 每个 batch 固定混入 `2` 个 replay 样本
- 其余样本来自当前阶段新数据

默认 batch size 是 `32`，所以实际是：

- `30` 个当前阶段样本
- `2` 个 replay 样本

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L49)

它的作用是：

- 保证每个 batch 都有旧样本
- 但 replay 比例又不会高到干扰当前任务学习

## 3. 所有实验 batch 内都有做均衡采样吗？

不是。

分两种情况：

1. 没有固定 replay 混合时  
如果 `balance_sample=True`，会走 `WeightedRandomSampler`

代码在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L702)

2. 使用 `replay_batch_size > 0` 时  
这时 loader 变成 `FixedReplayDataLoader`，主要靠“固定 replay/new 比例”来控制 batch 结构，不再走那个 class-balanced sampler。

所以更准确地说：

- 早期很多实验用了 batch 内类均衡采样
- 后期 `replaymix2` 主线不再额外用那个 sampler

## 4. oldweight25 是什么逻辑？

`oldweight25` 表示：

- `old_class_weight_power = 2.5`

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L577)

旧类 loss 权重大致按下面的方式分配：

\[
w_i \propto (old\_k - i)^p
\]

其中：

- `old_k` 是当前旧类数量
- `i` 是旧类索引
- `p` 就是 `oldweight`

这意味着：

- 越老的类，权重越大
- 在 `task3` 时，最老的 `A/B` 会被更重点保护

## 5. 为什么会去试 `10,12,14`、`10,12,12` 这种分阶段 epoch？

这是从现象出发，不是从理论直接推出来的。

原因是：

- `stage1` 最简单，类少，也还没有遗忘问题
- `stage2` 和 `stage3` 才开始真正出现旧类保持与跨被试分布偏移

所以直觉上应当：

- 给后两阶段更多训练预算
- 但也不能无限增加，否则容易过拟合当前阶段、反而伤旧类

后来的实验说明：

- “后两阶段稍微多训一点”是对的
- 但不是越长越好
- 最后最稳的成熟主线反而落在 `10,12,12`

## 6. normalized NME 是什么？为什么提升这么大？

普通 NME 会直接用特征和 prototype 做距离比较。  
`normalized NME` 则是：

- 先把每个样本特征做 L2 归一化
- 再把每个类 prototype 也做 L2 归一化
- 再做最近均值分类

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1568)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1417)

它有效的核心原因是：

- EEG 跨被试特征的幅值漂移比较大
- 但类别信息更稳定地体现在方向结构上
- 归一化后，分类更像是在比较“方向”，而不是比较“长度”

所以这一步对 prototype geometry 修正非常直接，提升也就比较大。

## 7. hybrid NME+logits 是什么？

这是为了解决：

- 训练时优化 `fc logits`
- 测试时主要用 `NME`

两者不一致的问题。

做法是：

- 同时算 `NME scores` 和 `fc scores`
- 先对每个样本的两组分数分别标准化
- 然后做线性融合：

\[
score = \alpha \cdot score_{nme} + (1-\alpha)\cdot score_{fc}
\]

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1568)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1600)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1627)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1703)

## 8. stage3 feature distill 是什么？为什么只在 stage3 蒸馏？

这是 feature-level distillation。

做法是：

- 用当前模型抽特征
- 用上一阶段冻结模型抽特征
- 对旧类样本做 feature cosine consistency

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L659)

为什么只在 `stage3` 蒸馏：

- `stage2` 时遗忘还没那么严重
- `stage3` 同时面对：
  - 最老的 `A/B`
  - 中间类 `C`
  - 新类 `D`
- 这是特征空间最容易整体漂移的时候

实验也说明：

- 全阶段蒸馏不如集中在 `stage3`

## 9. task-affine_s3 是什么？

它是在 `PatchEmbedding` 中卷积特征之后、BN 之后加的一组任务专属仿射参数：

\[
x = x \cdot scale + bias
\]

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L117)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L165)

`task-affine_s3` 的意思是：

- 这组任务专属 affine 只从 `stage3` 开始启用

后来的更强版本是 `affine2`：

- 也就是从 `stage2` 开始启用

## 10. new_only prototype blend 是什么？

prototype blend 的意思是：

- 一方面用 exemplar memory 算 prototype
- 一方面用当前阶段数据算 prototype
- 然后做加权融合

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1427)

`new_only` 的意思是：

- 不是对所有当前类都做 blend
- 只对真正新增的类做 blend

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1458)

这个设计的直觉是：

- 旧类 prototype 已经相对稳定
- 真正不稳定的是刚加入的新类
- 所以只修新类更合理

## 11. task_shared adapter 用到过吗？最优方法里有吗？测试时 task-spec 模块怎么选？

`shared adapter` 用过，但只试过一条比较早的 short：

- `no_contrastive_shared8_task16_s2_lr15_short`
- 结果：`85.65 / 55.86 / 42.28`

结论是：

- 没有进入后期最优主线
- 当前最优方法里 **没有** `shared adapter`

测试时 task-specific 模块怎么选：

- 不是按样本动态选
- 而是按“当前阶段”统一选

代码在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L347)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L434)

也就是说：

- `stage1` 测试时统一用 task 0 参数
- `stage2` 测试时统一用 task 1 参数
- `stage3` 测试时统一用 task 2 参数

最终 `task3` 全测试集也是统一用 `stage3` 这套参数，不会对 A/B/C/D 再分别切不同 adapter。

## 12. stage1 也加 Adapter，尝试过吗？效果怎么样？

尝试过。

一个很重要的点是：

- 早期证明 adapter 有用的代表实验是 `adapter16_s2`
- 但后期真正最优的方法里，实验名只有 `adapter16`，没有 `s2`
- 这意味着后期最优主线里的 adapter 实际上就是 **从 stage1 开始启用**

所以结论不是“stage1 不能加 adapter”，而是：

- 早期为了稳，先从 `stage2` 开始试
- 后期在更强主线里，`stage1` 也加 adapter 反而是可以接受甚至更优的

## 13. stage3-only hybrid 这个 hybrid，有没有在阶段1和2尝试过？效果怎么样？

有试过更早启用的版本。

可以分成三种理解：

1. 从 `stage2` 就启用 hybrid  
这就是实验名里没有 `s3` 的 `hybrid` 版本。

代表 full：

- `lwf0175T15_normnmehybrid_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm`
- `85.72 / 59.31 / 48.59`

2. 从 `stage3` 才启用 hybrid  
代表 full：

- `lwf0175T15_normnmehybrid_s3_adapter16_mem36_oldweight3375_stage101212_stagefd0025_taskaffine_s3_confirm`
- `85.72 / 59.62 / 48.59`

比较结果很清楚：

- `task3` 持平
- `task2` 反而是 `stage3-only hybrid` 更高

所以后来保留的是：

- `stage3-only hybrid`

3. 从 `stage1` 就启用 hybrid  
这个方向基本没有意义，也没有成为主实验。

原因很简单：

- `stage1` 只有最初两类
- 这时 old/new classifier mismatch 还不是真问题
- hybrid 的价值主要出现在后续增量阶段，尤其 `stage3`

所以它不是“没想到”，而是：

- 从问题本身看，`stage1` 不值得优先做 hybrid
- 实验上也证明 `stage3-only` 更合理
