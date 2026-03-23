# 当前主线方法说明（固定预算口径）

日期：`2026-03-23`

## 1. 本文档的目的

这份文档专门解释当前固定预算标准下的主线方法：

- 它到底由哪些组件组成
- 每个组件在什么阶段生效
- 具体改了什么 loss / prototype / classifier
- 哪些类别在什么阶段被特殊处理
- 对应的代码位置在哪里

目标是把方法写得比实验总结更细，方便后续：

- 写论文方法章节
- 做答辩 PPT
- 回看代码时快速定位

## 2. 当前主比较标准

当前主比较标准已经统一为：

- `stage_epochs = [10, 12, 12]`
- `screen = 1 seed`
- `full = 3 seeds`

原因是：

- 这个口径更符合“连续学习固定增量训练预算”的比较习惯
- 最近那批默认 `30/30/30` 的结果，会把“方法效果”和“训练时长效果”混在一起
- 对当前主线来说，`10,12,12` 更能反映旧任务保持与新任务学习之间的真实平衡

## 3. 当前最可信主线

当前固定预算下最可信的 full 是：

- `s2ace00_ptblendnew02_stage101212_confirm`
- `84.18 / 60.70 / 49.40`

如果只看 one-seed short，同口径主线是：

- `s2ace00_ptblendnew02_stage101212_short`
- `84.72 / 60.96 / 50.54`

## 4. 场景回顾

当前默认持续学习场景是：

1. `stage1`
   - train：subjects `1,2,3`
   - classes：`A,B`
2. `stage2`
   - train：subjects `4,5,6`
   - classes：`B,C`
3. `stage3`
   - train：subjects `7,8,9`
   - classes：`C,D`

类别映射：

- `A = Left Hand`
- `B = Right Hand`
- `C = Both Feet`
- `D = Tongue`

## 5. 当前主线背后的核心问题判断

最近这条主线不是靠继续做 `task3` 小修补得出来的，而是先把问题重新定义清楚了。

### 5.1 为什么 `task2` 是真正瓶颈

我们额外跑过 `subjects 4,5,6` 的 joint 训练上界。结果说明：

- `task2` 现在卡在 `60.x`
- 不是因为 `4/5/6` 完全学不会
- 而是 continual 场景下，旧类 `A=Left Hand` 没有成功迁移到新域 `4/5/6`

换句话说：

> `task2` 的核心问题，不是简单的“旧类会忘”，  
> 而是：当前 task 里没有 `A`，但模型又在当前样本上不断把 `A` 当成负类压低。

这就是后面 `s2ace00` 的出发点。

## 6. 当前主线的完整组成

当前主线可以拆成下面几部分。

### 6.1 `s2ace00`：stage2 asymmetric BCE

这是当前主线里最重要的新点。

配置含义：

- `stage_absent_old_current_weights = [1, 0, 1]`

解释：

- `stage1`: 正常 BCE
- `stage2`: 对“当前 task 中缺失的旧类”不再施加当前样本负压力
- `stage3`: 正常 BCE

在当前场景里，`stage2` 当前类是 `B,C`，旧类是 `A,B`，所以：

- `stage2` 里真正被特殊处理的是 `A`
- 在 `stage2` 的当前样本上，不再让 `A` 承受 BCE 负项

### 6.2 `normalized NME`

当前主线保留了 `normalized NME`：

- 测试时先提特征
- 样本特征和 class prototype 都做 L2 normalize
- 再按最近均值分类

它的意义是：

- 把 prototype classifier 的几何从“同时受长度和方向影响”
- 改成更偏“方向相似性”

对于跨被试 EEG，这通常比原始 NME 更稳。

### 6.3 `hybrid NME + logits`

当前主线不是只用 NME，也不是只用 FC logits，而是两者融合：

- 一边是 prototype / NME 的几何信息
- 一边是训练得到的 FC decision boundary

具体做法是：

1. 分别算 `nme_scores` 和 `fc_scores`
2. 对每个样本做行归一化
3. 按 `alpha` 融合
4. 用 calibration set 自动选最优 `alpha`

也就是说：

> 它不是手工拍脑袋给一个固定权重，而是每阶段自动校准。

### 6.4 `new_only prototype blend @ stage3`

当前主线在 `stage3` 做 prototype refinement，但已经从更复杂的 `split ptblend` 简化成：

- `current_prototype_blend_scope = new_only`
- `current_prototype_blend_alpha = 0.2`
- `current_prototype_blend_start_task = 3`

意思是：

- 只在 `stage3`
- 只修真正新增类 `D=Tongue` 的 prototype
- 不去动所有当前类

这个设计的直觉是：

- memory 里的旧类 prototype 大体已经可靠
- 真正最不稳定的是最后新加入的类
- 所以只修新增类，更符合类增量语义，也比 `split/current/overlap` 那些版本更简洁

### 6.5 `task adapter`

当前主线保留：

- `task adapter`
- `dim = 16`

它的本质是：

- 在 embedding 输出后、进入 transformer 前
- 为每个阶段提供一套小型残差瓶颈模块

当前代码里：

- 按阶段统一切换，不是按样本动态路由
- 它是 **stage-conditioned parameterization**
- 不是类别标签泄漏

### 6.6 `task affine`

当前主线也保留：

- `task affine`

它加在 `PatchEmbedding` 里，具体是：

- 卷积 + BN 后
- 对当前阶段施加一组可学习的 `scale + bias`

它的作用更像：

- 给不同增量阶段一点轻量的分布校正自由度

它不是当前主线的绝对核心，但有中等增益。

### 6.7 `LwF`

当前主线保留显式 `LwF`：

- `lambda = 0.175`
- `T = 1.5`

它的作用不是替代 iCaRL 原生 soft target，而是叠加在其上的一层显式旧类蒸馏。

### 6.8 `stage3-only feature distill`

当前主线保留：

- `stage_feature_distill_lambdas = [0, 0, 0.03]`

也就是：

- `stage1` 不用
- `stage2` 不用
- `stage3` 才启用

它只对：

- `old_mask = labels < old_k`

这部分旧类样本生效，做 feature-level cosine consistency。

它现在更像：

- 最后阶段稳定器

而不是一个真正可推广到所有阶段的通用组件。

### 6.9 `oldweight = 3.125`

当前主线保留：

- `old_class_weight_power = 3.125`

这个机制是：

- 对旧类 BCE 部分做按“旧类年龄”递增的加权
- 越老的类权重越高

在当前场景里，它的作用主要是：

- 保护 `A/B` 这种更老的类

## 7. 这个方法在三个阶段分别做了什么

这一节按阶段拆开讲。

### 7.1 `stage1`

当前训练类：

- `A,B`

这时没有旧类，所以：

- `s2ace00` 不起作用
- `LwF` 不起作用
- `feature distill` 不起作用

这阶段主要是在已有预训练 backbone 上，建立初始的 `A/B` 表示和 memory。

### 7.2 `stage2`

当前训练类：

- `B,C`

测试类：

- `A,B,C`

这是当前主线最关键的阶段，因为：

- 当前样本中 `A` 完全缺失
- 但模型又必须保住 `A`
- 同时还要适应新域 `subjects 4,5,6`

在这个阶段里，关键动作是：

1. **`s2ace00`**
   - `A` 在当前样本 BCE 中不再被当负类压下去
2. replay
   - memory 中的 `A/B` exemplar 被继续带回 batch
3. iCaRL soft target + LwF
   - 继续维持旧类输出
4. oldweight
   - 对更老旧类给更高权重

所以 `stage2` 的核心不是“加更多花活”，而是：

> 不要让缺失旧类 `A` 在当前 `B/C` 样本上持续被错误压制。

### 7.3 `stage3`

当前训练类：

- `C,D`

测试类：

- `A,B,C,D`

这时问题变成：

- old/new/overlap 关系更复杂
- 既有缺失旧类 `A,B`
- 也有 overlap 类 `C`
- 再加一个新类 `D`

当前主线在这里采取的是：

1. 不把 `s2ace` 机械推广到 `stage3`
2. 保留 `hybrid NME+logits`
3. 保留 `stage3-only feature distill`
4. 用 `new_only ptblend` 只修新类 `D`

这也是为什么最近实验说明：

- `s2ace` 在 `stage2` 很有效
- 但简单推广成 `stage3 asymmetric BCE` 并没有带来收益

## 8. 这个方法里，哪些类别的 loss 被“特殊处理”了

这一节单独回答“哪些类别被怎样处理”。

### 8.1 `stage2 asymmetric BCE`

在 `stage2`：

- 当前类：`B,C`
- 旧类：`A,B`
- 缺失旧类：`A`

所以：

- `A` 在当前 `B/C` 样本上的 BCE 负项被置零
- `B` 不会被屏蔽，因为它是 overlap/current 类

所以更准确地说：

> 被特殊处理的不是“所有旧类”，而是 **当前 task 中缺失的旧类**。  
> 在这个场景下，`stage2` 实际上就是在特殊处理 `A=Left Hand`。

### 8.2 `stage3`

在 `stage3`：

- 当前类：`C,D`
- 缺失旧类：`A,B`

理论上可以把同样的 asymmetric BCE 推广到这里。  
我已经试过：

- `stage3 absent-old weight = 0.25`
- `stage3 absent-old weight = 0.0`

但都没有超过同口径主线。  
所以当前结论是：

- `stage2` 的缺失旧类负压制问题是特别强的结构性问题
- `stage3` 不能简单套用同一规则

## 9. 代码映射

下面把当前主线的重要机制映射到代码位置。

### 9.1 配置入口

环境变量读取在：

- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L149)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L155)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L159)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L165)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L169)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L182)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L191)

主要包括：

- `ICARL_STAGE_ABSENT_OLD_CURRENT_WEIGHTS`
- `ICARL_OLD_CLASS_WEIGHT_POWER`
- `ICARL_USE_NORMALIZED_NME`
- `ICARL_USE_HYBRID_NME_LOGITS`
- `ICARL_CURRENT_PROTOTYPE_BLEND_*`
- `ICARL_TASK_ADAPTER_*`
- `ICARL_USE_TASK_AFFINE`

模型构造与总配置打印在：

- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L225)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L241)
- [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py#L263)

### 9.2 `stage2 asymmetric BCE`

核心函数：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L473)

关键逻辑：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L479)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L494)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L496)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L502)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L507)

这一段就是：

1. 先算 `bce_matrix`
2. 找出当前阶段里缺失的旧类
3. 对当前样本上这些类的 BCE loss 乘一个更小的系数

### 9.3 oldweight

旧类权重在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L509)

### 9.4 feature distill

旧类 feature cosine distill 在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L549)

关键点：

- 只对 `old_mask = labels < old_k` 生效

### 9.5 prototype blend

prototype 计算与 blend 在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1143)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1165)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1174)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1188)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1191)

### 9.6 hybrid NME + logits

NME score / FC score / fusion / calibration 在：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1348)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1366)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1375)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1402)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L1445)

### 9.7 task adapter / task affine

`task adapter` 定义在：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L9)

`task affine` 定义在：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L65)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L113)

embedding 后进入 adapter 的前向路径在：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L271)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L273)
- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L275)

阶段切换在：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L296)

## 10. 当前对主线方法的最终判断

现在比较清楚的结论是：

1. 真正最有价值的新点，不是继续修 `stage3`
2. 而是把 `stage2` 的核心问题重新定义成：
   - 缺失旧类 `A` 在当前 `B/C` 样本上被错误压制
3. `s2ace00` 正是对这个问题的最小、最经典、也最有效的修正
4. 修完这个以后，`stage3` 端只需要保留较简洁的：
   - `new_only ptblend`
   - `hybrid`
   - `stage3 feature distill`

所以当前主线最适合的叙事不是：

- “我们又加了很多 stage3 trick”

而是：

- “我们先用一个更通用、更文献可解释的 `stage2 missing-class treatment` 修掉真正瓶颈，  
  再用少量必要的 stage3 稳定组件完成最终平衡。”

## 11. 一句话总结

如果只用一句话概括当前方法，可以写成：

> 当前方法的核心是在 `stage2` 用 asymmetric BCE 阻止缺失旧类 `A` 被当前样本错误压制，再结合 `normalized NME + hybrid` 的稳定决策、`new_only prototype blend` 的最后阶段新类修正，以及少量阶段条件化参数与蒸馏项，构成一个比早期 `task3 specialist` 路线更通用、也更容易解释的持续学习方法。
