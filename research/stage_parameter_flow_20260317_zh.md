# 分阶段参数流说明

日期：`2026-03-17`

这份文档专门解释：

- 整个 `iCaRL_codex` 的主干代码是怎么跑的
- 每个阶段到底切换了哪些参数
- `task-specific adapter / task-affine` 在后续阶段还有没有用
- 这是否构成 task id 泄漏

## 1. 整体流程

当前实现的主流程是：

1. 在 [main.py](/data1/bochen/cbcontinual/iCaRL_codex/main.py) 中读取环境变量，构建模型和 `CBiCaRL`
2. 对每个 `seed`
3. 对每个 `stage=1,2,3`
4. 调用 [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py) 的 `beforeTrain(stage)`
5. 在 `beforeTrain(stage)` 里：
   - 设置当前阶段训练/测试被试和类别
   - 构建 dataloader
   - 如果 `stage>1`，复制一份 `prev_model` 作为 teacher
   - 把当前模型的分类头扩展到新类别数
   - 关键一步：`self.model.feature.set_current_task(self.stage - 1)`
6. 然后进入 `train()`
7. 每个 stage 结束后更新 exemplar、class mean、结果记录
8. 最终评估用的是 `NME / hybrid` 这一套分类逻辑

关键调用位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L324)
- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L353)

## 2. 模型结构里哪些部分是“阶段相关”的

当前 `mlm_mask` 主体由这些部分组成：

- `PatchEmbedding`
- `TransformerEncoder`
- `clshead`
- 可选的轻量 task-specific 模块：
  - `task adapter`
  - `shared adapter`
  - `task prompt`
  - `task LoRA`
  - `task affine`
  - `task BN`

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L322)

## 3. `current_task` 是怎么传进去的

每个阶段开始时，会执行：

```python
self.model.feature.set_current_task(self.stage - 1)
```

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L353)

然后在特征提取器里：

- `embedding.set_current_task(task_id)`
- transformer 里所有带 `set_current_task` 的子模块都切过去
- `shared_adapter / task_adapter / task_prompt` 也切过去

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L434)

因此三阶段对应关系是：

- `stage1 -> current_task = 0`
- `stage2 -> current_task = 1`
- `stage3 -> current_task = 2`

## 4. 每个阶段到底会用哪套参数

### 4.1 task adapter

`TaskEmbeddingAdapter` 内部是：

- 一个共享 `LayerNorm`
- `num_tasks` 套 `down`
- `num_tasks` 套 `up`

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L13)

前向逻辑是：

```python
hidden = self.down[self.current_task](x_norm)
return x + self.up[self.current_task](hidden)
```

也就是说：

- `stage1` 只用 `down[0], up[0]`
- `stage2` 只用 `down[1], up[1]`
- `stage3` 只用 `down[2], up[2]`

如果 `start_task` 还没到，就整个 adapter 直接跳过。

### 4.2 task affine

`PatchEmbedding` 里有：

- `task_scale[num_tasks, 128]`
- `task_bias[num_tasks, 128]`

前向逻辑是：

```python
scale = self.task_scale[self.current_task]
bias = self.task_bias[self.current_task]
x = x * scale + bias
```

代码位置：

- [mlm.py](/data1/bochen/cbcontinual/iCaRL_codex/mlm.py#L165)

所以：

- `stage1` 用 `task_scale[0], task_bias[0]`
- `stage2` 用 `task_scale[1], task_bias[1]`
- `stage3` 用 `task_scale[2], task_bias[2]`

### 4.3 task BN / task prompt / task LoRA

这些模块也是完全同样的逻辑：

- 按当前 `current_task` 只取一套参数
- 不会在同一次前向里同时混用多套

## 5. 那是不是 `stage3` 训练时，`task1/task2` 的 adapter 参数就没用了？

结论是：

- **对 `stage3` 这一阶段的前向计算来说，是的，旧 task 的 adapter 参数不再被使用**
- **它们也不会在 `stage3` 继续得到梯度更新**

原因很直接：

1. 前向里只索引 `self.current_task`
2. `stage3` 时 `current_task=2`
3. 所以只会走第 3 套参数

例如 `task adapter`：

- `stage3` 只会走 `down[2], up[2]`
- `down[0], up[0], down[1], up[1]` 都不会参与当前前向

这意味着：

- 它们虽然还在模型里
- 但对 `stage3` 的训练和测试都不直接起作用

## 6. 那旧 task 参数会不会还被优化器更新？

基本不会。

虽然优化器把整组参数都放进去了，但梯度只会出现在“这次前向真正被用到的参数”上。

当前训练代码里，优化器分组只是按：

- `adapter_params`
- `base_params`

来分学习率，不会细分到“只更新当前 task 的第几套参数”。

代码位置：

- [iCaRL.py](/data1/bochen/cbcontinual/iCaRL_codex/iCaRL.py#L767)

但因为前向只用了当前 task 的那一套，所以：

- `stage3` 时旧 adapter 参数没有梯度
- 因而也不会被实际更新

## 7. 这说明了什么

这说明当前这套 `task-specific adapter / affine` 的真实作用，更像是：

- **阶段特定的优化支架**
- 而不是“最终推理时一直联合工作的多专家系统”

更直白地说：

- `stage1` 时，task0 参数帮助把第一阶段训好
- `stage2` 时，task1 参数帮助把第二阶段训好
- `stage3` 时，真正决定最后结果的，主要是 task2 参数

所以你会有一种“前两套参数后面岂不是没用了”的感觉，这是对的。

它们不是完全没价值，因为：

- 它们在各自阶段参与过训练
- 影响过当时学到的 backbone / classifier / exemplar 选择 / teacher 模型

但从最终 `stage3` 推理本身看：

- 旧 task 的 adapter / affine 参数本身不会直接被调用

## 8. 这算不算设计问题

从研究角度说，这更像是一个**设计 tradeoff**，不是简单的 bug。

好处：

- 很容易实现
- 对跨阶段分布偏移有帮助
- 不需要复杂的样本级 task 路由

问题：

- 最终 `stage3` 推理时，旧 task 分支不再直接发挥作用
- 因而这些模块更像“阶段条件化参数”，不是长期共享知识库
- 如果你特别强调“单一统一模型最终推理”，它就不够干净

## 9. 这算不算 task id 泄漏

不算标签泄漏，但算阶段条件化。

因为：

- 它没有用类别真值
- 没有用样本级 task 标签
- 只是用“当前系统处在第几个增量阶段”这个全局信息

更准确的表述是：

- **stage-conditioned parameters**

不是：

- sample-wise task routing
- label leakage

## 10. 论文里怎么写更稳妥

建议这样写：

> 本文中引入的 task-affine 与 task-specific adapter 并不使用样本级任务标签进行动态路由，而仅依赖当前增量阶段这一全局状态，因此不属于类别标签泄漏；但它们本质上属于阶段条件化参数设计。

## 11. 如果想进一步验证这个问题，后面最值得补的对照

可以补三类实验：

1. `stage3` 测试时禁用 task-specific adapter / affine  
看最终 `task3` 掉多少。

2. 只保留 shared adapter，不保留 task-specific adapter  
看能不能接近当前 best。

3. 把 task-specific adapter 改成“共享主 adapter + 小 residual gate”  
这样旧阶段信息在最终阶段也仍然能间接参与。

这三类实验都能帮助你回答一个更本质的问题：

- 当前提升到底来自“合理利用阶段信息”
- 还是来自“task-conditioned 参数本身”
