import os
import copy
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from network import Network
from midata import MIData
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn import functional as F
import gc
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from utils import process_and_replace_loader, process_data_chn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FixedReplayDataLoader:
    def __init__(
        self,
        new_dataset,
        replay_dataset,
        batch_size,
        replay_batch_size,
        pin_memory=True,
    ):
        self.new_dataset = new_dataset
        self.replay_dataset = replay_dataset
        self.batch_size = batch_size
        self.replay_batch_size = max(0, min(replay_batch_size, batch_size - 1))
        self.new_batch_size = batch_size - self.replay_batch_size
        self.pin_memory = pin_memory
        self.drop_last = False
        self.num_workers = 0
        self.dataset = ConcatDataset([new_dataset, replay_dataset])

    def __len__(self):
        return int(np.ceil(len(self.new_dataset) / max(1, self.new_batch_size)))

    def __iter__(self):
        new_loader = DataLoader(
            self.new_dataset,
            batch_size=self.new_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
        replay_loader = DataLoader(
            self.replay_dataset,
            batch_size=self.replay_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
        ) if self.replay_batch_size > 0 else None

        replay_iter = cycle(replay_loader) if replay_loader is not None else None

        for new_batch in new_loader:
            new_x, new_y, new_s = new_batch[:3]
            new_extra = list(new_batch[3:])
            if replay_iter is None:
                if new_extra:
                    yield (new_x, new_y, new_s, *new_extra)
                else:
                    yield new_x, new_y, new_s
                continue

            replay_batch = next(replay_iter)
            replay_x, replay_y, replay_s = replay_batch[:3]
            replay_extra = list(replay_batch[3:])
            merged = [
                torch.cat([new_x, replay_x], dim=0),
                torch.cat([new_y, replay_y], dim=0),
                torch.cat([new_s, replay_s], dim=0),
            ]
            for new_item, replay_item in zip(new_extra, replay_extra):
                merged.append(torch.cat([new_item, replay_item], dim=0))
            perm = torch.randperm(merged[1].size(0))
            yield tuple(item[perm] for item in merged)

class CBiCaRL:
    def __init__(self, seed, result_dir, data_path, is_cross_session, numclass, feature_extractor, \
        batch_size, memory_size, balance_sample, balance_power, replay_batch_size, is_contrastive_loss, lambda_contrastive_loss, temperature,\
        use_normalized_nme, \
        use_hybrid_nme_logits, hybrid_start_task, hybrid_alpha_min, hybrid_alpha_max, hybrid_alpha_steps, hybrid_old_weight, \
        use_current_prototype_blend, current_prototype_blend_alpha, current_prototype_blend_start_task, current_prototype_blend_scope, \
        exemplar_mode, exemplar_mode_start_task, \
        task_adapter_lr_mult, \
        use_lwf, lwf_lambda, lwf_T, stage_lwf_lambdas, use_feature_distill, feature_distill_lambda, stage_feature_distill_lambdas, weighted_crossentropy, old_class_weight_power, stage_old_class_weight_powers, \
            epochs, stage_epochs, learning_rate, is_align, log, current_date):
        super().__init__()

        self.seed = seed
        self.result_dir = result_dir
        self.epochs = epochs
        self.stage_epochs = stage_epochs
        self.learning_rate = learning_rate
        self.model = Network(numclass,feature_extractor)
        self.stage = None
        self.numclass = numclass
        self.log = log
        self.is_align = is_align
        self.current_date = current_date

        self.dataset = MIData(seed=self.seed, data_path=data_path, \
            is_cross_session=is_cross_session, trials_persession=288, is_align=is_align)

        self.batch_size = batch_size
        self.balance_sample =balance_sample
        self.balance_power = balance_power
        self.replay_batch_size = replay_batch_size

        self.train_loader=None
        self.test_loader=None

        # 重放参数
        self.memory_size = memory_size
        self.exemplar_set = []
        self.class_mean_set = []
        self.class_radius_set = []

        self.is_contrastive_loss = is_contrastive_loss
        self.lambda_contrastive_loss = lambda_contrastive_loss
        self.temperature = temperature
        self.use_normalized_nme = use_normalized_nme
        self.use_hybrid_nme_logits = use_hybrid_nme_logits
        self.hybrid_start_task = hybrid_start_task
        self.hybrid_alpha_min = hybrid_alpha_min
        self.hybrid_alpha_max = hybrid_alpha_max
        self.hybrid_alpha_steps = hybrid_alpha_steps
        self.hybrid_old_weight = hybrid_old_weight
        self.hybrid_alpha = 1.0
        self.use_current_prototype_blend = use_current_prototype_blend
        self.current_prototype_blend_alpha = current_prototype_blend_alpha
        self.current_prototype_blend_start_task = current_prototype_blend_start_task
        self.current_prototype_blend_scope = current_prototype_blend_scope
        self.exemplar_mode = exemplar_mode
        self.exemplar_mode_start_task = exemplar_mode_start_task
        self.task_adapter_lr_mult = task_adapter_lr_mult

        # LwF参数
        self.prev_model = None
        self.use_lwf = use_lwf
        self.lwf_lambda = lwf_lambda
        self.lwf_T = lwf_T
        self.stage_lwf_lambdas = stage_lwf_lambdas
        self.use_feature_distill = use_feature_distill
        self.feature_distill_lambda = feature_distill_lambda
        self.stage_feature_distill_lambdas = stage_feature_distill_lambdas

        self.weighted_crossentropy = weighted_crossentropy
        self.old_class_weight_power = old_class_weight_power
        self.stage_old_class_weight_powers = stage_old_class_weight_powers
        self.class_weights = None

        self.counts_train_perclass = np.zeros(shape=(4,)) # 用于统计累积各类别训练样本数目

        # 当前阶段训练和测试的被试index
        self.train_idt = None
        self.test_idt = None

    def beforeTrain(self, stage):
        self.stage = stage # stage id
        # 修改
        self.train_idt = np.arange(self.stage * 3 - 2, self.stage * 3 + 1)
        self.test_idt = np.arange(1, self.stage * 3 + 1)

        stage_log = f'Stage: {self.stage}, numclass: {self.numclass}'
        stage_log = '==================' + stage_log + '=================='
        self.log.record(stage_log)
        print(stage_log)

        train_class_list = np.array([self.numclass-2, self.numclass-1])
        test_class_list = np.arange(self.numclass)

        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(self.train_idt, self.test_idt, train_class_list, test_class_list, self.balance_sample)
        
        if self.stage > 1:
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.to(device)
            self.prev_model.eval()
            for p in self.prev_model.parameters():
                p.requires_grad = False
            self.model.Incremental_learning(self.numclass)

        if hasattr(self.model.feature, "set_current_task"):
            self.model.feature.set_current_task(self.stage - 1)
        self.model.train()
        self.model.to(device)
        
    def get_exampler_dataset(self):
        """
            返回exampler的dataset
        """

        if len(self.exemplar_set) == 0:
            return None
        ex_datas = []
        ex_labels = []
        for label, exemplar_objs in enumerate(self.exemplar_set):
            class_data = torch.stack([torch.as_tensor(o) for o in exemplar_objs]).float()
            ex_datas.append(class_data)
            ex_labels.append(torch.full((len(exemplar_objs),), label, dtype=torch.long))
        ex_datas = torch.cat(ex_datas, dim=0)
        ex_labels = torch.cat(ex_labels, dim=0)
        exemplar_subjects = torch.full((ex_labels.shape[0],), -1, dtype=torch.long)
        return TensorDataset(ex_datas, ex_labels, exemplar_subjects)

    def _get_train_and_test_dataloader(self, train_idt, test_idt, train_class_list, test_class_list, balance_sample=False):
        
        X_train_raw, y_train, s_train = self.dataset.get_train_data(
            train_idt, train_class_list, return_subject_ids=True, apply_align=False
        )
        X_train = X_train_raw
        if self.is_align:
            X_train = self.dataset.get_train_data(
                train_idt, train_class_list, return_subject_ids=False, apply_align=True
            )[0]
        X_test, y_test = self.dataset.get_test_data(test_idt, test_class_list)

        subject_info = f"Selected subjects for train: {train_idt}"
        shape_info = f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        self.log.record(subject_info)
        self.log.record(shape_info)
        print(subject_info)
        print(shape_info)

        Xtr = process_data_chn(X_train)
        Ytr = torch.tensor(y_train, dtype=torch.long)
        Str = torch.tensor(s_train, dtype=torch.long)
        Xte = process_data_chn(X_test)
        Yte = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(Xtr.float(), Ytr, Str)
        test_dataset = TensorDataset(Xte, Yte)

        exampler_dataset = self.get_exampler_dataset()
        replay_dataset = None
        if exampler_dataset is not None:
            replay_x, replay_y, replay_s = exampler_dataset.tensors
            replay_dataset = TensorDataset(process_data_chn(replay_x.numpy()).float(), replay_y, replay_s)

        if exampler_dataset is not None:
            train_dataset = ConcatDataset([train_dataset, replay_dataset])
            shape_info = f"After replay, total num of trials for train: {len(train_dataset)}"
            self.log.record(shape_info)
            print(shape_info)

        self.class_weights = self._compute_class_weights(train_dataset)

        if replay_dataset is not None and self.replay_batch_size > 0:
            train_loader = FixedReplayDataLoader(
                new_dataset=TensorDataset(Xtr.float(), Ytr, Str),
                replay_dataset=replay_dataset,
                batch_size=self.batch_size,
                replay_batch_size=self.replay_batch_size,
                pin_memory=True,
            )
        elif balance_sample:
            train_loader = self._balance_sample_train_loader(train_dataset=train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return train_loader, test_loader

    def _compute_class_weights(self, train_dataset):
        all_labels = []
        for i in range(len(train_dataset)):
            label = train_dataset[i][1]
            all_labels.append(label.item())
        if not all_labels:
            return None

        all_labels = torch.tensor(all_labels, dtype=torch.long)
        class_counts = torch.bincount(all_labels, minlength=self.numclass).float()
        valid_mask = class_counts > 0
        if valid_mask.sum().item() == 0:
            return None

        class_weights = torch.ones_like(class_counts)
        class_weights[valid_mask] = 1.0 / torch.pow(class_counts[valid_mask], self.balance_power)
        class_weights = class_weights / class_weights[valid_mask].mean()
        return class_weights.to(device)

    def _compute_old_class_weights(self, old_k):
        current_power = self.old_class_weight_power
        if self.stage_old_class_weight_powers and len(self.stage_old_class_weight_powers) >= self.stage:
            current_power = float(self.stage_old_class_weight_powers[self.stage - 1])
        if old_k <= 0 or current_power <= 0:
            return None
        weights = torch.tensor(
            [(old_k - idx) ** current_power for idx in range(old_k)],
            dtype=torch.float32,
            device=device,
        )
        return weights / weights.mean()

    def _compute_bce_loss(self, logits, target, old_k=None):
        bce_matrix = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        if self.weighted_crossentropy and self.class_weights is not None:
            active_weights = self.class_weights[:logits.size(1)].view(1, -1)
            bce_matrix = bce_matrix * active_weights

        if old_k is not None and old_k > 0:
            old_class_weights = self._compute_old_class_weights(old_k)
            if old_class_weights is not None:
                bce_matrix[:, :old_k] = bce_matrix[:, :old_k] * old_class_weights.view(1, -1)
        return bce_matrix.mean()

    def _feature_distillation_loss(self, student_features, teacher_features, labels, old_k):
        if not self.use_feature_distill or teacher_features is None or old_k <= 0:
            return None
        old_mask = labels < old_k
        if old_mask.sum().item() == 0:
            return None
        student_old = F.normalize(student_features[old_mask], p=2, dim=1)
        teacher_old = F.normalize(teacher_features[old_mask], p=2, dim=1)
        return (1.0 - F.cosine_similarity(student_old, teacher_old, dim=1)).mean()

    def _balance_sample_train_loader(self,train_dataset):
        '''
            batch内均衡采样实现
        '''
        # 1. 获取整个 train_dataset 中所有的标签
        all_labels = []
        for i in range(len(train_dataset)):
            label = train_dataset[i][1]
            all_labels.append(label.item())
        all_labels = torch.tensor(all_labels)

        # 2. 统计当前训练集中，每个类别有多少个样本
        class_counts = torch.bincount(all_labels)
        # 过滤掉 count 为 0 的类别（防止除以0）
        classes = torch.nonzero(class_counts).squeeze()

        # 3. 计算类别的权重（样本越少，权重越大）
        # 比如 A有12个，B有120个，C有108个。那么 A 的权重就是 1/12，B 是 1/120...
        class_weights = torch.zeros_like(class_counts, dtype=torch.float)
        for c in classes:
            # class_weights[c] = 1.0 / class_counts[c]
            # 使用平方根平滑，减少 A 类的过高采样率
            class_weights[c] = 1.0 / torch.pow(class_counts[c], self.balance_power)

        # 4. 为数据集中的【每一个样本】分配权重
        sample_weights = class_weights[all_labels]

        # 5. 创建加权采样器 (replacement=True 允许重复采样少量的旧类别样本)
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), # 一个 epoch 采样的总次数，保持和数据集大小一致即可
            replacement=True
        )
        # ==================================================

        # 注意：使用了 sampler 之后，shuffle 必须设置为 False！
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=sampler,           
            drop_last=False, 
            pin_memory=True
        )

        return train_loader 

    def train(self):
        current_epochs = self.epochs
        if self.stage_epochs and len(self.stage_epochs) >= self.stage:
            current_epochs = int(self.stage_epochs[self.stage - 1])

        # 创建 TensorBoard writer
        ea_status = "EA" if self.is_align else "noEA"
        replay_status = f"buffersize{self.memory_size}" if self.memory_size > 0 else "noReplay"
        run_name = f"S{self.stage}_seed{self.seed}_{replay_status}_{ea_status}_{self.current_date}"
        log_dir = os.path.join('./tensorboard_logs', str(self.current_date), run_name)
        writer = SummaryWriter(log_dir=log_dir)

        # optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # for n, p in self.model.named_parameters():
        #     print(n, p.requires_grad)
        
        # input(' ')

        adapter_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(tag in name for tag in ('task_adapter', 'shared_adapter', 'task_prompt', 'lora_')):
                adapter_params.append(param)
            else:
                base_params.append(param)

        param_groups = []
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.learning_rate,
                'weight_decay': 0.0001,
            })
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': self.learning_rate * self.task_adapter_lr_mult,
                'weight_decay': 0.0001,
            })

        optimizer = optim.Adam(param_groups)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=current_epochs, T_mult=1, eta_min=1e-6
        )

        train_loader_len = len(self.train_loader)

        for epoch in range(current_epochs):
            
            # accumulators for monitoring
            epoch_train_losses = []
            self.model.train()
            
            for step, batch in enumerate(self.train_loader):
                if len(batch) == 3:
                    x, y, subject_ids = batch
                    subject_ids = subject_ids.to(device)
                else:
                    x, y = batch
                    subject_ids = None
                x = x.unsqueeze(1).to(device)  # (B,1,Chans,Samples)
                y = y.to(device)
                
                features, logits = self.model(x, return_feat=True)

                # iCaRL原版
                target = torch.zeros_like(logits, dtype=torch.float32).to(device)
                
                if self.prev_model is not None:
                    self.prev_model.eval()
                    with torch.no_grad():
                        old_logits = self.prev_model(x)      # (B, old_k)
                        old_features = self.prev_model.feature_extractor(x)
                        old_prob = torch.sigmoid(old_logits) # (B, old_k)
                    old_k = old_prob.size(1)
                    # 覆盖 target 的旧类部分为教师输出（论文 iCaRL 的做法）
                    target[:, :old_k] = old_prob
                
                target.scatter_(1, y.reshape(-1,1), 1.0)
                
                loss_bce = self._compute_bce_loss(logits, target, old_k if self.prev_model is not None else None)

                if self.is_contrastive_loss:
                    loss_con = self.supervised_contrastive_loss(features, y, temperature=self.temperature)
                    loss = loss_bce + self.lambda_contrastive_loss * loss_con
                else:
                    loss = loss_bce

                if self.use_lwf and self.prev_model is not None:
                    current_lwf_lambda = self.lwf_lambda
                    if self.stage_lwf_lambdas and len(self.stage_lwf_lambdas) >= self.stage:
                        current_lwf_lambda = float(self.stage_lwf_lambdas[self.stage - 1])
                    student_old_logits = logits[:, :old_k]
                    teacher_old_probs = torch.sigmoid(old_logits / self.lwf_T)
                    student_old_probs = torch.sigmoid(student_old_logits / self.lwf_T)
                    loss_lwf = F.binary_cross_entropy(student_old_probs, teacher_old_probs)
                    loss = loss + current_lwf_lambda * (self.lwf_T ** 2) * loss_lwf

                if self.prev_model is not None:
                    loss_feat = self._feature_distillation_loss(features, old_features, y, old_k)
                    if loss_feat is not None:
                        current_feature_distill_lambda = self.feature_distill_lambda
                        if self.stage_feature_distill_lambdas and len(self.stage_feature_distill_lambdas) >= self.stage:
                            current_feature_distill_lambda = float(self.stage_feature_distill_lambdas[self.stage - 1])
                        loss = loss + current_feature_distill_lambda * loss_feat

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # 使用 epoch + progress 作为 step 的参数：scheduler.step(epoch + step/len_loader)
                scheduler.step(epoch + float(step) / float(train_loader_len))

                # 记录当前 lr（第 0 个 param_group）
                current_lr = optimizer.param_groups[0]['lr']
                global_step = epoch * (train_loader_len or 1) + step
                writer.add_scalar('LR/param_group0', current_lr, global_step)

                epoch_train_losses.append(loss.item())

            # 每 epoch 评估一次在当前任务测试集上的表现
            overall_acc = self._test(self.test_loader,return_perclass=False)
            epoch_train_loss = np.mean(epoch_train_losses)

            train_str = f'epoch:{epoch+1},train avg loss:{epoch_train_loss},acc:{overall_acc}'
            self.log.record(train_str)
            print(train_str)

            # 记录当前epoch的loss到TensorBoard
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
            writer.add_scalar('Acc/val', overall_acc, epoch)

    def supervised_contrastive_loss(self, features, labels, temperature=0.07):
        """
        健壮版的 SupCon Loss
        features: (batch_size, feature_dim) - 必须是 L2 归一化前的特征或之后的都可以，内部会再做一次保证
        labels: (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. 强制 L2 归一化 (映射到单位超球面)
        features = F.normalize(features, p=2, dim=1)
        
        # 2. 计算标签掩码 (mask[i, j] = 1 如果 label[i] == label[j] 否则 0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 3. 计算余弦相似度并除以温度系数
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
        
        # --- 关键修正 1：数值稳定性 (Max Trick) 防止 exp 溢出 ---
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 4. 消除自身与自身的对比 (对角线设为 0)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask  # 真正的正样本 mask (排除自己)

        # 5. 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # 加 1e-12 防止 log(0) 出现 NaN
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 6. 计算每个样本的平均正样本 Loss
        # --- 关键修正 2：防止 batch 内某类别只有 1 个样本导致 mask.sum(1) 为 0 ---
        mask_sum = mask.sum(1)
        # 如果 mask_sum 为 0，说明这个样本在 batch 里没有同类，Loss 给 0
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)

        # 由于是最小化 Loss，所以取负号
        loss = -mean_log_prob_pos
        
        # 仅计算那些在 batch 内有正样本参与的 anchor 的 loss
        valid_loss = loss[mask_sum > 0]
        if valid_loss.numel() > 0:
            return valid_loss.mean()
        else:
            # 极端情况：整个 batch 所有样本都没同类
            warning_info = f"There is no same class in one batch!"
            self.log.record(warning_info)
            print(warning_info)
            return torch.tensor(0.0).to(device)

    def _test(self, test_loader, return_perclass=False):
        """
        评估模型并可选返回每个类别的准确率。

        参数：
            backbone, classifier: 模型部分（backbone + linear）
            data_loader: torch DataLoader，返回 (x, y)
            device: torch.device
            class_indices: 若 eval_mode=='task-available'，则为当前指定的输出任务头
            eval_mode: 'task-unavailable' (默认) 或 'task-available'。
                - task-unavailable: 直接用全局 logits argmax 作为预测。
                - task-available: 仅把 class_indices 视为合法预测，若 argmax 不在其内则算错。
            return_per_class: 若 True，返回 (overall_acc, per_class_acc_dict, per_class_counts)
            flag: 调试用，打印一些信息

        返回：
            如果 return_per_class == False:
                返回 overall_acc (float, 百分比)
            否则返回 (overall_acc, per_class_acc_dict, per_class_counts)
                - per_class_acc_dict: {class_id: accuracy_percent or np.nan}
                - per_class_counts: {class_id: n_samples}
        """

        self.model.eval()

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for xb, yb in test_loader:
                # xb: (N, Chans, Samples) in your pipeline -> prepare_input
                xb = xb.to(device).unsqueeze(1)   # (N,1,Chans,Samples)  （或调用你的prepare_input）
                yb = yb.to(device)

                logits = self.model(xb)        # shape (N, total_classes)
                preds = logits.argmax(dim=1).cpu().numpy()  # numpy (N,)
                trues = yb.cpu().numpy()

                all_preds.append(preds)
                all_trues.append(trues)
                    
            # 把测试集上所有的预测和真实标签合并
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

            # 计算正确的数目
            correct_mask = (all_preds == all_trues)

            total = all_trues.shape[0]
            overall_acc = 100.0 * correct_mask.sum() / total if total > 0 else 0.0

            # print("============test============")
            # print(all_preds[:10])
            # print(all_trues[:10])
            # print(correct_mask[:10])
            # input(' ')

            # 需要返回每个类别上的准确率
            classes = np.unique(all_trues)

            per_class_acc = {}
            # 每个类别的样本数目
            per_class_counts = {}
            for c in classes:
                # 先统计每个类别有多少样本
                mask = (all_trues == c)
                n = mask.sum()
                per_class_counts[c] = int(n)
                
                per_class_acc[c] = 100.0 * ((all_preds[mask] == all_trues[mask]).sum()) / n

        self.model.train()

        # 只返回整体准确率
        if not return_perclass:
            return overall_acc
        else:
            return overall_acc, per_class_acc

    def afterTrain(self):
        self.model.eval()
        class_budgets = self._compute_memory_budgets()
        self._reduce_exemplar_sets(class_budgets)

        # 按照类别选择样本重放
        start_idx = 0 if self.stage == 1 else self.numclass-1
        for i in range(start_idx, self.numclass):
            construct_info = f'construct class {i} examplar:'
            self.log.record(construct_info)
            print(construct_info)
            class_list = np.arange(i, i+1)
            X_train, _ = self.dataset.get_train_data(self.train_idt, class_list, return_subject_ids=False, apply_align=True)
            self._construct_exemplar_set(X_train, class_budgets[i])

        # 计算类别均值并评估
        self.compute_exemplar_class_mean()
        self._fit_hybrid_fusion_calibration()
        self._eval_mean()
        # 评估每个被试的每个类别
        subject_class_acc_matrix = self._eval_cnn_by_sub()

        stage_finish_log = f'Stage: {self.stage} finish'
        stage_log = '==================' + stage_finish_log + '=================='
        self.log.record(stage_log)
        print(stage_log)

        self.numclass += 1

        # 清理显存
        gc.collect()
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return subject_class_acc_matrix

    def _compute_memory_budgets(self):
        if self.numclass <= 0:
            return []
        base = self.memory_size // self.numclass
        budgets = [base for _ in range(self.numclass)]
        for idx in range(self.memory_size - sum(budgets)):
            budgets[idx % self.numclass] += 1
        return budgets

    def _reduce_exemplar_sets(self, class_budgets):
        '''
            减少前面类别的重放样本
        '''
        for index in range(len(self.exemplar_set)):
            budget = class_budgets[index]
            self.exemplar_set[index] = self.exemplar_set[index][:budget]
            reduce_info = f'Size of class {index} examplar: {str(len(self.exemplar_set[index]))}'
            self.log.record(reduce_info)
            print(reduce_info)

    def _construct_exemplar_set(self, X_train, m):
        '''
            构建重放样本集(同一个类别),这里的X_train还没变换到45通道,存放的也是22通道的样本
            1.计算训练样本特征均值
            2.迭代更新,选择距离中心最近的样本
        '''
        X_initial = X_train
        X_train = process_data_chn(X_train)

        class_mean, feature_extractor_output, _, _ = self.compute_class_mean(X_train.unsqueeze(1))
        exemplar = []
        now_class_mean = np.zeros((1, feature_extractor_output.shape[1]))
        selected_indices = []
        selected_mask = np.zeros(feature_extractor_output.shape[0], dtype=bool)
        max_pick = min(int(m), int(feature_extractor_output.shape[0]))
        for i in range(max_pick):
            candidate_mean_error = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            candidate_mean_error = np.linalg.norm(candidate_mean_error, axis=1)
            candidate_mean_error[selected_mask] = np.inf
            index = int(np.argmin(candidate_mean_error))
            selected_mask[index] = True
            selected_indices.append(index)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(X_initial[index])

        exampler_info = f"the size of exemplar :{(str(len(exemplar)))}" 
        self.log.record(exampler_info)
        print(exampler_info)
        self.exemplar_set.append(exemplar)

    def compute_class_mean(self, x):
        '''
            计算类别特征中心向量
            输入:x,在MIRepNet的时候需要提前变换到45通道
            输出:类别的均值向量和特征提取器的输出 (batch_size * emb_dim)
        '''
        x = x.to(device)
        feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        if self.use_normalized_nme:
            feature_extractor_output = feature_extractor_output / (np.linalg.norm(feature_extractor_output, axis=1, keepdims=True) + 1e-12)
        class_mean = np.mean(feature_extractor_output, axis=0)
        if self.use_normalized_nme:
            class_mean = class_mean / (np.linalg.norm(class_mean, keepdims=True) + 1e-12)
        class_radius = np.linalg.norm(feature_extractor_output - class_mean, axis=1).mean()
        class_radius = float(max(class_radius, 1e-12))
        class_var = np.maximum(np.var(feature_extractor_output, axis=0), 1e-12)
        return class_mean, feature_extractor_output, class_radius, class_var
    
    def compute_exemplar_class_mean(self):
        '''
            计算buffer中各个类别的特征均值向量
        '''
        self.class_mean_set = []
        self.class_radius_set = []
        self.class_var_set = []
        for index in range(len(self.exemplar_set)):
            mean_info = f"compute the class mean of class {str(index)}"
            self.log.record(mean_info)
            print(mean_info)
            #exemplar=self.train_dataset.get_image_class(index)
            # exemplar = torch.stack(exemplar, dim=0)
            exemplar = np.array(self.exemplar_set[index])
            exemplar = process_data_chn(exemplar)
            exemplar = exemplar.unsqueeze(1).to(device)

            class_mean, _, class_radius, class_var = self.compute_class_mean(exemplar)
            self.class_mean_set.append(class_mean)
            self.class_radius_set.append(class_radius)
            self.class_var_set.append(class_var)

        if (
            self.use_current_prototype_blend
            and self.stage is not None
            and self.stage >= self.current_prototype_blend_start_task
        ):
            if self.stage == 1:
                current_classes = np.arange(self.numclass)
            elif self.current_prototype_blend_scope == 'overlap_only':
                current_classes = np.array([self.numclass - 2], dtype=np.int64)
            elif self.current_prototype_blend_scope == 'new_only':
                current_classes = np.array([self.numclass - 1], dtype=np.int64)
            else:
                current_classes = np.array([self.numclass - 2, self.numclass - 1], dtype=np.int64)

            blend_alpha = float(self.current_prototype_blend_alpha)
            for cls_idx in current_classes:
                if cls_idx >= len(self.class_mean_set):
                    continue
                train_mean, train_radius, train_var = self._compute_current_blend_target(cls_idx, device)
                if train_mean is None:
                    continue
                blended_mean = blend_alpha * self.class_mean_set[cls_idx] + (1.0 - blend_alpha) * train_mean
                if self.use_normalized_nme:
                    blended_mean = blended_mean / (np.linalg.norm(blended_mean, keepdims=True) + 1e-12)
                self.class_mean_set[cls_idx] = blended_mean
                self.class_radius_set[cls_idx] = blend_alpha * self.class_radius_set[cls_idx] + (1.0 - blend_alpha) * train_radius
                self.class_var_set[cls_idx] = blend_alpha * self.class_var_set[cls_idx] + (1.0 - blend_alpha) * train_var

    def _compute_current_blend_target(self, cls_idx, device):
        X_train_cls, _ = self.dataset.get_train_data(self.train_idt, np.array([cls_idx]))
        if X_train_cls is None or len(X_train_cls) == 0:
            return None, None, None
        X_train_cls = process_data_chn(X_train_cls).unsqueeze(1).to(device)
        train_mean, _, train_radius, train_var = self.compute_class_mean(X_train_cls)
        return train_mean, train_radius, train_var

    def _subsample_per_class(self, x_np, y_np, max_per_class):
        if x_np is None or y_np is None or len(y_np) == 0:
            return None, None
        keep_indices = []
        rng = np.random.RandomState(self.seed * 100 + self.stage)
        for cls in np.unique(y_np):
            cls_indices = np.where(y_np == cls)[0]
            if len(cls_indices) > max_per_class:
                cls_indices = rng.choice(cls_indices, size=max_per_class, replace=False)
            keep_indices.append(np.sort(cls_indices))
        if not keep_indices:
            return None, None
        keep_indices = np.concatenate(keep_indices, axis=0)
        return x_np[keep_indices], y_np[keep_indices]

    def _build_bias_calibration_set(self):
        xs = []
        ys = []

        exemplar_dataset = self.get_exampler_dataset()
        if exemplar_dataset is not None:
            ex_x, ex_y, _ = exemplar_dataset.tensors[:3]
            ex_x_np = ex_x.detach().cpu().numpy()
            ex_y_np = ex_y.detach().cpu().numpy()
            ex_x_np, ex_y_np = self._subsample_per_class(ex_x_np, ex_y_np, max_per_class=64)
            if ex_x_np is not None:
                xs.append(ex_x_np)
                ys.append(ex_y_np)

        current_classes = np.array([self.numclass - 2, self.numclass - 1]) if self.stage > 1 else np.arange(self.numclass)
        current_x, current_y = self.dataset.get_train_data(self.train_idt, current_classes)
        current_x, current_y = self._subsample_per_class(current_x, current_y, max_per_class=96)
        if current_x is not None:
            xs.append(current_x)
            ys.append(current_y)

        if not xs:
            return None, None

        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        return x_all, y_all

    def _compute_nme_scores(self, test):
        feature = self.model.feature_extractor(test).detach().cpu().numpy()
        if self.use_normalized_nme:
            feature = feature / (np.linalg.norm(feature, axis=1, keepdims=True) + 1e-12)
        class_mean_set = np.array(self.class_mean_set)
        if class_mean_set.size == 0:
            return np.empty((feature.shape[0], 0), dtype=np.float64), feature

        dist = np.linalg.norm(feature[:, None, :] - class_mean_set[None, :, :], ord=2, axis=2)
        return -dist, feature

    def _normalize_score_rows(self, scores):
        if scores.size == 0:
            return scores
        row_mean = scores.mean(axis=1, keepdims=True)
        row_std = scores.std(axis=1, keepdims=True)
        return (scores - row_mean) / (row_std + 1e-12)

    def _compute_fc_scores(self, test):
        logits = self.model(test).detach().cpu().numpy()
        if logits.size == 0:
            return logits
        num_classes = len(self.class_mean_set)
        if num_classes > 0 and logits.shape[1] > num_classes:
            logits = logits[:, :num_classes]
        return logits

    def _compute_hybrid_scores(self, test):
        nme_scores, feature = self._compute_nme_scores(test)
        if not self.use_hybrid_nme_logits or self.stage is None or self.stage < self.hybrid_start_task or nme_scores.size == 0:
            return nme_scores, feature

        fc_scores = self._compute_fc_scores(test)
        if fc_scores.size == 0 or fc_scores.shape != nme_scores.shape:
            return nme_scores, feature

        nme_scores = self._normalize_score_rows(nme_scores)
        fc_scores = self._normalize_score_rows(fc_scores)
        scores = self.hybrid_alpha * nme_scores + (1.0 - self.hybrid_alpha) * fc_scores
        return scores, feature

    def _macro_accuracy(self, y_true, y_pred, class_ids):
        accs = []
        for cls in class_ids:
            mask = y_true == cls
            if mask.sum() <= 0:
                continue
            accs.append(float((y_pred[mask] == y_true[mask]).mean()))
        if not accs:
            return 0.0
        return float(np.mean(accs))

    def _fit_hybrid_fusion_calibration(self):
        self.hybrid_alpha = 1.0

        if (
            not self.use_hybrid_nme_logits
            or self.stage <= 1
            or self.stage < self.hybrid_start_task
            or len(self.class_mean_set) <= 1
        ):
            return

        x_cal, y_cal = self._build_bias_calibration_set()
        if x_cal is None:
            return

        x_cal_tensor = process_data_chn(x_cal).unsqueeze(1).to(device)
        nme_scores, _ = self._compute_nme_scores(x_cal_tensor)
        fc_scores = self._compute_fc_scores(x_cal_tensor)
        y_cal = np.asarray(y_cal, dtype=np.int64)
        if (
            nme_scores.size == 0
            or fc_scores.size == 0
            or nme_scores.shape != fc_scores.shape
            or len(y_cal) != nme_scores.shape[0]
        ):
            return

        nme_scores = self._normalize_score_rows(nme_scores)
        fc_scores = self._normalize_score_rows(fc_scores)
        old_k = len(self.class_mean_set) - 1

        class_ids = np.unique(y_cal)
        base_pred = nme_scores.argmax(axis=1)
        base_old = self._macro_accuracy(y_cal, base_pred, np.arange(old_k))
        base_all = self._macro_accuracy(y_cal, base_pred, np.unique(y_cal))
        best_obj = self.hybrid_old_weight * base_old + (1.0 - self.hybrid_old_weight) * base_all
        best_alpha = 1.0

        alpha_values = np.linspace(self.hybrid_alpha_min, self.hybrid_alpha_max, int(self.hybrid_alpha_steps))
        for alpha in alpha_values:
            fused = alpha * nme_scores + (1.0 - alpha) * fc_scores
            pred = fused.argmax(axis=1)
            old_macro = self._macro_accuracy(y_cal, pred, np.arange(old_k))
            all_macro = self._macro_accuracy(y_cal, pred, class_ids)
            obj = self.hybrid_old_weight * old_macro + (1.0 - self.hybrid_old_weight) * all_macro
            if obj > best_obj + 1e-8:
                best_obj = obj
                best_alpha = float(alpha)

        self.hybrid_alpha = best_alpha
        hybrid_info = (
            f"hybrid nme-logits calibration stage {self.stage}: old_k={old_k}, "
            f"alpha={self.hybrid_alpha:.4f}, objective={best_obj:.4f}"
        )
        self.log.record(hybrid_info)
        print(hybrid_info)

    def _eval_mean(self):
        """
            使用类别均值向量分类的测试函数
        """

        self.model.eval()

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                # xb: (N, Chans, Samples) in your pipeline -> prepare_input
                xb = xb.to(device).unsqueeze(1)   # (N,1,Chans,Samples)  （或调用你的prepare_input）
                yb = yb.to(device)

                preds = self.classify(xb)
                trues = yb.cpu().numpy()

                all_preds.append(preds)
                all_trues.append(trues)
                    
            # 把测试集上所有的预测和真实标签合并
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

        # 计算各项指标
        self._print_results(all_trues, all_preds)
    
    def _eval_cnn_by_sub(self):
        """
            评测每个被试的每个类别准确率
        """
        self.model.eval()

        # 1. 初始化结果矩阵：[被试数量, 4个类别]
        # 假设 self.test_idt 包含所有被试 ID，例如 9 个被试
        num_subjects = len(self.test_idt)
        num_classes = self.numclass
        # 初始化为 0.0，建议用 nan 方便排查是否有未覆盖的情况，但在 BCI 场景 0.0 也行
        subject_class_acc_matrix = np.zeros((num_subjects, num_classes))

        all_feats_for_tsne = []
        all_labels_for_tsne = []
        all_subs_for_tsne = []

        for s_idx, sid in enumerate(self.test_idt):

            all_preds = []
            all_trues = []

            if sid in [1, 2, 3]: sub_group = 'Sub123'
            elif sid in [4, 5, 6]: sub_group = 'Sub456'
            else: sub_group = 'Sub789'
            
            sub_loader = self._get_test_dataloader(list((sid,)),self.numclass)
            
            sub_loader = process_and_replace_loader(
                sub_loader, 
                ischangechn=True, 
                dataset='BNCI2014001-4'
            )

            for _, (inputs, targets) in enumerate(sub_loader):
                inputs = inputs.unsqueeze(1).to(device)
                targets = targets.to(device)

                with torch.no_grad():
                    preds, feat = self.classify(inputs, return_feat=True)
                    trues = targets.cpu().numpy()

                all_preds.append(preds)
                all_trues.append(trues)

                all_feats_for_tsne.append(feat)
                all_labels_for_tsne.append(trues)
                all_subs_for_tsne.extend([sub_group] * len(trues))
                            
            # 把测试集上所有的预测和真实标签合并
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_trues, axis=0)

            # print('In _eval_cnn')
            # print(f"all_preds: {y_pred[:100]}")
            # print(f"all_trues: {y_true[:100]}")

            for cls_idx in range(num_classes):
                # 找到真实标签为该类别的索引
                indices = np.where(y_true == cls_idx)[0]
                
                if len(indices) > 0:
                    # 计算准确率：预测正确的数量 / 该类别的样本总数
                    acc = (y_pred[indices] == y_true[indices]).sum() / len(indices)
                    subject_class_acc_matrix[s_idx, cls_idx] = acc
                else:
                    # 如果测试集中该被试没有这个动作的数据（通常不会发生），设为 NaN 或 0
                    subject_class_acc_matrix[s_idx, cls_idx] = 0.0 # 或者 np.nan
            
            start_info = '*'*10 + f'sub:{sid} result' + '*'*10
            self.log.record(start_info)
            print(start_info)
            # 计算各项指标
            self._print_results(y_true, y_pred)
            end_info = '*'*10 + f'sub:{sid} result end' + '*'*10
            self.log.record(end_info)
            print(end_info)

        # --- 循环结束后，调用绘图 ---
        all_feats_for_tsne = np.concatenate(all_feats_for_tsne, axis=0)
        all_labels_for_tsne = np.concatenate(all_labels_for_tsne, axis=0)
        
        self.plot_tsne(all_feats_for_tsne, all_labels_for_tsne, all_subs_for_tsne, stage_note="TestSet")

        # 返回计算好的矩阵
        return subject_class_acc_matrix
    
    def classify(self, test, return_feat=False):
        scores, feature = self._compute_hybrid_scores(test)
        result = np.argmax(scores, axis=1)
        if return_feat:
            return torch.tensor(result), feature
        else:
            return torch.tensor(result)
    
    def plot_tsne(self, features, labels, subject_ids, stage_note="TestSet"):
        """
        features: (N, embed_dim)
        labels: (N,) - 0, 1, 2, 3 (A, B, C, D)
        subject_ids: (N,) - 用于区分来源
        """
        print("Generating t-SNE plot...")
        # 1. 降维
        tsne = TSNE(n_components=2, init='pca', random_state=self.seed)
        X_embedded = tsne.fit_transform(features)

        x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
        X_embedded = (X_embedded - x_min) / (x_max - x_min + 1e-12)

        # 2. 准备数据框，方便 seaborn 绘图
        # 将数字标签转为字母标签，增强可读性
        class_map = {0: 'left hand', 1: 'right hand', 2: 'both feet', 3: 'tongue'}
        df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            'Class': [class_map[l] for l in labels],
            'SubjectGroup': subject_ids # 标记是 sub123, sub456 还是 sub789
        })

        my_palette = {
            'left hand': '#F30505',  # 鲜红
            'right hand': '#67D65C',  # 
            'both feet': '#4326EF',  # 钢蓝
            'tongue': '#CAD246'   # 橙黄
        }

        # 3. 绘图
        plt.figure(figsize=(10, 8))
        # 使用不同的颜色代表 Class，不同的形状(style)代表 SubjectGroup
        sns.scatterplot(
            data=df, x='x', y='y', 
            hue='Class', style='SubjectGroup',
            palette='viridis', s=60, alpha=0.7
        )

        # 强制类别顺序，确保 legend 顺序也是 A->B->C->D
        hue_order = ['left hand','right hand','both feet','tongue'][:self.numclass] 
        
        # 也可以自定义形状，确保不同来源的被试一眼就能分清
        # 'o' 圆圈, 'X' 叉, 's' 正方形, '^' 三角形
        marker_map = {
            'Sub123': 'o',
            'Sub456': 'X',
            'Sub789': 's'
        }
        
        # 额外：把类中心（exemplar mean）也画进去（用五角星表示）
        # 如果你有 class_mean_set，也可以降维后画出来，看聚类是否围绕中心
        
        # 4. 绘图
        plt.figure(figsize=(12, 9))
        sns.set_style("whitegrid") # 添加网格线方便对齐观察

        ax = sns.scatterplot(
            data=df, x='x', y='y', 
            hue='Class', 
            style='SubjectGroup',
            palette=my_palette,    # 使用自定义颜色
            hue_order=hue_order,   # 固定颜色顺序
            markers=marker_map,    # 使用自定义形状
            s=100,                 # 稍微调大点，BCI特征点多时更清晰
            alpha=0.6,             # 增加透明度，观察重叠程度
            edgecolor='w',         # 给点加白边，重叠时更好分
            linewidth=0.5
        )

        # 使用 ax.set_title 设置标题
        ax.set_title(f"t-SNE Visualization - Stage {self.stage}", fontsize=15)
        
        # 使用 ax.legend 设置图例
        ax.legend(title='Category & Source', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        # 保存并显示
        save_path = f"{self.result_dir}/tsne_seed_{self.seed}_stage_{self.stage}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
        plt.show()
    
    def _print_results(self, y_true, y_pred):
        """
        打印详细的各个类别准确率
        """
        # 注意：如果在早期阶段（如 Task 1），y_true 里可能没有 'Both Feet' 和 'Tongue'
        # classification_report 会针对缺失类别报警告或显示 0.00，这是正常的
        class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']
        
        # 使用 sklearn 生成详细报告
        # labels 参数确保即使某些类没出现，也能按固定顺序输出报告
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        # 只打印实际存在的类别的名字
        target_names = [class_names[i] for i in sorted(unique_labels)]
        
        report = classification_report(y_true, y_pred, labels=sorted(unique_labels), target_names=target_names, digits=4)
        print(report)
        if hasattr(self, 'log'): 
            self.log.record(report)

        # 手动计算总平均准确率
        total_acc = (y_true == y_pred).sum() / len(y_true)
        avg_acc_info = f"Final Total Accuracy: {total_acc * 100:.2f}%"
        if hasattr(self, 'log'):
            self.log.record(avg_acc_info)
        print(avg_acc_info)

        # 计算每个类别的准确率
        for i, name in enumerate(class_names):
            # 只统计测试集中真实存在的类别
            idx = np.where(y_true == i)[0]
            if len(idx) > 0:
                class_acc = (y_true[idx] == y_pred[idx]).sum() / len(idx)
                class_acc_info = f"Accuracy for {name}: {class_acc * 100:.2f}%"
                if hasattr(self, 'log'):
                    self.log.record(class_acc_info)
                print(class_acc_info)
        
        return y_pred, y_true

    # def _testbeforetask(self):
    #     # 测试在之前所有任务上的准确率
    #     A_current_stage = np.zeros((self.numclass-1, self.numclass-1))
    #     for i in range(1, self.stage):
            
    #         # 获取当前测试被试的id
    #         test_idt = np.arange(i*3) + 1
            
    #         test_loader = self._get_test_dataloader(test_idt=test_idt, num_class=i+1)

    #         val_info = f"test on Stage {i} task, subjects:{test_idt}"
    #         self.log.record(val_info)
    #         print(val_info)

    #         test_acc, test_perclass = self._test(test_loader,num_class=i+1,return_perclass=True)
    #         self.model.eval()
    #         #第j阶段在taski-1上的表现
    #         A_current_stage[i-1,i-1] = test_acc
            
    #         # 记录每个类别的结果
    #         for c, acc in test_perclass.items():
    #             self.A_perclass[self.idx_perclass, c] = acc
    #         self.idx_perclass += 1

    #         current_info = f"Acc on stage {i}:{test_acc}"
    #         self.log.record(current_info)
    #         print(current_info)

    #     return A_current_stage

    def _get_test_dataloader(self, test_idt, numclass):
        class_list = np.arange(numclass)
        X_test, y_test = self.dataset.get_test_data(test_idt, class_list)
        # print(f"sid:{test_idt},test shape:{X_test.shape}, {y_test.shape}")

        Xte = torch.tensor(X_test, dtype=torch.float32)
        Yte = torch.tensor(y_test, dtype=torch.long)

        test_dataset = TensorDataset(Xte, Yte)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return test_loader

    # def _testcurrenttask(self, A_current_stage):
    #     # 记录当前阶段测试集上的表现，包含每个被试、每个类别上的表现
    #     # 每三个被试算一次平均准确率加到A_stage里面（最后一列）
    #     # 并且记录每个被试在当前阶段测试任务的准确率
    #     # 并且记录在每个类别上的准确率
    #     mean_acc = 0

    #     current_info = 'test on current stage task'
    #     self.log.record(current_info)
    #     print(current_info)

    #     for i, sid in enumerate(self.test_idt):

    #         sub_loader = self._get_test_dataloader(list((sid,)),self.numclass)

    #         acc_sub, acc_perclass = self._test(sub_loader,self.numclass,return_perclass=True)
    #         self.model.eval()
    #         mean_acc += acc_sub

    #         # 记录每个类别的结果
    #         for c, acc in acc_perclass.items():
    #             self.A_perclass[self.idx_perclass, c] += acc

    #         self.acc_persub[self.stage-1,int(sid)-1] = float(acc_sub)
    #         # 打印并记录（被试编号以 1 起始显示更直观）

    #         line = f" Subject S{sid}: acc={acc_sub:.4f}"
    #         self.log.record(line)
    #         print(line)
            
    #         if (i+1) % 3 == 0:
    #             # A_current_stage[(i + 1) // 3 - 1, -1] = mean_acc / 3
    #             mean_acc = 0

    #     self.A_perclass[self.idx_perclass] /= len(self.test_idt)
    #     self.idx_perclass += 1
        
    #     return A_current_stage
