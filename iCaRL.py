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
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from utils import extract_labels_from_dataset,\
        process_and_replace_loader, process_data_chn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CBiCaRL:
    def __init__(self, seed, result_dir, data_path, is_cross_session, numclass, feature_extractor, \
        batch_size, memory_size, balance_sample, is_contrastive_loss, lambda_contrastive_loss, temperature,\
        use_proto_align, proto_align_lambda, \
        use_lwf, lwf_lambda, lwf_T, weighted_crossentropy, \
            epochs, learning_rate, is_align, log, current_date):
        super().__init__()

        self.seed = seed
        self.result_dir = result_dir
        self.epochs = epochs
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

        self.train_loader=None
        self.test_loader=None

        # 重放参数
        self.memory_size = memory_size
        self.exemplar_set = []
        self.class_mean_set = []

        self.is_contrastive_loss = is_contrastive_loss
        self.lambda_contrastive_loss = lambda_contrastive_loss
        self.temperature = temperature
        self.use_proto_align = use_proto_align
        self.proto_align_lambda = proto_align_lambda
        self.old_class_prototypes = None

        # LwF参数
        self.prev_model = None
        self.use_lwf = use_lwf
        self.lwf_lambda = lwf_lambda
        self.lwf_T = lwf_T

        self.weighted_crossentropy = weighted_crossentropy
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
        
        # Preprocess data
        self.train_loader = process_and_replace_loader(
            self.train_loader, 
            ischangechn=True, 
            dataset='BNCI2014001-4'
        )
        self.test_loader = process_and_replace_loader(
            self.test_loader, 
            ischangechn=True, 
            dataset='BNCI2014001-4'
        )

        if self.stage > 1:
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.to(device)
            self.prev_model.eval()
            for p in self.prev_model.parameters():
                p.requires_grad = False
            if len(self.class_mean_set) > 0:
                old_prototypes = torch.tensor(np.array(self.class_mean_set), dtype=torch.float32)
                self.old_class_prototypes = F.normalize(old_prototypes, p=2, dim=1).to(device)
            else:
                self.old_class_prototypes = None
            self.model.Incremental_learning(self.numclass)
        else:
            self.old_class_prototypes = None
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
            # exemplar_objs 是该类别的样本数组 [N, C, T]
            class_data = torch.stack([torch.as_tensor(o) for o in exemplar_objs]).float()
            ex_datas.append(class_data)
            ex_labels.append(torch.full((len(exemplar_objs),), label, dtype=torch.long))
        
        # 合并所有旧类数据
        ex_datas = torch.cat(ex_datas, dim=0)
        ex_labels = torch.cat(ex_labels, dim=0)
        
        # 构建旧样本数据集
        exemplar_dataset = TensorDataset(ex_datas, ex_labels)
        return exemplar_dataset

    def _get_train_and_test_dataloader(self, train_idt, test_idt, train_class_list, test_class_list, balance_sample=False):
        
        X_train, y_train = self.dataset.get_train_data(train_idt, train_class_list)
        X_test, y_test = self.dataset.get_test_data(test_idt, test_class_list)

        subject_info = f"Selected subjects for train: {train_idt}"
        shape_info = f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        self.log.record(subject_info)
        self.log.record(shape_info)
        print(subject_info)
        print(shape_info)

        Xtr = torch.tensor(X_train, dtype=torch.float32)
        Ytr = torch.tensor(y_train, dtype=torch.long)
        Xte = torch.tensor(X_test, dtype=torch.float32)
        Yte = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(Xtr, Ytr)
        test_dataset = TensorDataset(Xte, Yte)

        # 将重放样本加入训练集
        exampler_dataset = self.get_exampler_dataset()
        if exampler_dataset is not None:
            train_dataset = ConcatDataset([train_dataset, exampler_dataset])
            shape_info = f"After replay, total num of trials for train: {len(train_dataset)}"
            self.log.record(shape_info)
            print(shape_info)

        if balance_sample:
            train_loader = self._balance_sample_train_loader(train_dataset=train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return train_loader, test_loader

    def _balance_sample_train_loader(self,train_dataset):
        '''
            batch内均衡采样实现
        '''
        # 1. 获取整个 train_dataset 中所有的标签
        all_labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
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
            class_weights[c] = 1.0 / torch.pow(class_counts[c], 0.5)

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

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0001
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.epochs, T_mult=1, eta_min=1e-6
        )

        train_loader_len = len(self.train_loader)

        for epoch in range(self.epochs):
            
            # accumulators for monitoring
            epoch_train_losses = []
            self.model.train()
            
            for step, (x, y) in enumerate(self.train_loader):
                # print(x.shape, y.shape)
                # torch.Size([32, 22, 1001]) torch.Size([32])
                x = x.unsqueeze(1).to(device)  # (B,1,Chans,Samples)
                y = y.to(device)
                
                features, logits = self.model(x, return_feat=True)

                # iCaRL原版
                target = torch.zeros_like(logits, dtype=torch.float32).to(device)
                
                if self.prev_model is not None:
                    self.prev_model.eval()
                    with torch.no_grad():
                        old_logits = self.prev_model(x)      # (B, old_k)
                        old_prob = torch.sigmoid(old_logits) # (B, old_k)
                    old_k = old_prob.size(1)
                    # 覆盖 target 的旧类部分为教师输出（论文 iCaRL 的做法）
                    target[:, :old_k] = old_prob
                
                target.scatter_(1, y.reshape(-1,1), 1.0)
                
                loss_bce = F.binary_cross_entropy_with_logits(logits, target)

                if self.is_contrastive_loss:
                    loss_con = self.supervised_contrastive_loss(features, y, temperature=self.temperature)
                    loss = loss_bce + self.lambda_contrastive_loss * loss_con
                else:
                    loss = loss_bce

                if self.use_proto_align and self.old_class_prototypes is not None:
                    loss_proto = self.prototype_alignment_loss(features, y, self.old_class_prototypes)
                    loss = loss + self.proto_align_lambda * loss_proto

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

    def prototype_alignment_loss(self, features, labels, prototypes):
        old_class_count = prototypes.size(0)
        old_mask = labels < old_class_count
        if old_mask.sum().item() == 0:
            return torch.tensor(0.0, device=features.device)

        old_features = F.normalize(features[old_mask], p=2, dim=1)
        old_labels = labels[old_mask]
        target_proto = prototypes[old_labels]
        cosine = F.cosine_similarity(old_features, target_proto, dim=1)
        return (1.0 - cosine).mean()

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
        m = int(self.memory_size/self.numclass)
        # m = min(m, )
        self._reduce_exemplar_sets(m)

        # 按照类别选择样本重放
        start_idx = 0 if self.stage == 1 else self.numclass-1
        for i in range(start_idx, self.numclass):
            construct_info = f'construct class {i} examplar:'
            self.log.record(construct_info)
            print(construct_info)
            class_list = np.arange(i, i+1)
            X_train, _ = self.dataset.get_train_data(self.train_idt, class_list)
            self._construct_exemplar_set(X_train,m)

        # 计算类别均值并评估
        self.compute_exemplar_class_mean()
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

    def _reduce_exemplar_sets(self, m):
        '''
            减少前面类别的重放样本
        '''
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            reduce_info = f'Size of class {index} examplar: {str(len(self.exemplar_set[index]))}'
            self.log.record(reduce_info)
            print(reduce_info)

    def _construct_exemplar_set(self, X_train, m):
        '''
            构建重放样本集(同一个类别),这里的X_train还没变换到45通道,存放的也是22通道的样本
            1.计算训练样本特征均值
            2.迭代更新,选择距离中心最近的样本
        '''
        # 先变换到45通道，并备份
        X_initial = X_train
        X_train = process_data_chn(X_train)

        class_mean, feature_extractor_output = self.compute_class_mean(X_train.unsqueeze(1))
        exemplar = []
        now_class_mean = np.zeros((1, 256))
     
        for i in range(m):
            # shape：batch_size*256
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
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
        # feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output
    
    def compute_exemplar_class_mean(self):
        '''
            计算buffer中各个类别的特征均值向量
        '''
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            mean_info = f"compute the class mean of class {str(index)}"
            self.log.record(mean_info)
            print(mean_info)
            #exemplar=self.train_dataset.get_image_class(index)
            # exemplar = torch.stack(exemplar, dim=0)
            exemplar = np.array(self.exemplar_set[index])
            exemplar = process_data_chn(exemplar)
            exemplar = exemplar.unsqueeze(1).to(device)

            class_mean, _ = self.compute_class_mean(exemplar)
            self.class_mean_set.append(class_mean)

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
        result = []
        # test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        feature = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in feature:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
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
