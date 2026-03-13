from numpy.random import noncentral_chisquare
import pandas as pd
import torch
import numpy as np
import os
import torch.nn as nn
import random
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch.nn.functional as F
from channel_list import *
from scipy.spatial.distance import cdist
from scipy.linalg import fractional_matrix_power

def save_results(all_seeds_results, result_dir):
    """
    all_seeds_results: [Seed_Num, Stage_Num] 的嵌套列表
    每个元素是形状为 (Subject_Num, Class_Num) 的 ndarray
    """
    # 1. 数据准备
    # 将列表转换为 3D 张量: (Seed_Num, Subject_Num, Class_Num)
    # 例如: (5, 9, 4)

    num_stages = len(all_seeds_results[0])
    # 完整的类别名称定义
    full_class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']

    # 遍历每一个阶段 (Task)
    for stage_idx in range(num_stages):
        stage_results_list = [seed_res[stage_idx] for seed_res in all_seeds_results]
        all_results_np = np.stack(stage_results_list, axis=0) * 100

        # 获取当前阶段的维度信息
        num_seeds, num_subjects, current_num_classes = all_results_np.shape

        # -------------------------------------------------------------
        # 2. 计算各个维度的 Mean 和 Std
        # -------------------------------------------------------------

        # A. 主体部分 (Subject x Class)
        # 沿 Seed (axis=0) 聚合
        body_mean = np.mean(all_results_np, axis=0)  # Shape: (9, 4)
        body_std = np.std(all_results_np, axis=0)    # Shape: (9, 4)

        # B. 右侧新增列：每个被试的平均准确率 (Subject Avg)
        # 先沿 Class (axis=2) 求均值，得到 (Seed, Subject)，再沿 Seed 求 Mean/Std
        sub_avg_data = np.mean(all_results_np, axis=2) 
        col_mean = np.mean(sub_avg_data, axis=0)     # Shape: (9,)
        col_std = np.std(sub_avg_data, axis=0)       # Shape: (9,)

        # C. 底部新增行：每个类别的平均准确率 (Class Avg)
        # 先沿 Subject (axis=1) 求均值，得到 (Seed, Class)，再沿 Seed 求 Mean/Std
        cls_avg_data = np.mean(all_results_np, axis=1)
        row_mean = np.mean(cls_avg_data, axis=0)     # Shape: (4,)
        row_std = np.std(cls_avg_data, axis=0)       # Shape: (4,)

        # D. 右下角：全局平均准确率 (Total Avg)
        # 先沿 Subject 和 Class 同时求均值，得到 (Seed,)，再沿 Seed 求 Mean/Std
        total_avg_data = np.mean(all_results_np, axis=(1, 2))
        total_mean = np.mean(total_avg_data)         # Scalar
        total_std = np.std(total_avg_data)           # Scalar

        # -------------------------------------------------------------
        # 3. 构建 DataFrame 并格式化
        # -------------------------------------------------------------
        
        current_class_names = full_class_names[:current_num_classes]
        # class_names = ['Left Hand', 'Right Hand']
        # 扩展列名
        columns = current_class_names + ['Subject Avg']
        # 扩展行索引
        subject_ids = [f'Subject_{i+1}' for i in range(body_mean.shape[0])] + ['Class Avg']

        # 创建一个空的 DataFrame用于存放最终字符串
        df_combined = pd.DataFrame(index=subject_ids, columns=columns)

        # --- 填充主体部分 (9x4) ---
        for r in range(body_mean.shape[0]):
            for c in range(body_mean.shape[1]):
                df_combined.iloc[r, c] = f"{body_mean[r, c]:.2f}±{body_std[r, c]:.2f}"

        # --- 填充右侧 "Subject Avg" 列 (前9行) ---
        for r in range(len(col_mean)):
            df_combined.iloc[r, -1] = f"{col_mean[r]:.2f}±{col_std[r]:.2f}"

        # --- 填充底部 "Class Avg" 行 (前4列) ---
        for c in range(len(row_mean)):
            df_combined.iloc[-1, c] = f"{row_mean[c]:.2f}±{row_std[c]:.2f}"

        # --- 填充右下角 "Total Avg" ---
        df_combined.iloc[-1, -1] = f"{total_mean:.2f}±{total_std:.2f}"

        # -------------------------------------------------------------
        # 4. (可选) 构建纯数值的 Mean 表，方便后续画图
        # -------------------------------------------------------------
        # 为了方便，先把数据拼起来
        # 拼列
        full_mean = np.c_[body_mean, col_mean]
        full_std = np.c_[body_std, col_std]
        # 拼行 (注意要先把 row_mean 加一个 total_mean 拼成 (5,) 再叠上去)
        bottom_row_mean = np.append(row_mean, total_mean)
        bottom_row_std = np.append(row_std, total_std)
        
        full_mean = np.vstack([full_mean, bottom_row_mean])
        full_std = np.vstack([full_std, bottom_row_std])

        df_mean_raw = pd.DataFrame(full_mean, index=subject_ids, columns=columns)
        df_std_raw = pd.DataFrame(full_std, index=subject_ids, columns=columns)

        # -------------------------------------------------------------
        # 5. 保存
        # -------------------------------------------------------------
        file_name = f'stage_{stage_idx+1}_classes_{current_num_classes}.xlsx'
        save_path = os.path.join(result_dir, file_name)

        with pd.ExcelWriter(save_path) as writer:
            df_combined.to_excel(writer, sheet_name='Mean_Std_Combined')
            df_mean_raw.to_excel(writer, sheet_name='Raw_Mean')
            df_std_raw.to_excel(writer, sheet_name='Raw_Std')

        print(f"Stage {stage_idx+1} 结果已保存至: {save_path}")

    print("\n所有阶段统计完成。")

def create_folder(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def fix_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 用于weighted-crossentropy，统计各个类别的样本数目
def extract_labels_from_dataset(ds):
    # 尝试高效路径：TensorDataset 有 .tensors
    try:
        from torch.utils.data import ConcatDataset
        if hasattr(ds, 'tensors'):
            # TensorDataset
            y_tensor = ds.tensors[1]
            return y_tensor.detach().cpu().numpy()
        elif isinstance(ds, ConcatDataset):
            parts = []
            for sub in ds.datasets:
                if hasattr(sub, 'tensors'):
                    parts.append(sub.tensors[1].detach().cpu().numpy())
                else:
                    # 最后退回到逐样本遍历（较慢）
                    tmp = []
                    for _, yy in sub:
                        tmp.append(int(yy.cpu().numpy()))
                    if len(tmp) > 0:
                        parts.append(np.array(tmp, dtype=np.int64))
            if len(parts) == 0:
                # 最后退回全遍历
                tmp = []
                for _, yy in ds:
                    tmp.append(int(yy.cpu().numpy()))
                return np.array(tmp, dtype=np.int64)
            else:
                return np.concatenate(parts, axis=0)
        else:
            # 普通 Dataset，逐样本读取 labels（可能慢）
            labels = []
            for _, yy in ds:
                labels.append(int(yy.detach().cpu().numpy()))
            return np.array(labels, dtype=np.int64)
    except Exception:
        # 极端异常时退回遍历
        labels = []
        for _, yy in ds:
            labels.append(int(yy.detach().cpu().numpy()))
        return np.array(labels, dtype=np.int64)

def process_data_chn(data):
    print("before processed：", data.shape)
    channels_names = BNCI2014001_chn_names
    processed_data = pad_missing_channels_diff(data,use_channels_names,channels_names)
    print("after processed：", processed_data.shape)
    return torch.tensor(processed_data, dtype=torch.float32)
    
def process_and_replace_loader(loader,ischangechn,dataset):
    all_data = []
    all_labels = []
    for i in range(len(loader.dataset)):
        data, label = loader.dataset[i]
        all_data.append(data.numpy())
        all_labels.append(label)
    
    # data_np = np.stack(all_data, axis=0)
    processed_data = np.stack(all_data, axis=0).astype(np.float32)
    labels_tensor = torch.stack(all_labels)
    
    # processed_data = EA(data_np).astype(np.float32)  
    
    if ischangechn:
        print("before processed：", processed_data.shape)
        if dataset == 'BNCI2014001':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'BNCI2014004':
            channels_names = BNCI2014004_chn_names
        elif dataset == 'BNCI2014001-4':
            channels_names = BNCI2014001_chn_names
        elif dataset == 'AlexMI':
            channels_names = AlexMI_chn_names
        elif dataset =='BNCI2015001':
            channels_names = BNCI2015001_chn_names
        processed_data = pad_missing_channels_diff(processed_data,use_channels_names,channels_names)
        print("after processed：", processed_data.shape)
    new_dataset = TensorDataset(
        torch.from_numpy(processed_data).float(),  
        labels_tensor
    )
    
    loader_args = {
        'batch_size': loader.batch_size,
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
    }

    if isinstance(loader.sampler, WeightedRandomSampler):
        new_sampler = WeightedRandomSampler(
            weights=loader.sampler.weights.clone().detach(),
            num_samples=loader.sampler.num_samples,
            replacement=loader.sampler.replacement,
        )
        loader_args['sampler'] = new_sampler
        loader_args['shuffle'] = False
    elif isinstance(loader.sampler, RandomSampler):
        loader_args['shuffle'] = True
    elif isinstance(loader.sampler, SequentialSampler):
        loader_args['shuffle'] = False
    else:
        loader_args['shuffle'] = False
    
    return torch.utils.data.DataLoader(new_dataset, **loader_args)

def pad_missing_channels_diff(x, target_channels, actual_channels):
    B, C, T = x.shape
    num_target = len(target_channels)
    
    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])

    target_pos = np.array([channel_positions[ch] for ch in target_channels])
    
    W = np.zeros((num_target, C))
    for i, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            W[i, src_idx] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1 / (dist + 1e-6)  
            weights /= weights.sum()     
            W[i] = weights
    
    padded = np.zeros((B, num_target, T))
    for b in range(B):
        padded[b] = W @ x[b]  
    
    return padded

