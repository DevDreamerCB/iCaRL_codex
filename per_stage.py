import torch
import numpy as np
import random
from datetime import datetime

from iCaRL import iCaRLmodel
from EEGNet import EEGNet
from mlm import mlm_mask
from utils import fix_random_seed, save_results
from LogRecord import LogRecord

data_path = '/data1/bochen/continental_leaning/cross_session_data/'
# data_path = '/data1/bochen/continental_leaning/data37/'
# data_path = '/data1/bochen/cbcontinual/data37'
# pretrain_path = '/data1/bochen/MIRepNet/weight/MIRepNet.pth'

numclass=2
batch_size=32
num_seeds=3
epochs=50
learning_rate=0.001

is_cross_session = True

# Replay参数
per_subj_class = 0
replay_strategy = 'random'

# LwF参数
use_lwf = False
lwf_lambda = 0.1
lwf_T = 2.0

is_align = False
is_task_available = False

# 是否考虑样本
weighted_crossentropy = False

current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = './logs/' + str(current_date) + '/'

total_acc_persub = []
A_stage1 = []
A_stage2 = []
A_stage3 = []
A_perclasses = []

for seed in range(1, num_seeds+1):

    fix_random_seed(seed)

    # 初始化日志
    result_dir = './logs/' + str(current_date) + '/'
    log = LogRecord(result_dir, '2014001', 'EEGNet', is_align)
    log.log_init()

    feature_extractor=EEGNet(n_classes=numclass, Chans=22, Samples=1001, kernLength=64,F1=16, D=2, F2=32, dropoutRate=0.5)
    # feature_extractor=mlm_mask(emb_size=256, depth=6, n_classes=2, pretrain=pretrain_path,pretrainmode=False)

    model=iCaRLmodel(seed,data_path,is_cross_session,numclass,\
        feature_extractor,batch_size,\
        per_subj_class, replay_strategy,\
        use_lwf, lwf_lambda, lwf_T, weighted_crossentropy,\
        epochs,learning_rate,is_align,is_task_available,log,current_date)

    model.beforeTrain(1)
    model.train()
    acc_persub, A, A_perclass = model.afterTrain()

    total_acc_persub.append(acc_persub)
    A_stage1.append(A[1])
    A_perclasses.append(A_perclass)

import os 
import pandas as pd
save_dir = os.path.join(result_dir, 'A_matrices')
os.makedirs(save_dir, exist_ok=True)

# 2) 计算均值与标准差矩阵并保存
mean_A_stage1 = np.mean(A_stage1, axis=0)  
std_A_stage1 = np.std(A_stage1, axis=0)    
mean_acc_persub = np.mean(total_acc_persub, axis=0)
std_acc_persub = np.std(total_acc_persub, axis=0)
mean_acc_perclass = np.mean(A_perclasses, axis=0)
std_acc_perclass = np.std(A_perclasses, axis=0)

A_info_1 = f"Stage 1: mean:{mean_A_stage1}, std:{std_A_stage1}"
acc_persub_info = f"acc_persub: mean:{mean_acc_persub}, std:{std_acc_persub}"

# args.log.record(A_info_1)
# args.log.record(A_info_2)
# args.log.record(A_info_3)
# args.log.record(acc_persub_info)
print(A_info_1)
print(acc_persub_info)

# 保存各个阶段的平均准确率
# 定义行列标签
col_labels = ["4分类"]
row_labels = ["123456789"]

# --- 辅助函数：把 numpy 数组稳健转为 DataFrame，兼容 0D/1D/2D ---
def arr_to_df(arr, row_labels=None, col_labels=None, float_format=None):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    elif arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))  # 视为 1 行
    # now arr.ndim == 2
    r, c = arr.shape

    # 准备行列标签：优先使用用户提供的标签，短则补齐为默认名
    if row_labels is None:
        row_idx = [f"R{i+1}" for i in range(r)]
    else:
        # 若提供标签比需要的少，补齐；若多则截断
        row_idx = list(row_labels[:r]) + [f"R{i+1}" for i in range(len(row_labels), r)]

    if col_labels is None:
        col_idx = [f"C{j+1}" for j in range(c)]
    else:
        col_idx = list(col_labels[:c]) + [f"C{j+1}" for j in range(len(col_labels), c)]

    df = pd.DataFrame(arr, index=row_idx, columns=col_idx)
    if float_format is not None:
        # 返回格式化副本（不改变原始数值）
        df = df.round(float_format)
    return df

# --- 5) 构造 DataFrame（mean & std） ---
df_mean_stage1 = arr_to_df(mean_A_stage1, row_labels[:1], col_labels[:1])
df_std_stage1  = arr_to_df(std_A_stage1,  row_labels[:1], col_labels[:1])

# --- 6) 写入 Excel（注意使用 save_dir 变量，不要写成字符串 'save_dir/...') ---
excel_path = os.path.join(save_dir, 'A_matrices_by_stage.xlsx')
try:
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_mean_stage1.to_excel(writer, sheet_name='Stage1_mean')
        
        df_std_stage1.to_excel(writer,  sheet_name='Stage1_std')

    print(f"Saved Excel to: {excel_path}")
except Exception as e:
    # 捕获并记录异常，避免程序无提示地挂掉
    err_msg = f"Failed to save Excel ({excel_path}): {e}"
    print(err_msg)

# 记录每个被试的结果
row_labels = ['stage1']
col_labels = ['sub1','sub2','sub3','sub4','sub5','sub6','sub7','sub8','sub9','avg']

mean_df = pd.DataFrame(mean_acc_persub, index=row_labels, columns=col_labels)
std_df  = pd.DataFrame(std_acc_persub,  index=row_labels, columns=col_labels)

mean_df.to_csv(os.path.join(save_dir, 'persub_mean.csv'))
std_df.to_csv(os.path.join(save_dir, 'persub_std.csv'))

# 记录每个类别的结果
row_labels = ['stage1']
col_labels = ['left_hand', 'right_hand', 'feet', 'tongue']

mean_df = pd.DataFrame(mean_acc_perclass, index=row_labels, columns=col_labels)
std_df  = pd.DataFrame(std_acc_perclass,  index=row_labels, columns=col_labels)

mean_df.to_csv(os.path.join(save_dir, 'perclass_mean.csv'))
std_df.to_csv(os.path.join(save_dir, 'perclass_std.csv'))

print("Done!")
