import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)

def _get_env_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)

def _get_env_bool(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_env_optional_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value.strip().lower() in {"none", "cpu"}:
        return None
    return int(value)


def _get_env_str(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _configure_trainable_params(feature_extractor, mode):
    mode = (mode or "all").strip().lower()
    if mode == "all":
        return

    for _, param in feature_extractor.named_parameters():
        param.requires_grad = False

    if mode == "embedding_only":
        for name, param in feature_extractor.named_parameters():
            if name.startswith("embedding."):
                param.requires_grad = True
    elif mode == "embedding_transformer":
        for name, param in feature_extractor.named_parameters():
            if name.startswith("embedding.") or name.startswith("transformer."):
                param.requires_grad = True
    else:
        raise ValueError(f"Unsupported ICARL_TRAINABLE_PART: {mode}")

GPU_ID = _get_env_optional_int("ICARL_GPU_ID", 6)

if GPU_ID is None:
    # 不设置 CUDA_VISIBLE_DEVICES，让系统所有 GPU 都可见（或只用 CPU）
    pass
else:
    # 把要用的 GPU 暴露为可见（注意：这个必须在 import torch 之前）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import numpy as np
import random
from datetime import datetime

from iCaRL import CBiCaRL
from mlm import mlm_mask
from utils import fix_random_seed, save_results
from LogRecord import LogRecord

data_path = os.getenv('ICARL_DATA_PATH', '/data1/bochen/continental_leaning/cross_session_data/')
# data_path = '/data1/bochen/continental_leaning/data37/'
# data_path = '/data1/bochen/cbcontinual/data37'

pretrain_path = os.getenv('ICARL_PRETRAIN_PATH', '/data1/bochen/MIRepNet/weight/MIRepNet.pth')

numclass = _get_env_int('ICARL_INIT_NUMCLASS', 2)
batch_size = _get_env_int('ICARL_BATCH_SIZE', 32)
balance_sample = _get_env_bool('ICARL_BALANCE_SAMPLE', True)
num_stages = _get_env_int('ICARL_NUM_STAGES', 3)
num_seeds = _get_env_int('ICARL_NUM_SEEDS', 3)
epochs = _get_env_int('ICARL_EPOCHS', 30)
learning_rate = _get_env_float('ICARL_LR', 0.001)

is_cross_session = _get_env_bool('ICARL_CROSS_SESSION', True)

# Replay参数
memory_size = _get_env_int('ICARL_MEMORY_SIZE', 24)

is_contrastive_loss = _get_env_bool('ICARL_USE_CONTRASTIVE', True)
lambda_contrastive_loss = _get_env_float('ICARL_CONTRASTIVE_LAMBDA', 0.1)
temperature = _get_env_float('ICARL_TEMPERATURE', 0.3)

# LwF参数
use_lwf = _get_env_bool('ICARL_USE_LWF', False)
lwf_lambda = _get_env_float('ICARL_LWF_LAMBDA', 0.1)
lwf_T = _get_env_float('ICARL_LWF_T', 2.0)

is_align = _get_env_bool('ICARL_USE_ALIGN', True)

weighted_crossentropy = _get_env_bool('ICARL_WEIGHTED_CE', False)
trainable_part = _get_env_str('ICARL_TRAINABLE_PART', 'all')
use_proto_align = _get_env_bool('ICARL_USE_PROTO_ALIGN', False)
proto_align_lambda = _get_env_float('ICARL_PROTO_ALIGN_LAMBDA', 0.1)
use_task_adapter = _get_env_bool('ICARL_USE_TASK_ADAPTER', False)
task_adapter_dim = _get_env_int('ICARL_TASK_ADAPTER_DIM', 32)

run_tag = os.getenv('ICARL_RUN_TAG', '').strip()
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
if run_tag:
    current_date = f"{current_date}_{run_tag}"
result_dir = './logs/' + str(current_date) + '/'

all_seeds_results=[]

for seed in range(1, num_seeds+1):

    fix_random_seed(seed)

    # 初始化日志
    result_dir = './logs/' + str(current_date) + '/'
    log = LogRecord(result_dir, '2014001', 'MIRepNet', is_align)
    log.log_init()

    state_log = f'Replay memory size:{memory_size}, learning_rate:{learning_rate}, epochs:{epochs}, \
        is_cross_session:{is_cross_session}, is_balance_sample:{balance_sample}, is_contrastive_loss:{is_contrastive_loss},\
            lambda_contrastive_loss = {lambda_contrastive_loss}, temperature = {temperature}, trainable_part = {trainable_part}, \
                use_proto_align = {use_proto_align}, proto_align_lambda = {proto_align_lambda}, \
                    use_task_adapter = {use_task_adapter}, task_adapter_dim = {task_adapter_dim}'
    log.record(state_log)
    print(state_log)

    # feature_extractor=EEGNet(n_classes=numclass, Chans=22, Samples=1001, kernLength=64,F1=16, D=2, F2=32, dropoutRate=0.5)
    feature_extractor=mlm_mask(
        emb_size=256,
        depth=6,
        n_classes=2,
        pretrain=pretrain_path,
        pretrainmode=False,
        use_task_adapter=use_task_adapter,
        adapter_dim=task_adapter_dim,
        num_tasks=num_stages,
    )
    _configure_trainable_params(feature_extractor, trainable_part)

    model=CBiCaRL(seed,result_dir, data_path, is_cross_session, numclass,\
        feature_extractor,batch_size,\
        memory_size, balance_sample,is_contrastive_loss, lambda_contrastive_loss, temperature, \
        use_proto_align, proto_align_lambda, \
        use_lwf, lwf_lambda, lwf_T, weighted_crossentropy,\
        epochs,learning_rate,is_align,log,current_date)

    current_seed_stage_results = []

    for stage in range(1, num_stages+1):

        model.beforeTrain(stage)
        model.train()
        result_matrix = model.afterTrain()
        current_seed_stage_results.append(result_matrix)

    all_seeds_results.append(current_seed_stage_results)
    print(f"Seed {seed} finished.")

save_results(all_seeds_results,result_dir)
