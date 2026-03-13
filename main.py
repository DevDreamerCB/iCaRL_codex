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


def _get_env_int_list(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return [int(item.strip()) for item in value.split(',') if item.strip()]


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
balance_power = _get_env_float('ICARL_BALANCE_POWER', 0.5)
replay_batch_size = _get_env_int('ICARL_REPLAY_BATCH_SIZE', 0)
num_stages = _get_env_int('ICARL_NUM_STAGES', 3)
num_seeds = _get_env_int('ICARL_NUM_SEEDS', 3)
epochs = _get_env_int('ICARL_EPOCHS', 30)
stage_epochs = _get_env_int_list('ICARL_STAGE_EPOCHS', None)
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
task_adapter_dropout = _get_env_float('ICARL_TASK_ADAPTER_DROPOUT', 0.1)
task_adapter_start_task = _get_env_int('ICARL_TASK_ADAPTER_START_TASK', 0)
task_adapter_lr_mult = _get_env_float('ICARL_TASK_ADAPTER_LR_MULT', 1.0)
use_shared_adapter = _get_env_bool('ICARL_USE_SHARED_ADAPTER', False)
shared_adapter_dim = _get_env_int('ICARL_SHARED_ADAPTER_DIM', 16)
shared_adapter_dropout = _get_env_float('ICARL_SHARED_ADAPTER_DROPOUT', 0.1)
shared_adapter_start_task = _get_env_int('ICARL_SHARED_ADAPTER_START_TASK', 0)
use_task_prompt = _get_env_bool('ICARL_USE_TASK_PROMPT', False)
task_prompt_len = _get_env_int('ICARL_TASK_PROMPT_LEN', 4)
task_prompt_start_task = _get_env_int('ICARL_TASK_PROMPT_START_TASK', 0)
use_task_lora = _get_env_bool('ICARL_USE_TASK_LORA', False)
task_lora_rank = _get_env_int('ICARL_TASK_LORA_RANK', 4)
task_lora_alpha = _get_env_float('ICARL_TASK_LORA_ALPHA', 1.0)
task_lora_dropout = _get_env_float('ICARL_TASK_LORA_DROPOUT', 0.0)
task_lora_start_task = _get_env_int('ICARL_TASK_LORA_START_TASK', 0)
use_task_affine = _get_env_bool('ICARL_USE_TASK_AFFINE', False)
task_affine_start_task = _get_env_int('ICARL_TASK_AFFINE_START_TASK', 0)
use_task_bn = _get_env_bool('ICARL_USE_TASK_BN', False)
task_bn_start_task = _get_env_int('ICARL_TASK_BN_START_TASK', 0)

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

    state_log = f'Replay memory size:{memory_size}, learning_rate:{learning_rate}, epochs:{epochs}, stage_epochs:{stage_epochs}, \
        is_cross_session:{is_cross_session}, is_balance_sample:{balance_sample}, balance_power:{balance_power}, replay_batch_size:{replay_batch_size}, is_contrastive_loss:{is_contrastive_loss},\
            lambda_contrastive_loss = {lambda_contrastive_loss}, temperature = {temperature}, weighted_crossentropy = {weighted_crossentropy}, trainable_part = {trainable_part}, \
                use_proto_align = {use_proto_align}, proto_align_lambda = {proto_align_lambda}, \
                    use_task_adapter = {use_task_adapter}, task_adapter_dim = {task_adapter_dim}, \
                        task_adapter_dropout = {task_adapter_dropout}, task_adapter_start_task = {task_adapter_start_task}, task_adapter_lr_mult = {task_adapter_lr_mult}, \
                            use_shared_adapter = {use_shared_adapter}, shared_adapter_dim = {shared_adapter_dim}, \
                                shared_adapter_dropout = {shared_adapter_dropout}, shared_adapter_start_task = {shared_adapter_start_task}, \
                                    use_task_prompt = {use_task_prompt}, task_prompt_len = {task_prompt_len}, \
                                        task_prompt_start_task = {task_prompt_start_task}, \
                                            use_task_lora = {use_task_lora}, task_lora_rank = {task_lora_rank}, \
                                                task_lora_alpha = {task_lora_alpha}, task_lora_dropout = {task_lora_dropout}, \
                                                    task_lora_start_task = {task_lora_start_task}, \
                            use_task_affine = {use_task_affine}, task_affine_start_task = {task_affine_start_task}, \
                                use_task_bn = {use_task_bn}, task_bn_start_task = {task_bn_start_task}'
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
        adapter_dropout=task_adapter_dropout,
        num_tasks=num_stages,
        adapter_start_task=task_adapter_start_task,
        use_shared_adapter=use_shared_adapter,
        shared_adapter_dim=shared_adapter_dim,
        shared_adapter_dropout=shared_adapter_dropout,
        shared_adapter_start_task=shared_adapter_start_task,
        use_task_prompt=use_task_prompt,
        task_prompt_len=task_prompt_len,
        task_prompt_start_task=task_prompt_start_task,
        use_task_lora=use_task_lora,
        task_lora_rank=task_lora_rank,
        task_lora_alpha=task_lora_alpha,
        task_lora_dropout=task_lora_dropout,
        task_lora_start_task=task_lora_start_task,
        use_task_affine=use_task_affine,
        affine_start_task=task_affine_start_task,
        use_task_bn=use_task_bn,
        bn_start_task=task_bn_start_task,
    )
    _configure_trainable_params(feature_extractor, trainable_part)

    model=CBiCaRL(seed,result_dir, data_path, is_cross_session, numclass,\
        feature_extractor,batch_size,\
        memory_size, balance_sample, balance_power, replay_batch_size, is_contrastive_loss, lambda_contrastive_loss, temperature, \
        use_proto_align, proto_align_lambda, \
        task_adapter_lr_mult, \
        use_lwf, lwf_lambda, lwf_T, weighted_crossentropy,\
        epochs, stage_epochs, learning_rate,is_align,log,current_date)

    current_seed_stage_results = []

    for stage in range(1, num_stages+1):

        model.beforeTrain(stage)
        model.train()
        result_matrix = model.afterTrain()
        current_seed_stage_results.append(result_matrix)

    all_seeds_results.append(current_seed_stage_results)
    print(f"Seed {seed} finished.")

save_results(all_seeds_results,result_dir)
