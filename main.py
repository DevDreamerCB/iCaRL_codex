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


def _get_env_float_list(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return [float(item.strip()) for item in value.split(',') if item.strip()]


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
use_age_replay = _get_env_bool('ICARL_USE_AGE_REPLAY', False)
age_replay_power = _get_env_float('ICARL_AGE_REPLAY_POWER', 1.0)
use_replay_lr_flip = _get_env_bool('ICARL_USE_REPLAY_LR_FLIP', False)
replay_lr_flip_prob = _get_env_float('ICARL_REPLAY_LR_FLIP_PROB', 0.5)
replay_lr_flip_start_task = _get_env_int('ICARL_REPLAY_LR_FLIP_START_TASK', 3)
use_replay_mixup = _get_env_bool('ICARL_USE_REPLAY_MIXUP', False)
replay_mixup_alpha = _get_env_float('ICARL_REPLAY_MIXUP_ALPHA', 0.2)
replay_mixup_lambda = _get_env_float('ICARL_REPLAY_MIXUP_LAMBDA', 0.5)
replay_mixup_start_task = _get_env_int('ICARL_REPLAY_MIXUP_START_TASK', 3)
use_replay_logits_distill = _get_env_bool('ICARL_USE_REPLAY_LOGITS_DISTILL', False)
replay_logits_lambda = _get_env_float('ICARL_REPLAY_LOGITS_LAMBDA', 0.5)
replay_logits_start_task = _get_env_int('ICARL_REPLAY_LOGITS_START_TASK', 2)
use_replay_hardness = _get_env_bool('ICARL_USE_REPLAY_HARDNESS', False)
replay_hardness_power = _get_env_float('ICARL_REPLAY_HARDNESS_POWER', 1.0)
replay_hardness_start_task = _get_env_int('ICARL_REPLAY_HARDNESS_START_TASK', 3)
use_replay_repeat = _get_env_bool('ICARL_USE_REPLAY_REPEAT', False)
replay_repeat_lambda = _get_env_float('ICARL_REPLAY_REPEAT_LAMBDA', 0.5)
replay_repeat_start_task = _get_env_int('ICARL_REPLAY_REPEAT_START_TASK', 3)
use_replay_global_ea = _get_env_bool('ICARL_USE_REPLAY_GLOBAL_EA', False)
replay_global_ea_start_task = _get_env_int('ICARL_REPLAY_GLOBAL_EA_START_TASK', 3)
num_stages = _get_env_int('ICARL_NUM_STAGES', 3)
num_seeds = _get_env_int('ICARL_NUM_SEEDS', 3)
epochs = _get_env_int('ICARL_EPOCHS', 30)
stage_epochs = _get_env_int_list('ICARL_STAGE_EPOCHS', None)
learning_rate = _get_env_float('ICARL_LR', 0.001)

is_cross_session = _get_env_bool('ICARL_CROSS_SESSION', True)

# Replay参数
memory_size = _get_env_int('ICARL_MEMORY_SIZE', 24)
use_age_memory = _get_env_bool('ICARL_USE_AGE_MEMORY', False)
age_memory_power = _get_env_float('ICARL_AGE_MEMORY_POWER', 1.0)

is_contrastive_loss = _get_env_bool('ICARL_USE_CONTRASTIVE', True)
lambda_contrastive_loss = _get_env_float('ICARL_CONTRASTIVE_LAMBDA', 0.1)
temperature = _get_env_float('ICARL_TEMPERATURE', 0.3)

# LwF参数
use_lwf = _get_env_bool('ICARL_USE_LWF', False)
lwf_lambda = _get_env_float('ICARL_LWF_LAMBDA', 0.1)
lwf_T = _get_env_float('ICARL_LWF_T', 2.0)
stage_lwf_lambdas = _get_env_float_list('ICARL_STAGE_LWF_LAMBDAS', None)
use_feature_distill = _get_env_bool('ICARL_USE_FEATURE_DISTILL', False)
feature_distill_lambda = _get_env_float('ICARL_FEATURE_DISTILL_LAMBDA', 0.1)
stage_feature_distill_lambdas = _get_env_float_list('ICARL_STAGE_FEATURE_DISTILL_LAMBDAS', None)

is_align = _get_env_bool('ICARL_USE_ALIGN', True)

weighted_crossentropy = _get_env_bool('ICARL_WEIGHTED_CE', False)
old_class_weight_power = _get_env_float('ICARL_OLD_CLASS_WEIGHT_POWER', 0.0)
stage_old_class_weight_powers = _get_env_float_list('ICARL_STAGE_OLD_CLASS_WEIGHT_POWERS', None)
trainable_part = _get_env_str('ICARL_TRAINABLE_PART', 'all')
use_proto_align = _get_env_bool('ICARL_USE_PROTO_ALIGN', False)
proto_align_lambda = _get_env_float('ICARL_PROTO_ALIGN_LAMBDA', 0.1)
use_normalized_nme = _get_env_bool('ICARL_USE_NORMALIZED_NME', False)
use_diag_cov_nme = _get_env_bool('ICARL_USE_DIAG_COV_NME', False)
diag_cov_nme_shrink = _get_env_float('ICARL_DIAG_COV_NME_SHRINK', 0.1)
use_radius_nme = _get_env_bool('ICARL_USE_RADIUS_NME', False)
radius_nme_power = _get_env_float('ICARL_RADIUS_NME_POWER', 1.0)
use_age_nme = _get_env_bool('ICARL_USE_AGE_NME', False)
age_nme_power = _get_env_float('ICARL_AGE_NME_POWER', 0.0)
use_group_bias_calibration = _get_env_bool('ICARL_USE_GROUP_BIAS_CALIBRATION', False)
group_bias_alpha_min = _get_env_float('ICARL_GROUP_BIAS_ALPHA_MIN', 0.85)
group_bias_alpha_max = _get_env_float('ICARL_GROUP_BIAS_ALPHA_MAX', 1.35)
group_bias_alpha_steps = _get_env_int('ICARL_GROUP_BIAS_ALPHA_STEPS', 11)
group_bias_beta_min = _get_env_float('ICARL_GROUP_BIAS_BETA_MIN', -0.25)
group_bias_beta_max = _get_env_float('ICARL_GROUP_BIAS_BETA_MAX', 0.05)
group_bias_beta_steps = _get_env_int('ICARL_GROUP_BIAS_BETA_STEPS', 13)
group_bias_old_weight = _get_env_float('ICARL_GROUP_BIAS_OLD_WEIGHT', 0.6)
use_weight_align = _get_env_bool('ICARL_USE_WEIGHT_ALIGN', False)
weight_align_start_task = _get_env_int('ICARL_WEIGHT_ALIGN_START_TASK', 2)
use_hybrid_nme_logits = _get_env_bool('ICARL_USE_HYBRID_NME_LOGITS', False)
hybrid_start_task = _get_env_int('ICARL_HYBRID_START_TASK', 2)
hybrid_alpha_min = _get_env_float('ICARL_HYBRID_ALPHA_MIN', 0.0)
hybrid_alpha_max = _get_env_float('ICARL_HYBRID_ALPHA_MAX', 1.0)
hybrid_alpha_steps = _get_env_int('ICARL_HYBRID_ALPHA_STEPS', 11)
hybrid_old_weight = _get_env_float('ICARL_HYBRID_OLD_WEIGHT', 0.6)
use_subject_class_align = _get_env_bool('ICARL_USE_SUBJECT_CLASS_ALIGN', False)
subject_class_align_lambda = _get_env_float('ICARL_SUBJECT_CLASS_ALIGN_LAMBDA', 0.05)
use_current_prototype_blend = _get_env_bool('ICARL_USE_CURRENT_PROTOTYPE_BLEND', False)
current_prototype_blend_alpha = _get_env_float('ICARL_CURRENT_PROTOTYPE_BLEND_ALPHA', 0.5)
current_prototype_blend_start_task = _get_env_int('ICARL_CURRENT_PROTOTYPE_BLEND_START_TASK', 1)
current_prototype_blend_mode = _get_env_str('ICARL_CURRENT_PROTOTYPE_BLEND_MODE', 'global')
current_prototype_blend_scope = _get_env_str('ICARL_CURRENT_PROTOTYPE_BLEND_SCOPE', 'current')
use_prototype_neighbor_calibration = _get_env_bool('ICARL_USE_PROTOTYPE_NEIGHBOR_CALIBRATION', False)
prototype_neighbor_calibration_beta = _get_env_float('ICARL_PROTOTYPE_NEIGHBOR_CALIBRATION_BETA', 0.2)
exemplar_mode = _get_env_str('ICARL_EXEMPLAR_MODE', 'legacy_herding')
exemplar_mode_start_task = _get_env_int('ICARL_EXEMPLAR_MODE_START_TASK', 1)
exemplar_diversity_lambda = _get_env_float('ICARL_EXEMPLAR_DIVERSITY_LAMBDA', 0.1)
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

    state_log = f'Replay memory size:{memory_size}, use_age_memory:{use_age_memory}, age_memory_power:{age_memory_power}, learning_rate:{learning_rate}, epochs:{epochs}, stage_epochs:{stage_epochs}, \
        is_cross_session:{is_cross_session}, is_balance_sample:{balance_sample}, balance_power:{balance_power}, replay_batch_size:{replay_batch_size}, use_age_replay:{use_age_replay}, age_replay_power:{age_replay_power}, use_replay_lr_flip:{use_replay_lr_flip}, replay_lr_flip_prob:{replay_lr_flip_prob}, replay_lr_flip_start_task:{replay_lr_flip_start_task}, use_replay_mixup:{use_replay_mixup}, replay_mixup_alpha:{replay_mixup_alpha}, replay_mixup_lambda:{replay_mixup_lambda}, replay_mixup_start_task:{replay_mixup_start_task}, use_replay_logits_distill:{use_replay_logits_distill}, replay_logits_lambda:{replay_logits_lambda}, replay_logits_start_task:{replay_logits_start_task}, use_replay_hardness:{use_replay_hardness}, replay_hardness_power:{replay_hardness_power}, replay_hardness_start_task:{replay_hardness_start_task}, use_replay_repeat:{use_replay_repeat}, replay_repeat_lambda:{replay_repeat_lambda}, replay_repeat_start_task:{replay_repeat_start_task}, use_replay_global_ea:{use_replay_global_ea}, replay_global_ea_start_task:{replay_global_ea_start_task}, is_contrastive_loss:{is_contrastive_loss},\
            lambda_contrastive_loss = {lambda_contrastive_loss}, temperature = {temperature}, weighted_crossentropy = {weighted_crossentropy}, old_class_weight_power = {old_class_weight_power}, use_feature_distill = {use_feature_distill}, feature_distill_lambda = {feature_distill_lambda}, trainable_part = {trainable_part}, \
                stage_old_class_weight_powers = {stage_old_class_weight_powers}, \
                stage_lwf_lambdas = {stage_lwf_lambdas}, \
                stage_feature_distill_lambdas = {stage_feature_distill_lambdas}, \
                use_proto_align = {use_proto_align}, proto_align_lambda = {proto_align_lambda}, use_normalized_nme = {use_normalized_nme}, \
                    use_diag_cov_nme = {use_diag_cov_nme}, diag_cov_nme_shrink = {diag_cov_nme_shrink}, \
                    use_radius_nme = {use_radius_nme}, radius_nme_power = {radius_nme_power}, use_age_nme = {use_age_nme}, age_nme_power = {age_nme_power}, \
                    use_group_bias_calibration = {use_group_bias_calibration}, group_bias_alpha_min = {group_bias_alpha_min}, group_bias_alpha_max = {group_bias_alpha_max}, \
                        group_bias_alpha_steps = {group_bias_alpha_steps}, group_bias_beta_min = {group_bias_beta_min}, group_bias_beta_max = {group_bias_beta_max}, \
                            group_bias_beta_steps = {group_bias_beta_steps}, group_bias_old_weight = {group_bias_old_weight}, \
                    use_weight_align = {use_weight_align}, weight_align_start_task = {weight_align_start_task}, \
                    use_hybrid_nme_logits = {use_hybrid_nme_logits}, hybrid_start_task = {hybrid_start_task}, hybrid_alpha_min = {hybrid_alpha_min}, hybrid_alpha_max = {hybrid_alpha_max}, \
                        hybrid_alpha_steps = {hybrid_alpha_steps}, hybrid_old_weight = {hybrid_old_weight}, \
                    use_subject_class_align = {use_subject_class_align}, subject_class_align_lambda = {subject_class_align_lambda}, \
                    use_current_prototype_blend = {use_current_prototype_blend}, current_prototype_blend_alpha = {current_prototype_blend_alpha}, current_prototype_blend_start_task = {current_prototype_blend_start_task}, current_prototype_blend_mode = {current_prototype_blend_mode}, current_prototype_blend_scope = {current_prototype_blend_scope}, \
                    use_prototype_neighbor_calibration = {use_prototype_neighbor_calibration}, prototype_neighbor_calibration_beta = {prototype_neighbor_calibration_beta}, \
                    exemplar_mode = {exemplar_mode}, exemplar_mode_start_task = {exemplar_mode_start_task}, exemplar_diversity_lambda = {exemplar_diversity_lambda}, \
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
        memory_size, use_age_memory, age_memory_power, balance_sample, balance_power, replay_batch_size, use_age_replay, age_replay_power, use_replay_lr_flip, replay_lr_flip_prob, replay_lr_flip_start_task, use_replay_mixup, replay_mixup_alpha, replay_mixup_lambda, replay_mixup_start_task, use_replay_logits_distill, replay_logits_lambda, replay_logits_start_task, use_replay_hardness, replay_hardness_power, replay_hardness_start_task, use_replay_repeat, replay_repeat_lambda, replay_repeat_start_task, use_replay_global_ea, replay_global_ea_start_task, is_contrastive_loss, lambda_contrastive_loss, temperature, \
        use_proto_align, proto_align_lambda, use_normalized_nme, use_diag_cov_nme, diag_cov_nme_shrink, use_radius_nme, radius_nme_power, use_age_nme, age_nme_power, \
        use_group_bias_calibration, group_bias_alpha_min, group_bias_alpha_max, group_bias_alpha_steps, group_bias_beta_min, group_bias_beta_max, group_bias_beta_steps, group_bias_old_weight, \
        use_weight_align, weight_align_start_task, \
        use_hybrid_nme_logits, hybrid_start_task, hybrid_alpha_min, hybrid_alpha_max, hybrid_alpha_steps, hybrid_old_weight, \
        use_subject_class_align, subject_class_align_lambda, \
        use_current_prototype_blend, current_prototype_blend_alpha, current_prototype_blend_start_task, current_prototype_blend_mode, current_prototype_blend_scope, use_prototype_neighbor_calibration, prototype_neighbor_calibration_beta, \
        exemplar_mode, exemplar_mode_start_task, exemplar_diversity_lambda, \
        task_adapter_lr_mult, \
        use_lwf, lwf_lambda, lwf_T, stage_lwf_lambdas, use_feature_distill, feature_distill_lambda, stage_feature_distill_lambdas, weighted_crossentropy, old_class_weight_power, stage_old_class_weight_powers,\
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
