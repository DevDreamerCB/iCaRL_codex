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
    elif mode == "peft_only":
        peft_tags = (
            "shared_adapter.",
            "task_adapter.",
            "embedding.task_bn.",
            "embedding.task_scale",
            "embedding.task_bias",
        )
        for name, param in feature_extractor.named_parameters():
            if any(tag in name for tag in peft_tags):
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
stage_replay_batch_sizes = _get_env_int_list('ICARL_STAGE_REPLAY_BATCH_SIZES', None)
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
stage_lwf_lambdas = _get_env_float_list('ICARL_STAGE_LWF_LAMBDAS', None)
use_feature_distill = _get_env_bool('ICARL_USE_FEATURE_DISTILL', False)
feature_distill_lambda = _get_env_float('ICARL_FEATURE_DISTILL_LAMBDA', 0.1)
stage_feature_distill_lambdas = _get_env_float_list('ICARL_STAGE_FEATURE_DISTILL_LAMBDAS', None)
exclusive_old_feature_distill_boost = _get_env_float('ICARL_EXCLUSIVE_OLD_FEATURE_DISTILL_BOOST', 1.0)
stage_exclusive_old_feature_distill_boosts = _get_env_float_list('ICARL_STAGE_EXCLUSIVE_OLD_FEATURE_DISTILL_BOOSTS', None)
overlap_align_lambda = _get_env_float('ICARL_OVERLAP_ALIGN_LAMBDA', 0.0)
stage_overlap_align_lambdas = _get_env_float_list('ICARL_STAGE_OVERLAP_ALIGN_LAMBDAS', None)

is_align = _get_env_bool('ICARL_USE_ALIGN', True)

weighted_crossentropy = _get_env_bool('ICARL_WEIGHTED_CE', False)
old_class_weight_power = _get_env_float('ICARL_OLD_CLASS_WEIGHT_POWER', 0.0)
stage_old_class_weight_powers = _get_env_float_list('ICARL_STAGE_OLD_CLASS_WEIGHT_POWERS', None)
exclusive_old_boost = _get_env_float('ICARL_EXCLUSIVE_OLD_BOOST', 1.0)
stage_exclusive_old_boosts = _get_env_float_list('ICARL_STAGE_EXCLUSIVE_OLD_BOOSTS', None)
absent_old_current_weight = _get_env_float('ICARL_ABSENT_OLD_CURRENT_WEIGHT', 1.0)
stage_absent_old_current_weights = _get_env_float_list('ICARL_STAGE_ABSENT_OLD_CURRENT_WEIGHTS', None)
exclusive_old_replay_boost = _get_env_float('ICARL_EXCLUSIVE_OLD_REPLAY_BOOST', 1.0)
stage_exclusive_old_replay_boosts = _get_env_float_list('ICARL_STAGE_EXCLUSIVE_OLD_REPLAY_BOOSTS', None)
trainable_part = _get_env_str('ICARL_TRAINABLE_PART', 'all')
use_normalized_nme = _get_env_bool('ICARL_USE_NORMALIZED_NME', False)
use_hybrid_nme_logits = _get_env_bool('ICARL_USE_HYBRID_NME_LOGITS', False)
hybrid_start_task = _get_env_int('ICARL_HYBRID_START_TASK', 2)
hybrid_alpha_min = _get_env_float('ICARL_HYBRID_ALPHA_MIN', 0.0)
hybrid_alpha_max = _get_env_float('ICARL_HYBRID_ALPHA_MAX', 1.0)
hybrid_alpha_steps = _get_env_int('ICARL_HYBRID_ALPHA_STEPS', 11)
hybrid_old_weight = _get_env_float('ICARL_HYBRID_OLD_WEIGHT', 0.6)
hybrid_focus_classes = _get_env_int_list('ICARL_HYBRID_FOCUS_CLASSES', None)
hybrid_focus_weight = _get_env_float('ICARL_HYBRID_FOCUS_WEIGHT', 0.0)
hybrid_class_bias_gamma = _get_env_float('ICARL_HYBRID_CLASS_BIAS_GAMMA', 0.0)
use_current_prototype_blend = _get_env_bool('ICARL_USE_CURRENT_PROTOTYPE_BLEND', False)
current_prototype_blend_alpha = _get_env_float('ICARL_CURRENT_PROTOTYPE_BLEND_ALPHA', 0.5)
current_prototype_blend_start_task = _get_env_int('ICARL_CURRENT_PROTOTYPE_BLEND_START_TASK', 1)
current_prototype_blend_scope = _get_env_str('ICARL_CURRENT_PROTOTYPE_BLEND_SCOPE', 'current')
current_prototype_blend_overlap_alpha = _get_env_float('ICARL_CURRENT_PROTOTYPE_BLEND_OVERLAP_ALPHA', -1.0)
current_prototype_blend_new_alpha = _get_env_float('ICARL_CURRENT_PROTOTYPE_BLEND_NEW_ALPHA', -1.0)
use_prototype_drift_comp = _get_env_bool('ICARL_USE_PROTOTYPE_DRIFT_COMP', False)
prototype_drift_beta = _get_env_float('ICARL_PROTOTYPE_DRIFT_BETA', 0.5)
prototype_drift_start_task = _get_env_int('ICARL_PROTOTYPE_DRIFT_START_TASK', 2)
overlap_transport_beta = _get_env_float('ICARL_OVERLAP_TRANSPORT_BETA', 0.0)
stage_overlap_transport_betas = _get_env_float_list('ICARL_STAGE_OVERLAP_TRANSPORT_BETAS', None)
exemplar_mode = _get_env_str('ICARL_EXEMPLAR_MODE', 'legacy_herding')
exemplar_mode_start_task = _get_env_int('ICARL_EXEMPLAR_MODE_START_TASK', 1)
use_task_adapter = _get_env_bool('ICARL_USE_TASK_ADAPTER', False)
task_adapter_dim = _get_env_int('ICARL_TASK_ADAPTER_DIM', 32)
task_adapter_dropout = _get_env_float('ICARL_TASK_ADAPTER_DROPOUT', 0.1)
task_adapter_start_task = _get_env_int('ICARL_TASK_ADAPTER_START_TASK', 0)
task_adapter_lr_mult = _get_env_float('ICARL_TASK_ADAPTER_LR_MULT', 1.0)
use_shared_adapter = _get_env_bool('ICARL_USE_SHARED_ADAPTER', False)
shared_adapter_dim = _get_env_int('ICARL_SHARED_ADAPTER_DIM', 16)
shared_adapter_dropout = _get_env_float('ICARL_SHARED_ADAPTER_DROPOUT', 0.1)
shared_adapter_start_task = _get_env_int('ICARL_SHARED_ADAPTER_START_TASK', 0)
use_task_affine = _get_env_bool('ICARL_USE_TASK_AFFINE', False)
task_affine_start_task = _get_env_int('ICARL_TASK_AFFINE_START_TASK', 0)
use_task_bn = _get_env_bool('ICARL_USE_TASK_BN', False)
task_bn_start_task = _get_env_int('ICARL_TASK_BN_START_TASK', 0)
use_stage2_lr_mirror_aug = _get_env_bool('ICARL_USE_STAGE2_LR_MIRROR_AUG', False)
stage2_lr_mirror_aug_ratio = _get_env_float('ICARL_STAGE2_LR_MIRROR_AUG_RATIO', 0.5)
use_subject_reweight = _get_env_bool('ICARL_USE_SUBJECT_REWEIGHT', False)
subject_reweight_power = _get_env_float('ICARL_SUBJECT_REWEIGHT_POWER', 1.0)
subject_reweight_start_task = _get_env_int('ICARL_SUBJECT_REWEIGHT_START_TASK', 2)
subject_reweight_end_task = _get_env_int('ICARL_SUBJECT_REWEIGHT_END_TASK', 99)
subject_reweight_ema = _get_env_float('ICARL_SUBJECT_REWEIGHT_EMA', 0.9)
stage_balanced_ft_epochs = _get_env_int_list('ICARL_STAGE_BALANCED_FT_EPOCHS', None)
balanced_ft_lr_scale = _get_env_float('ICARL_BALANCED_FT_LR_SCALE', 0.1)
balanced_ft_classifier_only = _get_env_bool('ICARL_BALANCED_FT_CLASSIFIER_ONLY', False)
current_class_ce_weight = _get_env_float('ICARL_CURRENT_CLASS_CE_WEIGHT', 0.0)
stage_current_class_ce_weights = _get_env_float_list('ICARL_STAGE_CURRENT_CLASS_CE_WEIGHTS', None)

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

    state_log = (
        f"mem={memory_size}, lr={learning_rate}, epochs={epochs}, stage_epochs={stage_epochs}, "
        f"balance={balance_sample}/{balance_power}, replay_batch={replay_batch_size}, stage_replay={stage_replay_batch_sizes}, "
        f"contrastive={is_contrastive_loss}, lwf={use_lwf}@{lwf_lambda}, lwf_T={lwf_T}, "
        f"feat_distill={use_feature_distill}@{feature_distill_lambda}, oldweight={old_class_weight_power}, exold={exclusive_old_boost}, exreplay={exclusive_old_replay_boost}, "
        f"norm_nme={use_normalized_nme}, hybrid={use_hybrid_nme_logits}, ptblend={use_current_prototype_blend}, "
        f"task_adapter={use_task_adapter}:{task_adapter_dim}, shared_adapter={use_shared_adapter}:{shared_adapter_dim}, "
        f"task_affine={use_task_affine}, task_bn={use_task_bn}, "
        f"s2_mirror_aug={use_stage2_lr_mirror_aug}:{stage2_lr_mirror_aug_ratio}, "
        f"balanced_ft={stage_balanced_ft_epochs}@{balanced_ft_lr_scale}, bft_head={balanced_ft_classifier_only}, "
        f"currce={current_class_ce_weight}/{stage_current_class_ce_weights}, exemplar_mode={exemplar_mode}"
    )
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
        use_task_affine=use_task_affine,
        affine_start_task=task_affine_start_task,
        use_task_bn=use_task_bn,
        bn_start_task=task_bn_start_task,
    )
    _configure_trainable_params(feature_extractor, trainable_part)

    model=CBiCaRL(seed,result_dir, data_path, is_cross_session, numclass,\
        feature_extractor,batch_size,\
        memory_size, balance_sample, balance_power, replay_batch_size, is_contrastive_loss, lambda_contrastive_loss, temperature, \
        use_normalized_nme, \
        use_hybrid_nme_logits, hybrid_start_task, hybrid_alpha_min, hybrid_alpha_max, hybrid_alpha_steps, hybrid_old_weight, hybrid_focus_classes, hybrid_focus_weight, hybrid_class_bias_gamma, \
        use_current_prototype_blend, current_prototype_blend_alpha, current_prototype_blend_start_task, current_prototype_blend_scope, \
        current_prototype_blend_overlap_alpha, current_prototype_blend_new_alpha, \
        use_prototype_drift_comp, prototype_drift_beta, prototype_drift_start_task, overlap_transport_beta, stage_overlap_transport_betas, \
        exemplar_mode, exemplar_mode_start_task, \
        task_adapter_lr_mult, \
        use_lwf, lwf_lambda, lwf_T, stage_lwf_lambdas, use_feature_distill, feature_distill_lambda, stage_feature_distill_lambdas, exclusive_old_feature_distill_boost, stage_exclusive_old_feature_distill_boosts, overlap_align_lambda, stage_overlap_align_lambdas, weighted_crossentropy, old_class_weight_power, stage_old_class_weight_powers, exclusive_old_boost, stage_exclusive_old_boosts, absent_old_current_weight, stage_absent_old_current_weights, exclusive_old_replay_boost, stage_exclusive_old_replay_boosts,\
        use_stage2_lr_mirror_aug, stage2_lr_mirror_aug_ratio, \
        use_subject_reweight, subject_reweight_power, subject_reweight_start_task, subject_reweight_end_task, subject_reweight_ema, \
        stage_balanced_ft_epochs, balanced_ft_lr_scale, balanced_ft_classifier_only, current_class_ce_weight, stage_current_class_ce_weights, \
        stage_replay_batch_sizes, epochs, stage_epochs, learning_rate,is_align,log,current_date)

    current_seed_stage_results = []

    for stage in range(1, num_stages+1):

        model.beforeTrain(stage)
        model.train()
        result_matrix = model.afterTrain()
        current_seed_stage_results.append(result_matrix)

    all_seeds_results.append(current_seed_stage_results)
    print(f"Seed {seed} finished.")

save_results(all_seeds_results,result_dir)
