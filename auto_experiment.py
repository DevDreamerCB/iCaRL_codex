import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

from export_metrics import append_experiment_row, parse_run_dir, write_latest_md, load_experiment_rows


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "memory_used": int(used),
                "memory_total": int(total),
                "util": int(util),
            }
        )
    return gpus


def choose_gpu(max_used_mb: int, max_util: int):
    gpus = query_gpus()
    candidates = [
        g for g in gpus if g["memory_used"] <= max_used_mb and g["util"] <= max_util
    ]
    if not candidates:
        candidates = sorted(gpus, key=lambda g: (g["memory_used"], g["util"]))
        return candidates[0]
    return sorted(candidates, key=lambda g: (g["memory_used"], g["util"]))[0]


def run_experiment(args):
    base_dir = Path(__file__).resolve().parent
    mode = "full" if args.full else "screen"
    if args.full:
        if args.seeds is None:
            args.seeds = 3
        if args.epochs is None:
            args.epochs = 30
    else:
        if args.seeds is None:
            args.seeds = 1
        if args.epochs is None:
            args.epochs = 10

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.note:
        run_tag = f"{run_tag}_{args.note}"

    if args.gpu is None:
        gpu_info = choose_gpu(args.max_used_mb, args.max_util)
        gpu_id = gpu_info["index"]
    else:
        gpu_info = None
        gpu_id = args.gpu

    env = os.environ.copy()
    env["ICARL_GPU_ID"] = str(gpu_id)
    env["ICARL_NUM_SEEDS"] = str(args.seeds)
    env["ICARL_EPOCHS"] = str(args.epochs)
    if args.stage_epochs:
        env["ICARL_STAGE_EPOCHS"] = args.stage_epochs
    env["ICARL_RUN_TAG"] = run_tag
    env["ICARL_USE_CONTRASTIVE"] = "true" if args.use_contrastive else "false"
    env["ICARL_BALANCE_SAMPLE"] = "true" if args.balance_sample else "false"
    env["ICARL_BALANCE_POWER"] = str(args.balance_power)
    env["ICARL_REPLAY_BATCH_SIZE"] = str(args.replay_batch_size)
    env["ICARL_USE_AGE_REPLAY"] = "true" if args.use_age_replay else "false"
    env["ICARL_AGE_REPLAY_POWER"] = str(args.age_replay_power)
    env["ICARL_USE_ALIGN"] = "true" if args.use_align else "false"
    env["ICARL_MEMORY_SIZE"] = str(args.memory_size)
    env["ICARL_USE_AGE_MEMORY"] = "true" if args.use_age_memory else "false"
    env["ICARL_AGE_MEMORY_POWER"] = str(args.age_memory_power)
    env["ICARL_TRAINABLE_PART"] = args.trainable_part
    env["ICARL_WEIGHTED_CE"] = "true" if args.weighted_ce else "false"
    env["ICARL_OLD_CLASS_WEIGHT_POWER"] = str(args.old_class_weight_power)
    env["ICARL_USE_LWF"] = "true" if args.use_lwf else "false"
    env["ICARL_LWF_LAMBDA"] = str(args.lwf_lambda)
    env["ICARL_LWF_T"] = str(args.lwf_t)
    env["ICARL_USE_PROTO_ALIGN"] = "true" if args.use_proto_align else "false"
    env["ICARL_PROTO_ALIGN_LAMBDA"] = str(args.proto_align_lambda)
    env["ICARL_USE_TASK_ADAPTER"] = "true" if args.use_task_adapter else "false"
    env["ICARL_TASK_ADAPTER_DIM"] = str(args.task_adapter_dim)
    env["ICARL_TASK_ADAPTER_DROPOUT"] = str(args.task_adapter_dropout)
    env["ICARL_TASK_ADAPTER_START_TASK"] = str(args.task_adapter_start_task)
    env["ICARL_TASK_ADAPTER_LR_MULT"] = str(args.task_adapter_lr_mult)
    env["ICARL_USE_SHARED_ADAPTER"] = "true" if args.use_shared_adapter else "false"
    env["ICARL_SHARED_ADAPTER_DIM"] = str(args.shared_adapter_dim)
    env["ICARL_SHARED_ADAPTER_DROPOUT"] = str(args.shared_adapter_dropout)
    env["ICARL_SHARED_ADAPTER_START_TASK"] = str(args.shared_adapter_start_task)
    env["ICARL_USE_TASK_PROMPT"] = "true" if args.use_task_prompt else "false"
    env["ICARL_TASK_PROMPT_LEN"] = str(args.task_prompt_len)
    env["ICARL_TASK_PROMPT_START_TASK"] = str(args.task_prompt_start_task)
    env["ICARL_USE_TASK_LORA"] = "true" if args.use_task_lora else "false"
    env["ICARL_TASK_LORA_RANK"] = str(args.task_lora_rank)
    env["ICARL_TASK_LORA_ALPHA"] = str(args.task_lora_alpha)
    env["ICARL_TASK_LORA_DROPOUT"] = str(args.task_lora_dropout)
    env["ICARL_TASK_LORA_START_TASK"] = str(args.task_lora_start_task)
    env["ICARL_USE_TASK_AFFINE"] = "true" if args.use_task_affine else "false"
    env["ICARL_TASK_AFFINE_START_TASK"] = str(args.task_affine_start_task)
    env["ICARL_USE_TASK_BN"] = "true" if args.use_task_bn else "false"
    env["ICARL_TASK_BN_START_TASK"] = str(args.task_bn_start_task)

    if args.lr is not None:
        env["ICARL_LR"] = str(args.lr)

    log_capture = base_dir / "metrics" / "last_run_stdout.log"
    with log_capture.open("w", encoding="utf-8") as f:
        subprocess.run(
            ["python", "main.py"],
            cwd=base_dir,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    candidate_dirs = sorted(
        [p for p in (base_dir / "logs").iterdir() if p.is_dir() and p.name.endswith(run_tag)]
    )
    if not candidate_dirs:
        raise RuntimeError(f"No log directory found for run tag {run_tag}")

    log_dir = candidate_dirs[-1]
    parsed = parse_run_dir(log_dir)
    csv_path = base_dir / "metrics" / "experiments.csv"
    append_experiment_row(
        csv_path,
        parsed,
        mode=mode,
        gpu=str(gpu_id),
        seeds=str(args.seeds),
        note=args.note,
        hypothesis=args.hypothesis,
        change_summary=args.change_summary,
        status="completed",
    )
    rows = load_experiment_rows(csv_path)
    write_latest_md(rows, base_dir / "metrics" / "latest.md")

    print(f"run_tag={run_tag}")
    print(f"gpu={gpu_id}")
    if gpu_info is not None:
        print(f"gpu_memory_used_mb={gpu_info['memory_used']}")
        print(f"gpu_util={gpu_info['util']}")
    print(f"stage1_total={parsed['stage1_total']}")
    print(f"stage2_total={parsed['stage2_total']}")
    print(f"stage3_total={parsed['stage3_total']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--change-summary", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--max-used-mb", type=int, default=3000)
    parser.add_argument("--max-util", type=int, default=20)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stage-epochs", default="")
    parser.add_argument("--memory-size", type=int, default=24)
    parser.add_argument("--use-age-memory", dest="use_age_memory", action="store_true")
    parser.add_argument("--no-use-age-memory", dest="use_age_memory", action="store_false")
    parser.set_defaults(use_age_memory=False)
    parser.add_argument("--age-memory-power", type=float, default=1.0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--trainable-part", default="all")
    parser.add_argument("--use-weighted-ce", dest="weighted_ce", action="store_true")
    parser.add_argument("--no-use-weighted-ce", dest="weighted_ce", action="store_false")
    parser.set_defaults(weighted_ce=False)
    parser.add_argument("--old-class-weight-power", type=float, default=0.0)
    parser.add_argument("--use-lwf", dest="use_lwf", action="store_true")
    parser.add_argument("--no-use-lwf", dest="use_lwf", action="store_false")
    parser.set_defaults(use_lwf=False)
    parser.add_argument("--lwf-lambda", type=float, default=0.1)
    parser.add_argument("--lwf-t", type=float, default=2.0)
    parser.add_argument("--use-proto-align", action="store_true")
    parser.add_argument("--no-use-proto-align", dest="use_proto_align", action="store_false")
    parser.set_defaults(use_proto_align=False)
    parser.add_argument("--proto-align-lambda", type=float, default=0.1)
    parser.add_argument("--use-task-adapter", action="store_true")
    parser.add_argument("--no-use-task-adapter", dest="use_task_adapter", action="store_false")
    parser.set_defaults(use_task_adapter=False)
    parser.add_argument("--task-adapter-dim", type=int, default=32)
    parser.add_argument("--task-adapter-dropout", type=float, default=0.1)
    parser.add_argument("--task-adapter-start-task", type=int, default=0)
    parser.add_argument("--task-adapter-lr-mult", type=float, default=1.0)
    parser.add_argument("--use-shared-adapter", action="store_true")
    parser.add_argument("--no-use-shared-adapter", dest="use_shared_adapter", action="store_false")
    parser.set_defaults(use_shared_adapter=False)
    parser.add_argument("--shared-adapter-dim", type=int, default=16)
    parser.add_argument("--shared-adapter-dropout", type=float, default=0.1)
    parser.add_argument("--shared-adapter-start-task", type=int, default=0)
    parser.add_argument("--use-task-prompt", action="store_true")
    parser.add_argument("--no-use-task-prompt", dest="use_task_prompt", action="store_false")
    parser.set_defaults(use_task_prompt=False)
    parser.add_argument("--task-prompt-len", type=int, default=4)
    parser.add_argument("--task-prompt-start-task", type=int, default=0)
    parser.add_argument("--use-task-lora", action="store_true")
    parser.add_argument("--no-use-task-lora", dest="use_task_lora", action="store_false")
    parser.set_defaults(use_task_lora=False)
    parser.add_argument("--task-lora-rank", type=int, default=4)
    parser.add_argument("--task-lora-alpha", type=float, default=1.0)
    parser.add_argument("--task-lora-dropout", type=float, default=0.0)
    parser.add_argument("--task-lora-start-task", type=int, default=0)
    parser.add_argument("--use-task-affine", action="store_true")
    parser.add_argument("--no-use-task-affine", dest="use_task_affine", action="store_false")
    parser.set_defaults(use_task_affine=False)
    parser.add_argument("--task-affine-start-task", type=int, default=0)
    parser.add_argument("--use-task-bn", action="store_true")
    parser.add_argument("--no-use-task-bn", dest="use_task_bn", action="store_false")
    parser.set_defaults(use_task_bn=False)
    parser.add_argument("--task-bn-start-task", type=int, default=0)
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--no-use-contrastive", dest="use_contrastive", action="store_false")
    parser.set_defaults(use_contrastive=True)
    parser.add_argument("--balance-sample", action="store_true")
    parser.add_argument("--no-balance-sample", dest="balance_sample", action="store_false")
    parser.set_defaults(balance_sample=True)
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--replay-batch-size", type=int, default=0)
    parser.add_argument("--use-age-replay", dest="use_age_replay", action="store_true")
    parser.add_argument("--no-use-age-replay", dest="use_age_replay", action="store_false")
    parser.set_defaults(use_age_replay=False)
    parser.add_argument("--age-replay-power", type=float, default=1.0)
    parser.add_argument("--use-align", action="store_true")
    parser.add_argument("--no-use-align", dest="use_align", action="store_false")
    parser.set_defaults(use_align=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
