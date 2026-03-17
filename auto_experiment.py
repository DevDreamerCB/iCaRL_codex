import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

from export_metrics import append_experiment_row, parse_run_dir, write_latest_md, load_experiment_rows


def append_research_result(tsv_path: Path, parsed: dict, *, mode: str, note: str, hypothesis: str, change_summary: str, status: str):
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not tsv_path.exists():
        tsv_path.write_text(
            "\t".join(
                [
                    "timestamp",
                    "mode",
                    "note",
                    "task1",
                    "task2",
                    "task3",
                    "score",
                    "status",
                    "hypothesis",
                    "change_summary",
                ]
            ) + "\n",
            encoding="utf-8",
        )
    score = ""
    if parsed.get("stage1_total") != "" or parsed.get("stage2_total") != "" or parsed.get("stage3_total") != "":
        s1 = float(parsed.get("stage1_total") or 0.0)
        s2 = float(parsed.get("stage2_total") or 0.0)
        s3 = float(parsed.get("stage3_total") or 0.0)
        score = str(round(0.2 * s1 + 0.3 * s2 + 0.5 * s3, 2))
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    mode,
                    note,
                    str(parsed.get("stage1_total", "")),
                    str(parsed.get("stage2_total", "")),
                    str(parsed.get("stage3_total", "")),
                    score,
                    status,
                    hypothesis,
                    change_summary,
                ]
            ) + "\n"
        )


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
    env["ICARL_USE_ALIGN"] = "true" if args.use_align else "false"
    env["ICARL_MEMORY_SIZE"] = str(args.memory_size)
    env["ICARL_TRAINABLE_PART"] = args.trainable_part
    env["ICARL_WEIGHTED_CE"] = "true" if args.weighted_ce else "false"
    env["ICARL_OLD_CLASS_WEIGHT_POWER"] = str(args.old_class_weight_power)
    if args.stage_old_class_weight_powers:
        env["ICARL_STAGE_OLD_CLASS_WEIGHT_POWERS"] = args.stage_old_class_weight_powers
    env["ICARL_USE_LWF"] = "true" if args.use_lwf else "false"
    env["ICARL_LWF_LAMBDA"] = str(args.lwf_lambda)
    env["ICARL_LWF_T"] = str(args.lwf_t)
    if args.stage_lwf_lambdas:
        env["ICARL_STAGE_LWF_LAMBDAS"] = args.stage_lwf_lambdas
    env["ICARL_USE_FEATURE_DISTILL"] = "true" if args.use_feature_distill else "false"
    env["ICARL_FEATURE_DISTILL_LAMBDA"] = str(args.feature_distill_lambda)
    if args.stage_feature_distill_lambdas:
        env["ICARL_STAGE_FEATURE_DISTILL_LAMBDAS"] = args.stage_feature_distill_lambdas
    env["ICARL_USE_NORMALIZED_NME"] = "true" if args.use_normalized_nme else "false"
    env["ICARL_USE_HYBRID_NME_LOGITS"] = "true" if args.use_hybrid_nme_logits else "false"
    env["ICARL_HYBRID_START_TASK"] = str(args.hybrid_start_task)
    env["ICARL_HYBRID_ALPHA_MIN"] = str(args.hybrid_alpha_min)
    env["ICARL_HYBRID_ALPHA_MAX"] = str(args.hybrid_alpha_max)
    env["ICARL_HYBRID_ALPHA_STEPS"] = str(args.hybrid_alpha_steps)
    env["ICARL_HYBRID_OLD_WEIGHT"] = str(args.hybrid_old_weight)
    env["ICARL_USE_CURRENT_PROTOTYPE_BLEND"] = "true" if args.use_current_prototype_blend else "false"
    env["ICARL_CURRENT_PROTOTYPE_BLEND_ALPHA"] = str(args.current_prototype_blend_alpha)
    env["ICARL_CURRENT_PROTOTYPE_BLEND_START_TASK"] = str(args.current_prototype_blend_start_task)
    env["ICARL_CURRENT_PROTOTYPE_BLEND_SCOPE"] = args.current_prototype_blend_scope
    env["ICARL_EXEMPLAR_MODE"] = args.exemplar_mode
    env["ICARL_EXEMPLAR_MODE_START_TASK"] = str(args.exemplar_mode_start_task)
    env["ICARL_USE_TASK_ADAPTER"] = "true" if args.use_task_adapter else "false"
    env["ICARL_TASK_ADAPTER_DIM"] = str(args.task_adapter_dim)
    env["ICARL_TASK_ADAPTER_DROPOUT"] = str(args.task_adapter_dropout)
    env["ICARL_TASK_ADAPTER_START_TASK"] = str(args.task_adapter_start_task)
    env["ICARL_TASK_ADAPTER_LR_MULT"] = str(args.task_adapter_lr_mult)
    env["ICARL_USE_SHARED_ADAPTER"] = "true" if args.use_shared_adapter else "false"
    env["ICARL_SHARED_ADAPTER_DIM"] = str(args.shared_adapter_dim)
    env["ICARL_SHARED_ADAPTER_DROPOUT"] = str(args.shared_adapter_dropout)
    env["ICARL_SHARED_ADAPTER_START_TASK"] = str(args.shared_adapter_start_task)
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
    append_research_result(
        base_dir / "research" / "results.tsv",
        parsed,
        mode=mode,
        note=args.note,
        hypothesis=args.hypothesis,
        change_summary=args.change_summary,
        status="completed",
    )

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
    parser.add_argument("--lr", type=float)
    parser.add_argument("--trainable-part", default="all")
    parser.add_argument("--use-weighted-ce", dest="weighted_ce", action="store_true")
    parser.add_argument("--no-use-weighted-ce", dest="weighted_ce", action="store_false")
    parser.set_defaults(weighted_ce=False)
    parser.add_argument("--old-class-weight-power", type=float, default=0.0)
    parser.add_argument("--stage-old-class-weight-powers", default="")
    parser.add_argument("--use-lwf", dest="use_lwf", action="store_true")
    parser.add_argument("--no-use-lwf", dest="use_lwf", action="store_false")
    parser.set_defaults(use_lwf=False)
    parser.add_argument("--lwf-lambda", type=float, default=0.1)
    parser.add_argument("--lwf-t", type=float, default=2.0)
    parser.add_argument("--stage-lwf-lambdas", default="")
    parser.add_argument("--use-feature-distill", dest="use_feature_distill", action="store_true")
    parser.add_argument("--no-use-feature-distill", dest="use_feature_distill", action="store_false")
    parser.set_defaults(use_feature_distill=False)
    parser.add_argument("--feature-distill-lambda", type=float, default=0.1)
    parser.add_argument("--stage-feature-distill-lambdas", default="")
    parser.add_argument("--use-normalized-nme", dest="use_normalized_nme", action="store_true")
    parser.add_argument("--no-use-normalized-nme", dest="use_normalized_nme", action="store_false")
    parser.set_defaults(use_normalized_nme=False)
    parser.add_argument("--use-hybrid-nme-logits", dest="use_hybrid_nme_logits", action="store_true")
    parser.add_argument("--no-use-hybrid-nme-logits", dest="use_hybrid_nme_logits", action="store_false")
    parser.set_defaults(use_hybrid_nme_logits=False)
    parser.add_argument("--hybrid-start-task", type=int, default=2)
    parser.add_argument("--hybrid-alpha-min", type=float, default=0.0)
    parser.add_argument("--hybrid-alpha-max", type=float, default=1.0)
    parser.add_argument("--hybrid-alpha-steps", type=int, default=11)
    parser.add_argument("--hybrid-old-weight", type=float, default=0.6)
    parser.add_argument("--use-current-prototype-blend", dest="use_current_prototype_blend", action="store_true")
    parser.add_argument("--no-use-current-prototype-blend", dest="use_current_prototype_blend", action="store_false")
    parser.set_defaults(use_current_prototype_blend=False)
    parser.add_argument("--current-prototype-blend-alpha", type=float, default=0.5)
    parser.add_argument("--current-prototype-blend-start-task", type=int, default=1)
    parser.add_argument("--current-prototype-blend-scope", default="current")
    parser.add_argument("--exemplar-mode", default="legacy_herding")
    parser.add_argument("--exemplar-mode-start-task", type=int, default=1)
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
    parser.add_argument("--use-align", action="store_true")
    parser.add_argument("--no-use-align", dest="use_align", action="store_false")
    parser.set_defaults(use_align=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
