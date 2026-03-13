import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
METRICS_CSV = BASE_DIR / "metrics" / "experiments.csv"
LATEST_MD = BASE_DIR / "metrics" / "latest.md"
RESEARCH_DIR = BASE_DIR / "research"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
NOTES_MD = RESEARCH_DIR / "notes.md"
LOG_PATH = BASE_DIR / "metrics" / "overnight_runner.log"


def load_rows():
    if not METRICS_CSV.exists():
        return []
    with METRICS_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_research_files():
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
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
            )
            + "\n",
            encoding="utf-8",
        )
    if not NOTES_MD.exists():
        NOTES_MD.write_text(
            "# Overnight Notes\n\n"
            "- Current baseline focus: `oldweight2` for task3 retention.\n"
            "- Promotion rule: screen candidates with strong `task3` or score are escalated to full confirm.\n",
            encoding="utf-8",
        )


def parse_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def best_full(rows):
    full_rows = [r for r in rows if r.get("mode") == "full" and r.get("stage3_total")]
    if not full_rows:
        return None
    return max(full_rows, key=lambda r: float(r.get("score") or -1))


def best_screen(rows):
    screen_rows = [r for r in rows if r.get("mode") == "screen" and r.get("stage3_total")]
    if not screen_rows:
        return None
    return max(screen_rows, key=lambda r: float(r.get("score") or -1))


def already_ran(rows, note):
    return any(r.get("note") == note for r in rows)


def append_research_result(row):
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    row.get("mode", ""),
                    row.get("note", ""),
                    row.get("stage1_total", ""),
                    row.get("stage2_total", ""),
                    row.get("stage3_total", ""),
                    row.get("score", ""),
                    row.get("status", ""),
                    row.get("hypothesis", ""),
                    row.get("change_summary", ""),
                ]
            )
            + "\n"
        )


def append_research_note(row, verdict):
    with NOTES_MD.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {datetime.now().isoformat(timespec='seconds')} `{row.get('note', '')}`\n"
            f"- verdict: {verdict}\n"
            f"- mode: {row.get('mode', '')}\n"
            f"- task1/task2/task3: {row.get('stage1_total', '')} / {row.get('stage2_total', '')} / {row.get('stage3_total', '')}\n"
            f"- score: {row.get('score', '')}\n"
            f"- why: {row.get('hypothesis', '')}\n"
            f"- change: {row.get('change_summary', '')}\n"
        )


def git_checkpoint(note):
    try:
        subprocess.run(["git", "add", "metrics/experiments.csv", "metrics/latest.md", "research/results.tsv", "research/notes.md"], cwd=BASE_DIR, check=True)
        status = subprocess.check_output(["git", "status", "--short"], cwd=BASE_DIR, text=True).strip()
        if not status:
            return
        subprocess.run(["git", "commit", "-m", f"Overnight research: {note}"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "push"], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[overnight] git checkpoint failed for {note}: {exc}", flush=True)


def run_auto_experiment(candidate, full=False):
    note = candidate["note_confirm"] if full else candidate["note_short"]
    cmd = ["python", "auto_experiment.py"]
    if full:
        cmd.append("--full")
    else:
        cmd.extend(["--epochs", "10"])
    cmd.extend(
        [
            "--note",
            note,
            "--hypothesis",
            candidate["hypothesis_confirm"] if full else candidate["hypothesis_short"],
            "--change-summary",
            candidate["change_summary_confirm"] if full else candidate["change_summary_short"],
            "--no-use-contrastive",
            "--use-lwf",
            "--lwf-lambda",
            str(candidate.get("lwf_lambda", 0.15)),
            "--lwf-t",
            str(candidate.get("lwf_t", 2.0)),
            "--use-task-adapter",
            "--task-adapter-dim",
            str(candidate.get("task_adapter_dim", 16)),
            "--task-adapter-start-task",
            str(candidate.get("task_adapter_start_task", 1)),
            "--memory-size",
            str(candidate.get("memory_size", 36)),
            "--replay-batch-size",
            str(candidate.get("replay_batch_size", 2)),
            "--old-class-weight-power",
            str(candidate.get("old_class_weight_power", 0.0)),
            "--max-used-mb",
            str(candidate.get("max_used_mb", 3000)),
            "--max-util",
            str(candidate.get("max_util", 20)),
        ]
    )
    if candidate.get("use_age_memory"):
        cmd.extend(["--use-age-memory", "--age-memory-power", str(candidate.get("age_memory_power", 0.5))])
    if candidate.get("use_task_bn"):
        cmd.extend(["--use-task-bn", "--task-bn-start-task", str(candidate.get("task_bn_start_task", 1))])
    if candidate.get("memory_size") is not None:
        cmd.extend(["--memory-size", str(candidate["memory_size"])])

    print(f"[overnight] running {'full' if full else 'screen'}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def candidate_plan():
    return [
        {
            "note_short": "lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_short",
            "note_confirm": "lwf015_replaymix2_adapter16_mem36_oldweight2_agemem025_confirm",
            "hypothesis_short": "a milder age-aware exemplar budget may keep most of the task3 gain from the combined method while recovering more task2 than power 0.5",
            "change_summary_short": "true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.0 plus age-aware exemplar memory budgets power 0.25",
            "hypothesis_confirm": "the mild age-memory combination may be the most stable way to add exemplar bias on top of oldweight2 without over-hurting task2",
            "change_summary_confirm": "promote combined oldweight2+age-memory(0.25) candidate to 3-seed confirm",
            "use_age_memory": True,
            "age_memory_power": 0.25,
            "old_class_weight_power": 2.0,
        },
        {
            "note_short": "lwf015_replaymix2_adapter16_mem36_oldweight25_short",
            "note_confirm": "lwf015_replaymix2_adapter16_mem36_oldweight25_confirm",
            "hypothesis_short": "if power 2.0 still under-regularizes the oldest class, a slightly stronger old-class weight of 2.5 may push task3 further without the instability of changing replay or memory budgets",
            "change_summary_short": "true LwF on replaymix2+adapter16+mem36 with old-class BCE power 2.5",
            "hypothesis_confirm": "oldweight 2.5 may improve task3 retention beyond oldweight2 if the gain is coming from stronger oldest-class protection rather than overfitting",
            "change_summary_confirm": "promote old-class BCE power 2.5 candidate to 3-seed confirm",
            "old_class_weight_power": 2.5,
        },
        {
            "note_short": "lwf015_replaymix2_adapter16_mem42_oldweight2_short",
            "note_confirm": "lwf015_replaymix2_adapter16_mem42_oldweight2_confirm",
            "hypothesis_short": "once oldweight2 stabilizes forgetting, a modestly larger memory budget of 42 may improve task2/task3 without the instability seen in earlier sampler variants",
            "change_summary_short": "true LwF on replaymix2+adapter16 with memory size 42 and old-class BCE power 2.0",
            "hypothesis_confirm": "the corrected pipeline may finally benefit from a slightly larger memory when paired with oldweight2 rather than sampler changes",
            "change_summary_confirm": "promote memory42+oldweight2 candidate to 3-seed confirm",
            "memory_size": 42,
            "old_class_weight_power": 2.0,
        },
        {
            "note_short": "lwf015_replaymix2_adapter16_mem42_oldweight2_agemem025_short",
            "note_confirm": "lwf015_replaymix2_adapter16_mem42_oldweight2_agemem025_confirm",
            "hypothesis_short": "memory42 plus a mild age-aware memory bias may create a better task2/task3 tradeoff than either one alone under the oldweight2 training regime",
            "change_summary_short": "true LwF on replaymix2+adapter16 with memory size 42, old-class BCE power 2.0, and age-aware exemplar memory power 0.25",
            "hypothesis_confirm": "a slightly larger memory with mild age bias may be the most thesis-friendly extension of the current oldweight2 line if the short-run gain holds",
            "change_summary_confirm": "promote memory42+oldweight2+age-memory(0.25) candidate to 3-seed confirm",
            "memory_size": 42,
            "old_class_weight_power": 2.0,
            "use_age_memory": True,
            "age_memory_power": 0.25,
        },
    ]


def should_promote(row, rows):
    best_full_row = best_full(rows)
    best_screen_row = best_screen(rows)
    stage2 = parse_float(row, "stage2_total") or 0.0
    stage3 = parse_float(row, "stage3_total") or 0.0
    score = parse_float(row, "score") or 0.0
    best_full_stage3 = parse_float(best_full_row, "stage3_total") if best_full_row else 0.0
    best_full_score = parse_float(best_full_row, "score") if best_full_row else 0.0
    best_screen_stage3 = parse_float(best_screen_row, "stage3_total") if best_screen_row else 0.0
    best_screen_stage2 = parse_float(best_screen_row, "stage2_total") if best_screen_row else 0.0

    if stage3 >= best_full_stage3 + 0.25:
        return True
    if score >= best_full_score + 0.15:
        return True
    if stage3 >= best_screen_stage3 - 0.15 and stage2 >= best_screen_stage2 - 0.75:
        return True
    return False


def latest_row_by_note(rows, note):
    matches = [r for r in rows if r.get("note") == note]
    return matches[-1] if matches else None


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    deadline = datetime.now() + timedelta(hours=hours)
    ensure_research_files()

    stale_rounds = 0
    baseline_best_full = best_full(load_rows())
    baseline_score = parse_float(baseline_best_full, "score") if baseline_best_full else 0.0
    baseline_task3 = parse_float(baseline_best_full, "stage3_total") if baseline_best_full else 0.0

    for candidate in candidate_plan():
        if datetime.now() >= deadline:
            break

        rows = load_rows()
        if already_ran(rows, candidate["note_short"]):
            print(f"[overnight] skip existing screen: {candidate['note_short']}", flush=True)
        else:
            run_auto_experiment(candidate, full=False)

        rows = load_rows()
        screen_row = latest_row_by_note(rows, candidate["note_short"])
        if screen_row is None:
            continue
        append_research_result(screen_row)

        if should_promote(screen_row, rows):
            append_research_note(screen_row, "promoted_to_confirm")
            if not already_ran(rows, candidate["note_confirm"]) and datetime.now() < deadline:
                run_auto_experiment(candidate, full=True)
                rows = load_rows()
                full_row = latest_row_by_note(rows, candidate["note_confirm"])
                if full_row is not None:
                    append_research_result(full_row)
                    append_research_note(full_row, "confirm_completed")
                    git_checkpoint(candidate["note_confirm"])
                    latest_best = best_full(rows)
                    latest_score = parse_float(latest_best, "score") if latest_best else 0.0
                    latest_task3 = parse_float(latest_best, "stage3_total") if latest_best else 0.0
                    if latest_score > baseline_score + 1e-6 or latest_task3 > baseline_task3 + 1e-6:
                        baseline_score = max(baseline_score, latest_score)
                        baseline_task3 = max(baseline_task3, latest_task3)
                        stale_rounds = 0
                    else:
                        stale_rounds += 1
        else:
            append_research_note(screen_row, "rejected_after_screen")
            git_checkpoint(candidate["note_short"])
            stale_rounds += 1

        if stale_rounds >= 6:
            print("[overnight] stale threshold reached, stopping nightly search.", flush=True)
            break

        time.sleep(5)

    print(f"[overnight] finished at {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
