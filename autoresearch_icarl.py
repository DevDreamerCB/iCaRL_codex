import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from export_metrics import load_experiment_rows, write_latest_md


BASE_DIR = Path(__file__).resolve().parent
METRICS_CSV = BASE_DIR / "metrics" / "experiments.csv"
LATEST_MD = BASE_DIR / "metrics" / "latest.md"
RESEARCH_DIR = BASE_DIR / "research"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
NOTES_MD = RESEARCH_DIR / "notes.md"
PROGRAM_MD = RESEARCH_DIR / "program.md"
STATE_JSON = RESEARCH_DIR / "state.json"
LOG_PATH = BASE_DIR / "metrics" / "autoresearch.log"
EVENTS_JSONL = BASE_DIR / "metrics" / "events.jsonl"

DEFAULT_DIRECTION_FAMILIES = {
    "normalized_nme_schedule_line": ["oldweight_tune", "balance_tune", "schedule_tune", "lwf_tune"],
    "prototype_calibration_line": ["age_nme", "radius_nme"],
    "replay_loss_line": ["replay_tune", "lwf_tune", "balance_tune"],
}


def log(msg: str):
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_float(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def load_state():
    return json.loads(STATE_JSON.read_text(encoding="utf-8"))


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def emit_event(event_type, payload):
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": event_type,
        "payload": payload,
    }
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


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


def make_signature(candidate):
    fields = [
        candidate["family"],
        f"lwf{candidate['lwf_lambda']}",
        f"old{candidate['old_class_weight_power']}",
        f"stage{'-'.join(str(x) for x in candidate['stage_epochs'])}",
        f"replay{candidate['replay_batch_size']}",
        f"balance{candidate['balance_power']}",
        f"norm{int(candidate['use_normalized_nme'])}",
        f"radius{int(candidate['use_radius_nme'])}_{candidate['radius_nme_power']}",
        f"agenme{int(candidate['use_age_nme'])}_{candidate['age_nme_power']}",
    ]
    return "|".join(fields)


def format_note(candidate, suffix):
    parts = [f"lwf{str(candidate['lwf_lambda']).replace('.', '')}"]
    if candidate["use_normalized_nme"]:
        parts.append("normnme")
    if candidate["use_radius_nme"]:
        parts.append(f"radius{str(candidate['radius_nme_power']).replace('.', '')}")
    if candidate["use_age_nme"]:
        parts.append(f"agenme{str(candidate['age_nme_power']).replace('.', '')}")
    parts.extend(
        [
            "adapter16",
            f"mem{candidate['memory_size']}",
            f"oldweight{str(candidate['old_class_weight_power']).replace('.', '')}",
            f"stage{''.join(str(x) for x in candidate['stage_epochs'])}",
            suffix,
        ]
    )
    return "_".join(parts)


def build_candidates(state):
    base = deepcopy(state["active_line"])
    candidates = []
    allowed_families = state.get("candidate_families") or DEFAULT_DIRECTION_FAMILIES.get(
        state.get("active_direction", "normalized_nme_schedule_line"),
        ["oldweight_tune", "age_nme", "radius_nme", "balance_tune", "schedule_tune", "lwf_tune", "replay_tune"],
    )

    def add_candidate(*, family, note_hint, updates, hypothesis, change_summary):
        if family not in allowed_families:
            return
        cand = deepcopy(base)
        cand.update(updates)
        cand["family"] = family
        cand["note_hint"] = note_hint
        cand["hypothesis"] = hypothesis
        cand["change_summary"] = change_summary
        candidates.append(cand)

    add_candidate(
        family="oldweight_tune",
        note_hint="oldweight225",
        updates={"old_class_weight_power": 2.25},
        hypothesis="slightly relaxing old-class weighting may improve task2 while keeping most of the task3 gain on the current best line",
        change_summary="tune old-class BCE power from 2.5 to 2.25 on the current normalized-NME schedule line",
    )
    add_candidate(
        family="oldweight_tune",
        note_hint="oldweight275",
        updates={"old_class_weight_power": 2.75},
        hypothesis="slightly stronger old-class weighting may improve task3 if the current best line is still under-protecting the oldest classes",
        change_summary="tune old-class BCE power from 2.5 to 2.75 on the current normalized-NME schedule line",
    )
    add_candidate(
        family="age_nme",
        note_hint="agenme01",
        updates={"use_age_nme": True, "age_nme_power": 0.1},
        hypothesis="a very small age-aware prototype distance bias may recover old-class retention in task3 without distorting task2 too much",
        change_summary="add mild age-aware NME scaling with power 0.1 on the current best line",
    )
    add_candidate(
        family="age_nme",
        note_hint="agenme02",
        updates={"use_age_nme": True, "age_nme_power": 0.2},
        hypothesis="a slightly stronger age-aware prototype distance bias may further improve task3 if 0.1 is too weak",
        change_summary="add age-aware NME scaling with power 0.2 on the current best line",
    )
    add_candidate(
        family="age_nme",
        note_hint="agenme005",
        updates={"use_age_nme": True, "age_nme_power": 0.05},
        hypothesis="an even milder age-aware prototype bias may preserve task2 better while still improving task3 retention",
        change_summary="add age-aware NME scaling with power 0.05 on the current best line",
    )
    add_candidate(
        family="radius_nme",
        note_hint="radius05",
        updates={"use_radius_nme": True, "radius_nme_power": 0.5},
        hypothesis="radius-normalized NME may correct class-spread mismatch at inference time while preserving fixed-memory comparability",
        change_summary="add radius-normalized NME with power 0.5 on the current best line",
    )
    add_candidate(
        family="radius_nme",
        note_hint="radius025",
        updates={"use_radius_nme": True, "radius_nme_power": 0.25},
        hypothesis="a weaker radius-normalized NME may avoid over-correcting class distances while still reducing prototype spread bias",
        change_summary="add radius-normalized NME with power 0.25 on the current best line",
    )
    add_candidate(
        family="schedule_tune",
        note_hint="bal025_stage",
        updates={"balance_power": 0.25},
        hypothesis="a milder class-balance exponent may recover task2 while keeping the stage-scheduled task3 gain",
        change_summary="tune balance power to 0.25 on the current best line",
    )
    add_candidate(
        family="schedule_tune",
        note_hint="stage101213",
        updates={"stage_epochs": [10, 12, 13]},
        hypothesis="slightly reducing the last-stage training budget may preserve most of the task3 gain while improving generalization stability",
        change_summary="tune stage schedule from [10,12,14] to [10,12,13] on the current best line",
    )
    add_candidate(
        family="lwf_tune",
        note_hint="lwf01_stage",
        updates={"lwf_lambda": 0.1},
        hypothesis="a milder LwF term on the stage-scheduled line may recover task2 while preserving the new task3 geometry",
        change_summary="tune LwF from 0.15 to 0.1 on the current normalized-NME schedule line",
    )
    add_candidate(
        family="lwf_tune",
        note_hint="lwf02_stage",
        updates={"lwf_lambda": 0.2},
        hypothesis="a slightly stronger LwF term on the stage-scheduled line may improve old-class retention if the current line still under-regularizes stage3",
        change_summary="tune LwF from 0.15 to 0.2 on the current normalized-NME schedule line",
    )
    add_candidate(
        family="replay_tune",
        note_hint="replay1_stage",
        updates={"replay_batch_size": 1},
        hypothesis="a lighter replay mixture may improve task2 while keeping most of the current task3 gain",
        change_summary="tune replay mix from 2 to 1 on the current normalized-NME schedule line",
    )
    add_candidate(
        family="replay_tune",
        note_hint="replay3_stage",
        updates={"replay_batch_size": 3},
        hypothesis="a slightly stronger replay mixture may improve task3 if the current line still underuses replay in later stages",
        change_summary="tune replay mix from 2 to 3 on the current normalized-NME schedule line",
    )

    seen = set()
    filtered = []
    rejected = set(state.get("rejected_signatures", []))
    accepted = set(state.get("accepted_signatures", []))
    for cand in candidates:
        sig = make_signature(cand)
        if sig in seen or sig in rejected or sig in accepted:
            continue
        seen.add(sig)
        filtered.append(cand)
    return filtered


def has_note(rows, note):
    return any(r.get("note") == note for r in rows)


def latest_by_note(rows, note):
    matches = [r for r in rows if r.get("note") == note]
    return matches[-1] if matches else None


def build_command(candidate, mode):
    note = format_note(candidate, "confirm" if mode == "full" else "short")
    cmd = [
        "python",
        "auto_experiment.py",
        "--note",
        note,
        "--hypothesis",
        candidate["hypothesis"],
        "--change-summary",
        candidate["change_summary"],
        "--memory-size",
        str(candidate["memory_size"]),
        "--use-lwf",
        "--lwf-lambda",
        str(candidate["lwf_lambda"]),
        "--lwf-t",
        "2.0",
        "--use-normalized-nme",
        "--use-task-adapter",
        "--task-adapter-dim",
        str(candidate["task_adapter_dim"]),
        "--task-adapter-start-task",
        str(candidate["task_adapter_start_task"]),
        "--replay-batch-size",
        str(candidate["replay_batch_size"]),
        "--old-class-weight-power",
        str(candidate["old_class_weight_power"]),
        "--balance-power",
        str(candidate["balance_power"]),
        "--stage-epochs",
        ",".join(str(x) for x in candidate["stage_epochs"]),
        "--no-use-contrastive",
        "--max-used-mb",
        "3000",
        "--max-util",
        "20",
    ]
    if mode == "full":
        cmd.append("--full")
    else:
        cmd.extend(["--epochs", "10", "--seeds", "1"])
    if candidate["use_radius_nme"]:
        cmd.extend(["--use-radius-nme", "--radius-nme-power", str(candidate["radius_nme_power"])])
    if candidate["use_age_nme"]:
        cmd.extend(["--use-age-nme", "--age-nme-power", str(candidate["age_nme_power"])])
    return note, cmd


def run_candidate(candidate, mode):
    note, cmd = build_command(candidate, mode)
    log(f"run {mode}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=BASE_DIR, check=True)
    rows = load_experiment_rows(METRICS_CSV)
    write_latest_md(rows, LATEST_MD)
    row = latest_by_note(rows, note)
    if row is None:
        raise RuntimeError(f"missing row for note={note}")
    emit_event(
        "run_completed",
        {
            "mode": mode,
            "note": note,
            "family": candidate["family"],
            "signature": make_signature(candidate),
            "task1": row.get("stage1_total", ""),
            "task2": row.get("stage2_total", ""),
            "task3": row.get("stage3_total", ""),
            "score": row.get("score", ""),
        },
    )
    return row


def should_promote(screen_row, rows):
    best_full_row = best_full(rows)
    best_screen_row = best_screen(rows)
    stage2 = parse_float(screen_row, "stage2_total") or 0.0
    stage3 = parse_float(screen_row, "stage3_total") or 0.0
    score = parse_float(screen_row, "score") or 0.0
    best_full_stage3 = parse_float(best_full_row, "stage3_total") if best_full_row else 0.0
    best_full_score = parse_float(best_full_row, "score") if best_full_row else 0.0
    best_screen_stage3 = parse_float(best_screen_row, "stage3_total") if best_screen_row else 0.0
    best_screen_stage2 = parse_float(best_screen_row, "stage2_total") if best_screen_row else 0.0
    return (
        stage3 >= best_full_stage3 + 0.5
        or score >= best_full_score + 0.25
        or (stage3 >= best_screen_stage3 - 0.1 and stage2 >= best_screen_stage2 - 0.5)
    )


def is_improved(confirm_row, best_full_row):
    if best_full_row is None:
        return True
    c2 = parse_float(confirm_row, "stage2_total") or 0.0
    c3 = parse_float(confirm_row, "stage3_total") or 0.0
    cs = parse_float(confirm_row, "score") or 0.0
    b2 = parse_float(best_full_row, "stage2_total") or 0.0
    b3 = parse_float(best_full_row, "stage3_total") or 0.0
    bs = parse_float(best_full_row, "score") or 0.0
    if c3 >= b3 + 0.25 and c2 >= b2 - 1.5:
        return True
    if cs >= bs + 0.15:
        return True
    return False


def append_note(title, verdict, row):
    with NOTES_MD.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {datetime.now().isoformat(timespec='seconds')} `{title}`\n"
            f"- verdict: {verdict}\n"
            f"- task1/task2/task3: {row.get('stage1_total', '')} / {row.get('stage2_total', '')} / {row.get('stage3_total', '')}\n"
            f"- score: {row.get('score', '')}\n"
            f"- note: {row.get('note', '')}\n"
        )


def update_active_line_from_candidate(state, candidate, confirm_row):
    state["active_line"].update(
        {
            "memory_size": candidate["memory_size"],
            "lwf_lambda": candidate["lwf_lambda"],
            "old_class_weight_power": candidate["old_class_weight_power"],
            "stage_epochs": candidate["stage_epochs"],
            "replay_batch_size": candidate["replay_batch_size"],
            "balance_power": candidate["balance_power"],
            "use_normalized_nme": True,
            "use_radius_nme": candidate["use_radius_nme"],
            "radius_nme_power": candidate["radius_nme_power"],
            "use_age_nme": candidate["use_age_nme"],
            "age_nme_power": candidate["age_nme_power"],
            "use_task_adapter": True,
            "task_adapter_dim": candidate["task_adapter_dim"],
            "task_adapter_start_task": candidate["task_adapter_start_task"],
        }
    )
    state["best_full_note"] = confirm_row["note"]
    state["last_direction"] = candidate["family"]


def git_checkpoint(note):
    try:
        subprocess.run(
            [
                "git",
                "add",
                "metrics/experiments.csv",
                "metrics/latest.md",
                "research/results.tsv",
                "research/notes.md",
                "research/program.md",
                "research/state.json",
                "research/summary_20260314.md",
                "main.py",
                "iCaRL.py",
                "auto_experiment.py",
                "autoresearch_icarl.py",
                "run_autoresearch.sh",
            ],
            cwd=BASE_DIR,
            check=True,
        )
        status = subprocess.check_output(["git", "status", "--short"], cwd=BASE_DIR, text=True).strip()
        if not status:
            return
        subprocess.run(["git", "commit", "-m", f"AutoResearch: {note}"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "push"], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        log(f"git checkpoint failed: {exc}")


def main():
    hours = 10.0
    max_cycles = None
    sleep_on_idle = True
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        hours = float(args.pop(0))
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--hours":
            hours = float(args[idx + 1])
            idx += 2
            continue
        if arg == "--max-cycles":
            max_cycles = int(args[idx + 1])
            idx += 2
            continue
        if arg == "--no-sleep-on-idle":
            sleep_on_idle = False
            idx += 1
            continue
        raise ValueError(f"Unknown arg: {arg}")

    deadline = datetime.now() + timedelta(hours=hours)
    state = load_state()
    rows = load_experiment_rows(METRICS_CSV)
    write_latest_md(rows, LATEST_MD)
    cycles = 0

    while datetime.now() < deadline:
        if max_cycles is not None and cycles >= max_cycles:
            break
        rows = load_experiment_rows(METRICS_CSV)
        best_full_row = best_full(rows)
        candidates = build_candidates(state)
        if not candidates:
            log("no new candidates available; stopping")
            break

        progressed = False
        for candidate in candidates:
            if datetime.now() >= deadline:
                break
            signature = make_signature(candidate)

            screen_note, _ = build_command(candidate, "screen")
            if has_note(rows, screen_note):
                screen_row = latest_by_note(rows, screen_note)
            else:
                screen_row = run_candidate(candidate, "screen")
                rows = load_experiment_rows(METRICS_CSV)
                progressed = True

            if not should_promote(screen_row, rows):
                state["rejected_signatures"].append(signature)
                state["stale_rounds"] += 1
                append_note(screen_note, "screen_rejected", screen_row)
                emit_event(
                    "screen_rejected",
                    {
                        "note": screen_note,
                        "family": candidate["family"],
                        "signature": signature,
                        "task1": screen_row.get("stage1_total", ""),
                        "task2": screen_row.get("stage2_total", ""),
                        "task3": screen_row.get("stage3_total", ""),
                        "score": screen_row.get("score", ""),
                    },
                )
                save_state(state)
                continue

            confirm_note, _ = build_command(candidate, "full")
            if has_note(rows, confirm_note):
                confirm_row = latest_by_note(rows, confirm_note)
            else:
                confirm_row = run_candidate(candidate, "full")
                rows = load_experiment_rows(METRICS_CSV)
                progressed = True

            if is_improved(confirm_row, best_full_row):
                state["accepted_signatures"].append(signature)
                state["stale_rounds"] = 0
                update_active_line_from_candidate(state, candidate, confirm_row)
                append_note(confirm_note, "confirm_accepted", confirm_row)
                emit_event(
                    "confirm_accepted",
                    {
                        "note": confirm_note,
                        "family": candidate["family"],
                        "signature": signature,
                        "task1": confirm_row.get("stage1_total", ""),
                        "task2": confirm_row.get("stage2_total", ""),
                        "task3": confirm_row.get("stage3_total", ""),
                        "score": confirm_row.get("score", ""),
                    },
                )
                git_checkpoint(confirm_note)
                best_full_row = confirm_row
            else:
                state["rejected_signatures"].append(signature)
                state["stale_rounds"] += 1
                append_note(confirm_note, "confirm_rejected", confirm_row)
                emit_event(
                    "confirm_rejected",
                    {
                        "note": confirm_note,
                        "family": candidate["family"],
                        "signature": signature,
                        "task1": confirm_row.get("stage1_total", ""),
                        "task2": confirm_row.get("stage2_total", ""),
                        "task3": confirm_row.get("stage3_total", ""),
                        "score": confirm_row.get("score", ""),
                    },
                )

            save_state(state)
            break

        if not progressed:
            log("no new work performed in this cycle; sleeping briefly")
            if sleep_on_idle:
                time.sleep(10)
            state["stale_rounds"] += 1
            save_state(state)

        if state["stale_rounds"] >= 5:
            log("stale_rounds reached threshold; stopping for manual direction refresh")
            break
        cycles += 1


if __name__ == "__main__":
    main()
