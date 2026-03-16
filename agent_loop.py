import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from export_metrics import load_experiment_rows, write_latest_md


BASE_DIR = Path(__file__).resolve().parent
STATE_JSON = BASE_DIR / "research" / "state.json"
NOTES_MD = BASE_DIR / "research" / "notes.md"
MEMORY_MD = BASE_DIR / "research" / "MEMORY.md"
DAILY_MD = BASE_DIR / "research" / f"daily_{datetime.now().strftime('%Y%m%d')}.md"
LOG_PATH = BASE_DIR / "metrics" / "agent_loop.log"
METRICS_CSV = BASE_DIR / "metrics" / "experiments.csv"
LATEST_MD = BASE_DIR / "metrics" / "latest.md"
EVENTS_JSONL = BASE_DIR / "metrics" / "events.jsonl"
STATUS_MD = BASE_DIR / "metrics" / "agent_status.md"

DEFAULT_DIRECTION_FAMILIES = {
    "normalized_nme_schedule_line": ["oldweight_tune", "balance_tune", "schedule_tune", "lwf_tune"],
    "prototype_calibration_line": ["age_nme", "radius_nme"],
    "replay_loss_line": ["replay_tune", "lwf_tune", "balance_tune"],
}


def ensure_state_defaults(state):
    direction_queue = state.get("direction_queue") or list(DEFAULT_DIRECTION_FAMILIES)
    active_direction = state.get("active_direction") or direction_queue[0]
    candidate_families = state.get("candidate_families") or DEFAULT_DIRECTION_FAMILIES.get(
        active_direction, direction_queue
    )

    state.setdefault("version", 1)
    state.setdefault("best_full_note", "")
    state.setdefault("best_screen_note", "")
    state.setdefault("active_line", {})
    state["direction_queue"] = direction_queue
    state["active_direction"] = active_direction
    state["candidate_families"] = candidate_families
    state.setdefault("last_event_offset", 0)
    state.setdefault("family_stats", {})
    state.setdefault("rejected_signatures", [])
    state.setdefault("accepted_signatures", [])
    state.setdefault("stale_rounds", 0)
    state.setdefault("last_direction", active_direction)
    return state


def log(msg: str):
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    return json.loads(STATE_JSON.read_text(encoding="utf-8"))


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_note(title, body):
    with NOTES_MD.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {datetime.now().isoformat(timespec='seconds')} `{title}`\n"
            f"- {body}\n"
        )


def append_daily(body):
    with DAILY_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n- {body}\n")


def append_memory(body):
    with MEMORY_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n- {body}\n")


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


def parse_metric(value):
    if value in ("", None):
        return None
    return float(value)


def get_family_bucket(state, family):
    bucket = state["family_stats"].setdefault(
        family,
        {
            "accepted": 0,
            "confirm_rejected": 0,
            "screen_rejected": 0,
            "last_note": "",
            "last_task3": "",
            "last_score": "",
            "last_event": "",
        },
    )
    return bucket


def rotate_direction(state):
    queue = state.get("direction_queue") or list(DEFAULT_DIRECTION_FAMILIES)
    current = state.get("active_direction", queue[0])
    if current in queue:
        idx = queue.index(current)
        next_direction = queue[(idx + 1) % len(queue)]
    else:
        next_direction = queue[0]
    state["active_direction"] = next_direction
    state["candidate_families"] = DEFAULT_DIRECTION_FAMILIES[next_direction]
    state["stale_rounds"] = 0
    append_note(
        "direction_switch",
        f"switch to `{next_direction}` with candidate families `{state['candidate_families']}` after stale rounds",
    )
    append_daily(f"direction switch -> `{next_direction}` after stale rounds")
    log(f"switched direction to {next_direction}")


def family_priority(state, family):
    stats = state.get("family_stats", {}).get(family, {})
    last_score = parse_metric(stats.get("last_score"))
    accepted = int(stats.get("accepted", 0))
    confirm_rejected = int(stats.get("confirm_rejected", 0))
    screen_rejected = int(stats.get("screen_rejected", 0))
    return (
        accepted * 100.0
        + (last_score or 0.0)
        - confirm_rejected * 20.0
        - screen_rejected * 5.0
    )


def refresh_candidate_families(state):
    default_families = DEFAULT_DIRECTION_FAMILIES.get(state["active_direction"], [])
    merged = []
    seen = set()
    for family in list(state.get("candidate_families", [])) + list(default_families):
        if family not in seen:
            seen.add(family)
            merged.append(family)
    merged.sort(key=lambda family: (-family_priority(state, family), default_families.index(family)))
    state["candidate_families"] = merged
    return state


def run_one_cycle():
    subprocess.run(
        ["python", "autoresearch_icarl.py", "--hours", "1", "--max-cycles", "1", "--no-sleep-on-idle"],
        cwd=BASE_DIR,
        check=True,
    )


def load_events():
    if not EVENTS_JSONL.exists():
        return []
    events = []
    with EVENTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def append_event_memory(event):
    etype = event.get("type", "")
    payload = event.get("payload", {})
    note = payload.get("note", "")
    family = payload.get("family", "")
    t1 = payload.get("task1", "")
    t2 = payload.get("task2", "")
    t3 = payload.get("task3", "")
    score = payload.get("score", "")
    if etype == "confirm_accepted":
        append_daily(f"accepted `{note}` ({family}) -> {t1} / {t2} / {t3}, score {score}")
        append_memory(f"accepted `{note}` -> {t1} / {t2} / {t3}, score {score}")
    elif etype == "confirm_rejected":
        append_daily(f"rejected `{note}` ({family}) -> {t1} / {t2} / {t3}, score {score}")
    elif etype == "screen_rejected":
        append_daily(f"screen rejected `{note}` ({family}) -> {t1} / {t2} / {t3}, score {score}")
    elif etype == "run_completed":
        append_note("event_run_completed", f"`{note}` ({family}) -> {t1} / {t2} / {t3}, score {score}")


def update_family_stats(state, event):
    payload = event.get("payload", {})
    family = payload.get("family", "")
    if not family:
        return
    bucket = get_family_bucket(state, family)
    etype = event.get("type", "")
    if etype == "confirm_accepted":
        bucket["accepted"] += 1
    elif etype == "confirm_rejected":
        bucket["confirm_rejected"] += 1
    elif etype == "screen_rejected":
        bucket["screen_rejected"] += 1
    bucket["last_note"] = payload.get("note", "") or bucket["last_note"]
    bucket["last_task3"] = payload.get("task3", "") or bucket["last_task3"]
    bucket["last_score"] = payload.get("score", "") or bucket["last_score"]
    bucket["last_event"] = etype or bucket["last_event"]


def process_new_events(state):
    events = load_events()
    offset = int(state.get("last_event_offset", 0))
    new_events = events[offset:]
    if not new_events:
        return state, []

    bootstrap_only = offset == 0 and not state.get("family_stats")
    for event in new_events:
        update_family_stats(state, event)
        if not bootstrap_only:
            append_event_memory(event)

    state["last_event_offset"] = len(events)
    if bootstrap_only:
        append_note(
            "event_bootstrap",
            f"bootstrapped family stats from {len(new_events)} historical events without replaying old notes",
        )
        append_daily(f"bootstrapped family stats from {len(new_events)} historical events")
    return state, new_events


def format_best_line(row):
    if not row:
        return "- none"
    return (
        f"- `{row.get('note', '')}`\n"
        f"- task1 `{row.get('stage1_total', '')}` | task2 `{row.get('stage2_total', '')}` | task3 `{row.get('stage3_total', '')}`\n"
        f"- score `{row.get('score', '')}`"
    )


def write_status_md(state, rows, recent_events):
    best_full_row = best_full(rows)
    best_screen_row = best_screen(rows)
    lines = [
        "# Agent Status",
        "",
        f"- updated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- active direction: `{state.get('active_direction', '')}`",
        f"- candidate families: `{', '.join(state.get('candidate_families', []))}`",
        f"- stale rounds: `{state.get('stale_rounds', 0)}`",
        "",
        "## Best Full Confirm",
        format_best_line(best_full_row),
        "",
        "## Best Screen",
        format_best_line(best_screen_row),
        "",
        "## Family Stats",
    ]

    family_stats = state.get("family_stats", {})
    if family_stats:
        for family in state.get("candidate_families", []):
            stats = family_stats.get(family, {})
            lines.append(
                f"- `{family}`: accepted={stats.get('accepted', 0)}, "
                f"confirm_rejected={stats.get('confirm_rejected', 0)}, "
                f"screen_rejected={stats.get('screen_rejected', 0)}, "
                f"last_task3={stats.get('last_task3', '')}, "
                f"last_score={stats.get('last_score', '')}, "
                f"last_note=`{stats.get('last_note', '')}`"
            )
    else:
        lines.append("- no family stats yet")

    lines.extend(["", "## Recent Events"])
    if recent_events:
        for event in recent_events[-8:]:
            payload = event.get("payload", {})
            lines.append(
                f"- `{event.get('type', '')}` `{payload.get('note', '')}` "
                f"({payload.get('family', '')}) -> task3 `{payload.get('task3', '')}`, score `{payload.get('score', '')}`"
            )
    else:
        lines.append("- no new events in this controller window")

    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    deadline = datetime.now() + timedelta(hours=hours)
    state = ensure_state_defaults(load_state())
    write_status_md(state, load_experiment_rows(METRICS_CSV), [])

    while datetime.now() < deadline:
        state, events_before = process_new_events(state)
        state = refresh_candidate_families(state)
        save_state(state)
        rows_before = load_experiment_rows(METRICS_CSV)
        best_before = best_full(rows_before)
        best_before_note = best_before["note"] if best_before else ""

        run_one_cycle()

        rows_after = load_experiment_rows(METRICS_CSV)
        write_latest_md(rows_after, LATEST_MD)
        state = ensure_state_defaults(load_state())
        state, events_after = process_new_events(state)
        state = refresh_candidate_families(state)
        best_after = best_full(rows_after)
        best_screen_after = best_screen(rows_after)
        best_after_note = best_after["note"] if best_after else ""

        if best_after_note != best_before_note and best_after is not None:
            state["best_full_note"] = best_after_note
            state["stale_rounds"] = 0
            save_state(state)
            append_note(
                "best_update",
                f"new best full confirm `{best_after_note}` -> {best_after['stage1_total']} / {best_after['stage2_total']} / {best_after['stage3_total']}",
            )
            append_daily(
                f"new best full confirm `{best_after_note}` -> {best_after['stage1_total']} / {best_after['stage2_total']} / {best_after['stage3_total']}"
            )
            append_memory(
                f"current best full confirm: `{best_after_note}` -> {best_after['stage1_total']} / {best_after['stage2_total']} / {best_after['stage3_total']}"
            )
            log(f"new best full confirm: {best_after_note}")
        else:
            state["stale_rounds"] = int(state.get("stale_rounds", 0)) + 1

        if best_screen_after is not None:
            state["best_screen_note"] = best_screen_after.get("note", "")

        if state.get("stale_rounds", 0) >= 5:
            rotate_direction(state)

        write_status_md(state, rows_after, events_before + events_after)
        save_state(state)

        time.sleep(2)


if __name__ == "__main__":
    main()
