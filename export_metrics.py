import argparse
import csv
import re
from pathlib import Path


LOG_FILE_RE = re.compile(r"log_.*\.txt$")
DATE_DIR_RE = re.compile(r"^\d{8}_\d{6}.*$")
STAGE_START_RE = re.compile(r"Stage:\s*(\d+),\s*numclass:\s*(\d+)")
STAGE_END_RE = re.compile(r"Stage:\s*(\d+)\s+finish")
FINAL_ACC_RE = re.compile(r"Final Total Accuracy:\s*([0-9.]+)%")
STATE_RE = re.compile(
    r"Replay memory size:(?P<memory>\d+), learning_rate:(?P<lr>[0-9.]+), epochs:(?P<epochs>\d+),"
)


def parse_single_log(log_path: Path):
    stage_totals = {}
    current_stage = None
    pending_stage_total = None
    in_subject_block = False
    config = {"memory_size": "", "learning_rate": "", "epochs": ""}

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            state_match = STATE_RE.search(line)
            if state_match:
                config["memory_size"] = state_match.group("memory")
                config["learning_rate"] = state_match.group("lr")
                config["epochs"] = state_match.group("epochs")

            stage_match = STAGE_START_RE.search(line)
            if stage_match:
                current_stage = int(stage_match.group(1))
                pending_stage_total = None
                in_subject_block = False
                continue

            if "**********sub:" in line:
                in_subject_block = True
                continue

            if "result end" in line:
                in_subject_block = False
                continue

            acc_match = FINAL_ACC_RE.search(line)
            if acc_match and current_stage is not None and not in_subject_block:
                pending_stage_total = float(acc_match.group(1))
                continue

            end_match = STAGE_END_RE.search(line)
            if end_match:
                stage_id = int(end_match.group(1))
                if pending_stage_total is not None:
                    stage_totals[stage_id] = pending_stage_total
                current_stage = None
                pending_stage_total = None
                in_subject_block = False

    run_tag = log_path.parent.name
    return {
        "run_tag": run_tag,
        "log_file": str(log_path.relative_to(log_path.parents[1])),
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "memory_size": config["memory_size"],
        "stage1_total": stage_totals.get(1, ""),
        "stage2_total": stage_totals.get(2, ""),
        "stage3_total": stage_totals.get(3, ""),
    }


def find_logs(log_root: Path):
    logs = []
    if not log_root.exists():
        return logs
    for subdir in sorted(log_root.iterdir()):
        if not subdir.is_dir() or not DATE_DIR_RE.match(subdir.name):
            continue
        for child in sorted(subdir.iterdir()):
            if child.is_file() and LOG_FILE_RE.match(child.name):
                logs.append(child)
    return logs


def stage_count(row):
    return sum(1 for key in ("stage1_total", "stage2_total", "stage3_total") if row[key] != "")


def compute_score(row):
    s1 = float(row["stage1_total"]) if row["stage1_total"] != "" else 0.0
    s2 = float(row["stage2_total"]) if row["stage2_total"] != "" else 0.0
    s3 = float(row["stage3_total"]) if row["stage3_total"] != "" else 0.0
    if row["stage3_total"] != "":
        return round(0.2 * s1 + 0.3 * s2 + 0.5 * s3, 2)
    if row["stage2_total"] != "":
        return round(0.3 * s1 + 0.7 * s2, 2)
    if row["stage1_total"] != "":
        return round(s1, 2)
    return ""


def write_latest_md(rows, md_path: Path):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    complete_rows = [r for r in rows if r["stage3_total"] != ""]
    if complete_rows:
        latest = complete_rows[-1]
    elif rows:
        max_stage = max(stage_count(r) for r in rows)
        latest = [r for r in rows if stage_count(r) == max_stage][-1]
    else:
        latest = None

    lines = ["# Latest Metrics", ""]
    if latest is None:
        lines.append("No completed logs found.")
    else:
        lines.extend(
            [
                f"- run tag: `{latest['run_tag']}`",
                f"- epochs: `{latest.get('epochs', '')}`",
                f"- seeds: `{latest.get('seeds', '')}`",
                f"- gpu: `{latest.get('gpu', '')}`",
                f"- note: `{latest.get('note', '')}`",
                f"- task1 / stage1 total: `{latest['stage1_total']}`",
                f"- task2 / stage2 total: `{latest['stage2_total']}`",
                f"- task3 / stage3 total: `{latest['stage3_total']}`",
                f"- score: `{latest.get('score', '')}`",
                f"- log: `{latest['log_file']}`",
            ]
        )

        if complete_rows:
            best = max(complete_rows, key=lambda r: float(r.get("score", -1) or -1))
            lines.extend(
                [
                    "",
                    "## Best Completed Run So Far",
                    f"- run tag: `{best['run_tag']}`",
                    f"- note: `{best.get('note', '')}`",
                    f"- task3 total: `{best['stage3_total']}`",
                    f"- task2 total: `{best['stage2_total']}`",
                    f"- task1 total: `{best['stage1_total']}`",
                    f"- score: `{best.get('score', '')}`",
                ]
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment_row(
    csv_path: Path,
    parsed_row: dict,
    *,
    mode: str = "",
    gpu: str = "",
    seeds: str = "",
    note: str = "",
    hypothesis: str = "",
    change_summary: str = "",
    status: str = "",
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "mode",
        "gpu",
        "seeds",
        "epochs",
        "learning_rate",
        "memory_size",
        "stage1_total",
        "stage2_total",
        "stage3_total",
        "score",
        "status",
        "note",
        "hypothesis",
        "change_summary",
        "log_file",
    ]

    row = {
        "run_tag": parsed_row["run_tag"],
        "mode": mode,
        "gpu": gpu,
        "seeds": seeds,
        "epochs": parsed_row.get("epochs", ""),
        "learning_rate": parsed_row.get("learning_rate", ""),
        "memory_size": parsed_row.get("memory_size", ""),
        "stage1_total": parsed_row.get("stage1_total", ""),
        "stage2_total": parsed_row.get("stage2_total", ""),
        "stage3_total": parsed_row.get("stage3_total", ""),
        "score": compute_score(parsed_row),
        "status": status,
        "note": note,
        "hypothesis": hypothesis,
        "change_summary": change_summary,
        "log_file": parsed_row.get("log_file", ""),
    }

    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def load_experiment_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_experiment_rows(csv_path: Path, rows):
    fieldnames = [
        "run_tag",
        "mode",
        "gpu",
        "seeds",
        "epochs",
        "learning_rate",
        "memory_size",
        "stage1_total",
        "stage2_total",
        "stage3_total",
        "score",
        "status",
        "note",
        "hypothesis",
        "change_summary",
        "log_file",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def migrate_log_history(log_root: Path, csv_path: Path):
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = load_experiment_rows(csv_path)
        if existing and "mode" in existing[0]:
            return existing

    rows = []
    for parsed in [parse_single_log(p) for p in find_logs(log_root)]:
        rows.append(
            {
                "run_tag": parsed["run_tag"],
                "mode": "legacy",
                "gpu": "",
                "seeds": "",
                "epochs": parsed["epochs"],
                "learning_rate": parsed["learning_rate"],
                "memory_size": parsed["memory_size"],
                "stage1_total": parsed["stage1_total"],
                "stage2_total": parsed["stage2_total"],
                "stage3_total": parsed["stage3_total"],
                "score": compute_score(parsed),
                "status": "legacy",
                "note": "migrated from existing log",
                "hypothesis": "",
                "change_summary": "",
                "log_file": parsed["log_file"],
            }
        )

    write_experiment_rows(csv_path, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--csv", default="metrics/experiments.csv")
    parser.add_argument("--latest-md", default="metrics/latest.md")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    log_root = base_dir / args.log_root
    csv_path = base_dir / args.csv
    rows = migrate_log_history(log_root, csv_path)
    write_latest_md(rows, base_dir / args.latest_md)
    print(f"Prepared metrics files at {csv_path}")


if __name__ == "__main__":
    main()
