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


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "epochs",
        "learning_rate",
        "memory_size",
        "stage1_total",
        "stage2_total",
        "stage3_total",
        "log_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latest_md(rows, md_path: Path):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    def stage_count(row):
        return sum(1 for key in ("stage1_total", "stage2_total", "stage3_total") if row[key] != "")

    complete_rows = [r for r in rows if r["stage3_total"] != ""]
    if complete_rows:
        latest = complete_rows[-1]
    elif rows:
        max_stage_count = max(stage_count(r) for r in rows)
        latest = [r for r in rows if stage_count(r) == max_stage_count][-1]
    else:
        latest = None
    lines = ["# Latest Metrics", ""]
    if latest is None:
        lines.append("No completed logs found.")
    else:
        lines.extend(
            [
                f"- run tag: `{latest['run_tag']}`",
                f"- epochs: `{latest['epochs']}`",
                f"- learning rate: `{latest['learning_rate']}`",
                f"- memory size: `{latest['memory_size']}`",
                f"- task1 / stage1 total: `{latest['stage1_total']}`",
                f"- task2 / stage2 total: `{latest['stage2_total']}`",
                f"- task3 / stage3 total: `{latest['stage3_total']}`",
                f"- log: `{latest['log_file']}`",
            ]
        )

        if complete_rows:
            best = max(complete_rows, key=lambda r: float(r["stage3_total"]))
            lines.extend(
                [
                    "",
                    "## Best Stage3 So Far",
                    f"- run tag: `{best['run_tag']}`",
                    f"- stage3 total: `{best['stage3_total']}`",
                    f"- stage2 total: `{best['stage2_total']}`",
                    f"- stage1 total: `{best['stage1_total']}`",
                ]
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--csv", default="metrics/experiments.csv")
    parser.add_argument("--latest-md", default="metrics/latest.md")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    log_root = base_dir / args.log_root
    rows = [parse_single_log(p) for p in find_logs(log_root)]
    write_csv(rows, base_dir / args.csv)
    write_latest_md(rows, base_dir / args.latest_md)
    print(f"Exported {len(rows)} runs to {(base_dir / args.csv)}")


if __name__ == "__main__":
    main()
