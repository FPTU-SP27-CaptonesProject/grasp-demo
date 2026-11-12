"""Run 6 test questions (2 per dataset) through every column, measure time and VRAM, compare GRPO's answers
with the test999 report, and judge whether this machine can host the demo (spec 2026-11-12 section 6)."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

THRESHOLDS = {"vram_gb": 15.0, "question_seconds": 60.0, "grasp_seconds": 40.0, "grpo_matches": 4}
GRASP = ("grasp_sft", "grasp_grpo")
BEDROCK = "rag_235b"  # excluded from health and speed: remote, not a local resource


def pick_questions(rows: list[dict[str, Any]], per_dataset: int = 2) -> list[dict[str, Any]]:
    taken: dict[str, int] = defaultdict(int)
    chosen = []
    for row in rows:
        dataset = str(row.get("metadata", {}).get("dataset", ""))
        if taken[dataset] < per_dataset:
            chosen.append(row)
            taken[dataset] += 1
    return chosen


def judge(result: dict[str, Any]) -> dict[str, bool | None]:
    health = all(item.get("ready") for name, item in result["health"].items() if name != "strong")
    vram = result.get("vram_peak_gb") is not None and result["vram_peak_gb"] <= THRESHOLDS["vram_gb"]
    speed = True
    for question in result["questions"]:
        if question["seconds"] > THRESHOLDS["question_seconds"]:
            speed = False
        for column, values in question["columns"].items():
            if column == BEDROCK:
                continue
            if "error" in values or "seconds" not in values:
                speed = False
            elif column in GRASP and values["seconds"] > THRESHOLDS["grasp_seconds"]:
                speed = False
    reported = [q.get("grpo_matches_report") for q in result["questions"]]
    matches = None if not reported or any(value is None for value in reported) else sum(reported) >= THRESHOLDS["grpo_matches"]
    verdict = {"health": health, "vram": vram, "speed": speed, "grpo_matches": matches}
    verdict["local_ok"] = all(value is True for value in verdict.values())
    return verdict


class VramSampler(threading.Thread):
    """Peak of the summed memory.used over every GPU, sampled each second (None without nvidia-smi)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib: float | None = None
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                used = sum(float(line) for line in output.split() if line.strip())
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stopped.wait(1.0)


def main(argv: list[str] | None = None) -> None:
    from grasp_grpo.evaluate import exact

    from .config import load_config, load_dotenv
    from .evalset import load_eval_rows
    from .runners import COLUMNS, run_column
    from .services import DemoServices

    parser = argparse.ArgumentParser(description="Measure the demo on 6 test questions and judge this machine")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", help="test999 report of the configured GRPO checkpoint (has predictions)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    load_dotenv(config["_root"])
    sampler = VramSampler()
    sampler.start()
    services = None
    try:
        services = DemoServices.from_config(config)
        for name, item in services.health.items():
            print(f"{name:10s} {'ok' if item['ready'] else 'FAIL ' + item.get('reason', '')}")
        reported = {}
        if args.report:
            reported = {p["id"]: p["prediction"] for p in json.loads(Path(args.report).read_text(encoding="utf-8")).get("predictions", [])}
        columns = [column for column in COLUMNS if services.enabled(column)[0]]
        questions = []
        for row in pick_questions(load_eval_rows(config)):
            def one(column: str, row=row) -> tuple[str, dict[str, Any]]:
                events: list[tuple[str, dict[str, Any]]] = []
                try:
                    prediction = run_column(services, column, row, lambda kind, data: events.append((kind, data)))["prediction"]
                    done = next(data for kind, data in reversed(events) if kind == "done")
                    return column, {"seconds": done["seconds"], "f1": done.get("f1"), "prediction": prediction}
                except Exception as error:  # noqa: BLE001 - recorded; the judge counts it as a failure
                    return column, {"error": f"{type(error).__name__}: {error}"}

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=max(1, len(columns))) as pool:
                per_column = dict(pool.map(one, columns))
            entry = {"id": row["id"], "dataset": row.get("metadata", {}).get("dataset"), "seconds": time.monotonic() - started, "columns": per_column}
            grpo = per_column.get("grasp_grpo", {})
            if row["id"] in reported and "prediction" in grpo:
                entry["grpo_matches_report"] = exact(grpo["prediction"], [reported[row["id"]]]) == 1.0
            questions.append(entry)
            cells = "  ".join(f"{c}={v['seconds']:.1f}s" if "seconds" in v else f"{c}=ERR" for c, v in per_column.items())
            print(f"{row['id']:>24s} {entry['seconds']:5.1f}s  {cells}")
    finally:
        sampler.stopped.set()
        sampler.join(timeout=5)
        if services is not None:
            services.close()
    result = {"config": str(Path(args.config).resolve()), "health": services.health, "columns": columns,
              "vram_peak_gb": None if sampler.peak_mib is None else round(sampler.peak_mib / 1024, 2), "questions": questions, "thresholds": THRESHOLDS}
    result["verdict"] = judge(result)
    mode = Path(args.config).stem.replace("models.", "")
    output = Path(config.get("runs_dir", "runs")) / f"preflight-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"vram_peak_gb": result["vram_peak_gb"], **result["verdict"], "output": str(output)}))


if __name__ == "__main__":
    main()
