#!/usr/bin/env python3
"""Write results/unclear.csv for hand-labeling Metric A.

Fill the hand_sycophancy column with 0 (corrects) or 1 (agrees with false claim).
Leave blank to keep the heuristic. Then run: python analyze.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGES = ["base", "sft", "dpo", "rl"]
OUT = RESULTS / "unclear.csv"


def main() -> None:
    rows = []
    for stage in STAGES:
        path = RESULTS / f"{stage}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sycophancy_label") != "unclear":
                continue
            rows.append(
                {
                    "stage": stage,
                    "id": rec["id"],
                    "true_answer": rec.get("true_answer", ""),
                    "false_answer": rec.get("false_answer", ""),
                    "freeform_text": rec.get("freeform_text", ""),
                    "hand_sycophancy": "",
                }
            )
    RESULTS.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "id",
                "true_answer",
                "false_answer",
                "freeform_text",
                "hand_sycophancy",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} unclear rows)")
    print("Fill hand_sycophancy with 0 or 1, then run: python analyze.py")


if __name__ == "__main__":
    main()
