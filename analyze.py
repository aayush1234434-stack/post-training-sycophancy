#!/usr/bin/env python3
"""Tables for the write-up: rates, 2x2 override/erosion, true-control, private B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGES = ["base", "sft", "dpo", "rl"]
CHAT = ["sft", "dpo", "rl"]


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def apply_hand_labels(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    path = RESULTS / "unclear.csv"
    if df.empty or not path.exists():
        return df
    labels = pd.read_csv(path, dtype=str)
    if "hand_sycophancy" not in labels.columns:
        return df
    labels = labels[labels["stage"] == stage].copy()
    labels = labels[labels["hand_sycophancy"].isin(["0", "1"])]
    if labels.empty:
        return df
    df = df.copy()
    mapping = dict(zip(labels["id"], labels["hand_sycophancy"].astype(int)))
    mask = df["id"].map(mapping).notna()
    df.loc[mask, "sycophancy"] = df.loc[mask, "id"].map(mapping)
    df.loc[mask, "sycophancy_label"] = "hand"
    return df


def rate(series) -> tuple[float | None, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, 0
    return float(s.mean()), int(len(s))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summary_rows = []
    cell_rows = []

    for stage in STAGES:
        df = load_jsonl(RESULTS / f"{stage}.jsonl")
        if df.empty:
            continue
        df = apply_hand_labels(df, stage)
        known = df[df["neutral_correct"] == 1] if "neutral_correct" in df.columns else df
        priv = load_jsonl(RESULTS / f"{stage}_private.jsonl")
        if not priv.empty:
            known = known.merge(
                priv[["id", "private_truth"]],
                on="id",
                how="left",
            )

        a_rate, a_n = rate(known.get("sycophancy"))
        b_rate, b_n = rate(known.get("recoverable_truth"))
        ctrl_rate, ctrl_n = rate(known.get("true_control_agrees"))
        priv_rate, priv_n = rate(known.get("private_truth")) if "private_truth" in known.columns else (None, 0)
        n_unclear = int((known.get("sycophancy_label") == "unclear").sum()) if "sycophancy_label" in known.columns else 0

        summary_rows.append(
            {
                "stage": stage,
                "n_all": len(df),
                "n_known": int(len(known)),
                "n_scored_A": a_n,
                "n_unclear_A": n_unclear,
                "sycophancy_A": a_rate,
                "pressured_B": b_rate,
                "n_pressured_B": b_n,
                "private_B": priv_rate,
                "n_private_B": priv_n,
                "true_control_agree": ctrl_rate,
                "n_true_control": ctrl_n,
            }
        )

        scored = known.dropna(subset=["sycophancy", "recoverable_truth"])
        a1b1 = int(((scored["sycophancy"] == 1) & (scored["recoverable_truth"] == 1)).sum())
        a1b0 = int(((scored["sycophancy"] == 1) & (scored["recoverable_truth"] == 0)).sum())
        a0b1 = int(((scored["sycophancy"] == 0) & (scored["recoverable_truth"] == 1)).sum())
        a0b0 = int(((scored["sycophancy"] == 0) & (scored["recoverable_truth"] == 0)).sum())
        cell_rows.append(
            {
                "stage": stage,
                "n_2x2": len(scored),
                "override_A1_B1": a1b1,
                "erosion_A1_B0": a1b0,
                "honest_A0_B1": a0b1,
                "other_A0_B0": a0b0,
            }
        )

    if not summary_rows:
        raise SystemExit("no results/*.jsonl found")

    summary = pd.DataFrame(summary_rows)
    cells = pd.DataFrame(cell_rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    cells.to_csv(RESULTS / "table_override.csv", index=False)

    print("=== rates (known-fact items) ===")
    print(summary.to_string(index=False))
    print("\n=== 2x2: free-form A vs pressured forced-choice B ===")
    print(cells.to_string(index=False))
    print(f"\nwrote {RESULTS / 'summary.csv'}")
    print(f"wrote {RESULTS / 'table_override.csv'}")

    missing_priv = [s for s in CHAT if not (RESULTS / f"{s}_private.jsonl").exists()]
    if missing_priv:
        print(f"\nprivate B not run yet for: {', '.join(missing_priv)}")
        print("Colab: python eval.py --stage sft --mode private --load-4bit")


if __name__ == "__main__":
    main()
