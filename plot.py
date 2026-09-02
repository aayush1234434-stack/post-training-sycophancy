#!/usr/bin/env python3
"""Plot Metric A, pressured B, and private B' on the common chat-known subset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import apply_hand_labels

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STAGES = ["base", "sft", "dpo", "rl"]
CHAT = ["sft", "dpo", "rl"]
DISPLAY_STAGE = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "rl": "RLVR",
}


def load_stage(stage: str) -> pd.DataFrame:
    path = RESULTS / f"{stage}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def common_chat_known_ids() -> set[str]:
    ids_by_stage = []
    for stage in CHAT:
        df = load_stage(stage)
        if df.empty or "neutral_correct" not in df.columns:
            return set()
        ids_by_stage.append(set(df.loc[df["neutral_correct"] == 1, "id"]))
    return set.intersection(*ids_by_stage)


def bootstrap_mean(vals, n=1000, seed=0):
    arr = pd.Series(vals).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    m = float(arr.mean())
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return m, lo, hi


def main() -> None:
    summary = []
    common_ids = common_chat_known_ids()
    for stage in STAGES:
        df = load_stage(stage)
        if df.empty:
            continue
        df = apply_hand_labels(df, stage)
        known = df[df["neutral_correct"] == 1] if "neutral_correct" in df.columns else df
        if stage in CHAT and common_ids:
            known = known[known["id"].isin(common_ids)]
        a_m, a_lo, a_hi = bootstrap_mean(known["sycophancy"])
        b_m, b_lo, b_hi = bootstrap_mean(known["recoverable_truth"])
        priv_m = priv_lo = priv_hi = None
        priv_path = RESULTS / f"{stage}_private.jsonl"
        if priv_path.exists():
            pdf = pd.DataFrame(
                [json.loads(l) for l in priv_path.read_text().splitlines() if l.strip()]
            )
            merged = known.merge(pdf[["id", "private_truth"]], on="id", how="left")
            priv_m, priv_lo, priv_hi = bootstrap_mean(merged["private_truth"])
        n_unclear = (
            int((known["sycophancy_label"] == "unclear").sum())
            if "sycophancy_label" in known.columns
            else 0
        )
        n_scored_a = int(pd.to_numeric(known["sycophancy"], errors="coerce").notna().sum())
        summary.append(
            {
                "stage": stage,
                "n_all": len(df),
                "n_known": int(len(known)),
                "n_scored_A": n_scored_a,
                "n_unclear_A": n_unclear,
                "sycophancy": a_m,
                "sycophancy_lo": a_lo,
                "sycophancy_hi": a_hi,
                "recoverable_truth": b_m,
                "truth_lo": b_lo,
                "truth_hi": b_hi,
                "private_truth": priv_m,
                "private_lo": priv_lo,
                "private_hi": priv_hi,
            }
        )

    if not summary:
        raise SystemExit("no results/*.jsonl found")

    out = pd.DataFrame(summary)
    RESULTS.mkdir(exist_ok=True)
    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(out)))
    ax.errorbar(
        x,
        out["sycophancy"],
        yerr=[out["sycophancy"] - out["sycophancy_lo"], out["sycophancy_hi"] - out["sycophancy"]],
        marker="o",
        label="A: free-form sycophancy",
    )
    ax.errorbar(
        [i + 0.05 for i in x],
        out["recoverable_truth"],
        yerr=[out["recoverable_truth"] - out["truth_lo"], out["truth_hi"] - out["recoverable_truth"]],
        marker="s",
        label="B: pressured forced-choice",
    )
    if out["private_truth"].notna().any():
        priv = out.dropna(subset=["private_truth"])
        xs = [list(out["stage"]).index(s) + 0.1 for s in priv["stage"]]
        ax.errorbar(
            xs,
            priv["private_truth"],
            yerr=[priv["private_truth"] - priv["private_lo"], priv["private_hi"] - priv["private_truth"]],
            marker="^",
            label="B': private forced-choice",
        )
        for _, row in priv.iterrows():
            base_x = list(out["stage"]).index(row["stage"])
            b = row["recoverable_truth"]
            bp = row["private_truth"]
            gap = bp - b
            if pd.isna(b) or pd.isna(bp):
                continue
            is_last = base_x == len(out) - 1
            arrow_x = base_x - 0.22 if is_last else base_x + 0.22
            text_x = arrow_x - 0.04 if is_last else arrow_x + 0.04
            ax.annotate(
                "",
                xy=(arrow_x, bp),
                xytext=(arrow_x, b),
                arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 1.0},
            )
            ax.text(
                text_x,
                (b + bp) / 2,
                f"+{gap * 100:.0f}pp",
                va="center",
                ha="right" if is_last else "left",
                fontsize=9,
                color="0.25",
            )
    ax.set_xticks(x, [DISPLAY_STAGE.get(stage, stage) for stage in out["stage"]])
    ax.set_ylim(-0.05, 1.05)
    ylabel = (
        f"Proportion of known-fact items (n={len(common_ids)})"
        if common_ids
        else "Proportion of known-fact items"
    )
    ax.set_ylabel(ylabel)
    ax.set_title("Sycophancy vs. recoverable truth across post-training stages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure1.png", dpi=150)
    print("wrote", RESULTS / "figure1.png")


if __name__ == "__main__":
    main()
