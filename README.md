# Dual-metric sycophancy across post-training

MATS application project: on a public OLMo-3 7B Instruct ladder (base → SFT → DPO → RL), when free-form sycophancy rises, does recoverable truth stay intact (**override**), fall (**erosion**), or split by stage?

**No training.** Inference only.

## Start here (today)

1. Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Colab (or copy the cells).
2. Runtime → GPU → T4.
3. Set `STAGE = "sft"` and `SMOKE = True`.
4. Run all. You should get 3 scored items in ~10–20 minutes (first download is slow).
5. If that works, set `SMOKE = False` and run **one full stage**, then change `STAGE` and repeat.

Details: [`START.md`](START.md) · science spec: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md)

## Commands (Colab or any GPU box)

```bash
python eval.py --stage sft --limit 3 --load-4bit
python eval.py --stage sft --load-4bit
python plot.py
```

Stages: `base` `sft` `dpo` `rl`

## Outputs

`results/<stage>.jsonl` → `results/summary.csv` + `results/figure1.png`
