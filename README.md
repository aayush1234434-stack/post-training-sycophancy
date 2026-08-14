# Dual-metric sycophancy across post-training

MATS application project: on OLMo-3 7B Instruct (base → SFT → DPO → RL), when free-form sycophancy happens, is recoverable truth intact (override) or falling (erosion)?

**No training.** Inference only.

## What to run now

Main 4-stage eval is done. Remaining empirical work:

1. **Laptop:** `python list_unclear.py` then fill `results/unclear.csv`
2. **Colab:** private forced-choice (no user opinion), chat stages only
3. **Laptop:** `python analyze.py` then `python plot.py`

See [`START.md`](START.md). Science spec: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md)

```bash
python eval.py --stage sft --mode private --load-4bit
python analyze.py
python plot.py
```
