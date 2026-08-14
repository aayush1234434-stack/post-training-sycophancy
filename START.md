# Start today

You already locked the question. This is the mechanical start.

## 0. Push this code to GitHub

From this folder:

```bash
git add .
git commit -m "Add dual-metric sycophancy eval scaffold."
git remote add origin https://github.com/aayush1234434-stack/post-training-sycophancy.git
git branch -M main
git push -u origin main
```

If this folder is not the GitHub clone, copy everything into
`~/Desktop/post-training-sycophancy/post-training-sycophancy` and push from there.

Skip `git remote add` if origin already exists.

## 1. Colab smoke test (do this before anything else)

1. [colab.research.google.com](https://colab.research.google.com) → New notebook.
2. Runtime → Change runtime type → **T4 GPU**. Confirm with:

```python
import torch
print(torch.cuda.get_device_name(0))
```

If this errors, you did not get a GPU. Try later or use Colab Pro / Vast.

3. Upload `notebooks/colab_run.ipynb` **or** paste its cells.
4. Set `STAGE = "sft"` and `SMOKE = True`.
5. Run all cells.

Success = a `results/sft.jsonl` with 3 lines, each with `sycophancy` and `recoverable_truth`.

**Do not start the full 60-item run until the smoke test works.**

## 2. Full run (one stage per Colab session)

Same notebook, `SMOKE = False`.

Order: `sft` → `dpo` → `rl` → `base`.

SFT first because it is a chat model and the easiest sanity check. Base last: no chat template, more likely to look messy.

After each stage, the notebook copies `results/<stage>.jsonl` to Google Drive. If Colab dies, re-run the same stage: `eval.py` **resumes** and skips IDs already in the jsonl.

## 3. After 2+ stages, plot locally (no GPU)

```bash
python plot.py
```

You need `pandas matplotlib numpy`. This laptop can do this.

## If something breaks

| Symptom | Fix |
|---------|-----|
| CUDA out of memory | Keep `--load-4bit`. Do not load two models. Restart runtime. |
| No GPU | Runtime → T4. If still CPU, wait / Pro / Vast. |
| `Olmo3` / transformers error | `pip install -U transformers` (need ≥ 4.57). |
| Slow first run | First download is several GB. Later stages in the same session are faster only if HF cache survives — it often does not. Budget 10 min download per new checkpoint. |
| Scorer says `unclear` | Fine for smoke test. We hand-check 20 later. |

Do not add metrics, probes, or a second model family until Figure 1 exists.
