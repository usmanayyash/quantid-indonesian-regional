# Decision stability of quantized small language models on Indonesian and ten regional languages

Companion data and code for the article "Accuracy Hides Decision Flips: Evaluating Quantized Small Language Models on Indonesian and Ten Regional Languages" (submitted to *Language Resources and Evaluation*).

## Contents

- `code/`: evaluation scripts.
  - `qid_eval.py`: IndoMMLU, NusaX (Indonesian), TyDiQA (Indonesian) and the speed/energy benchmark.
  - `qid_multi.py`: NusaX in twelve languages and TyDiQA (English).
  - `qid_app.py`: offline classroom assistant with benchmark mode.
  - `qid_data.py`: data preparation.
  - `qid_report.py`: summary tables.
- `results/`: per-item predictions, scores and telemetry for every configuration.
  - `{model}_{precision}.json`: Indonesian tasks.
  - `multi_{model}_{precision}.json`: twelve NusaX languages and English TyDiQA. Each `nusax_{language}_items` list holds the sentence id, predicted label, gold label and correctness.
  - `app_*.json`: classroom assistant benchmark.
- `analysis/`: statistical analysis.
  - `mstats.py` and its output `mstats.json`: McNemar tests with Holm correction, bootstrap intervals, share of changed decisions, label shift and Spearman correlations.
  - `overlap.json`: word overlap with Indonesian per language.

## Models and data

The models come from Hugging Face:

- `unsloth/gemma-3-1b-it`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

The scripts quantize them with bitsandbytes (LLM.int8() and NF4). The test data come from IndoMMLU, NusaX and TyDiQA-GoldP; please obtain them from their original authors and cite them.

## Hardware

NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.7.1, CUDA 11.8.
