# ThreatFort-LLM

**Adversarial Robustness Benchmarking Suite for Large Language Models**

ThreatFort-LLM evaluates LLM robustness against adversarial prompt attacks and trains a QLoRA adversarial prompt classifier.

## Overview

1. **Multi-Attack Benchmark**: evaluation against GCG, AutoDAN, PAIR, and the expanded prompt-attack taxonomy.
2. **Adversarial Prompt Classifier**: a Llama 3.2 3B Instruct QLoRA sequence classifier trained from `test.ipynb` on `newDataset/processed`.

## Current Training Flow

Training happens in Google Colab with `test.ipynb`.

The notebook clones `https://github.com/Jaypatil588/robustbench-llm.git` branch `main` into `/content/threatfort_repo`, then copies these cloned repo files into `/content/threatfort_runtime/processed/`:

```text
newDataset/processed/train.jsonl
newDataset/processed/val.jsonl
newDataset/processed/test.jsonl
```

Training reads the copied runtime files. Local evaluation reads:

```text
newDataset/processed/test.jsonl
```

## Project Structure

```text
threatfort/
├── benchmark/
│   ├── evaluate.py
│   ├── baseline_comparison.py
│   └── visualize.py
├── classifier/
│   ├── app.py
│   ├── pipeline.py
│   ├── train.py
│   ├── evaluate_adapter.py
│   ├── offline_inference.py
│   └── download_and_eval_unseen.py
├── newDataset/
│   ├── processed/
│   ├── prompt_attack_groq_data/
│   └── unseen_validation/
├── reports/
│   ├── prompt_attack_taxonomy.md
│   └── prompt_dataset_balancing_report.md
├── test.ipynb
├── results/
└── requirements.txt
```

## Quick Start

### 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train in Colab

1. Open `test.ipynb` in Google Colab.
2. Set runtime to `T4 GPU`.
3. Run all cells. The notebook clones the repo and copies `newDataset/processed` into the Colab runtime automatically.
5. Download `classifier_adapter.zip`.
6. Extract it into:

```text
models/classifier/final_adapter/
```

The adapter must contain `adapter_config.json`.

### 3. Evaluate Locally

```bash
python3 classifier/pipeline.py eval-adapter
```

or:

```bash
python3 classifier/evaluate_adapter.py
```

### 4. Run Benchmark

```bash
python3 benchmark/evaluate.py --n-adversarial 100 --n-benign 50
```

## Model

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Training method | QLoRA sequence classification |
| Quantization | 4-bit NF4 + double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Input | `prompt` only |
| Target | binary `label` only: `0` benign, `1` adversarial |
| Target modules | all linear modules in `test.ipynb` |
| Epochs | up to 5 with early stopping |
| Effective batch size | 16 |
| Max sequence length | 512 |

## Current Dataset

| Split | Samples | Adversarial | Benign |
|-------|--------:|------------:|-------:|
| Train | 7,341 | 3,744 | 3,597 |
| Validation | 914 | 466 | 448 |
| Test | 943 | 481 | 462 |
| **Total** | **9,198** | **4,691** | **4,507** |

The processed data combines the baseline adversarial/benign corpus with the report-defined 23 prompt-attack taxonomy datasets.

## Reports

- `reports/prompt_attack_taxonomy.md`
- `reports/prompt_dataset_balancing_report.md`

The taxonomy JSON files under `newDataset/prompt_attack_groq_data/` are currently balanced 1:1 by label.
