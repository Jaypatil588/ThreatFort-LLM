# RobustBench-LLM

**Adversarial Robustness Benchmarking Suite for Large Language Models**

A benchmarking framework to evaluate LLM robustness against adversarial prompt attacks, featuring automated attack evaluation across multiple models and a QLoRA fine-tuned adversarial prompt classifier.

## Overview

RobustBench-LLM addresses a critical gap in LLM safety evaluation by providing:

1. **Multi-Attack Benchmark**: Systematic evaluation of open-source LLMs against three state-of-the-art adversarial attack methods (GCG, AutoDAN, PAIR)
2. **Adversarial Prompt Classifier**: A QLoRA fine-tuned Llama 3.2 1B model that detects adversarial inputs with **99.4% accuracy** on held-out test data and **95.6% accuracy** on completely unseen out-of-distribution datasets

## Classifier Results

### In-Distribution Test Set (518 samples)

The classifier was evaluated on a held-out test split from the same distribution as the training data.

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.4%** (515/518) |
| Adversarial Recall | 98.9% (265/268) |
| Benign Recall | 100.0% (250/250) |
| False Positives | 0 |
| False Negatives | 3 |

#### Per-Attack-Type Accuracy

| Attack Type | Accuracy | Samples |
|-------------|----------|---------|
| GCG (gradient-based suffix) | 98.4% | 192 |
| AutoDAN (jailbreak template) | 100.0% | 30 |
| PAIR (iterative refinement) | 100.0% | 46 |
| Benign (no attack) | 100.0% | 250 |

#### Confusion Matrix

|  | Predicted Adversarial | Predicted Benign |
|--|----------------------|-----------------|
| **True Adversarial** | 265 | 3 |
| **True Benign** | 0 | 250 |

> **Zero false positives** — the model never incorrectly flags a benign prompt as adversarial on the in-distribution test set.

---

### Out-of-Distribution Validation (4,740 unseen samples)

To validate generalization, the classifier was evaluated on **completely unseen datasets** that were NOT used during training:

- **[lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat)**: 4,640 real user conversations from Vicuna/ChatGPT annotated for toxicity and jailbreaking
- **[SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests)**: 100 clear-cut unsafe prompts designed to test basic safety boundaries

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **95.6%** (4,533/4,740) |
| Adversarial Recall | 81.1% (142/175) |
| Benign Recall | 96.2% (4,391/4,565) |

#### Per-Source Accuracy

| Dataset | Accuracy | Samples | Notes |
|---------|----------|---------|-------|
| ToxicChat | 96.2% | 4,640 | Real-world user conversations |
| SimpleSafetyTests | 71.0% | 100 | Direct harmful requests without jailbreak wrapping |

#### Confusion Matrix

|  | Predicted Adversarial | Predicted Benign | Predicted Unknown |
|--|----------------------|-----------------|-------------------|
| **True Adversarial** | 142 | 33 | 0 |
| **True Benign** | 166 | 4,391 | 8 |

#### Analysis

- **Strong real-world generalization**: 96.2% accuracy on real user conversations the model has never seen
- **Weakness on direct harm prompts**: SimpleSafetyTests (71%) contains plain harmful requests (e.g., "How to make a bomb") without jailbreak template wrapping — the training data is skewed toward jailbreak-wrapped adversarial prompts
- **False positive pattern**: The model over-flags roleplay and creative-writing prompts that superficially resemble jailbreak attempts (e.g., "roleplay a murder suspect", "write as Walter White")

## Attack Methods

| Attack | Type | Description |
|--------|------|-------------|
| **GCG** | Gradient-based | Greedy Coordinate Gradient — appends optimized adversarial suffixes |
| **AutoDAN** | Generation-based | Automatically generates "Do Anything Now" style jailbreaks |
| **PAIR** | Iterative refinement | Uses an attacker LLM to iteratively refine jailbreak prompts |

## Project Structure

```
robustbench/
├── benchmark/
│   ├── evaluate.py               # Run attacks against models via Groq API
│   ├── baseline_comparison.py    # Compare model robustness
│   └── visualize.py              # Generate charts and result tables
├── classifier/
│   ├── train.py                  # QLoRA fine-tuning script (GPU)
│   ├── evaluate_adapter.py       # Evaluate adapter on held-out test set (MPS/CPU)
│   └── download_and_eval_unseen.py  # Download unseen datasets & evaluate OOD
├── data/
│   ├── collect_datasets.py       # Collect adversarial & benign datasets
│   ├── processed/                # Train/val/test JSONL splits
│   └── unseen_validation/        # Out-of-distribution validation data
├── notebooks/
│   └── train_classifier.ipynb    # Google Colab notebook for GPU training
├── models/
│   └── classifier/final_adapter/ # Trained QLoRA adapter weights
├── results/                      # Evaluation results and figures
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Dataset

```bash
python data/collect_datasets.py
```

Downloads and processes 5,173 adversarial and benign prompt pairs from:
- AdvBench (GCG attacks) + synthetic adversarial suffixes
- AutoDAN-style jailbreak templates
- PAIR conversational jailbreaks
- ChatGPT Jailbreak Prompts & prompt injection datasets
- Stanford Alpaca (benign)

### 3. Train Classifier (Google Colab)

1. Open `notebooks/train_classifier.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Run all cells — training completes in ~25 minutes on T4
4. The notebook will download the adapter weights as a zip file
5. Extract to `models/classifier/final_adapter/`

### 4. Evaluate Classifier

```bash
# In-distribution evaluation (held-out test set)
python classifier/evaluate_adapter.py

# Out-of-distribution evaluation (unseen datasets)
python classifier/download_and_eval_unseen.py
```

### 5. Run Benchmark

```bash
# Set your API key
cp .env.example .env
# Edit .env with your Groq API key

# Run benchmark
python benchmark/evaluate.py --n-adversarial 100 --n-benign 50
```

### 6. Generate Visualizations

```bash
python benchmark/visualize.py
```

## Technical Details

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.2 1B Instruct |
| Quantization | 4-bit NormalFloat (NF4) + double quantization |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q, k, v, o, gate, up, down projections (all-linear) |
| Trainable Params | 11.3M / 1.25B (0.90%) |
| Training Epochs | 3 |
| Learning Rate | 2e-4 (cosine schedule, 3% warmup) |
| Effective Batch Size | 16 (4 × 4 gradient accumulation) |
| Max Sequence Length | 512 |
| Gradient Checkpointing | Enabled (non-reentrant) |
| Training Time | ~25 min on T4 GPU |

### Training Dataset Composition

| Category | Count | Sources |
|----------|-------|---------|
| GCG adversarial | 1,912 | AdvBench + synthetic suffixes |
| AutoDAN adversarial | 341 | DAN template variations |
| PAIR adversarial | 420 | Conversational jailbreaks |
| Benign | 2,500 | Alpaca, OpenAssistant |
| **Total** | **5,173** | 80/10/10 train/val/test split |

### Unseen Validation Dataset

| Source | Adversarial | Benign | Total |
|--------|-------------|--------|-------|
| lmsys/toxic-chat | 91 | 4,549 | 4,640 |
| SimpleSafetyTests | 100 | 0 | 100 |
| **Total** | **175** | **4,565** | **4,740** |

## References

- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) (GCG)
- [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned LLMs](https://arxiv.org/abs/2310.04451) (AutoDAN)
- [Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.08419) (PAIR)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (QLoRA)
- [HarmBench: A Standardized Evaluation Framework](https://arxiv.org/abs/2402.04249) (HarmBench)
- [ToxicChat: Unveiling Hidden Challenges of Toxicity Detection](https://arxiv.org/abs/2310.17389) (ToxicChat)

## License

MIT
