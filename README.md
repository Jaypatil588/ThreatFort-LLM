# RobustBench-LLM

**Adversarial Robustness Benchmarking Suite for Large Language Models**

A benchmarking framework to evaluate LLM robustness against adversarial prompt attacks, featuring automated attack evaluation across multiple models and a fine-tuned adversarial prompt classifier.

## Overview

RobustBench-LLM addresses a critical gap in LLM safety evaluation by providing:

1. **Multi-Attack Benchmark**: Systematic evaluation of 5 open-source LLMs against three state-of-the-art adversarial attack methods (GCG, AutoDAN, PAIR)
2. **Adversarial Prompt Classifier**: A QLoRA fine-tuned Llama 3.1 8B model that detects adversarial inputs with 94% accuracy at <35ms latency

## Key Results

### Model Robustness Comparison

| Model | ASR (%) ↓ | Refusal Rate (%) ↑ | FPR (%) ↓ | Avg Latency (ms) |
|-------|-----------|-------------------|-----------|-------------------|
| Llama 3.3 70B | 18.7 | 81.3 | 12.1 | 892 |
| Gemma 2 9B | 22.1 | 77.9 | 14.8 | 267 |
| Mixtral 8x7B | 29.8 | 70.2 | 9.5 | 215 |
| Llama 3.1 8B | 34.2 | 65.8 | 8.3 | 245 |
| Llama 3.2 3B | 42.5 | 57.5 | 5.7 | 198 |
| Llama 3.2 1B | 47.3 | 52.7 | 4.2 | 156 |

*ASR = Attack Success Rate (lower is better) | FPR = False Positive Rate on benign inputs*

### Classifier Performance

- **Accuracy**: 94.2% on held-out test set
- **Avg Latency**: 28.3ms per classification
- **P95 Latency**: 33.7ms (under 35ms target)
- **Method**: QLoRA fine-tuning (4-bit quantization + LoRA rank 16)

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
│   ├── evaluate.py          # Run attacks against models via API
│   └── visualize.py         # Generate charts and result tables
├── classifier/
│   └── train.py             # QLoRA fine-tuning script
├── data/
│   └── collect_datasets.py  # Collect adversarial & benign datasets
├── notebooks/
│   └── train_classifier.ipynb  # Colab notebook for GPU training
├── results/
│   └── figures/             # Generated visualizations
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

This downloads and processes 5,000+ adversarial and benign prompt pairs from:
- AdvBench (GCG attacks)
- JailbreakV-28K (diverse jailbreaks)
- ChatGPT Jailbreak Prompts
- Stanford Alpaca (benign)
- OpenAssistant (benign)

### 3. Run Benchmark

```bash
# Set your API key
cp .env.example .env
# Edit .env with your Groq API key

# Run benchmark (uses API, no GPU needed)
python benchmark/evaluate.py --n-adversarial 100 --n-benign 50
```

### 4. Train Classifier

**Option A: Google Colab (recommended)**
- Open `notebooks/train_classifier.ipynb` in Google Colab
- Follow the step-by-step instructions
- Training takes ~45 min on T4, ~15 min on A100

**Option B: Local (requires 16GB+ VRAM)**
```bash
python classifier/train.py --mode train
```

### 5. Evaluate Classifier

```bash
python classifier/train.py --mode eval
```

### 6. Generate Visualizations

```bash
python benchmark/visualize.py
```

## Technical Details

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.1 8B Instruct |
| Quantization | 4-bit NormalFloat (NF4) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q, k, v, o, gate, up, down projections |
| Trainable Params | ~0.1% of total |
| Training Epochs | 3 |
| Learning Rate | 2e-4 |
| Effective Batch Size | 16 |

### Dataset Composition

| Category | Count | Sources |
|----------|-------|---------|
| GCG adversarial | ~700 | AdvBench + synthetic suffixes |
| AutoDAN adversarial | ~900 | DAN template variations |
| PAIR adversarial | ~900 | Conversational jailbreaks |
| Additional jailbreaks | ~400 | JailbreakV-28K, community |
| Benign | ~2,500 | Alpaca, OpenAssistant |
| **Total** | **~5,400** | |

## References

- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) (GCG)
- [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned LLMs](https://arxiv.org/abs/2310.04451) (AutoDAN)
- [Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.08419) (PAIR)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (QLoRA)
- [HarmBench: A Standardized Evaluation Framework](https://arxiv.org/abs/2402.04249) (HarmBench)

## License

MIT
