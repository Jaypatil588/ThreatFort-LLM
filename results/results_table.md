# ThreatFort-LLM Results

## Model Robustness Comparison

| Model | ASR (%) ↓ | Refusal Rate (%) ↑ | FPR (%) ↓ | Avg Latency (ms) |
|-------|-----------|-------------------|-----------|-------------------|
| llama-3.1-70b | 18.7 | 81.3 | 12.1 | 892 |
| gemma-2-9b | 22.1 | 77.9 | 14.8 | 267 |
| qwen-2.5-7b | 29.8 | 70.2 | 9.5 | 215 |
| llama-3.1-8b | 34.2 | 65.8 | 8.3 | 245 |
| mistral-7b | 42.5 | 57.5 | 5.7 | 198 |
| phi-3-mini | 47.3 | 52.7 | 4.2 | 156 |

*ASR = Attack Success Rate (lower is better)*
*FPR = False Positive Rate on benign inputs (lower is better)*
