"""
Evaluate the QLoRA adapter on the test set.
Runs on Mac (MPS/CPU) without bitsandbytes — loads model in fp32.
"""

import json
import time
import sys
import os
from pathlib import Path
from collections import defaultdict

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Authenticate with HuggingFace for gated model access
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set in .env — gated model access may fail")

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTER_PATH = PROJECT_ROOT / "models" / "classifier" / "final_adapter"
TEST_PATH    = PROJECT_ROOT / "data" / "processed" / "test.jsonl"

# ─── Inference template (matches training format) ────────────────────────────

INFERENCE_TEMPLATE = (
    "System: You are a security classifier. "
    "Answer with just 'adversarial' or 'benign'.\n"
    "User: {prompt}\n"
    "Assistant:"
)


def load_model(adapter_path: str):
    """Load base model + LoRA adapter on MPS or CPU (no bitsandbytes needed)."""
    
    with open(Path(adapter_path) / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]
    
    print(f"Loading base model: {base_model_id}")
    
    # Pick device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    # Load in fp32 on Mac (no quantization needed for inference eval)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map={"": device},
    )
    
    # Merge the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # Fold adapter into base weights for speed
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded and adapter merged. Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, device


def classify(model, tokenizer, device, prompt: str) -> dict:
    """Run a single classification inference."""
    
    input_text = INFERENCE_TEMPLATE.format(prompt=prompt[:800])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    latency_ms = (time.time() - start) * 1000
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    
    if "adversarial" in raw:
        label = "adversarial"
    elif "benign" in raw:
        label = "benign"
    else:
        label = "unknown"
    
    return {"prediction": label, "raw_output": raw, "latency_ms": latency_ms}


def main():
    adapter_path = str(ADAPTER_PATH)
    test_path    = str(TEST_PATH)
    
    # Load model
    model, tokenizer, device = load_model(adapter_path)
    
    # Load test data
    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"\nTest samples: {len(test_data)}")
    
    # Run evaluation
    correct = 0
    total   = 0
    latencies = []
    
    # Track per-class and per-attack-type metrics
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    attack_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(int)  # (true, pred) -> count
    errors = []
    
    for sample in tqdm(test_data, desc="Evaluating"):
        result = classify(model, tokenizer, device, sample["prompt"])
        
        true_label = sample["label"]
        pred_label = result["prediction"]
        is_correct = pred_label == true_label
        
        if is_correct:
            correct += 1
        else:
            errors.append({
                "prompt": sample["prompt"][:120],
                "true": true_label,
                "pred": pred_label,
                "raw": result["raw_output"],
                "attack_type": sample.get("attack_type", "unknown"),
            })
        
        total += 1
        latencies.append(result["latency_ms"])
        
        class_stats[true_label]["total"] += 1
        if is_correct:
            class_stats[true_label]["correct"] += 1
        
        attack_type = sample.get("attack_type", "unknown")
        attack_stats[attack_type]["total"] += 1
        if is_correct:
            attack_stats[attack_type]["correct"] += 1
        
        confusion[(true_label, pred_label)] += 1
    
    # ─── Results ──────────────────────────────────────────────────────────
    
    accuracy = correct / total * 100
    avg_lat  = sum(latencies) / len(latencies)
    p50_lat  = sorted(latencies)[int(0.50 * len(latencies))]
    p95_lat  = sorted(latencies)[int(0.95 * len(latencies))]
    
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY: {accuracy:.1f}%  ({correct}/{total})")
    print(f"{'='*60}")
    
    print(f"\n  Latency — Avg: {avg_lat:.0f}ms | P50: {p50_lat:.0f}ms | P95: {p95_lat:.0f}ms")
    
    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        acc = s["correct"] / s["total"] * 100
        print(f"    {cls:>12s}: {acc:5.1f}%  ({s['correct']}/{s['total']})")
    
    # Confusion matrix
    labels = sorted(set(k[0] for k in confusion) | set(k[1] for k in confusion))
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "              " + "".join(f"{l:>12s}" for l in labels)
    print(header)
    for true_l in labels:
        row = f"    {true_l:>10s}"
        for pred_l in labels:
            row += f"{confusion.get((true_l, pred_l), 0):>12d}"
        print(row)
    
    # Per-attack-type accuracy
    print(f"\n  Per-Attack-Type Accuracy:")
    for atk in sorted(attack_stats.keys()):
        s = attack_stats[atk]
        acc = s["correct"] / s["total"] * 100
        print(f"    {atk:>20s}: {acc:5.1f}%  ({s['correct']}/{s['total']})")
    
    # Show some errors
    if errors:
        print(f"\n  Sample Errors (first 10):")
        for e in errors[:10]:
            print(f"    [{e['attack_type']:>10s}] true={e['true']}, pred={e['pred']}, raw='{e['raw']}'")
            print(f"              prompt: {e['prompt']}...")
    
    print(f"\n{'='*60}")
    
    # Save results
    results_path = PROJECT_ROOT / "results" / "adapter_eval.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": round(accuracy, 2),
        "total_samples": total,
        "correct": correct,
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(p50_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "per_class": {k: {"accuracy": round(v["correct"]/v["total"]*100, 2), **v} for k, v in class_stats.items()},
        "per_attack_type": {k: {"accuracy": round(v["correct"]/v["total"]*100, 2), **v} for k, v in attack_stats.items()},
        "confusion_matrix": {f"{k[0]}_as_{k[1]}": v for k, v in confusion.items()},
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
