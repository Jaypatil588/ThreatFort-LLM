"""
Evaluate the QLoRA sequence-classification adapter on the held-out test set.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set in .env; gated model access may fail")

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTER_PATH = PROJECT_ROOT / "models" / "classifier" / "final_adapter"
TEST_PATH = PROJECT_ROOT / "newDataset" / "processed" / "test.jsonl"
MAX_SEQ_LENGTH = 512

LABEL_TO_ID = {"benign": 0, "adversarial": 1}
ID_TO_LABEL = {0: "benign", 1: "adversarial"}


def load_model(adapter_path: str):
    with open(Path(adapter_path) / "adapter_config.json", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading sequence classifier base model: {base_model_id}")
    print(f"Device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        torch_dtype=torch.float32,
        device_map={"": device},
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer, device


def classify(model, tokenizer, device, prompt: str) -> dict:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency_ms = (time.time() - start) * 1000

    pred_id = int(outputs.logits.argmax(dim=-1).item())
    return {
        "prediction": ID_TO_LABEL[pred_id],
        "raw_output": [round(float(x), 4) for x in outputs.logits[0].detach().cpu()],
        "latency_ms": latency_ms,
    }


def load_test_data(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            missing = {"prompt", "label"} - set(row)
            if missing:
                raise KeyError(f"{path}:{line_number} missing required field(s): {sorted(missing)}")
            if row["label"] not in LABEL_TO_ID:
                raise ValueError(f"{path}:{line_number} invalid label: {row['label']!r}")
            rows.append(row)
    return rows


def main():
    model, tokenizer, device = load_model(str(ADAPTER_PATH))
    test_data = load_test_data(TEST_PATH)
    print(f"\nTest samples: {len(test_data)}")

    correct = 0
    latencies = []
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    attack_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(int)
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

        latencies.append(result["latency_ms"])
        class_stats[true_label]["total"] += 1
        attack_type = sample.get("attack_type", "unknown")
        attack_stats[attack_type]["total"] += 1
        if is_correct:
            class_stats[true_label]["correct"] += 1
            attack_stats[attack_type]["correct"] += 1
        confusion[(true_label, pred_label)] += 1

    total = len(test_data)
    accuracy = correct / total * 100
    avg_lat = sum(latencies) / len(latencies)
    p50_lat = sorted(latencies)[int(0.50 * len(latencies))]
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]

    print(f"\n{'=' * 60}")
    print(f"  OVERALL ACCURACY: {accuracy:.1f}%  ({correct}/{total})")
    print(f"{'=' * 60}")
    print(f"\n  Latency - Avg: {avg_lat:.0f}ms | P50: {p50_lat:.0f}ms | P95: {p95_lat:.0f}ms")

    print("\n  Per-Class Accuracy:")
    for cls in sorted(class_stats):
        s = class_stats[cls]
        print(f"    {cls:>12s}: {s['correct'] / s['total'] * 100:5.1f}%  ({s['correct']}/{s['total']})")

    print("\n  Per-Attack-Type Accuracy:")
    for atk in sorted(attack_stats):
        s = attack_stats[atk]
        print(f"    {atk:>24s}: {s['correct'] / s['total'] * 100:5.1f}%  ({s['correct']}/{s['total']})")

    if errors:
        print("\n  Sample Errors (first 10):")
        for e in errors[:10]:
            print(f"    [{e['attack_type']:>10s}] true={e['true']}, pred={e['pred']}, logits={e['raw']}")
            print(f"              prompt: {e['prompt']}...")

    results_path = PROJECT_ROOT / "results" / "adapter_eval.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": round(accuracy, 2),
        "total_samples": total,
        "correct": correct,
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(p50_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "per_class": {k: {"accuracy": round(v["correct"] / v["total"] * 100, 2), **v} for k, v in class_stats.items()},
        "per_attack_type": {k: {"accuracy": round(v["correct"] / v["total"] * 100, 2), **v} for k, v in attack_stats.items()},
        "confusion_matrix": {f"{k[0]}_as_{k[1]}": v for k, v in confusion.items()},
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
