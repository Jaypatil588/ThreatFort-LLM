import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_TO_ID = {"benign": 0, "adversarial": 1}
ID_TO_LABEL = {0: "benign", 1: "adversarial"}
MAX_SEQ_LENGTH = 512


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuned classifier offline on Mac")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to test")
    parser.add_argument("--adapter-dir", type=str, default="models/classifier/final_adapter", help="Path to your adapter")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this local inference script.")
    device = "mps"

    adapter_config_path = Path(args.adapter_dir) / "adapter_config.json"
    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency_ms = (time.time() - start) * 1000

    pred_id = int(outputs.logits.argmax(dim=-1).item())
    logits = [round(float(x), 4) for x in outputs.logits[0].detach().cpu()]

    print(pred_id)


if __name__ == "__main__":
    main()
