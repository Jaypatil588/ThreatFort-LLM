"""
QLoRA fine-tuning for the ThreatFort binary prompt classifier.

This trains Llama 3.2 3B Instruct as a sequence classifier: the model receives
only the prompt text and optimizes only the binary label.
"""

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "classifier"
DATA_DIR = Path(__file__).parent.parent / "newDataset" / "processed"

LABEL_TO_ID = {"benign": 0, "adversarial": 1}
ID_TO_LABEL = {0: "benign", 1: "adversarial"}

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

EPOCHS = 5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.03


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            missing = {"prompt", "label"} - set(row)
            if missing:
                raise KeyError(f"{path}:{line_number} missing required field(s): {sorted(missing)}")
            if row["label"] not in LABEL_TO_ID:
                raise ValueError(f"{path}:{line_number} invalid label: {row['label']!r}")
            rows.append({"prompt": row["prompt"], "label": LABEL_TO_ID[row["label"]]})
    return rows


def load_training_data(tokenizer) -> tuple[Dataset, Dataset]:
    train_raw = Dataset.from_list(load_jsonl(DATA_DIR / "train.jsonl"))
    val_raw = Dataset.from_list(load_jsonl(DATA_DIR / "val.jsonl"))

    def tokenize(batch):
        return tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    train_dataset = train_raw.map(tokenize, batched=True, remove_columns=["prompt"])
    val_dataset = val_raw.map(tokenize, batched=True, remove_columns=["prompt"])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def load_base_model():
    print("Loading sequence classifier with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def attach_lora(model):
    print("Attaching LoRA adapters for sequence classification...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        modules_to_save=["score"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def train(model, tokenizer, train_dataset, val_dataset):
    print("Starting classifier training...")
    output_dir = str(OUTPUT_DIR)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        fp16=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration / 60:.1f} minutes")

    adapter_dir = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    return trainer


def load_classifier(adapter_path: str | None = None):
    adapter_path = adapter_path or str(OUTPUT_DIR / "final_adapter")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def classify_prompt(model, tokenizer, prompt: str) -> dict:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency_ms = (time.time() - start) * 1000

    pred_id = int(outputs.logits.argmax(dim=-1).item())
    return {
        "prediction": str(pred_id),
        "prediction_id": pred_id,
        "label": ID_TO_LABEL[pred_id],
        "logits": [round(float(x), 4) for x in outputs.logits[0].detach().cpu()],
        "latency_ms": round(latency_ms, 2),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train/run adversarial prompt classifier")
    parser.add_argument("--mode", choices=["train", "classify", "eval"], default="train")
    parser.add_argument("--prompt", type=str, help="Prompt to classify (for --mode classify)")
    parser.add_argument("--adapter-path", type=str, help="Path to trained adapter")
    args = parser.parse_args()

    if args.mode == "train":
        model, tokenizer = load_base_model()
        train_dataset, val_dataset = load_training_data(tokenizer)
        model = attach_lora(model)
        train(model, tokenizer, train_dataset, val_dataset)
    elif args.mode == "classify":
        if not args.prompt:
            raise ValueError("--prompt is required for classify mode")
        model, tokenizer = load_classifier(args.adapter_path)
        result = classify_prompt(model, tokenizer, args.prompt)
        print(result["prediction"])
    elif args.mode == "eval":
        model, tokenizer = load_classifier(args.adapter_path)
        test_data = load_jsonl(DATA_DIR / "test.jsonl")

        correct = 0
        latencies = []
        from tqdm import tqdm
        for sample in tqdm(test_data, desc="Evaluating"):
            result = classify_prompt(model, tokenizer, sample["prompt"])
            if result["prediction_id"] == sample["label"]:
                correct += 1
            latencies.append(result["latency_ms"])

        accuracy = correct / len(test_data) * 100
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        print(f"\nAccuracy: {accuracy:.1f}%")
        print(f"Avg Latency: {avg_latency:.1f}ms")
        print(f"P95 Latency: {p95_latency:.1f}ms")
        print(f"Total samples: {len(test_data)}")


if __name__ == "__main__":
    main()
