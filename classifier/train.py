"""
QLoRA Fine-Tuning Script for Adversarial Prompt Classifier

This script fine-tunes Llama 3.2 3B Instruct to classify prompts as adversarial or benign.
Uses QLoRA (4-bit quantization + Low-Rank Adaptation) to make it trainable on a
single consumer GPU.

Can run locally if you have a GPU, or use the accompanying Colab notebook.

Why Llama 3.2 3B:
  - Matches the Google Colab training notebook and local evaluation pipeline.
  - 4-bit QLoRA keeps the VRAM footprint practical on a Colab T4.
  - The adapter must be evaluated against the same base model it was trained on.
  - Still uses QLoRA, same technique, same resume story

Why QLoRA works here:
  - Full fine-tuning the base model is expensive; QLoRA freezes it in 4-bit and trains adapters.
  - QLoRA freezes the base model in 4-bit and only trains ~0.1% of params via LoRA
  - Quality loss from quantization is minimal for classification tasks
"""

import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "classifier"
DATA_DIR = Path(__file__).parent.parent / "newDataset" / "processed"

# LoRA hyperparameters
LORA_R = 16         # Rank of adapter matrices (higher = more params = better but slower)
LORA_ALPHA = 32     # Scaling factor (typically 2x rank)
LORA_DROPOUT = 0.05 # Dropout for regularization
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj",       # MLP layers
]

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2  # Effective batch size = 8 * 2 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.03


# ─── Data Formatting ─────────────────────────────────────────────────────────

CLASSIFY_PROMPT_TEMPLATE = (
    "System: You are a security classifier. "
    "Answer with just 'adversarial' or 'benign'.\n"
    "User: {prompt}\n"
    "Assistant: {label}"
)

CLASSIFY_INFERENCE_TEMPLATE = (
    "System: You are a security classifier. "
    "Answer with just 'adversarial' or 'benign'.\n"
    "User: {prompt}\n"
    "Assistant:"
)


def load_training_data() -> tuple[Dataset, Dataset]:
    """Load and format training data for SFT."""
    
    def load_jsonl(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_raw = load_jsonl(DATA_DIR / "train.jsonl")
    val_raw = load_jsonl(DATA_DIR / "val.jsonl")
    
    def format_samples(samples):
        formatted = []
        for s in samples:
            text = CLASSIFY_PROMPT_TEMPLATE.format(
                prompt=s["prompt"][:800],  # Truncate long prompts
                label=s["label"]
            )
            formatted.append({"text": text})
        return formatted
    
    train_formatted = format_samples(train_raw)
    val_formatted = format_samples(val_raw)
    
    print(f"Training samples: {len(train_formatted)}")
    print(f"Validation samples: {len(val_formatted)}")
    
    return Dataset.from_list(train_formatted), Dataset.from_list(val_formatted)


# ─── Model Setup ─────────────────────────────────────────────────────────────

def load_base_model():
    """Load the base model with 4-bit quantization (the Q in QLoRA)."""
    
    print("Loading base model with 4-bit quantization...")
    
    # BitsAndBytesConfig is the quantization config
    # This compresses each weight from 16-bit float → 4-bit integer
    # Reduces memory from ~16GB → ~4GB, with minimal quality loss
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NormalFloat4 - best 4-bit format
        bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16 for speed
        bnb_4bit_use_double_quant=True,    # Quantize the quantization constants too
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically places layers on available GPUs
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def attach_lora(model):
    """Attach LoRA adapters to the model (the LoRA in QLoRA)."""
    
    print("Attaching LoRA adapters...")
    
    # LoRA works by adding small trainable matrices to frozen layers:
    # Instead of: output = W * input  (W is frozen base-model params)
    # We do:      output = W * input + (A @ B) * input  (A,B are tiny trainable matrices)
    # If W is 2048x2048 and rank=16, then A is 2048x16 and B is 16x2048
    # That's ~65K params instead of 4.1M per layer -- ~100x reduction
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameter count
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_dataset, val_dataset):
    """Run QLoRA fine-tuning."""
    
    print("Starting training...")
    output_dir = str(OUTPUT_DIR)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
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
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",  # Set to "wandb" if you want experiment tracking
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    
    print(f"\nTraining completed in {duration/60:.1f} minutes")
    
    # Save the LoRA adapter weights (small — typically ~50MB)
    adapter_dir = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    
    print(f"Adapter saved to {adapter_dir}")
    
    return trainer


# ─── Inference ───────────────────────────────────────────────────────────────

def load_classifier(adapter_path: str = None):
    """Load the fine-tuned classifier for inference."""
    
    adapter_path = adapter_path or str(OUTPUT_DIR / "final_adapter")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    model.eval()
    return model, tokenizer


def classify_prompt(model, tokenizer, prompt: str) -> dict:
    """
    Classify a single prompt as adversarial or benign.
    Returns the classification and latency.
    """
    # Format the input
    input_text = CLASSIFY_INFERENCE_TEMPLATE.format(prompt=prompt[:800])
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Time the inference
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # Only need "adversarial" or "benign" — a few tokens
            do_sample=False,
            temperature=1.0,
        )
    latency_ms = (time.time() - start) * 1000
    
    # Decode only the new tokens (the classification label)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    
    # Normalize prediction
    if "adversarial" in prediction:
        label = "adversarial"
    elif "benign" in prediction:
        label = "benign"
    else:
        label = "unknown"
    
    return {
        "prediction": label,
        "raw_output": prediction,
        "latency_ms": round(latency_ms, 2),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train/run adversarial prompt classifier")
    parser.add_argument("--mode", choices=["train", "classify", "eval"], default="train")
    parser.add_argument("--prompt", type=str, help="Prompt to classify (for --mode classify)")
    parser.add_argument("--adapter-path", type=str, help="Path to trained adapter")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_dataset, val_dataset = load_training_data()
        model, tokenizer = load_base_model()
        model = attach_lora(model)
        train(model, tokenizer, train_dataset, val_dataset)
        
    elif args.mode == "classify":
        if not args.prompt:
            print("Error: --prompt required for classify mode")
            return
        model, tokenizer = load_classifier(args.adapter_path)
        result = classify_prompt(model, tokenizer, args.prompt)
        print(f"Classification: {result['prediction']}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        
    elif args.mode == "eval":
        # Evaluate on test set
        model, tokenizer = load_classifier(args.adapter_path)
        
        test_data = []
        with open(DATA_DIR / "test.jsonl") as f:
            for line in f:
                test_data.append(json.loads(line))
        
        correct = 0
        total = 0
        latencies = []
        
        from tqdm import tqdm
        for sample in tqdm(test_data, desc="Evaluating"):
            result = classify_prompt(model, tokenizer, sample["prompt"])
            if result["prediction"] == sample["label"]:
                correct += 1
            total += 1
            latencies.append(result["latency_ms"])
        
        accuracy = correct / total * 100
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        
        print(f"\n{'='*40}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Avg Latency: {avg_latency:.1f}ms")
        print(f"P95 Latency: {p95_latency:.1f}ms")
        print(f"Total samples: {total}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
