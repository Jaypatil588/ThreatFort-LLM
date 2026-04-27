import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def format_prompt(text: str) -> str:
    return f"System: You are a security classifier. Answer with just 'adversarial' or 'benign'.\nUser: {text}\nAssistant:"

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuned classifier offline on Mac")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to test")
    parser.add_argument("--adapter-dir", type=str, default="models/classifier/final_adapter", help="Path to your unzipped Colab adapter")
    args = parser.parse_args()

    # Determine backend (Apple Silicon MPS, or fallback to CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading using device: {device}...")

    # Load 1B Base Model (Using bfloat16 instead of 4-bit because bitsandbytes doesn't work on Mac locally)
    # 1B model is tiny (~2.5GB) so it easily fits in normal Mac RAM!
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # We load full 16-bit precision
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device}  # Bind to Mac Silicon
        )
    except Exception as e:
        print(f"Failed to load base model. Make sure you accepted the Llama license online and ran `huggingface-cli login` on your Mac. Error: {e}")
        return

    # Attach your trained adapter
    print(f"Attaching your custom adapter from {args.adapter_dir}...")
    try:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    except Exception as e:
        print(f"Could not load adapter! Did you unzip your Colab download into {args.adapter_dir}? Error: {e}")
        return
        
    print("\nModel loaded successfully! Running inference...\n")

    formatted = format_prompt(args.prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5,     # We only need 1 word ('adversarial' or 'benign')
            temperature=0.0,      # Deterministic 
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract just the answer
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result.split("Assistant:")[-1].strip().lower()

    print("="*50)
    print(f"Prompt: '{args.prompt}'")
    print(f"Classification: [ {answer.upper()} ]")
    print("="*50)

if __name__ == "__main__":
    main()
