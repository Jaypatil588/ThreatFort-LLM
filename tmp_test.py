import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

print("Imported successfully. Creating dummy dataset...")
dummy_dataset = Dataset.from_list([{"text": "System: Security classifier.\nUser: Hello\nAssistant: benign"}])

print("Loading dummy model...")
MODEL_ID = "facebook/opt-125m" # small accessible model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

print("Attaching LoRA...")
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'], # safe proxy for target_modules
    lora_dropout=0.05, 
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)

print("Preparing SFTConfig...")
training_args = SFTConfig(
    output_dir='./classifier_output',
    max_seq_length=512,
    num_train_epochs=3, 
    per_device_train_batch_size=16, 
    learning_rate=2e-4,
    fp16=False, # cpu test
    dataset_text_field='text', 
    report_to='none', 
    save_strategy='epoch', 
    logging_steps=10
)

print("Initializing SFTTrainer...")
try:
    trainer = SFTTrainer(
        model=model, 
        args=training_args, 
        train_dataset=dummy_dataset, 
        eval_dataset=dummy_dataset, 
        processing_class=tokenizer            
    )
    print("SUCCESS: SFTTrainer successfully initialized without any syntax/argument errors!")
except Exception as e:
    import traceback
    traceback.print_exc()
