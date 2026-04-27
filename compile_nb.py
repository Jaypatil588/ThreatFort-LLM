import json

path = "/Users/senpai/Desktop/projects/robustbench/notebooks/train_classifier.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

code = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        block = "".join(cell["source"])
        # Strip magics and GUI uploads
        if "!pip" in block: continue
        block = block.replace("from google.colab import files", "files = type('Mock', (), {'download': lambda x: None})()")
        block = block.replace("login()", "")
        block = block.replace("files.upload()", "uploaded = {}")
        if "torch.cuda" in block:
            block = block.replace("print(f\"GPU: {torch.cuda.get_device_name(0)}\")", "")
            block = block.replace("print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")", "")
        
        # 1. Downgrade model to an open un-gated CPU-friendly model
        block = block.replace("'meta-llama/Llama-3.2-1B-Instruct'", "'facebook/opt-125m'")
        
        # 2. Rip out CUDA 4-bit precision 
        if "BitsAndBytesConfig" in block:
            import re
            block = re.sub(r'bnb_config\s*=\s*BitsAndBytesConfig\([^)]+\)', 'bnb_config = None', block)
            block = block.replace("from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig", "from transformers import AutoModelForCausalLM, AutoTokenizer")
            block = block.replace("quantization_config=bnb_config,", "torch_dtype=torch.float16,")
        
        # 3. Prevent SFTTrainer from starting an hour-long loop, just verify init
        if "trainer.train()" in block:
            block = block.replace("trainer.train()", "print('\\n=== HIT TRAINER.TRAIN() BOUNDARY WITH NO CRASHES ===')\nexit(0)")
            
        code.append(block)

full_script = "\n".join(code)

with open("/Users/senpai/Desktop/projects/robustbench/raw_nb_test.py", "w") as f:
    f.write(full_script)
