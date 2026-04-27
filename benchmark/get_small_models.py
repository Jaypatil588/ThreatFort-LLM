import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def list_small_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("Please set GROQ_API_KEY in your .env file first!")
        return

    client = Groq(api_key=api_key)
    try:
        models = client.models.list().data
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return
        
    print("\nGroq Models with <= 8B parameters:")
    print("-" * 50)
    
    small_models = {}
    for model in models:
        model_id = model.id.lower()
        # Look for 8b, 7b, 3b, 1b, etc in the ID
        if any(size in model_id for size in ["8b", "7b", "3b", "1b", "mini"]):
            if "whisper" not in model_id and "embed" not in model_id and "guard" not in model_id:
                sizes = [s for s in ["8b", "7b", "3b", "1b", "mini"] if s in model_id]
                if sizes:
                    name = model_id.split("-")[0] + "-" + sizes[0]
                    small_models[name] = model.id
                    print(f"Found: {model.id}")
                
    print("\nUpdated Dictionary for evaluate.py:")
    print("GROQ_MODELS = {")
    for k, v in small_models.items():
        print(f'    "{k}": "{v}",')
    print("}")

if __name__ == "__main__":
    list_small_groq_models()
