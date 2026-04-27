import json

path = "/Users/senpai/Desktop/projects/robustbench/notebooks/train_classifier.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "torch==2.4.0" in source:
            source = source.replace("torch==2.4.0 ", "torch ")
            cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in source.split("\n")]
            cell["source"] = [s[:-1] if i == len(cell["source"])-1 and s.endswith("\n") else s for i, s in enumerate(cell["source"])]

with open(path, "w") as f:
    json.dump(nb, f, indent=2)

print("Removed torch==2.4.0 from notebook to prevent torchvision crash.")
