import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
default_notebook = os.path.normpath(os.path.join(script_dir, '../baseline_evaluation.ipynb'))
notebook_path = sys.argv[1] if len(sys.argv) > 1 else default_notebook

if not os.path.exists(notebook_path):
    raise FileNotFoundError(f"Notebook file not found at: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'artifact_paths' in source or 'print(' in source:
            print(f"==== CELL {i} ====")
            print(source)
