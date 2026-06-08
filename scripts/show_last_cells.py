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

code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
print("==== SECOND TO LAST CELL ====")
if len(code_cells) >= 2:
    print("".join(code_cells[-2].get('source', [])))
print("==== LAST CELL ====")
if len(code_cells) >= 1:
    print("".join(code_cells[-1].get('source', [])))
