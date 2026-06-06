import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Add parent path to import evolution_db
sys.path.append(str(Path(__file__).parent))
import evolution_db as db
import pipeline as pl

load_dotenv()

app = FastAPI(
    title="ThreatFort-LLM Curation Dashboard",
    description="Human-in-the-loop active learning interface for the Llama 3.2 3B Safety Classifier"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to track background training tasks
training_state = {
    "status": "idle",  # "idle", "running", "completed", "failed"
    "started_at": None,
    "log_history": []
}

# --- Pydantic Schemas ---
class GenerateRequest(BaseModel):
    subtype: str
    prompt: Optional[str] = None
    count: Optional[int] = 5

class CurationActionRequest(BaseModel):
    status: str  # "approved" or "rejected"
    prompt_text: Optional[str] = None
    split: Optional[str] = "train"  # "train", "val", "test"

# --- Database Setup during startup ---
@app.on_event("startup")
def startup_event():
    db.init_db()

# --- Backend API Endpoints ---

@app.get("/api/stats")
def get_stats():
    """Fetch database metrics, splits overview, and run histories."""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM dataset_prompts")
        total_prompts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dataset_prompts WHERE is_anchor = TRUE")
        anchor_prompts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dataset_prompts WHERE is_anchor = FALSE")
        synthetic_prompts = cursor.fetchone()[0]
        
        # Split distribution
        cursor.execute("SELECT split, COUNT(*) FROM dataset_prompts GROUP BY split")
        split_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Label distribution
        cursor.execute("SELECT label, COUNT(*) FROM dataset_prompts GROUP BY label")
        label_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        
        runs = db.get_latest_runs(limit=10)
        
        # Get subtype breakdown of the latest run if exists
        latest_breakdown = []
        if runs:
            latest_breakdown = db.get_run_subtype_breakdown(runs[0]["run_id"])
            
        return {
            "success": True,
            "total_prompts": total_prompts,
            "anchor_prompts": anchor_prompts,
            "synthetic_prompts": synthetic_prompts,
            "splits": split_counts,
            "labels": label_counts,
            "runs": runs,
            "latest_run_breakdown": latest_breakdown,
            "training_status": training_state
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/logs")
def get_logs(error_type: Optional[str] = "slips", limit: Optional[int] = 100):
    """Fetch prediction logs and classifier slips."""
    try:
        logs = db.get_evaluation_logs(error_type=error_type, limit=limit)
        return {"success": True, "logs": logs}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/queue")
def get_queue():
    """Fetch pending prompts waiting in the curation queue."""
    try:
        queue = db.get_curation_queue(status="pending")
        return {"success": True, "queue": queue}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/generate-variants")
def generate_variants(req: GenerateRequest):
    """Call Groq to generate mutations and stage them in curation_queue."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured in env.")
        
    client = Groq(api_key=api_key)
    
    try:
        # If a seed prompt is provided, use targeted generation; else generate generally for subtype
        if req.prompt:
            print(f"Generating benign/adversarial pair variants for prompt: {req.prompt[:50]}")
            # Generate adversarial variants
            adv_prompts = pl.call_groq_generator(client, req.subtype, label="adversarial", count=req.count)
        else:
            print(f"Generating generic mutations for subtype: {req.subtype}")
            adv_prompts = pl.call_groq_generator(client, req.subtype, label="adversarial", count=req.count)
            
        staged_count = 0
        for p in adv_prompts:
            benign_counterpart = pl.call_groq_pair_balancer(client, req.subtype, p)
            if benign_counterpart:
                db.add_to_curation_queue(p, "adversarial", req.subtype, "manual_curation_dashboard")
                db.add_to_curation_queue(benign_counterpart, "benign", req.subtype, "manual_curation_dashboard")
                staged_count += 2
                
        return {"success": True, "message": f"Successfully generated and staged {staged_count} prompts."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/queue/{item_id}/action")
def curation_action(item_id: int, req: CurationActionRequest):
    """Approve or reject a prompt from the curation queue."""
    try:
        if req.status == "approved":
            # If the user edited the text in the dashboard, update it first
            if req.prompt_text:
                conn = db.get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE curation_queue SET prompt = %s WHERE id = %s",
                    (req.prompt_text, item_id)
                )
                conn.commit()
                cursor.close()
                conn.close()
                
            success = db.approve_curated_prompt(item_id, split=req.split)
            if success:
                # Sync back to filesystem JSON
                export_subtype_json_from_db(item_id)
            return {"success": success}
        elif req.status == "rejected":
            db.reject_curated_prompt(item_id)
            return {"success": True}
        else:
            raise HTTPException(status_code=400, detail="Invalid action status.")
    except Exception as e:
        return {"success": False, "error": str(e)}

def export_subtype_json_from_db(prompt_id):
    """Fetch all approved, non-anchor prompts for a specific subtype and export them back to prompt_attack_groq_data json files."""
    conn = db.get_db_connection()
    cursor = conn.cursor()
    try:
        # Get the subtype of approved prompt
        cursor.execute("SELECT attack_type FROM curation_queue WHERE id = %s", (prompt_id,))
        row = cursor.fetchone()
        if not row:
            return
        subtype = row[0]
        
        # Fetch all database prompts for this subtype that are NOT anchor
        cursor.execute(
            """
            SELECT prompt, label, attack_type, source
            FROM dataset_prompts
            WHERE attack_type = %s AND is_anchor = FALSE
            """,
            (subtype,)
        )
        records = [{"prompt": r[0], "label": r[1], "attack_type": r[2], "source": r[3]} for r in cursor.fetchall()]
        
        # Overwrite the filesystem taxonomy JSON file
        filepath = pl.TAXONOMY_DIR / f"{subtype}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
        print(f"Sync complete. Exported {len(records)} active prompts to {filepath.name}")
    except Exception as e:
        print(f"Error syncing database with JSON: {e}")
    finally:
        cursor.close()
        conn.close()

# --- Background Retraining Routine ---
def background_train_pipeline():
    global training_state
    training_state["status"] = "running"
    training_state["started_at"] = datetime.now().isoformat()
    training_state["log_history"] = []
    
    def log(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        training_state["log_history"].append(f"[{timestamp}] {msg}")
        print(msg)
        
    try:
        # 1. Prep splits
        log("Phase 1: Preparing training splits from database...")
        pl.prep_splits()
        log("Splits export completed successfully under newDataset/processed/.")
        
        # 2. Output step-by-step Colab instructions
        log("==================================================================")
        log("SUCCESS: DATASET PREPARED FOR GOOGLE COLAB!")
        log("==================================================================")
        log("Please follow these steps to train Llama 3.2 3B on Colab T4 GPU:")
        log("1. Open test.ipynb in Google Colab.")
        log("2. Set runtime type to 'T4 GPU' (Runtime > Change runtime type).")
        log("3. Run the setup and authentication cells.")
        log("4. The notebook clones the GitHub repo and copies newDataset/processed into the Colab runtime automatically.")
        log("5. Run the QLoRA training cells (includes early stopping).")
        log("6. The notebook will automatically download 'classifier_adapter.zip'.")
        log("7. Extract 'classifier_adapter.zip' into: models/classifier/final_adapter/")
        log("8. Run local evaluation to verify: python3 classifier/pipeline.py eval-adapter")
        log("==================================================================")
        
        training_state["status"] = "completed"
            
    except Exception as e:
        log(f"Fatal pipeline error: {str(e)}")
        training_state["status"] = "failed"

@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Asynchronously trigger the splits preparation and model training job."""
    global training_state
    if training_state["status"] == "running":
        return {"success": False, "message": "Training is already in progress."}
        
    background_tasks.add_task(background_train_pipeline)
    return {"success": True, "message": "Retraining loop triggered successfully in the background."}

# --- Frontend HTML Endpoint ---

@app.get("/")
def get_dashboard():
    """Serves the dashboard single page application."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThreatFort-LLM Curation Cockpit</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #070913;
            --panel-bg: rgba(15, 18, 36, 0.45);
            --border-glow: rgba(99, 102, 241, 0.15);
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --green-success: #10b981;
            --red-danger: #ef4444;
            --glass-blur: blur(16px);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(168, 53, 247, 0.15) 0px, transparent 50%);
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 24px 6%;
            border-bottom: 1px solid var(--border-glow);
            background: rgba(7, 9, 19, 0.8);
            backdrop-filter: var(--glass-blur);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        header h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tabs {
            display: flex;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px;
            border-radius: 30px;
            border: 1px solid var(--border-glow);
        }

        .tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .tab-btn.active, .tab-btn:hover {
            background: var(--accent-primary);
            color: white;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        }

        .container {
            padding: 30px 6%;
            max-width: 1600px;
            margin: 0 auto;
        }

        .view-panel {
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        .view-panel.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-glow);
            backdrop-filter: var(--glass-blur);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }

        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .card-title {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .card-value {
            font-size: 36px;
            font-weight: 800;
            font-family: 'Space Grotesk', sans-serif;
            color: white;
        }

        /* Overview Charts */
        .chart-row {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }

        /* Tables */
        .table-container {
            width: 100%;
            overflow-x: auto;
            margin-top: 15px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            padding: 14px 16px;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border-glow);
            font-size: 13px;
            text-transform: uppercase;
        }

        td {
            padding: 14px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
            font-size: 14px;
            vertical-align: middle;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.01);
        }

        .badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            display: inline-block;
        }

        .badge-adversarial {
            background: rgba(239, 68, 68, 0.15);
            color: var(--red-danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-benign {
            background: rgba(16, 185, 129, 0.15);
            color: var(--green-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        /* Staging queue styling */
        .queue-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .pair-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glow);
            border-radius: 12px;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            position: relative;
        }

        .pair-half {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .pair-half label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .pair-half textarea {
            width: 100%;
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-glow);
            border-radius: 8px;
            color: white;
            padding: 10px;
            font-family: inherit;
            font-size: 13px;
            resize: none;
            transition: border-color 0.3s ease;
        }

        .pair-half textarea:focus {
            border-color: var(--accent-primary);
            outline: none;
        }

        .pair-actions {
            grid-column: span 2;
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 15px;
        }

        /* Buttons */
        .btn {
            background: var(--accent-primary);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
            filter: brightness(1.1);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }

        .btn-danger {
            background: var(--red-danger);
        }

        .btn-danger:hover {
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        }

        /* Logs Console */
        .console {
            background: #000;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            color: var(--green-success);
            height: 300px;
            overflow-y: auto;
            margin-top: 15px;
            white-space: pre-wrap;
        }

        /* Generator Dialog */
        .dialog-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .dialog-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .dialog {
            background: var(--bg-dark);
            border: 1px solid var(--accent-primary);
            border-radius: 16px;
            width: 90%;
            max-width: 500px;
            padding: 30px;
            box-shadow: 0 0 50px rgba(99, 102, 241, 0.3);
            transform: scale(0.9);
            transition: transform 0.3s ease;
        }

        .dialog-overlay.active .dialog {
            transform: scale(1);
        }

        .dialog-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            margin-bottom: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 20px;
        }

        .form-group select, .form-group input {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glow);
            border-radius: 8px;
            color: white;
            padding: 12px;
            font-family: inherit;
        }

        .form-group select:focus, .form-group input:focus {
            border-color: var(--accent-primary);
            outline: none;
        }
    </style>
</head>
<body>

    <header>
        <h1>ThreatFort Cockpit</h1>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('overview')">Analytics</button>
            <button class="tab-btn" onclick="switchTab('logs')">Classifier Slips</button>
            <button class="tab-btn" onclick="switchTab('curation')">Curation Queue</button>
        </div>
        <button class="btn" onclick="triggerRetrain()">
            Retrain Adapter
        </button>
    </header>

    <div class="container">
        
        <!-- Tab: Overview -->
        <div id="overview" class="view-panel active">
            <div class="metrics-grid">
                <div class="card">
                    <div class="card-title">Total Dataset Size</div>
                    <div id="metric-total" class="card-value">-</div>
                </div>
                <div class="card">
                    <div class="card-title">Baseline Anchors</div>
                    <div id="metric-anchor" class="card-value">-</div>
                </div>
                <div class="card">
                    <div class="card-title">Evolved Samples</div>
                    <div id="metric-synthetic" class="card-value">-</div>
                </div>
                <div class="card">
                    <div class="card-title">Latest Run Acc</div>
                    <div id="metric-accuracy" class="card-value">-</div>
                </div>
            </div>

            <div class="chart-row">
                <div class="card" style="height: 400px;">
                    <div class="card-title">Model Evaluation History</div>
                    <canvas id="runsChart"></canvas>
                </div>
                <div class="card" style="height: 400px;">
                    <div class="card-title">Split Distribution</div>
                    <canvas id="splitChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Recent Run Evaluation Records</div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Run ID</th>
                                <th>Model Path</th>
                                <th>Test Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>Latency</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody id="runs-table-body">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Tab: Logs (Classifier Slips) -->
        <div id="logs" class="view-panel">
            <div class="card">
                <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Prompts that bypassed the classifier (Slips)</span>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Prompt Snippet</th>
                                <th>True Label</th>
                                <th>Prediction</th>
                                <th>Error Type</th>
                                <th>Latency</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="logs-table-body">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Tab: Curation Queue -->
        <div id="curation" class="view-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h2>Staged Mutation Review</h2>
                <button class="btn" onclick="openGenDialog()">
                    Generate New Variants
                </button>
            </div>
            <div id="queue-container" class="queue-grid">
                <!-- Populated dynamically with paired cards -->
            </div>
            
            <div class="card" style="margin-top: 30px;" id="training-console-panel">
                <div class="card-title">Orchestrator Pipeline Monitor</div>
                <div id="console-logs" class="console">Idle...</div>
            </div>
        </div>
    </div>

    <!-- Dialog: Generator -->
    <div class="dialog-overlay" id="gen-dialog">
        <div class="dialog">
            <h3 class="dialog-title">Generate Prompts via Groq</h3>
            <div class="form-group">
                <label>Taxonomy Subtype</label>
                <select id="gen-subtype">
                    <option value="system_override">System Override / Ignore Instructions</option>
                    <option value="developer_mode">Developer / Debug Mode Emulation</option>
                    <option value="preflight_hijack">Pre-flight Hijacking</option>
                    <option value="virtualization">Virtualization / Hypervisor Framing</option>
                    <option value="unrestricted_persona">Unrestricted Persona (DAN)</option>
                    <option value="narrative_wrapping">Narrative / Fictional Wrapping</option>
                    <option value="hypothetical_framing">Hypothetical / Counterfactuals</option>
                    <option value="roleplay_history">Roleplay / Historical</option>
                    <option value="emotional_manipulation">Emotional Manipulation</option>
                    <option value="binary_hex_base64">Binary / Hex / Base64</option>
                    <option value="cryptographic_ciphers">Cryptographic Ciphers</option>
                    <option value="low_resource_translation">Low-Resource Translation</option>
                    <option value="token_splitting">Token Splitting</option>
                    <option value="formatting_constraints">Formatting Constraints</option>
                    <option value="incremental_escalation">Incremental Escalation</option>
                    <option value="semantic_redefinition">Semantic Redefinition</option>
                    <option value="lost_in_the_middle">Lost-in-the-Middle</option>
                    <option value="invisible_css_font">Invisible CSS / Font</option>
                    <option value="instruction_poisoning">Instruction Poisoning</option>
                    <option value="gcg_optimization">GCG Optimization</option>
                    <option value="prefix_forcing">Prefix Forcing</option>
                    <option value="refusal_emulation">Refusal Emulation</option>
                    <option value="utility_paradox">Utility Paradox</option>
                </select>
            </div>
            <div class="form-group">
                <label>Number of pairs to generate</label>
                <input type="number" id="gen-count" value="5" min="1" max="25">
            </div>
            <div style="display:flex; justify-content:flex-end; gap:10px;">
                <button class="btn btn-secondary" onclick="closeGenDialog()">Cancel</button>
                <button class="btn" id="gen-btn" onclick="startGeneration()">Generate</button>
            </div>
        </div>
    </div>

    <script>
        let runsChart, splitChart;

        document.addEventListener('DOMContentLoaded', () => {
            fetchStats();
            fetchLogs();
            fetchQueue();
            setInterval(fetchStats, 5000); // Poll training status and stats
        });

        function switchTab(tabId) {
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        async function fetchStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (!data.success) return;

            document.getElementById('metric-total').innerText = data.total_prompts;
            document.getElementById('metric-anchor').innerText = data.anchor_prompts;
            document.getElementById('metric-synthetic').innerText = data.synthetic_prompts;
            
            if (data.runs && data.runs.length > 0) {
                const latestRun = data.runs[0];
                document.getElementById('metric-accuracy').innerText = (latestRun.test_accuracy * 100).toFixed(1) + '%';
            } else {
                document.getElementById('metric-accuracy').innerText = 'N/A';
            }

            renderRunsTable(data.runs);
            renderCharts(data);
            renderConsole(data.training_status);
        }

        async function fetchLogs() {
            const res = await fetch('/api/logs?error_type=slips&limit=50');
            const data = await res.json();
            if (!data.success) return;

            const tbody = document.getElementById('logs-table-body');
            tbody.innerHTML = '';
            
            data.logs.forEach(log => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${log.attack_type}</code></td>
                    <td title="${log.prompt}">${log.prompt.substring(0, 100)}...</td>
                    <td><span class="badge badge-${log.true_label}">${log.true_label}</span></td>
                    <td><span class="badge badge-${log.predicted_label}">${log.predicted_label}</span></td>
                    <td><span style="color:#ef4444">${log.error_type}</span></td>
                    <td>${Math.round(log.latency_ms)}ms</td>
                    <td>
                        <button class="btn btn-secondary" style="padding: 6px 12px; font-size:12px;" onclick="openGenDialog('${log.attack_type}', '${log.prompt}')">
                            Mutate
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function fetchQueue() {
            const res = await fetch('/api/queue');
            const data = await res.json();
            if (!data.success) return;

            const container = document.getElementById('queue-container');
            container.innerHTML = '';

            // Group by attack_type pairs
            const items = data.queue;
            const pairs = [];
            
            // Build matching pairs based on curation queue ordering (staged sequentially)
            for (let i = 0; i < items.length; i += 2) {
                if (i + 1 < items.length) {
                    pairs.push({
                        adv: items[i].label === 'adversarial' ? items[i] : items[i+1],
                        ben: items[i].label === 'benign' ? items[i] : items[i+1]
                    });
                }
            }

            if (pairs.length === 0) {
                container.innerHTML = '<div class="card" style="text-align:center; padding: 40px; color:var(--text-muted)">The curation queue is currently empty. Trigger new mutations to begin review.</div>';
                return;
            }

            pairs.forEach(pair => {
                const card = document.createElement('div');
                card.className = 'pair-card';
                card.innerHTML = `
                    <div class="pair-half">
                        <label style="color:var(--red-danger)">Adversarial Prompt (${pair.adv.attack_type})</label>
                        <textarea id="prompt-${pair.adv.id}">${pair.adv.prompt}</textarea>
                    </div>
                    <div class="pair-half">
                        <label style="color:var(--green-success)">Matching Benign Prompt</label>
                        <textarea id="prompt-${pair.ben.id}">${pair.ben.prompt}</textarea>
                    </div>
                    <div class="pair-actions">
                        <button class="btn btn-secondary" onclick="actionPair(${pair.adv.id}, ${pair.ben.id}, 'rejected')">Discard Pair</button>
                        <button class="btn" onclick="actionPair(${pair.adv.id}, ${pair.ben.id}, 'approved')">Approve & Stage</button>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        async function actionPair(advId, benId, action) {
            const advText = document.getElementById(`prompt-${advId}`).value;
            const benText = document.getElementById(`prompt-${benId}`).value;

            // Submit approval/rejection for both halves of the balanced pair
            await fetch(`/api/queue/${advId}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: action, prompt_text: advText })
            });

            await fetch(`/api/queue/${benId}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: action, prompt_text: benText })
            });

            fetchQueue();
            fetchStats();
        }

        function renderRunsTable(runs) {
            const tbody = document.getElementById('runs-table-body');
            tbody.innerHTML = '';
            if (!runs || runs.length === 0) return;

            runs.forEach(run => {
                const tr = document.createElement('tr');
                const isAdapter = run.run_id.startsWith('adapter-');
                const date = new Date(run.timestamp).toLocaleString();
                tr.innerHTML = `
                    <td><strong>${run.run_id.substring(0, 18)}...</strong></td>
                    <td title="${run.model_path}">${run.model_path.substring(0, 30)}...</td>
                    <td>${run.test_accuracy ? (run.test_accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${run.precision_score ? (run.precision_score * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${run.recall_score ? (run.recall_score * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${run.f1_score ? (run.f1_score * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${run.avg_latency_ms ? Math.round(run.avg_latency_ms) + 'ms' : 'N/A'}</td>
                    <td style="color:var(--text-muted)">${date}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderCharts(data) {
            // Split Chart
            const splitCtx = document.getElementById('splitChart').getContext('2d');
            if (splitChart) splitChart.destroy();
            
            const splits = data.splits || {};
            splitChart = new Chart(splitCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(splits).map(s => s.toUpperCase()),
                    datasets: [{
                        data: Object.values(splits),
                        backgroundColor: ['#6366f1', '#a855f7', '#10b981'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                    }
                }
            });

            // Runs Chart
            const runsCtx = document.getElementById('runsChart').getContext('2d');
            if (runsChart) runsChart.destroy();

            const runs = [...(data.runs || [])].reverse();
            runsChart = new Chart(runsCtx, {
                type: 'line',
                data: {
                    labels: runs.map(r => r.run_id.split('-').slice(2).join('-') || r.run_id.substring(0, 8)),
                    datasets: [
                        {
                            label: 'Accuracy',
                            data: runs.map(r => r.test_accuracy * 100),
                            borderColor: '#6366f1',
                            tension: 0.2
                        },
                        {
                            label: 'F1 Score',
                            data: runs.map(r => r.f1_score * 100),
                            borderColor: '#a855f7',
                            tension: 0.2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#94a3b8' } }
                    }
                }
            });
        }

        function renderConsole(status) {
            const consoleBox = document.getElementById('console-logs');
            if (status.status === 'running') {
                consoleBox.innerText = status.log_history.join('\\n');
                consoleBox.scrollTop = consoleBox.scrollHeight;
            } else if (status.status === 'completed') {
                consoleBox.innerText = 'Retraining completed successfully! New adapter loaded.';
            } else if (status.status === 'failed') {
                consoleBox.innerText = 'Training job failed. Check console logs for debugging.';
            } else {
                consoleBox.innerText = 'Idle. Ready to retrain.';
            }
        }

        function openGenDialog(subtype = 'system_override', prompt = '') {
            document.getElementById('gen-subtype').value = subtype;
            document.getElementById('gen-dialog').classList.add('active');
        }

        function closeGenDialog() {
            document.getElementById('gen-dialog').classList.remove('active');
        }

        async function startGeneration() {
            const subtype = document.getElementById('gen-subtype').value;
            const count = parseInt(document.getElementById('gen-count').value);
            
            const btn = document.getElementById('gen-btn');
            btn.innerText = 'Generating...';
            btn.disabled = true;

            const res = await fetch('/api/generate-variants', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subtype, count })
            });

            const data = await res.json();
            btn.innerText = 'Generate';
            btn.disabled = false;
            
            closeGenDialog();
            if (data.success) {
                switchTab('curation');
                fetchQueue();
            } else {
                alert('Generation failed: ' + data.error);
            }
        }

        async function triggerRetrain() {
            if (!confirm('Start Splits Rebuilding and QLoRA Fine-tuning? This runs asynchronously.')) return;
            const res = await fetch('/api/retrain', { method: 'POST' });
            const data = await res.json();
            alert(data.message);
            fetchStats();
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
