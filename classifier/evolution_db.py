import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Database Config from Env or defaults
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "threatfort")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

def get_db_connection(connect_to_default=False):
    """Establish a connection to the PostgreSQL database."""
    dbname = "postgres" if connect_to_default else DB_NAME
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD
    )

def init_db():
    """Ensure database and schema tables are initialized."""
    # 1. Check if database exists, create if not
    conn = get_db_connection(connect_to_default=True)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
    exists = cursor.fetchone()
    if not exists:
        print(f"Creating database '{DB_NAME}'...")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
    cursor.close()
    conn.close()

    # 2. Connect to database and create tables
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # Table: dataset_prompts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_prompts (
            id SERIAL PRIMARY KEY,
            prompt TEXT NOT NULL,
            label VARCHAR(20) NOT NULL,
            attack_type VARCHAR(50) NOT NULL,
            source VARCHAR(100) NOT NULL,
            split VARCHAR(10) NOT NULL,
            is_anchor BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_prompt UNIQUE (prompt)
        )
    """)

    # Table: evaluation_runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id VARCHAR(50) PRIMARY KEY,
            model_path VARCHAR(255) NOT NULL,
            test_accuracy FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            avg_latency_ms FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table: evaluation_logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_logs (
            id SERIAL PRIMARY KEY,
            prompt TEXT NOT NULL,
            true_label VARCHAR(20) NOT NULL,
            predicted_label VARCHAR(20) NOT NULL,
            error_type VARCHAR(20), -- 'false_positive', 'false_negative', or NULL
            attack_type VARCHAR(50) NOT NULL,
            latency_ms FLOAT,
            run_id VARCHAR(50) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table: curation_queue
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS curation_queue (
            id SERIAL PRIMARY KEY,
            prompt TEXT NOT NULL,
            label VARCHAR(20) NOT NULL,
            attack_type VARCHAR(50) NOT NULL,
            source VARCHAR(100) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_queue_prompt UNIQUE (prompt)
        )
    """)

    cursor.close()
    conn.close()
    print("PostgreSQL tables initialized successfully.")

# --- Prompts Operations ---

def add_prompt(prompt, label, attack_type, source, split, is_anchor):
    """Add a prompt to the training/eval dataset."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO dataset_prompts (prompt, label, attack_type, source, split, is_anchor)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (prompt) DO NOTHING
            """,
            (prompt, label, attack_type, source, split, is_anchor)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def clear_dataset_prompts(only_non_anchor=False):
    """Clear prompts dataset."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if only_non_anchor:
            cursor.execute("DELETE FROM dataset_prompts WHERE is_anchor = FALSE")
        else:
            cursor.execute("TRUNCATE TABLE dataset_prompts RESTART IDENTITY CASCADE")
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_dataset_prompts(split=None):
    """Fetch all dataset prompts."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        if split:
            cursor.execute("SELECT * FROM dataset_prompts WHERE split = %s", (split,))
        else:
            cursor.execute("SELECT * FROM dataset_prompts")
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

# --- Evaluation Runs Operations ---

def create_evaluation_run(run_id, model_path, test_accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, avg_latency=0.0):
    """Create an evaluation run record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO evaluation_runs (run_id, model_path, test_accuracy, precision_score, recall_score, f1_score, avg_latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                test_accuracy = EXCLUDED.test_accuracy,
                precision_score = EXCLUDED.precision_score,
                recall_score = EXCLUDED.recall_score,
                f1_score = EXCLUDED.f1_score,
                avg_latency_ms = EXCLUDED.avg_latency_ms
            """,
            (run_id, model_path, test_accuracy, precision, recall, f1, avg_latency)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def log_evaluation(prompt, true_label, predicted_label, error_type, attack_type, latency_ms, run_id):
    """Log an individual prediction to evaluation logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO evaluation_logs (prompt, true_label, predicted_label, error_type, attack_type, latency_ms, run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (prompt, true_label, predicted_label, error_type, attack_type, latency_ms, run_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_latest_runs(limit=10):
    """Fetch latest runs metadata."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT * FROM evaluation_runs ORDER BY timestamp DESC LIMIT %s", (limit,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def get_evaluation_logs(run_id=None, error_type=None, limit=250):
    """Fetch individual prediction logs."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        query = "SELECT * FROM evaluation_logs WHERE 1=1"
        params = []
        if run_id:
            query += " AND run_id = %s"
            params.append(run_id)
        if error_type:
            if error_type == "slips":
                query += " AND error_type IS NOT NULL"
            else:
                query += " AND error_type = %s"
                params.append(error_type)
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        cursor.execute(query, tuple(params))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

# --- Curation Queue Operations ---

def add_to_curation_queue(prompt, label, attack_type, source):
    """Add a generated prompt to the curation queue for review."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO curation_queue (prompt, label, attack_type, source, status)
            VALUES (%s, %s, %s, %s, 'pending')
            ON CONFLICT (prompt) DO NOTHING
            """,
            (prompt, label, attack_type, source)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_curation_queue(status='pending'):
    """Fetch staged prompts in the queue."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT * FROM curation_queue WHERE status = %s ORDER BY created_at DESC", (status,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def update_curation_status(prompt_id, status):
    """Approve or reject a prompt in the queue."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE curation_queue SET status = %s WHERE id = %s",
            (status, prompt_id)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def approve_curated_prompt(prompt_id, split='train'):
    """Approve a prompt: move it from curation_queue to dataset_prompts and mark status."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # Get details
        cursor.execute("SELECT * FROM curation_queue WHERE id = %s", (prompt_id,))
        item = cursor.fetchone()
        if not item:
            return False
        
        # Insert into dataset_prompts as non-anchor
        cursor.execute(
            """
            INSERT INTO dataset_prompts (prompt, label, attack_type, source, split, is_anchor)
            VALUES (%s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (prompt) DO UPDATE SET split = EXCLUDED.split
            """,
            (item['prompt'], item['label'], item['attack_type'], item['source'], split)
        )
        
        # Mark as approved in queue
        cursor.execute("UPDATE curation_queue SET status = 'approved' WHERE id = %s", (prompt_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def reject_curated_prompt(prompt_id):
    """Mark a prompt as rejected in the curation queue."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE curation_queue SET status = 'rejected' WHERE id = %s", (prompt_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

# --- Aggregates and Analytics ---

def get_high_error_subtypes(run_id=None, limit=5):
    """Identify which sub-attack types have the highest count of classification slips."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        query = """
            SELECT attack_type, COUNT(*) as error_count
            FROM evaluation_logs
            WHERE error_type IS NOT NULL
        """
        params = []
        if run_id:
            query += " AND run_id = %s"
            params.append(run_id)
        query += " GROUP BY attack_type ORDER BY error_count DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def get_run_subtype_breakdown(run_id):
    """Get attack success rate (ASR) per subtype for a specific run."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(
            """
            SELECT 
                attack_type,
                COUNT(*) as total_count,
                SUM(CASE WHEN error_type = 'false_negative' THEN 1 ELSE 0 END) as slips_count,
                ROUND(100.0 * SUM(CASE WHEN error_type = 'false_negative' THEN 0 ELSE 1 END) / COUNT(*), 2) as accuracy
            FROM evaluation_logs
            WHERE run_id = %s AND true_label = 'adversarial'
            GROUP BY attack_type
            """,
            (run_id,)
        )
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
