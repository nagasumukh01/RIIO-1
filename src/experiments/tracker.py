"""
Experiment tracking with SQLite backend.
Records all experiment metadata, configurations, and results.
"""

import os
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.utils.reproducibility import get_hardware_info

logger = get_logger("experiment_tracker")


class ExperimentTracker:
    """
    SQLite-based experiment tracker.
    Records complete experiment metadata for reproducibility.
    """

    def __init__(self, db_path: str = "artifacts/experiment_logs/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            status TEXT DEFAULT 'running',
            model_name TEXT,
            model_type TEXT,
            source_dataset TEXT,
            target_dataset TEXT,
            config TEXT,
            seed INTEGER,
            device TEXT,
            hardware_info TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_s REAL,
            checkpoint_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT,
            metric_name TEXT,
            metric_value REAL,
            dataset TEXT,
            split TEXT,
            epoch INTEGER,
            timestamp TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT,
            sample_path TEXT,
            video_id TEXT,
            frame_idx INTEGER,
            true_label INTEGER,
            predicted_label INTEGER,
            score REAL,
            dataset TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
        """)

        conn.commit()
        conn.close()

    def create_experiment(
        self,
        name: str,
        exp_type: str,
        model_name: str,
        model_type: str,
        source_dataset: str,
        target_dataset: Optional[str] = None,
        config: Optional[Dict] = None,
        seed: int = 42,
        device: str = "cpu",
    ) -> str:
        """Create a new experiment and return its ID."""
        exp_id = str(uuid.uuid4())[:8] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        hw_info = {}
        try:
            hw_info = get_hardware_info()
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO experiments
        (id, name, type, model_name, model_type, source_dataset, target_dataset,
         config, seed, device, hardware_info, start_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id, name, exp_type, model_name, model_type,
            source_dataset, target_dataset,
            json.dumps(config or {}), seed, device,
            json.dumps(hw_info), datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

        logger.info(f"Created experiment: {exp_id} ({name})")
        return exp_id

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        dataset: str = "test",
        split: str = "test",
        epoch: Optional[int] = None,
    ):
        """Log metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value,
                                   dataset, split, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, name, float(value), dataset, split, epoch, timestamp))

        conn.commit()
        conn.close()

    def complete_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        checkpoint_path: Optional[str] = None,
        duration_s: Optional[float] = None,
    ):
        """Mark an experiment as completed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE experiments
        SET status = ?, end_time = ?, checkpoint_path = ?, duration_s = ?
        WHERE id = ?
        """, (status, datetime.now().isoformat(), checkpoint_path, duration_s, experiment_id))
        conn.commit()
        conn.close()

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def get_all_experiments(self, exp_type: Optional[str] = None) -> List[Dict]:
        """Get all experiments, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if exp_type:
            cursor.execute(
                "SELECT * FROM experiments WHERE type = ? ORDER BY created_at DESC",
                (exp_type,)
            )
        else:
            cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_metrics(
        self,
        experiment_id: str,
        dataset: Optional[str] = None,
    ) -> List[Dict]:
        """Get metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if dataset:
            cursor.execute(
                "SELECT * FROM metrics WHERE experiment_id = ? AND dataset = ?",
                (experiment_id, dataset)
            )
        else:
            cursor.execute(
                "SELECT * FROM metrics WHERE experiment_id = ?",
                (experiment_id,)
            )

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_comparison_table(
        self,
        metric_names: List[str] = None,
        exp_type: Optional[str] = None,
    ) -> List[Dict]:
        """Get a comparison table of all experiments with key metrics."""
        if metric_names is None:
            metric_names = ['accuracy', 'auroc', 'f1_score', 'eer']

        experiments = self.get_all_experiments(exp_type)
        comparison = []

        for exp in experiments:
            row = {
                'id': exp['id'],
                'name': exp['name'],
                'model': exp['model_name'],
                'type': exp['model_type'],
                'source': exp['source_dataset'],
                'target': exp['target_dataset'],
                'status': exp['status'],
                'duration_s': exp['duration_s'],
            }

            metrics = self.get_metrics(exp['id'])
            for m in metrics:
                if m['metric_name'] in metric_names:
                    row[m['metric_name']] = m['metric_value']

            comparison.append(row)

        return comparison

    def export_results(self, output_dir: str = "artifacts/experiment_logs"):
        """Export all results to CSV and JSON."""
        import csv

        os.makedirs(output_dir, exist_ok=True)

        # Export experiments
        experiments = self.get_all_experiments()
        if experiments:
            with open(os.path.join(output_dir, 'experiments.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=experiments[0].keys())
                writer.writeheader()
                writer.writerows(experiments)

        with open(os.path.join(output_dir, 'experiments.json'), 'w') as f:
            json.dump(experiments, f, indent=2)

        # Export comparison table
        comparison = self.get_comparison_table()
        with open(os.path.join(output_dir, 'comparison_table.json'), 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Exported results to {output_dir}")
