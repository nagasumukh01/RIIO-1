"""Export metrics from persisted RobustDeep evaluation artifacts only."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "experiment_logs"
OUTPUT_DIR = ARTIFACTS_DIR / "results_export"


def _read_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, (dict, list)):
        return None, "Unsupported JSON root; expected object or array."
    return value, None


def _is_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_fields(values: dict[str, Any]) -> dict[str, int | float]:
    return {key: value for key, value in values.items() if _is_metric(value)}


def _source_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def collect_model_evaluations(artifacts_dir: Path = ARTIFACTS_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for path in sorted((artifacts_dir / "models").glob("*/training_summary.json")):
        summary, error = _read_json(path)
        if error:
            errors.append(f"{path}: {error}")
            continue
        if not isinstance(summary, dict) or summary.get("is_synthetic") is not False:
            continue
        metrics = summary.get("test_metrics")
        if not isinstance(metrics, dict) or metrics.get("split", "test") != "test":
            continue
        metrics = _metric_fields(metrics)
        if not metrics:
            continue
        rows.append({
            "model_id": summary.get("experiment_id") or path.parent.name,
            "dataset": summary.get("dataset"),
            "source": _source_path(path, artifacts_dir.parent),
            "metrics": metrics,
        })
    return rows, errors


def collect_cross_dataset_evaluations(
    logs_dir: Path = LOGS_DIR,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for path in sorted(logs_dir.rglob("cross_dataset_matrix*.json")):
        artifact, error = _read_json(path)
        if error:
            errors.append(f"{path}: {error}")
            continue
        results = artifact.get("results") if isinstance(artifact, dict) else None
        if not isinstance(results, dict):
            continue
        for model_id, model in results.items():
            if not isinstance(model, dict) or not isinstance(model.get("test_results"), dict):
                continue
            for test_dataset, values in model["test_results"].items():
                if not isinstance(values, dict):
                    continue
                metrics = _metric_fields(values)
                if metrics:
                    rows.append({
                        "model_id": model_id,
                        "train_dataset": model.get("train_dataset"),
                        "test_dataset": test_dataset,
                        "source": _source_path(path, logs_dir.parent.parent),
                        "metrics": metrics,
                    })
    return rows, errors


def collect_robustness_evaluations(
    logs_dir: Path = LOGS_DIR,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for path in sorted(logs_dir.glob("robustness*.json")):
        artifact, error = _read_json(path)
        if error:
            errors.append(f"{path}: {error}")
            continue
        if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
            continue
        for result in artifact["rows"]:
            if not isinstance(result, dict):
                continue
            if result.get("is_synthetic") is not False or result.get("split") != "test":
                continue
            metrics = _metric_fields(result)
            for key in ("dataset", "split", "stress", "checkpoint", "is_synthetic"):
                metrics.pop(key, None)
            if metrics:
                rows.append({
                    "dataset": result.get("dataset"),
                    "stress": result.get("stress"),
                    "source": _source_path(path, logs_dir.parent.parent),
                    "metrics": metrics,
                })
    return rows, errors


def _binary_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if value in (0, "REAL"):
        return "REAL"
    if value in (1, "FAKE"):
        return "FAKE"
    return None


def collect_diffusion_evaluation(path: Path = LOGS_DIR / "diffusion_test_eval.json") -> tuple[dict[str, Any] | None, str | None]:
    artifact, error = _read_json(path)
    if error:
        return None, error
    if not isinstance(artifact, dict) or not isinstance(artifact.get("per_sample_results"), list):
        return None, "Unsupported artifact: per-sample labels and predictions are required."

    sample_rows = artifact["per_sample_results"]
    if not sample_rows:
        return None, "No persisted per-sample predictions are available."

    ensemble_correct = 0
    model_counts: dict[str, list[int]] = {}
    for row in sample_rows:
        if not isinstance(row, dict):
            return None, "Unsupported artifact: malformed per-sample row."
        truth = _binary_label(row.get("ground_truth"))
        prediction = _binary_label(row.get("ensemble_verdict"))
        if truth is None or prediction is None:
            return None, "Unsupported artifact: every sample needs a valid ground truth and ensemble verdict."
        ensemble_correct += int(truth == prediction)

        model_results = row.get("models")
        if not isinstance(model_results, dict):
            return None, "Unsupported artifact: per-sample model predictions are missing."
        for model_id, model_row in model_results.items():
            if not isinstance(model_row, dict):
                continue
            model_prediction = _binary_label(model_row.get("pred"))
            if model_prediction is None:
                continue
            counts = model_counts.setdefault(model_id, [0, 0])
            counts[0] += int(model_prediction == truth)
            counts[1] += 1

    model_rows = [
        {
            "model_id": model_id,
            "source": _source_path(path, PROJECT_ROOT),
            "metrics": {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            },
        }
        for model_id, (correct, total) in sorted(model_counts.items())
        if total
    ]
    return {
        "status": "available",
        "source": _source_path(path, PROJECT_ROOT),
        "ensemble": {
            "correct": ensemble_correct,
            "total": len(sample_rows),
            "accuracy": ensemble_correct / len(sample_rows),
        },
        "models": model_rows,
        "note": "Accuracy recomputed from persisted per-sample ground truths and predictions.",
    }, None


def _section(rows: list[dict[str, Any]], errors: list[str], empty_message: str) -> dict[str, Any]:
    if rows:
        return {"status": "available", "rows": rows, "errors": errors}
    return {
        "status": "pending",
        "message": empty_message,
        "errors": errors,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row.keys()
        ))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "message"])
        writer.writeheader()
        writer.writerow({"status": "pending", "message": message})


def _flatten(section: dict[str, Any], metadata_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for row in section.get("rows", []):
        flattened = {key: row.get(key) for key in metadata_fields}
        flattened.update(row.get("metrics", {}))
        rows.append(flattened)
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_rows, model_errors = collect_model_evaluations()
    cross_rows, cross_errors = collect_cross_dataset_evaluations()
    robustness_rows, robustness_errors = collect_robustness_evaluations()
    diffusion, diffusion_error = collect_diffusion_evaluation()

    diffusion_section = diffusion or {
        "status": "pending",
        "message": diffusion_error or "No supported persisted diffusion evaluation is available.",
    }
    results = {
        "project": "RobustDeep",
        "model_evaluations": _section(
            model_rows, model_errors, "No supported real-data test_metrics artifacts were found."
        ),
        "cross_dataset_evaluations": _section(
            cross_rows, cross_errors, "No supported persisted cross-dataset evaluation artifacts were found."
        ),
        "robustness_evaluations": _section(
            robustness_rows, robustness_errors, "No supported persisted robustness evaluation artifacts were found."
        ),
        "diffusion_evaluation": diffusion_section,
    }

    _write_csv(
        OUTPUT_DIR / "model_metrics_summary.csv",
        _flatten(results["model_evaluations"], ("model_id", "dataset", "source")),
        results["model_evaluations"].get("message", "Model evaluation is pending."),
    )
    _write_csv(
        OUTPUT_DIR / "cross_dataset_transfer_matrix.csv",
        _flatten(results["cross_dataset_evaluations"], ("model_id", "train_dataset", "test_dataset", "source")),
        results["cross_dataset_evaluations"].get("message", "Cross-dataset evaluation is pending."),
    )
    _write_csv(
        OUTPUT_DIR / "robustness_perturbation_summary.csv",
        _flatten(results["robustness_evaluations"], ("dataset", "stress", "source")),
        results["robustness_evaluations"].get("message", "Robustness evaluation is pending."),
    )
    (OUTPUT_DIR / "master_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )

    report_lines = ["# RobustDeep Evaluation Results", ""]
    for name, section in results.items():
        if name == "project":
            continue
        report_lines.extend([f"## {name.replace('_', ' ').title()}", "", f"Status: **{section['status']}**"])
        if section["status"] == "pending":
            report_lines.extend(["", section["message"]])
        elif "rows" in section:
            for row in section["rows"]:
                report_lines.extend(["", f"- Source: `{row['source']}`", f"- Persisted metrics: `{json.dumps(row['metrics'], sort_keys=True)}`"])
        else:
            report_lines.extend(["", f"- Source: `{section['source']}`", f"- Ensemble metrics: `{json.dumps(section['ensemble'], sort_keys=True)}`"])
            report_lines.extend(
                f"- {row['model_id']} metrics: `{json.dumps(row['metrics'], sort_keys=True)}`"
                for row in section["models"]
            )
        report_lines.append("")
    (OUTPUT_DIR / "MODEL_RESULTS_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUTPUT_DIR / "SUMMARY_TABLES.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Results exported from persisted evaluation artifacts to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
