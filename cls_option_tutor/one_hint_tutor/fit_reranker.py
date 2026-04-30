from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _feature_names(rows: List[dict]) -> List[str]:
    names = set()
    for row in rows:
        for name in dict(row.get("reranker_features", {}) or {}):
            names.add(str(name))
    return sorted(names)


def _target_value(row: dict, target_field: str) -> float:
    if target_field in dict(row.get("labels", {}) or {}):
        return float(row["labels"][target_field])
    return float(row.get(target_field, 0.0))


def _matrix(rows: List[dict], feature_names: List[str]) -> tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    X = np.zeros((len(rows), len(feature_names)), dtype=float)
    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for row_idx, row in enumerate(rows):
        features = dict(row.get("reranker_features", {}) or {})
        for col_idx, name in enumerate(feature_names):
            X[row_idx, col_idx] = float(features.get(name, 0.0))
    for col_idx, name in enumerate(feature_names):
        col = X[:, col_idx]
        mean = float(np.mean(col))
        std = float(np.std(col))
        means[name] = mean
        scales[name] = 1.0 if std <= 1e-8 else std
        X[:, col_idx] = (col - mean) / scales[name]
    return X, means, scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a simple ridge reranker from candidate dataset rows.")
    parser.add_argument("--data", required=True, help="JSONL from export_candidate_dataset.py")
    parser.add_argument("--out", required=True, help="Output reranker model JSON")
    parser.add_argument("--target-field", default="search_target", choices=["search_target", "transfer_target"])
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Default planner blend weight to store in model")
    args = parser.parse_args()

    rows = _load_jsonl(args.data)
    if not rows:
        raise ValueError("No dataset rows found")
    feature_names = _feature_names(rows)
    if not feature_names:
        raise ValueError("Dataset does not contain reranker_features")
    X, means, scales = _matrix(rows, feature_names)
    y = np.asarray([_target_value(row, args.target_field) for row in rows], dtype=float)

    ridge = float(args.ridge)
    gram = X.T @ X
    eye = np.eye(gram.shape[0], dtype=float)
    weights_vec = np.linalg.solve(gram + ridge * eye, X.T @ y)
    bias = float(np.mean(y))
    weights = {name: float(weights_vec[idx]) for idx, name in enumerate(feature_names) if abs(float(weights_vec[idx])) > 1e-10}
    payload = {
        "target": "search" if str(args.target_field) == "search_target" else "transfer",
        "target_field": str(args.target_field),
        "alpha": float(args.alpha),
        "bias": bias,
        "weights": weights,
        "feature_means": means,
        "feature_scales": scales,
        "n_rows": len(rows),
        "ridge": ridge,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "features": len(feature_names), "out": str(out_path)}))


if __name__ == "__main__":
    main()
