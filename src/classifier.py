import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .features import extract_features
except ImportError:
    from features import extract_features

DEFAULT_FEATURES = Path("data/processed/features_otsu_v1.npz")
DEFAULT_OUT = Path("reports/classifier")
DEFAULT_MODEL = DEFAULT_OUT / "model.joblib"

SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_feature_dataset(features_path: Path) -> Dict[str, object]:
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset de features não encontrado: {path}\n"
            "Execute 'python build_features.py' primeiro."
        )

    data = np.load(path, allow_pickle=False)
    required = {
        "X",
        "y",
        "splits",
        "paths",
        "feature_names",
        "feature_groups_json",
        "feature_version",
        "method",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Arquivo de features incompleto, faltando: {sorted(missing)}")

    groups = {
        name: tuple(bounds)
        for name, bounds in json.loads(str(data["feature_groups_json"])).items()
    }

    meta_by_path: Dict[str, Dict[str, object]] = {}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        meta_by_path = {
            str(item.get("path")): item
            for item in payload.get("images", [])
            if item.get("path")
        }

    return {
        "X": data["X"],
        "y": data["y"],
        "splits": data["splits"],
        "paths": data["paths"],
        "feature_names": data["feature_names"],
        "groups": groups,
        "feature_version": str(data["feature_version"]),
        "method": str(data["method"]),
        "features_path": str(path),
        "meta_by_path": meta_by_path,
    }


def validate_dataset(dataset: Dict[str, object]) -> List[str]:
    x = dataset["X"]
    y = dataset["y"]
    splits = dataset["splits"]
    paths = dataset["paths"]
    names = dataset["feature_names"]

    if x.ndim != 2:
        raise ValueError(f"X deve ser 2D, recebido {x.shape}")
    if len(y) != x.shape[0] or len(splits) != x.shape[0] or len(paths) != x.shape[0]:
        raise ValueError("X, y, splits e paths têm tamanhos incompatíveis.")
    if len(names) != x.shape[1]:
        raise ValueError(
            f"feature_names ({len(names)}) != dimensões de X ({x.shape[1]})."
        )
    if not np.isfinite(x).all():
        raise ValueError("X contém NaN ou infinito.")
    if len(set(paths.tolist())) != len(paths):
        raise ValueError("Há caminhos duplicados no dataset.")

    labels = sorted(set(y.tolist()))
    if not labels:
        raise ValueError("Nenhuma classe encontrada.")
    return labels


def _make_models(seed: int, quick: bool) -> Dict[str, Tuple[Pipeline, Dict]]:
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    rf = Pipeline(
        [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            )
        ]
    )

    if quick:
        svm_grid = {"svm__C": [1, 10, 100], "svm__gamma": ["scale", "auto"]}
        rf_grid = {
            "rf__n_estimators": [200],
            "rf__max_depth": [None, 20],
            "rf__min_samples_leaf": [1, 2],
        }
    else:
        svm_grid = {
            "svm__C": [1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.005, 0.01],
        }
        rf_grid = {
            "rf__n_estimators": [200, 400],
            "rf__max_depth": [None, 20, 40],
            "rf__min_samples_leaf": [1, 2],
        }

    return {"svm": (svm, svm_grid), "rf": (rf, rf_grid)}


def _run_search(
    name: str,
    pipeline: Pipeline,
    param_grid: Dict,
    x: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_jobs: int,
) -> Tuple[Dict[str, object], GridSearchCV]:
    print(f"[{name}] buscando hiperparâmetros ({args_count(param_grid)} combinações x {cv.n_splits} folds)...")
    start = time.time()
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=SCORING,
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(x, y)
    elapsed = time.time() - start

    index = search.best_index_
    results = search.cv_results_
    record = {
        "best_params": search.best_params_,
        "n_candidates": len(results["params"]),
        "search_seconds": round(elapsed, 2),
        "cv_f1_macro": {
            "mean": float(results["mean_test_f1_macro"][index]),
            "std": float(results["std_test_f1_macro"][index]),
        },
        "cv_accuracy": {
            "mean": float(results["mean_test_accuracy"][index]),
            "std": float(results["std_test_accuracy"][index]),
        },
        "cv_balanced_accuracy": {
            "mean": float(results["mean_test_balanced_accuracy"][index]),
            "std": float(results["std_test_balanced_accuracy"][index]),
        },
    }
    print(
        f"[{name}] f1_macro={record['cv_f1_macro']['mean']:.4f} "
        f"+/- {record['cv_f1_macro']['std']:.4f} em {elapsed:.1f}s"
    )
    return record, search


def args_count(param_grid: Dict) -> int:
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def _select_best(searches: Dict[str, Dict[str, object]]) -> str:
    return max(
        searches,
        key=lambda name: (
            searches[name]["cv_f1_macro"]["mean"],
            searches[name]["cv_accuracy"]["mean"],
        ),
    )


def _refit_final(search: GridSearchCV, x: np.ndarray, y: np.ndarray) -> Pipeline:
    pipeline = clone(search.best_estimator_)
    if "svm" in pipeline.named_steps:
        svc = pipeline.named_steps["svm"]
        pipeline.set_params(
            svm=CalibratedClassifierCV(estimator=svc, ensemble=False, cv=5)
        )
    pipeline.fit(x, y)
    return pipeline


def _aligned_proba(
    pipeline: Pipeline, x: np.ndarray, labels: Sequence[str]
) -> Optional[np.ndarray]:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        proba = pipeline.predict_proba(x)
    except Exception:
        return None
    classes = list(pipeline.steps[-1][1].classes_)
    order = [classes.index(label) for label in labels]
    return proba[:, order]


def _top_confusions(
    matrix: np.ndarray, labels: Sequence[str], top_n: int = 8
) -> List[Dict[str, object]]:
    rows = []
    for i, true_label in enumerate(labels):
        total = int(matrix[i].sum())
        for j, pred_label in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            rows.append(
                {
                    "true": str(true_label),
                    "pred": str(pred_label),
                    "count": int(matrix[i, j]),
                    "share_of_true": float(matrix[i, j] / total) if total else 0.0,
                }
            )
    rows.sort(key=lambda row: (row["count"], row["share_of_true"]), reverse=True)
    return rows[:top_n]


def _evaluate_holdout(
    pipeline: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
) -> Tuple[Dict[str, object], np.ndarray, Optional[np.ndarray]]:
    start = time.time()
    predictions = pipeline.predict(x)
    elapsed = time.time() - start
    proba = _aligned_proba(pipeline, x, labels)

    per_class = classification_report(
        y,
        predictions,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y, predictions, labels=list(labels))

    metrics = {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_macro": float(
            precision_score(y, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y, predictions)),
        "predict_seconds": round(elapsed, 3),
    }

    return (
        {
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "top_confusions": _top_confusions(matrix, labels),
        },
        predictions,
        proba,
    )


def _run_ablation(
    dataset: Dict[str, object],
    base_pipeline: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_jobs: int,
) -> List[Dict[str, object]]:
    groups = dataset["groups"]
    subsets: List[Tuple[str, np.ndarray]] = [("all", np.arange(x.shape[1]))]
    for name, (start, end) in groups.items():
        subsets.append((name, np.arange(start, end)))

    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = np.arange(*groups[names[i]])
            second = np.arange(*groups[names[j]])
            subsets.append(
                (f"{names[i]}+{names[j]}", np.concatenate([first, second]))
            )

    base = clone(base_pipeline)

    rows = []
    for name, indices in subsets:
        result = cross_validate(
            base,
            x[:, indices],
            y,
            cv=cv,
            scoring=("accuracy", "f1_macro"),
            n_jobs=n_jobs,
        )
        rows.append(
            {
                "subset": name,
                "n_features": int(len(indices)),
                "f1_macro_mean": float(result["test_f1_macro"].mean()),
                "f1_macro_std": float(result["test_f1_macro"].std()),
                "accuracy_mean": float(result["test_accuracy"].mean()),
                "accuracy_std": float(result["test_accuracy"].std()),
            }
        )
        print(
            f"[ablação] {name:16s} ({len(indices):3d} feats) "
            f"f1_macro={rows[-1]['f1_macro_mean']:.4f}"
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(
    matrix: np.ndarray, labels: Sequence[str], output_path: Path
) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(np.float64),
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = (
        (axes[0], matrix.astype(np.float64), "Contagem", "d"),
        (axes[1], normalized, "Normalizada (linha)", ".2f"),
    )
    for ax, values, title, fmt in panels:
        top = float(values.max()) if values.size else 1.0
        image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=top or 1.0)
        ax.set_title(title)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        threshold = top * 0.55
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if value <= 0:
                    continue
                text = format(int(value), fmt) if fmt == "d" else format(value, fmt)
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Birdpedia - Matriz de confusão (holdout)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_per_class(
    per_class: Dict[str, Dict[str, float]],
    labels: Sequence[str],
    output_path: Path,
) -> None:
    entries = sorted(
        ((str(label), float(per_class[label]["f1-score"])) for label in labels),
        key=lambda item: item[1],
    )
    names = [item[0] for item in entries]
    values = [item[1] for item in entries]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names, values, color="#3b7dd8")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_title("Birdpedia - F1 por classe (holdout)")
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.01, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    metadata: Dict[str, object],
    searches: Dict[str, Dict[str, object]],
    holdout: Dict[str, object],
    ablation: List[Dict[str, object]],
    labels: Sequence[str],
) -> None:
    best = metadata["model_name"]
    cv_rows = []
    for name, record in searches.items():
        cv_rows.append(
            [
                name.upper() if name == "rf" else name,
                f"{record['cv_f1_macro']['mean']:.4f} +/- {record['cv_f1_macro']['std']:.4f}",
                f"{record['cv_accuracy']['mean']:.4f} +/- {record['cv_accuracy']['std']:.4f}",
                f"{record['cv_balanced_accuracy']['mean']:.4f} +/- {record['cv_balanced_accuracy']['std']:.4f}",
                json.dumps(record["best_params"], ensure_ascii=False),
            ]
        )

    metrics = holdout["metrics"]
    holdout_rows = [
        ["Acurácia", f"{metrics['accuracy']:.4f}"],
        ["Acurácia balanceada", f"{metrics['balanced_accuracy']:.4f}"],
        ["F1 macro", f"{metrics['f1_macro']:.4f}"],
        ["F1 weighted", f"{metrics['f1_weighted']:.4f}"],
        ["Precisão macro", f"{metrics['precision_macro']:.4f}"],
        ["Recall macro", f"{metrics['recall_macro']:.4f}"],
        ["Cohen's kappa", f"{metrics['kappa']:.4f}"],
        ["Amostras", metrics["n_samples"]],
    ]

    class_rows = []
    for label in labels:
        row = holdout["per_class"][str(label)]
        class_rows.append(
            [
                label,
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1-score']:.3f}",
                int(row["support"]),
            ]
        )

    confusion_rows = [
        [row["true"], row["pred"], row["count"], f"{row['share_of_true']:.1%}"]
        for row in holdout["top_confusions"]
    ]

    ablation_rows = [
        [
            row["subset"],
            row["n_features"],
            f"{row['f1_macro_mean']:.4f} +/- {row['f1_macro_std']:.4f}",
            f"{row['accuracy_mean']:.4f}",
        ]
        for row in ablation
    ]

    content = [
        "# Classificação - Birdpedia",
        "",
        f"Gerado em `{metadata['created_at']}`.",
        "",
        "## Dataset",
        "",
        f"- Features: `{metadata['features_file']}`",
        f"- Versão: `{metadata['feature_version']}` | segmentação: `{metadata['method']}`",
        f"- Treino: {metadata['n_train']} amostras | Holdout: {metadata['n_holdout']} amostras ({'+'.join(metadata['holdout_splits'])})",
        f"- Classes: {len(labels)} | folds: {metadata['folds']} | seed: {metadata['seed']}",
        "",
        f"## Validação cruzada no treino ({metadata['folds']}-fold)",
        "",
        _md_table(
            ["Modelo", "F1 macro", "Acurácia", "Acurácia balanceada", "Melhores parâmetros"],
            cv_rows,
        ),
        "",
        f"Modelo selecionado: **{best}**.",
        "",
        "## Holdout",
        "",
        _md_table(["Métrica", "Valor"], holdout_rows),
        "",
        "## Métricas por classe",
        "",
        _md_table(["Classe", "Precisão", "Recall", "F1", "Suporte"], class_rows),
        "",
        "## Principais confusões",
        "",
        _md_table(["Real", "Predito", "N", "% da classe"], confusion_rows)
        if confusion_rows
        else "Nenhuma confusão no holdout.",
        "",
    ]

    if ablation_rows:
        content.extend(
            [
                "## Ablação de características",
                "",
                _md_table(
                    ["Subconjunto", "Nº features", "F1 macro", "Acurácia"],
                    ablation_rows,
                ),
                "",
            ]
        )

    content.extend(
        [
            "## Artefatos",
            "",
            "- `model.joblib`: pipeline treinado",
            "- `model_metadata.json`: configuração do modelo",
            "- `metrics.json`: todas as métricas",
            "- `metrics_per_class.csv`, `model_comparison.csv`, `ablation.csv`",
            "- `holdout_predictions.csv`: predições individuais",
            "- `confusion_matrix.png`, `per_class_f1.png`",
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def save_artifacts(
    output_dir: Path,
    model: Pipeline,
    metadata: Dict[str, object],
    searches: Dict[str, Dict[str, object]],
    holdout: Dict[str, object],
    ablation: List[Dict[str, object]],
    labels: Sequence[str],
    predictions_rows: List[Dict[str, object]],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump({"pipeline": model, "metadata": metadata}, model_path)

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, ensure_ascii=False, indent=2)

    metrics_payload = {
        "generated_at": metadata["created_at"],
        "features_file": metadata["features_file"],
        "feature_version": metadata["feature_version"],
        "method": metadata["method"],
        "seed": metadata["seed"],
        "folds": metadata["folds"],
        "holdout_splits": metadata["holdout_splits"],
        "n_train": metadata["n_train"],
        "n_holdout": metadata["n_holdout"],
        "class_names": list(labels),
        "models": searches,
        "selected_model": metadata["model_name"],
        "holdout": holdout,
        "ablation": ablation,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics_payload), handle, ensure_ascii=False, indent=2)

    comparison_rows = []
    for name, record in searches.items():
        comparison_rows.append(
            {
                "model": name,
                "cv_f1_macro_mean": record["cv_f1_macro"]["mean"],
                "cv_f1_macro_std": record["cv_f1_macro"]["std"],
                "cv_accuracy_mean": record["cv_accuracy"]["mean"],
                "cv_accuracy_std": record["cv_accuracy"]["std"],
                "cv_balanced_mean": record["cv_balanced_accuracy"]["mean"],
                "cv_balanced_std": record["cv_balanced_accuracy"]["std"],
                "search_seconds": record["search_seconds"],
                "best_params": json.dumps(record["best_params"], ensure_ascii=False),
                "selected": name == metadata["model_name"],
            }
        )
    _write_csv(
        output_dir / "model_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    class_rows = []
    for label in labels:
        row = holdout["per_class"][str(label)]
        class_rows.append(
            {
                "class": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": int(row["support"]),
            }
        )
    _write_csv(
        output_dir / "metrics_per_class.csv",
        class_rows,
        ["class", "precision", "recall", "f1", "support"],
    )

    if ablation:
        _write_csv(
            output_dir / "ablation.csv",
            ablation,
            [
                "subset",
                "n_features",
                "f1_macro_mean",
                "f1_macro_std",
                "accuracy_mean",
                "accuracy_std",
            ],
        )

    _write_csv(
        output_dir / "holdout_predictions.csv",
        predictions_rows,
        [
            "path",
            "true",
            "pred",
            "confidence",
            "correct",
            "mask_area_fraction",
            "mask_fallback",
            "channel",
        ],
    )

    matrix = np.asarray(holdout["confusion_matrix"])
    _plot_confusion(matrix, labels, output_dir / "confusion_matrix.png")
    _plot_per_class(holdout["per_class"], labels, output_dir / "per_class_f1.png")
    _write_report(
        output_dir / "report.md",
        metadata,
        searches,
        holdout,
        ablation,
        labels,
    )

    return {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "metrics": str(metrics_path),
        "report": str(output_dir / "report.md"),
        "confusion_matrix": str(output_dir / "confusion_matrix.png"),
        "per_class_f1": str(output_dir / "per_class_f1.png"),
        "ablation": str(output_dir / "ablation.csv") if ablation else "",
    }


def predict_species(
    image_path: Path,
    model_path: Path = DEFAULT_MODEL,
    method: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, object]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}\n"
            "Execute 'python -m src.classifier' para treinar primeiro."
        )

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        metadata = artifact.get("metadata", {})
    else:
        pipeline = artifact
        metadata = {}

    feature_result = extract_features(
        str(image_path),
        method=method or metadata.get("method", "otsu"),
    )
    vector = feature_result["vector"]
    indices = metadata.get("feature_indices")
    if indices is not None:
        vector = vector[np.asarray(indices, dtype=int)]

    features = vector.reshape(1, -1)
    prediction = str(pipeline.predict(features)[0])
    classes = list(pipeline.steps[-1][1].classes_)

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(features)[0]
        except Exception:
            probabilities = None

    ranking = []
    confidence = None
    if probabilities is not None:
        order = np.argsort(probabilities)[::-1][: max(1, top_k)]
        ranking = [
            {"species": str(classes[i]), "probability": float(probabilities[i])}
            for i in order
        ]
        confidence = float(probabilities[classes.index(prediction)])

    meta = feature_result["meta"]
    return {
        "species": prediction,
        "confidence": confidence,
        "top_k": ranking,
        "image": str(image_path),
        "model": str(path),
        "feature_meta": {
            key: meta.get(key)
            for key in (
                "method",
                "channel",
                "mask_area_fraction",
                "mask_fallback",
                "local_mask_eroded",
                "segmentation_area_fraction",
            )
        },
    }


def cmd_predict(args: argparse.Namespace) -> int:
    try:
        result = predict_species(
            args.predict,
            model_path=args.model,
            method=args.predict_method,
        )
    except Exception as exc:
        print(f"[ERRO] {exc}")
        return 1
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        dataset = load_feature_dataset(args.features)
        labels = validate_dataset(dataset)
    except Exception as exc:
        print(f"[ERRO] {exc}")
        return 1

    x = dataset["X"]
    y = dataset["y"]
    splits = dataset["splits"]

    train_mask = splits == "train"
    holdout_mask = np.isin(splits, args.holdout)
    if not train_mask.any():
        print("[ERRO] Split 'train' vazio.")
        return 1
    if not holdout_mask.any():
        print(f"[ERRO] Holdout vazio: {args.holdout}")
        return 1

    x_train, y_train = x[train_mask], y[train_mask]
    x_holdout, y_holdout = x[holdout_mask], y[holdout_mask]

    train_labels = set(y_train.tolist())
    missing = sorted(set(y_holdout.tolist()) - train_labels)
    if missing:
        print(f"[ERRO] Classes no holdout ausentes no treino: {missing}")
        return 1

    counts = Counter(y_train.tolist())
    min_count = min(counts.values())
    if args.folds > min_count:
        print(f"[ERRO] folds={args.folds} maior que a menor classe ({min_count}).")
        return 1

    print("=" * 64)
    print("Birdpedia - treinamento e avaliação")
    print(f"Features : {dataset['features_path']}")
    print(f"Treino   : {x_train.shape[0]} amostras | Holdout: {x_holdout.shape[0]} ({'+'.join(args.holdout)})")
    print(f"Classes  : {len(train_labels)} | folds: {args.folds} | seed: {args.seed}")
    print("=" * 64)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    model_specs = _make_models(args.seed, args.quick)

    searches: Dict[str, Dict[str, object]] = {}
    search_objects: Dict[str, GridSearchCV] = {}
    for name in args.models:
        if name not in model_specs:
            print(f"[AVISO] Modelo desconhecido, ignorando: {name}")
            continue
        pipeline, grid = model_specs[name]
        record, search = _run_search(name, pipeline, grid, x_train, y_train, cv, args.n_jobs)
        searches[name] = record
        search_objects[name] = search

    if not searches:
        print("[ERRO] Nenhum modelo treinado.")
        return 1

    best_name = _select_best(searches)
    print(f"[seleção] {best_name} (critério: F1 macro na CV)")

    start = time.time()
    final_model = _refit_final(search_objects[best_name], x_train, y_train)
    final_fit_seconds = time.time() - start

    holdout, predictions, proba = _evaluate_holdout(
        final_model, x_holdout, y_holdout, labels
    )

    indices = np.arange(x.shape[1])
    feature_names = dataset["feature_names"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features_file": dataset["features_path"],
        "feature_version": dataset["feature_version"],
        "method": dataset["method"],
        "feature_names": [str(name) for name in feature_names],
        "feature_indices": indices.tolist(),
        "groups": {name: list(bounds) for name, bounds in dataset["groups"].items()},
        "class_names": [str(label) for label in labels],
        "model_name": best_name,
        "model_params": searches[best_name]["best_params"],
        "calibration": (
            "CalibratedClassifierCV(ensemble=False, cv=5)"
            if best_name == "svm"
            else "nativa (RandomForest.predict_proba)"
        ),
        "cv": searches[best_name],
        "holdout_splits": list(args.holdout),
        "n_train": int(x_train.shape[0]),
        "n_holdout": int(x_holdout.shape[0]),
        "final_fit_seconds": round(final_fit_seconds, 2),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "folds": args.folds,
        "git_commit": _git_commit(),
    }

    ablation: List[Dict[str, object]] = []
    if args.ablation and "svm" in search_objects:
        print("-" * 64)
        ablation = _run_ablation(
            dataset,
            search_objects["svm"].best_estimator_,
            x_train,
            y_train,
            cv,
            args.n_jobs,
        )

    holdout_paths = dataset["paths"][holdout_mask]
    meta_by_path = dataset["meta_by_path"]
    predictions_rows = []
    for index, path in enumerate(holdout_paths):
        true_label = str(y_holdout[index])
        pred_label = str(predictions[index])
        confidence = None
        if proba is not None:
            confidence = float(proba[index, list(labels).index(pred_label)])
        image_meta = meta_by_path.get(str(path), {})
        predictions_rows.append(
            {
                "path": str(path),
                "true": true_label,
                "pred": pred_label,
                "confidence": confidence,
                "correct": true_label == pred_label,
                "mask_area_fraction": image_meta.get("mask_area_fraction"),
                "mask_fallback": image_meta.get("mask_fallback"),
                "channel": image_meta.get("channel"),
            }
        )

    artifacts = save_artifacts(
        args.out,
        final_model,
        metadata,
        searches,
        holdout,
        ablation,
        labels,
        predictions_rows,
    )

    metrics = holdout["metrics"]
    print("-" * 64)
    print(f"[OK] Modelo: {best_name}")
    print(
        f"     CV F1 macro: {searches[best_name]['cv_f1_macro']['mean']:.4f} "
        f"+/- {searches[best_name]['cv_f1_macro']['std']:.4f}"
    )
    print(
        f"     Holdout: acurácia={metrics['accuracy']:.4f} | "
        f"F1 macro={metrics['f1_macro']:.4f} | "
        f"kappa={metrics['kappa']:.4f}"
    )
    print(f"     Artefatos: {Path(args.out)}")
    print(f"     Relatório: {artifacts['report']}")
    print("=" * 64)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina/avalia o classificador de espécies ou prediz uma imagem."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--holdout",
        nargs="+",
        default=["valid", "test"],
        choices=["valid", "test"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svm", "rf"],
        choices=["svm", "rf"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grid menor, para smoke test.",
    )
    parser.add_argument(
        "--ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compara subconjuntos de features (usa o melhor SVM).",
    )
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Se informado, apenas prediz esta imagem e sai.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--predict-method",
        choices=["otsu", "grabcut"],
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.predict is not None:
        return cmd_predict(args)
    return cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
