import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES_V1,
    FEATURE_VERSION,
    extract_features,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _init_worker() -> None:
    cv2.setNumThreads(1)


def _process(task: Tuple[str, str, str]) -> Dict[str, object]:
    path, split, method = task
    species = Path(path).parent.name
    try:
        result = extract_features(path, method=method)
        return {
            "ok": True,
            "path": path,
            "split": split,
            "species": species,
            "vector": result["vector"],
            "meta": result["meta"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": path,
            "split": split,
            "species": species,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def collect_tasks(
    data_dir: Path,
    splits: List[str],
    method: str,
    limit: int,
) -> List[Tuple[str, str, str]]:
    tasks: List[Tuple[str, str, str]] = []
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[AVISO] Split ausente, ignorando: {split_dir}")
            continue
        for species_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = sorted(
                p for p in species_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if limit > 0:
                images = images[:limit]
            tasks.extend((str(path), split, method) for path in images)
    return tasks


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


def build_summary(
    results: List[Dict[str, object]],
    failures: List[Dict[str, object]],
    args: argparse.Namespace,
    x: np.ndarray,
    splits: np.ndarray,
    species: np.ndarray,
    elapsed: float,
    workers: int,
) -> Dict[str, object]:
    counts: Dict[str, Dict[str, int]] = {}
    for name in sorted(set(species.tolist())):
        counts[name] = dict(Counter(splits[species == name].tolist()))

    meta_records = []
    areas = []
    fallback = 0
    for item in results:
        meta = item["meta"]
        areas.append(float(meta.get("mask_area_fraction") or 0.0))
        fallback += int(bool(meta.get("mask_fallback")))
        meta_records.append(
            {
                "path": item["path"],
                "split": item["split"],
                "species": item["species"],
                **meta,
            }
        )

    std = x.std(axis=0)
    constant = [FEATURE_NAMES_V1[i] for i in np.where(std < 1e-12)[0]]
    finite = bool(np.isfinite(x).all())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "method": args.method,
        "feature_version": FEATURE_VERSION,
        "data_dir": str(args.data_dir),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_failed": len(failures),
        "workers": workers,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_image": round(elapsed / max(1, x.shape[0]), 3),
        "counts": counts,
        "qc": {
            "finite": finite,
            "mask_area_mean": float(np.mean(areas)) if areas else 0.0,
            "mask_area_min": float(np.min(areas)) if areas else 0.0,
            "mask_area_max": float(np.max(areas)) if areas else 0.0,
            "mask_fallback": fallback,
            "constant_features": constant,
        },
        "failures": failures,
        "images": meta_records,
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai características de data/raw e salva o dataset em .npz."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--method",
        default="otsu",
        choices=["otsu", "grabcut"],
        help="Método de segmentação usado para gerar a máscara.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Saída .npz (padrão: data/processed/features_<method>_<version>.npz).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Processos paralelos (0 = min(8, núcleos disponíveis)).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Máximo de imagens por espécie/split (0 = todas; útil para smoke test).",
    )
    parser.add_argument("--quiet", action="store_true", help="Sem barra de progresso.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir = Path(args.data_dir)
    if not args.data_dir.exists():
        print(f"[ERRO] Pasta não encontrada: {args.data_dir}")
        print("Execute 'python setup_dataset.py' depois de extrair o archive/ do Kaggle.")
        return 1

    method = args.method
    out = args.out or Path("data/processed") / f"features_{method}_{FEATURE_VERSION}.npz"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(args.data_dir, args.splits, method, args.limit)
    if not tasks:
        print(f"[ERRO] Nenhuma imagem encontrada em {args.data_dir} para {args.splits}.")
        return 1

    workers = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks)))

    print("=" * 64)
    print("Birdpedia - extração de características")
    print(f"Método   : {method}")
    print(f"Entrada  : {args.data_dir.resolve()}")
    print(f"Saída    : {out.resolve()}")
    print(f"Imagens  : {len(tasks)} | workers: {workers}")
    print("=" * 64)

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    start = time.time()

    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
            futures = [executor.submit(_process, task) for task in tasks]
            with tqdm(
                total=len(futures),
                desc="features",
                unit="img",
                disable=args.quiet,
            ) as bar:
                for future in as_completed(futures):
                    result = future.result()
                    if result["ok"]:
                        results.append(result)
                    else:
                        failures.append(result)
                    bar.update(1)
    except KeyboardInterrupt:
        print("\n[!] Interrompido pelo usuário.")
        return 130

    elapsed = time.time() - start
    if not results:
        print("[ERRO] Todas as imagens falharam.")
        for failure in failures[:5]:
            print(f"  - {failure['path']}: {failure['error']}")
        return 1

    split_order = {split: i for i, split in enumerate(args.splits)}
    results.sort(key=lambda item: (split_order.get(item["split"], 0), item["path"]))

    x = np.vstack([item["vector"] for item in results]).astype(np.float64)
    y = np.array([item["species"] for item in results])
    split_array = np.array([item["split"] for item in results])
    paths = np.array([item["path"] for item in results])

    np.savez_compressed(
        out,
        X=x,
        y=y,
        splits=split_array,
        paths=paths,
        feature_names=np.array(FEATURE_NAMES_V1),
        feature_groups_json=np.array(json.dumps(FEATURE_GROUPS)),
        feature_version=np.array(FEATURE_VERSION),
        method=np.array(method),
    )

    summary = build_summary(results, failures, args, x, split_array, y, elapsed, workers)
    summary_path = out.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, ensure_ascii=False, indent=2)

    per_split = Counter(split_array.tolist())
    print("-" * 64)
    print(f"[OK] X={x.shape} | y={y.shape} | salvo em {out}")
    print(f"     Tempo: {elapsed:.1f}s ({elapsed / len(results):.3f}s/img)")
    print("     Por split:", ", ".join(f"{k}={v}" for k, v in sorted(per_split.items())))
    print("     Por classe:", ", ".join(f"{k}={v}" for k, v in sorted(Counter(y.tolist()).items())))
    print(
        "     Máscara: área média {:.2f} | fallback={} | finito={}".format(
            summary["qc"]["mask_area_mean"],
            summary["qc"]["mask_fallback"],
            summary["qc"]["finite"],
        )
    )
    print(f"     Features constantes: {len(summary['qc']['constant_features'])}")
    print(f"     Falhas: {len(failures)} | QC: {summary_path}")
    if failures:
        print("     Primeiras falhas:")
        for failure in failures[:5]:
            print(f"       - {failure['path']}: {failure['error']}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
