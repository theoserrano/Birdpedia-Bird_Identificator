import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_GROUPS, FEATURE_NAMES_V1, FEATURE_VERSION, extract_features

SPLITS = ("train", "valid", "test")


def main():
    parser = argparse.ArgumentParser(description="Extrai as características usando a imagem inteira, sem segmentação.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path(f"data/processed/features_sem_mascara_{FEATURE_VERSION}.npz"))
    args = parser.parse_args()

    tasks = [
        (path, split)
        for split in SPLITS
        for path in sorted((args.data_dir / split).glob("*/*.jpg"))
    ]
    full_mask = np.ones((224, 224), np.uint8) * 255

    vectors, labels, splits, paths = [], [], [], []
    for path, split in tqdm(tasks, desc="sem máscara"):
        result = extract_features(str(path), mask=full_mask)
        vectors.append(result["vector"])
        labels.append(path.parent.name)
        splits.append(split)
        paths.append(str(path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=np.vstack(vectors).astype(np.float64),
        y=np.array(labels),
        splits=np.array(splits),
        paths=np.array(paths),
        feature_names=np.array(FEATURE_NAMES_V1),
        feature_groups_json=np.array(json.dumps(FEATURE_GROUPS)),
        feature_version=np.array(FEATURE_VERSION),
        method=np.array("sem_mascara"),
    )
    print(f"[OK] {len(vectors)} imagens | salvo em {args.out}")


if __name__ == "__main__":
    main()
