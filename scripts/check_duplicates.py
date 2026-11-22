import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def thumbnail(path: Path, size: int = 16) -> np.ndarray:
    # Miniatura em tons de cinza, centrada e com norma 1: o produto escalar vira a correlação
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    v = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def list_images(root: Path, split: str):
    return sorted((root / split).glob("*/*.jpg"))


def find_duplicates(root: Path, threshold: float):
    train = list_images(root, "train")
    train_vecs = np.stack([thumbnail(p) for p in train])
    pairs = []
    for split in ("valid", "test"):
        for path in list_images(root, split):
            corr = train_vecs @ thumbnail(path)
            for j in np.flatnonzero(corr > threshold):
                pairs.append((path, train[j], float(corr[j])))
    return sorted(pairs, key=lambda t: -t[2])


def holdout_accuracy(predictions: Path, excluded):
    # Acurácia do holdout com e sem as imagens duplicadas
    excluded = {Path(p).as_posix() for p in excluded}
    with open(predictions, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if Path(r["path"].replace("\\", "/")).as_posix() not in excluded]
    acc = lambda rs: sum(r["correct"] == "True" for r in rs) / len(rs)
    return len(rows), acc(rows), len(kept), acc(kept)


def main():
    parser = argparse.ArgumentParser(description="Procura imagens de valid/test quase idênticas a imagens de train.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--out", type=Path, default=Path("reports/duplicatas.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("reports/classifier/holdout_predictions.csv"))
    args = parser.parse_args()

    pairs = find_duplicates(args.data_dir, args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["holdout", "treino", "correlacao"])
        for hold, train, corr in pairs:
            writer.writerow([hold.as_posix(), train.as_posix(), f"{corr:.4f}"])

    print(f"{len(pairs)} par(es) com correlação > {args.threshold}:")
    for hold, train, corr in pairs:
        print(f"  {hold.as_posix()}  <->  {train.as_posix()}  ({corr:.4f})")
    print(f"Salvo em {args.out}")

    if args.predictions.exists():
        n_all, acc_all, n_kept, acc_kept = holdout_accuracy(args.predictions, [h for h, _, _ in pairs])
        print(f"Holdout: {acc_all:.4f} em {n_all} imagens | sem duplicatas: {acc_kept:.4f} em {n_kept} imagens")


if __name__ == "__main__":
    main()
