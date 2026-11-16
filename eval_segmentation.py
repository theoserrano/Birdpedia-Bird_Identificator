import argparse
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from src.preprocessing import load_image, preprocess_pipeline
from src.segmentation import get_mask

GT_ROOT = Path("data/processed/gt_rembg")


def sample_images(split: str, n_per_species: int, seed: int = 0) -> List[Tuple[str, Path]]:
    rng = random.Random(seed)
    out = []
    for sp in sorted(p for p in (Path("data/raw") / split).iterdir() if p.is_dir()):
        imgs = sorted(sp.glob("*.jpg"))
        for path in rng.sample(imgs, min(n_per_species, len(imgs))):
            out.append((sp.name, path))
    return out


def reference_mask(path: Path, split: str, session=None) -> np.ndarray:
    cache = GT_ROOT / split / path.parent.name / (path.stem + ".png")
    if cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)

    from rembg import remove
    img_rgb = cv2.cvtColor(load_image(str(path)), cv2.COLOR_BGR2RGB)
    alpha = np.asarray(remove(img_rgb, session=session, only_mask=True))
    mask = np.where(alpha > 127, 255, 0).astype(np.uint8)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), mask)
    return mask


def iou_dice(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    p, r = pred > 0, ref > 0
    inter = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    total = p.sum() + r.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / total if total else 1.0
    return float(iou), float(dice)


def evaluate(
    samples: List[Tuple[str, Path]],
    split: str,
    mask_fn: Callable[[Dict], np.ndarray],
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    session = None
    if not all((GT_ROOT / split / sp / (p.stem + ".png")).exists() for sp, p in samples):
        from rembg import new_session
        session = new_session("u2net")

    per_species: Dict[str, List[Tuple[float, float]]] = {}
    for species, path in samples:
        ref = reference_mask(path, split, session)
        pred = mask_fn(preprocess_pipeline(str(path)))
        per_species.setdefault(species, []).append(iou_dice(pred, ref))

    result = {sp: {"iou": float(np.mean([v[0] for v in vals])),
                   "dice": float(np.mean([v[1] for v in vals]))}
              for sp, vals in per_species.items()}
    all_vals = [v for vals in per_species.values() for v in vals]
    result["MÉDIA"] = {"iou": float(np.mean([v[0] for v in all_vals])),
                       "dice": float(np.mean([v[1] for v in all_vals]))}

    if verbose:
        print(f"{'Espécie':<26}{'IoU':>7}{'Dice':>7}")
        for sp, m in result.items():
            print(f"{sp:<26}{m['iou']:>7.3f}{m['dice']:>7.3f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=20, help="imagens por espécie")
    parser.add_argument("--method", default="grabcut", choices=["grabcut", "otsu"])
    args = parser.parse_args()

    samples = sample_images(args.split, args.n)
    print(f"Avaliando method={args.method} em {len(samples)} imagens ({args.split})\n")
    evaluate(samples, args.split, lambda pre: get_mask(pre, method=args.method))


if __name__ == "__main__":
    main()
