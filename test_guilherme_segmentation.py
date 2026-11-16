import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation import apply_mask, get_bird_mask

CHANNEL_NAMES = {"s": "S (HSV)", "a": "a* (Lab)", "b": "b* (Lab)", "l": "L* (Lab)",
                 "h": "H (HSV)", "v": "V (HSV)", "gray": "Cinza",
                 "d": "D (dist. ao fundo)"}


def list_species_images(split_dir: Path):
    species = sorted(p for p in split_dir.iterdir() if p.is_dir())
    return {sp.name: sorted(sp.glob("*.jpg")) for sp in species}


def demo_grid(images_by_species, out_path: Path, method: str = "grabcut"):
    rows = [(name, imgs[0]) for name, imgs in images_by_species.items() if imgs]
    cols = ["Original", "Canal escolhido", "Otsu + polaridade", "Otsu + morfologia",
            "Máscara final", "Ave segmentada"]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3 * len(cols), 3 * len(rows)))
    axes = np.atleast_2d(axes)
    title = "Otsu + Morfologia + GrabCut" if method == "grabcut" else "Otsu + Morfologia Matemática"
    fig.suptitle(f"Birdpedia — Segmentação: {title} (Guilherme)",
                 fontsize=16, weight="bold", y=1.0)

    for r, (species, path) in enumerate(rows):
        mask, steps = get_bird_mask(str(path), return_steps=True, method=method)
        img = steps["preprocessed"]["filtered_bgr"]
        panels = [
            (cv2.cvtColor(steps["preprocessed"]["original_bgr"], cv2.COLOR_BGR2RGB), None),
            (steps["channel_image"], "gray"),
            (steps["otsu_polarity"], "gray"),
            (steps["otsu_mask"], "gray"),
            (mask, "gray"),
            (cv2.cvtColor(apply_mask(img, mask), cv2.COLOR_BGR2RGB), None),
        ]
        for c, (panel, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap=cmap)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11, weight="bold")
        axes[r, 0].text(-10, 112, species, rotation=90, va="center", ha="right", fontsize=9)
        axes[r, 1].set_xlabel("")
        axes[r, 1].text(3, 15, f"{CHANNEL_NAMES.get(steps['channel'], steps['channel'])}"
                        f"{' (inv)' if steps['inverted'] else ''}",
                        color="yellow", fontsize=8, backgroundcolor="black")
        axes[r, 4].text(3, 15, f"{steps['method']} | área {steps['area_fraction']:.0%}",
                        color="yellow", fontsize=8, backgroundcolor="black")

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grade de demonstração salva em: {out_path.resolve()}")


def area_stats(images_by_species, per_species: int, method: str = "grabcut"):
    print(f"\nEstatísticas de máscara ({per_species} imagens/espécie, method={method})")
    print("(centro = % de máscaras que cobrem o pixel central; proxy de 'pegou a ave')")
    print(f"{'Espécie':<26}{'área média':>11}{'mín':>7}{'máx':>7}{'suspeitas':>11}{'centro':>8}"
          f"{'grabcut':>9}  canais")
    total_center, total_n = 0, 0
    random.seed(0)
    for species, imgs in images_by_species.items():
        sample = random.sample(imgs, min(per_species, len(imgs)))
        areas, channels, center_hits, gc_used = [], Counter(), 0, 0
        for path in sample:
            mask, steps = get_bird_mask(str(path), return_steps=True, method=method)
            gc_used += int(steps["method"].startswith("grabcut"))
            h, w = mask.shape
            center_hits += int(mask[h // 2, w // 2] > 0)
            areas.append(steps["area_fraction"])
            channels[steps["channel"]] += 1
        areas = np.array(areas)
        suspicious = int(((areas < 0.05) | (areas > 0.85)).sum())
        ch = ", ".join(f"{k}:{v}" for k, v in channels.most_common())
        print(f"{species:<26}{areas.mean():>10.0%}{areas.min():>7.0%}{areas.max():>7.0%}"
              f"{suspicious:>11}{center_hits / len(sample):>8.0%}{gc_used:>9}  {ch}")
        total_center += center_hits
        total_n += len(sample)
    print(f"{'TOTAL':<26}{'':>36}{total_center / total_n:>8.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="reports/demonstracao_segmentacao.png")
    parser.add_argument("--method", default="grabcut", choices=["grabcut", "otsu"])
    parser.add_argument("--stats", type=int, default=0,
                        help="nº de imagens por espécie para estatísticas de área (0 = desliga)")
    args = parser.parse_args()

    split_dir = Path("data/raw") / args.split
    if not split_dir.exists():
        print(f"[ERRO] {split_dir} não encontrado! Execute setup_dataset.py primeiro.")
        return

    images = list_species_images(split_dir)
    demo_grid(images, Path(args.out), args.method)
    if args.stats > 0:
        area_stats(images, args.stats, args.method)


if __name__ == "__main__":
    main()
