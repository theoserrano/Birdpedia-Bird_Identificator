from typing import Dict, Iterable, Tuple, Union

import cv2
import numpy as np

from .preprocessing import preprocess_pipeline

DEFAULT_CANDIDATES: Tuple[str, ...] = ("s", "a", "b", "l", "d")

MIN_AREA_FRACTION = 0.02
MAX_AREA_FRACTION = 0.90

GRABCUT_MIN_AREA = 0.10

DEFAULT_SIGMA_FRAC = 0.25


def otsu_threshold(channel: np.ndarray) -> Tuple[np.ndarray, float, float]:
    thresh, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, float(thresh), otsu_separability(channel, thresh)


def otsu_separability(channel: np.ndarray, threshold: float) -> float:
    values = channel.astype(np.float64).ravel()
    total_var = values.var()
    if total_var < 1e-9:
        return 0.0

    fg = values > threshold
    w1 = fg.mean()
    w0 = 1.0 - w1
    if w0 == 0.0 or w1 == 0.0:
        return 0.0

    mu0 = values[~fg].mean()
    mu1 = values[fg].mean()
    return float(w0 * w1 * (mu0 - mu1) ** 2 / total_var)


def frame_mask(shape: Tuple[int, ...], border: int = 5) -> np.ndarray:
    h, w = shape[:2]
    b = max(1, min(border, h // 2, w // 2))
    frame = np.ones((h, w), dtype=bool)
    frame[b:h - b, b:w - b] = False
    return frame


def border_foreground_ratio(mask: np.ndarray, border: int = 5) -> float:
    return float((mask[frame_mask(mask.shape, border)] > 0).mean())


def center_weight_map(shape: Tuple[int, ...], sigma_frac: float = DEFAULT_SIGMA_FRAC) -> np.ndarray:
    h, w = shape[:2]
    sigma = sigma_frac * min(h, w)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    d2 = (ys - (h - 1) / 2.0) ** 2 + (xs - (w - 1) / 2.0) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2))


def center_foreground_ratio(mask: np.ndarray, weights: np.ndarray) -> float:
    return float((weights * (mask > 0)).sum() / weights.sum())


def background_distance_channel(
    lab: np.ndarray,
    border: int = 5,
    k: int = 3,
    min_cluster_frac: float = 0.15,
) -> np.ndarray:
    h, w = lab.shape[:2]
    lab_f = lab.astype(np.float32)
    lab_f[..., 0] *= 100.0 / 255.0
    lab_f[..., 1:] -= 128.0

    samples = lab_f[frame_mask((h, w), border)].reshape(-1, 3)

    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k) / len(labels)
    centers = centers[counts >= min_cluster_frac]
    if len(centers) == 0:
        centers = np.median(samples, axis=0, keepdims=True)

    pixels = lab_f.reshape(-1, 1, 3)
    dist = np.sqrt(((pixels - centers[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    dist = dist.reshape(h, w)
    top = np.percentile(dist, 99)
    if top < 1e-6:
        return np.zeros((h, w), dtype=np.uint8)
    return np.clip(dist * (255.0 / top), 0, 255).astype(np.uint8)


def score_channel(
    channel: np.ndarray,
    border: int = 5,
    weights: Union[np.ndarray, None] = None,
) -> Dict[str, Union[float, bool, np.ndarray]]:
    if weights is None:
        weights = center_weight_map(channel.shape)
    raw, thresh, eta = otsu_threshold(channel)
    center_ratio = center_foreground_ratio(raw, weights)
    border_ratio = border_foreground_ratio(raw, border)
    inverted = center_ratio < border_ratio
    mask = raw
    if inverted:
        mask = cv2.bitwise_not(raw)
        center_ratio, border_ratio = 1.0 - center_ratio, 1.0 - border_ratio
    area = float((mask > 0).mean())

    score = eta * (center_ratio - border_ratio)
    if not (MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION):
        score = 0.0

    return {
        "raw": raw,
        "mask": mask,
        "threshold": thresh,
        "separability": eta,
        "inverted": inverted,
        "area_fraction": area,
        "border_ratio": border_ratio,
        "center_ratio": center_ratio,
        "score": float(score),
    }


def select_best_channel(
    color_spaces: Dict[str, np.ndarray],
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    border: int = 5,
    weights: Union[np.ndarray, None] = None,
) -> Tuple[str, Dict[str, Dict]]:
    results = {name: score_channel(color_spaces[name], border, weights) for name in candidates}
    best = max(results, key=lambda name: (results[name]["score"], results[name]["separability"]))
    return best, results


def elliptical_kernel(size: int) -> np.ndarray:
    size = max(1, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def apply_opening(mask: np.ndarray, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, elliptical_kernel(ksize), iterations=iterations)


def apply_closing(mask: np.ndarray, ksize: int = 11, iterations: int = 1) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, elliptical_kernel(ksize), iterations=iterations)


def keep_central_component(mask: np.ndarray, weights: Union[np.ndarray, None] = None) -> np.ndarray:
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    if weights is None:
        weights = center_weight_map(mask.shape)
    weighted_area = np.bincount(labels.ravel(), weights=weights.ravel(), minlength=n_labels)
    best = 1 + int(np.argmax(weighted_area[1:]))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def refine_mask(
    mask: np.ndarray,
    open_ksize: int = 5,
    close_ksize: int = 11,
    weights: Union[np.ndarray, None] = None,
) -> Dict[str, np.ndarray]:
    opened = apply_opening(mask, open_ksize)
    closed = apply_closing(opened, close_ksize)
    largest = keep_central_component(closed, weights)
    filled = fill_holes(largest)
    return {"opened": opened, "closed": closed, "largest": largest, "filled": filled}


def build_grabcut_seeds(
    seed_mask: np.ndarray,
    weights: np.ndarray,
    margin_frac: float = 0.04,
    use_otsu_core: bool = False,
    core_ksize: int = 15,
    core_min_weight: float = 0.3,
) -> np.ndarray:
    h, w = seed_mask.shape
    m = max(1, int(round(margin_frac * min(h, w))))
    labels = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    labels[m:h - m, m:w - m] = cv2.GC_PR_FGD

    if use_otsu_core:
        core = cv2.erode(seed_mask, elliptical_kernel(core_ksize))
        labels[(core > 0) & (weights >= core_min_weight)] = cv2.GC_FGD
    return labels


def grabcut_refine(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
    weights: np.ndarray,
    iterations: int = 5,
    open_ksize: int = 5,
    close_ksize: int = 11,
    use_otsu_core: bool = False,
) -> Dict[str, np.ndarray]:
    seeds = build_grabcut_seeds(seed_mask, weights, use_otsu_core=use_otsu_core)
    if use_otsu_core and not (seeds == cv2.GC_FGD).any():
        return {}
    gc = seeds.copy()
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, gc, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return {}

    raw = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    refined = refine_mask(raw, open_ksize, close_ksize, weights)
    return {"seeds": seeds, "raw": raw, **refined}


def get_mask(
    preprocessed: Dict,
    channel: str = "auto",
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    open_ksize: int = 5,
    close_ksize: int = 11,
    border: int = 5,
    sigma_frac: float = DEFAULT_SIGMA_FRAC,
    method: str = "grabcut",
    grabcut_iterations: int = 5,
    return_steps: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    if method not in ("grabcut", "otsu"):
        raise ValueError(f"method deve ser 'grabcut' ou 'otsu', recebido: {method!r}")

    candidates = tuple(candidates)
    spaces = dict(preprocessed["color_spaces"])
    weights = center_weight_map(spaces["gray"].shape, sigma_frac)
    if channel == "d" or (channel == "auto" and "d" in candidates):
        spaces["d"] = background_distance_channel(spaces["lab"], border)

    if channel == "auto":
        chosen, scores = select_best_channel(spaces, candidates, border, weights)
    else:
        chosen = channel
        scores = {channel: score_channel(spaces[channel], border, weights)}

    result = scores[chosen]
    refined = refine_mask(result["mask"], open_ksize, close_ksize, weights)
    if (refined["filled"] > 0).mean() < MIN_AREA_FRACTION:
        no_open = refine_mask(result["mask"], 1, close_ksize, weights)
        if (no_open["filled"] > 0).sum() > (refined["filled"] > 0).sum():
            refined = no_open
    otsu_mask = refined["filled"]
    mask = otsu_mask

    grabcut, used_method = {}, "otsu"
    if method == "grabcut":
        for use_core, name, min_area in ((False, "grabcut", GRABCUT_MIN_AREA),
                                         (True, "grabcut+otsu", MIN_AREA_FRACTION)):
            grabcut = grabcut_refine(preprocessed["filtered_bgr"], otsu_mask, weights,
                                     grabcut_iterations, open_ksize, close_ksize, use_core)
            if not grabcut:
                continue
            gc_area = float((grabcut["filled"] > 0).mean())
            if min_area <= gc_area <= MAX_AREA_FRACTION:
                mask, used_method = grabcut["filled"], name
                break

    if not return_steps:
        return mask

    steps = {
        "method": used_method,
        "otsu_mask": otsu_mask,
        "grabcut": grabcut,
        "channel": chosen,
        "channel_image": spaces[chosen],
        "threshold": result["threshold"],
        "separability": result["separability"],
        "inverted": result["inverted"],
        "otsu_raw": result["raw"],
        "otsu_polarity": result["mask"],
        **refined,
        "scores": {name: r["score"] for name, r in scores.items()},
        "area_fraction": float((mask > 0).mean()),
    }
    return mask, steps


def get_bird_mask(
    image_input: Union[str, np.ndarray],
    return_steps: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    preprocessed = preprocess_pipeline(image_input)
    out = get_mask(preprocessed, return_steps=return_steps, **kwargs)
    if return_steps:
        mask, steps = out
        steps["preprocessed"] = preprocessed
        return mask, steps
    return out


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)
