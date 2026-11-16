from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .preprocessing import preprocess_pipeline
from .segmentation import get_bird_mask

FEATURE_VERSION = "v1"

H_BINS = 16
S_BINS = 16

LBP_P = 8
LBP_R = 1
LBP_BINS = LBP_P + 2

GLCM_LEVELS = 16
GLCM_DISTANCES = (1, 2)
GLCM_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))

MIN_MASK_AREA = 0.02
MAX_MASK_AREA = 0.90
MIN_LOCAL_MASK_AREA = 0.005

SHAPE_FEATURE_NAMES = tuple(
    [f"shape_hu_{i}" for i in range(7)]
    + [
        "shape_area_frac",
        "shape_perimeter_norm",
        "shape_circularity",
        "shape_extent",
        "shape_aspect_ratio",
        "shape_solidity",
        "shape_eccentricity",
    ]
)

COLOR_FEATURE_NAMES = tuple(
    [f"color_hs_hist_{i:02d}_{j:02d}" for i in range(H_BINS) for j in range(S_BINS)]
    + [
        "color_lab_l_mean",
        "color_lab_l_std",
        "color_lab_a_mean",
        "color_lab_a_std",
        "color_lab_b_mean",
        "color_lab_b_std",
    ]
)

TEXTURE_FEATURE_NAMES = tuple(
    [f"texture_lbp_{i}" for i in range(LBP_BINS)]
    + [
        "texture_glcm_contrast",
        "texture_glcm_dissimilarity",
        "texture_glcm_homogeneity",
        "texture_glcm_energy",
        "texture_glcm_correlation",
        "texture_glcm_asm",
    ]
)

FEATURE_NAMES_V1 = tuple(SHAPE_FEATURE_NAMES + COLOR_FEATURE_NAMES + TEXTURE_FEATURE_NAMES)


def _build_groups() -> Dict[str, Tuple[int, int]]:
    groups: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, names in (
        ("shape", SHAPE_FEATURE_NAMES),
        ("color", COLOR_FEATURE_NAMES),
        ("texture", TEXTURE_FEATURE_NAMES),
    ):
        groups[name] = (start, start + len(names))
        start += len(names)
    return groups


FEATURE_GROUPS = _build_groups()


def _binary_mask(mask: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    binary = (arr > 0).astype(np.uint8)
    resized = False
    if binary.shape != shape:
        binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        resized = True
    return binary, resized


def _largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    area = float(binary.sum())
    if area <= 0.0:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float64)

    moments = cv2.moments(binary, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    contour = _largest_contour(binary)
    if contour is None:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float64)

    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0
    if perimeter > 0.0:
        circularity = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))

    _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
    extent = area / float(bbox_w * bbox_h) if bbox_w > 0 and bbox_h > 0 else 0.0
    aspect_ratio = bbox_w / float(bbox_h) if bbox_h > 0 else 0.0

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = min(1.0, area / hull_area) if hull_area > 0.0 else 0.0

    m00 = moments["m00"] if moments["m00"] > 0.0 else 1.0
    mu20 = moments["mu20"] / m00
    mu02 = moments["mu02"] / m00
    mu11 = moments["mu11"] / m00
    eig = np.linalg.eigvalsh(np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64))
    eccentricity = 0.0
    if eig[1] > 0.0:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - eig[0] / eig[1])))

    return np.concatenate(
        [
            hu,
            np.array(
                [
                    area / float(h * w),
                    perimeter / float(2 * (h + w)),
                    circularity,
                    extent,
                    aspect_ratio,
                    solidity,
                    eccentricity,
                ],
                dtype=np.float64,
            ),
        ]
    )


def extract_color_features(hsv: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros(len(COLOR_FEATURE_NAMES), dtype=np.float64)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        binary * 255,
        [H_BINS, S_BINS],
        [0, 180, 0, 256],
    ).ravel()
    total = float(hist.sum())
    if total > 0.0:
        hist = hist / total

    pixels = lab[binary > 0].astype(np.float64)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    lab_stats = np.stack([means, stds], axis=1).ravel()

    return np.concatenate([hist.astype(np.float64), lab_stats])


def _masked_glcm_counts(
    quant: np.ndarray,
    mask: np.ndarray,
    dy: int,
    dx: int,
    levels: int,
) -> np.ndarray:
    h, w = quant.shape[:2]
    y0a, y1a = max(0, -dy), h - max(0, dy)
    x0a, x1a = max(0, -dx), w - max(0, dx)
    y0b, y1b = max(0, dy), h - max(0, -dy)
    x0b, x1b = max(0, dx), w - max(0, -dx)
    if y1a <= y0a or x1a <= x0a:
        return np.zeros((levels, levels), dtype=np.float64)

    a = quant[y0a:y1a, x0a:x1a].astype(np.int64)
    b = quant[y0b:y1b, x0b:x1b].astype(np.int64)
    valid = (mask[y0a:y1a, x0a:x1a] > 0) & (mask[y0b:y1b, x0b:x1b] > 0)
    if not valid.any():
        return np.zeros((levels, levels), dtype=np.float64)

    idx = a[valid] * levels + b[valid]
    counts = np.bincount(idx, minlength=levels * levels)
    return counts.reshape(levels, levels).astype(np.float64)


def _glcm_properties(matrix: np.ndarray) -> np.ndarray:
    levels = matrix.shape[0]
    i = np.arange(levels, dtype=np.float64)[:, None]
    j = np.arange(levels, dtype=np.float64)[None, :]
    diff = i - j

    contrast = float((matrix * diff ** 2).sum())
    dissimilarity = float((matrix * np.abs(diff)).sum())
    homogeneity = float((matrix / (1.0 + diff ** 2)).sum())
    asm = float((matrix ** 2).sum())
    energy = float(np.sqrt(asm))

    mu_i = float((matrix * i).sum())
    mu_j = float((matrix * j).sum())
    var_i = float((matrix * (i - mu_i) ** 2).sum())
    var_j = float((matrix * (j - mu_j) ** 2).sum())
    correlation = 0.0
    if var_i > 0.0 and var_j > 0.0:
        correlation = float(
            (matrix * (i - mu_i) * (j - mu_j)).sum() / np.sqrt(var_i * var_j)
        )

    return np.array(
        [contrast, dissimilarity, homogeneity, energy, correlation, asm],
        dtype=np.float64,
    )


def extract_texture_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros(len(TEXTURE_FEATURE_NAMES), dtype=np.float64)

    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    values = lbp[binary > 0].astype(np.int64)
    lbp_hist = np.bincount(values, minlength=LBP_BINS)[:LBP_BINS].astype(np.float64)
    if lbp_hist.sum() > 0.0:
        lbp_hist = lbp_hist / lbp_hist.sum()

    quant = np.clip(
        gray.astype(np.int32) * GLCM_LEVELS // 256,
        0,
        GLCM_LEVELS - 1,
    ).astype(np.uint8)

    matrices = []
    for distance in GLCM_DISTANCES:
        for off_y, off_x in GLCM_OFFSETS:
            counts = _masked_glcm_counts(
                quant,
                binary,
                off_y * distance,
                off_x * distance,
                GLCM_LEVELS,
            )
            total = float(counts.sum())
            if total <= 0.0:
                continue
            matrix = counts + counts.T
            matrix = matrix / matrix.sum()
            matrices.append(_glcm_properties(matrix))

    if matrices:
        glcm = np.mean(np.vstack(matrices), axis=0)
    else:
        glcm = np.zeros(6, dtype=np.float64)

    return np.concatenate([lbp_hist, glcm])


def extract_features(
    image_input: Union[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    method: str = "otsu",
) -> Dict[str, object]:
    """
    Extrai o vetor de características de uma imagem.

    Se ``mask`` não for informada, a segmentação é executada com ``method``
    (padrão ``"otsu"``). A máscara é validada por área; quando fica fora de
    [MIN_MASK_AREA, MAX_MASK_AREA], a imagem inteira é usada e a flag
    ``mask_fallback`` é marcada no metadado.
    """
    meta: Dict[str, object] = {
        "feature_version": FEATURE_VERSION,
        "path": image_input if isinstance(image_input, str) else None,
        "method": None,
        "channel": None,
        "separability": None,
    }

    if mask is None:
        mask, steps = get_bird_mask(image_input, return_steps=True, method=method)
        preprocessed = steps["preprocessed"]
        meta["method"] = steps.get("method", method)
        meta["channel"] = steps.get("channel")
        meta["separability"] = steps.get("separability")
        meta["segmentation_area_fraction"] = steps.get("area_fraction")
    else:
        preprocessed = preprocess_pipeline(image_input)
        meta["method"] = "provided"

    spaces = preprocessed["color_spaces"]
    shape = spaces["gray"].shape[:2]
    binary, resized = _binary_mask(mask, shape)

    raw_area = float(binary.mean())
    fallback = raw_area < MIN_MASK_AREA or raw_area > MAX_MASK_AREA
    if fallback:
        binary = np.ones(shape, dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary * 255, kernel, iterations=1)
    local_mask = (eroded > 0).astype(np.uint8)
    local_eroded = True
    if local_mask.mean() < MIN_LOCAL_MASK_AREA:
        local_mask = binary
        local_eroded = False

    shape_vector = extract_shape_features(binary)
    color_vector = extract_color_features(spaces["hsv"], spaces["lab"], local_mask)
    texture_vector = extract_texture_features(spaces["gray"], local_mask)
    vector = np.concatenate([shape_vector, color_vector, texture_vector]).astype(np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    meta.update(
        {
            "mask_area_fraction": raw_area,
            "mask_area_fraction_used": float(binary.mean()),
            "mask_fallback": bool(fallback),
            "mask_resized": bool(resized),
            "local_mask_eroded": bool(local_eroded),
        }
    )

    return {
        "vector": vector,
        "feature_names": FEATURE_NAMES_V1,
        "groups": FEATURE_GROUPS,
        "meta": meta,
    }
