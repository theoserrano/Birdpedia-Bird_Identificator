import os
from typing import Dict, Tuple, Union
import cv2
import numpy as np


def load_image(
    image_path: str,
    target_size: Union[Tuple[int, int], None] = (224, 224)
) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível decodificar a imagem: {image_path}")

    if target_size is not None and (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def apply_gaussian_blur(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma_x: float = 1.0
) -> np.ndarray:
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def convert_color_spaces(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)

    return {
        "bgr": image_bgr,
        "rgb": rgb,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
        "h": h,
        "s": s,
        "v": v,
        "l": l_channel,
        "a": a_channel,
        "b": b_channel,
    }


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_chan)

    lab_enhanced = cv2.merge((l_clahe, a_chan, b_chan))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)


def preprocess_pipeline(
    image_input: Union[str, np.ndarray],
    filter_type: str = "bilateral",
    use_clahe: bool = True,
    target_size: Union[Tuple[int, int], None] = (224, 224)
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    if isinstance(image_input, str):
        img_bgr = load_image(image_input, target_size=target_size)
    else:
        img_bgr = image_input.copy()
        if target_size is not None and (img_bgr.shape[1], img_bgr.shape[0]) != target_size:
            img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    if filter_type == "bilateral":
        filtered = apply_bilateral_filter(img_bgr, d=9, sigma_color=75.0, sigma_space=75.0)
    elif filter_type == "gaussian":
        filtered = apply_gaussian_blur(img_bgr, ksize=(5, 5), sigma_x=1.0)
    else:
        filtered = img_bgr.copy()

    if use_clahe:
        filtered = enhance_contrast_clahe(filtered, clip_limit=2.0, tile_grid_size=(8, 8))

    spaces = convert_color_spaces(filtered)

    return {
        "original_bgr": img_bgr,
        "filtered_bgr": filtered,
        "color_spaces": spaces,
    }
