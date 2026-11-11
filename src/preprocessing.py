"""
Módulo de Pré-processamento e Filtragem Espacial - Birdpedia
Responsável: Théo do Rego Barros Serrano

Este módulo implementa as técnicas fundamentais de PDI:
1. Carregamento e validação de imagens digitais.
2. Filtragem espacial:
   - Filtro Gaussiano: suavização linear e atenuação de ruído gaussiano.
   - Filtro Bilateral: suavização não linear preservando bordas (crucial para penas e bicos).
3. Conversão para múltiplos espaços de cores (RGB, HSV e CIELAB).
4. Realce e equalização adaptativa de contraste (CLAHE no canal de luminância L do Lab).
"""

import os
from typing import Dict, Tuple, Union
import cv2
import numpy as np


def load_image(
    image_path: str,
    target_size: Union[Tuple[int, int], None] = (224, 224)
) -> np.ndarray:
    """
    Carrega uma imagem a partir do caminho fornecido e opcionalmente redimensiona.

    Args:
        image_path: Caminho completo ou relativo para o arquivo de imagem.
        target_size: Tupla (largura, altura) para redimensionamento. Se None, mantém original.

    Returns:
        np.ndarray: Imagem no padrão BGR (uint8).

    Raises:
        FileNotFoundError: Se a imagem não for encontrada ou não puder ser lida pelo OpenCV.
    """
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
    """
    Aplica o filtro espacial Gaussiano para atenuação de ruído de alta frequência.
    Útil para reduzir ruídos pontuais e suavizar texturas indesejadas de fundo.

    Args:
        image: Imagem de entrada (BGR ou monocromática).
        ksize: Tamanho do kernel ímpar (largura, altura), ex: (5, 5).
        sigma_x: Desvio padrão do kernel Gaussiano.

    Returns:
        np.ndarray: Imagem suavizada pelo filtro Gaussiano.
    """
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Aplica o filtro Bilateral (Edge-Preserving Filter).
    Combina proximidade espacial com similaridade fotométrica (diferença de intensidade).
    Efeito em aves: suaviza ruídos de granulação e fundo, mantendo bordas nítidas
    do contorno da ave, bico e divisões de penas.

    Args:
        image: Imagem de entrada em BGR (uint8).
        d: Diâmetro de cada vizinhança de pixel usada durante a filtragem.
        sigma_color: Desvio padrão no espaço de cor (maior = mistura cores mais distantes).
        sigma_space: Desvio padrão no espaço de coordenadas (maior = influencia pixels mais distantes).

    Returns:
        np.ndarray: Imagem suavizada preservando bordas.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def convert_color_spaces(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Converte a imagem BGR original para os principais espaços de cores do PDI:
    - RGB: visualização padrão em bibliotecas gráficas (Matplotlib).
    - HSV (Hue, Saturation, Value): isola a informação cromática (H) da iluminação (V).
    - CIELAB (L*, a*, b*): perceptualmente uniforme, desacopla luminância (L) das
      cores oponentes verde-vermelho (a) e azul-amarelo (b).
    - GRAY: escala de cinza baseada na percepção de luminância humana.

    Args:
        image_bgr: Imagem de entrada no padrão OpenCV BGR (uint8).

    Returns:
        dict: Dicionário contendo as representações 'bgr', 'rgb', 'hsv', 'lab', 'gray'
              e seus respectivos canais individuais decompostos.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Decomposição dos canais individuais
    h, s, v = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)

    return {
        "bgr": image_bgr,
        "rgb": rgb,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
        # Canais HSV individuais
        "h": h,
        "s": s,
        "v": v,
        # Canais CIELAB individuais
        "l": l_channel,
        "a": a_channel,
        "b": b_channel,
    }


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica equalização de histograma adaptativa limitada por contraste (CLAHE)
    exclusivamente no canal de luminância L* do espaço CIELAB.
    Evita alterações indesejadas de matiz/saturação e suaviza sombras ou superexposição.

    Args:
        image: Imagem de entrada em BGR.
        clip_limit: Limite para o corte de contraste local (evita amplificação de ruído).
        tile_grid_size: Dimensões da grade de blocos contextuais (ex: 8x8).

    Returns:
        np.ndarray: Imagem com contraste equalizado no formato BGR.
    """
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
    """
    Pipeline unificado de pré-processamento para o projeto Birdpedia.
    Entrega todas as representações necessárias para os módulos de segmentação e features.

    Etapas executadas:
    1. Carga e redimensionamento da imagem.
    2. Suavização espacial (Bilateral ou Gaussiana) para redução de ruídos de fundo.
    3. Realce de luminância com CLAHE (opcional).
    4. Geração e separação em múltiplos espaços de cores (RGB, HSV, CIELAB).

    Args:
        image_input: Caminho para a imagem (str) ou array já carregado (np.ndarray).
        filter_type: 'bilateral' (recomendado) ou 'gaussian'.
        use_clahe: Se True, aplica equalização adaptativa no canal L*.
        target_size: Tupla (largura, altura) para padronizar dimensões.

    Returns:
        dict com:
            - 'original_bgr': Imagem original carregada
            - 'filtered_bgr': Imagem após filtragem espacial e CLAHE
            - 'color_spaces': Dicionário com 'rgb', 'hsv', 'lab', 'gray' e canais separados
    """
    # 1. Carregamento
    if isinstance(image_input, str):
        img_bgr = load_image(image_input, target_size=target_size)
    else:
        img_bgr = image_input.copy()
        if target_size is not None and (img_bgr.shape[1], img_bgr.shape[0]) != target_size:
            img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # 2. Filtragem espacial
    if filter_type == "bilateral":
        filtered = apply_bilateral_filter(img_bgr, d=9, sigma_color=75.0, sigma_space=75.0)
    elif filter_type == "gaussian":
        filtered = apply_gaussian_blur(img_bgr, ksize=(5, 5), sigma_x=1.0)
    else:
        filtered = img_bgr.copy()

    # 3. Realce de contraste adaptativo
    if use_clahe:
        filtered = enhance_contrast_clahe(filtered, clip_limit=2.0, tile_grid_size=(8, 8))

    # 4. Conversão para múltiplos espaços de cores
    spaces = convert_color_spaces(filtered)

    return {
        "original_bgr": img_bgr,
        "filtered_bgr": filtered,
        "color_spaces": spaces,
    }
