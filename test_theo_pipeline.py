"""
Script de Teste e Demonstração Visual do Pré-Processamento
Responsável: Théo do Rego Barros Serrano

Este script valida as funções de src/preprocessing.py e gera uma
figura comparativa ilustrando o efeito dos filtros e a separação dos
espaços de cores (RGB, HSV e CIELAB) para o relatório/slides do grupo.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from src.preprocessing import (
    load_image,
    apply_gaussian_blur,
    apply_bilateral_filter,
    convert_color_spaces,
    preprocess_pipeline,
)


def run_demo():
    # Procura por uma imagem de teste na pasta data/raw/train/AMERICAN FLAMINGO
    base_dir = Path("data/raw/train")
    sample_images = list(base_dir.glob("AMERICAN FLAMINGO/*.jpg"))

    if not sample_images:
        # Fallback para qualquer imagem encontrada
        sample_images = list(base_dir.glob("*/*.jpg"))

    if not sample_images:
        print("[ERRO] Nenhuma imagem encontrada em data/raw! Execute setup_dataset.py primeiro.")
        return

    img_path = str(sample_images[0])
    print(f"Testando pré-processamento com a imagem: {img_path}")

    # 1. Carrega imagem original
    img_bgr = load_image(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Aplica filtros para comparação visual
    img_gauss = apply_gaussian_blur(img_bgr, ksize=(7, 7), sigma_x=1.5)
    img_bilateral = apply_bilateral_filter(img_bgr, d=9, sigma_color=75.0, sigma_space=75.0)

    # 3. Executa o pipeline completo
    result = preprocess_pipeline(img_bgr, filter_type="bilateral", use_clahe=True)
    spaces = result["color_spaces"]

    # 4. Cria figura comparativa para relatórios e apresentação
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Birdpedia — Pipeline de Pré-processamento e Espaços de Cores (Théo)", fontsize=16, weight="bold")

    # Linha 1: Comparativo de Filtragem Espacial
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Original (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_gauss, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Filtro Gaussiano (k=7x7)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("3. Filtro Bilateral (Edge-Preserving)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(cv2.cvtColor(result["filtered_bgr"], cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("4. Bilateral + CLAHE (Final)")
    axes[0, 3].axis("off")

    # Linha 2: Espaço HSV (Hue, Saturation, Value)
    axes[1, 0].imshow(spaces["hsv"])
    axes[1, 0].set_title("HSV Completo")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(spaces["h"], cmap="hsv")
    axes[1, 1].set_title("Canal H (Matiz / Cor Pura)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(spaces["s"], cmap="gray")
    axes[1, 2].set_title("Canal S (Saturação / Pureza)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(spaces["v"], cmap="gray")
    axes[1, 3].set_title("Canal V (Brilho / Valor)")
    axes[1, 3].axis("off")

    # Linha 3: Espaço CIELAB (L*, a*, b*)
    axes[2, 0].imshow(spaces["lab"])
    axes[2, 0].set_title("CIELAB Completo")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(spaces["l"], cmap="gray")
    axes[2, 1].set_title("Canal L* (Luminância)")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(spaces["a"], cmap="coolwarm")
    axes[2, 2].set_title("Canal a* (Verde <-> Vermelho)")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(spaces["b"], cmap="viridis")
    axes[2, 3].set_title("Canal b* (Azul <-> Amarelo)")
    axes[2, 3].axis("off")

    plt.tight_layout()

    # Salva em pasta reports/
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demonstracao_pre_processamento.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Sucesso! Gráfico comparativo salvo em: {out_path.resolve()}")
    print("Todas as funções de pré-processamento funcionaram perfeitamente.")


if __name__ == "__main__":
    run_demo()
