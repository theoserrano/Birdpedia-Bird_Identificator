import os
import shutil
from pathlib import Path


ORIGEM = Path("./archive")
DESTINO = Path("./data/raw")

ESPECIES_ALVO = [
    "AMERICAN FLAMINGO",
    "AMERICAN GOLDFINCH",
    "AFRICAN EMERALD CUCKOO",
    "AFRICAN FIREFINCH",
    "ALEXANDRINE PARAKEET",
    "AFRICAN CROWNED CRANE",
    "ALBATROSS",
    "ALPINE CHOUGH",
    "AMERICAN AVOCET",
    "AMERICAN KESTREL",
]

DIVISOES = ["train", "test", "valid"]


def preparar_dataset():
    print("=" * 60)
    print("Iniciando filtragem do dataset Birdpedia (10 Espécies)")
    print(f"Origem : {ORIGEM.resolve()}")
    print(f"Destino: {DESTINO.resolve()}")
    print("=" * 60)

    if not ORIGEM.exists():
        print(f"\n[ERRO] A pasta de origem '{ORIGEM}' não foi encontrada!")
        print("Por favor, baixe o dataset no Kaggle e extraia a pasta 'archive' na raiz do projeto.")
        return

    total_copiados = 0

    for divisao in DIVISOES:
        pasta_div_origem = ORIGEM / divisao
        if not pasta_div_origem.exists():
            print(f"[AVISO] Divisão '{divisao}' não encontrada na origem. Pulando...")
            continue

        for especie in ESPECIES_ALVO:
            caminho_orig = pasta_div_origem / especie
            caminho_dest = DESTINO / divisao / especie

            if caminho_orig.exists():
                caminho_dest.mkdir(parents=True, exist_ok=True)
                qtd_arquivos = 0
                for arquivo in caminho_orig.glob("*.*"):
                    shutil.copy2(arquivo, caminho_dest / arquivo.name)
                    qtd_arquivos += 1
                total_copiados += qtd_arquivos
                print(f"[{divisao.upper()}] Copiada espécie '{especie}' ({qtd_arquivos} imagens)")
            else:
                print(f"[ALERTA] Espécie '{especie}' não encontrada em: {caminho_orig}")

    print("=" * 60)
    print(f"Concluído com sucesso! Total de {total_copiados} imagens organizadas em '{DESTINO}'.")
    print("A base está pronta para o pipeline de Pré-Processamento e PDI!")
    print("=" * 60)


if __name__ == "__main__":
    preparar_dataset()
