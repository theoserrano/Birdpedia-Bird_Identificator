# Birdpedia — Sistema Híbrido de Identificação de Espécies de Aves

Projeto da disciplina de **Processamento Digital de Imagens (PDI)**.

## Equipe
- **Théo do Rego Barros Serrano** (Pré-processamento, Filtragem Espacial e Espaços de Cores)
- **Guilherme da Rocha Lima Barros** (Segmentação e Morfologia Matemática)
- **Douglas Wesley** (Extração de Características, Modelos e Avaliação)

---

## 1. Configuração do Ambiente e Dataset

### 1.1 Instalação das Dependências
Instale as bibliotecas necessárias com:
```bash
pip install -r requirements.txt
```

### 1.2 Obtenção dos Dados
1. Baixe o dataset do Kaggle (20 species) e extraia a pasta `archive/` na raiz deste projeto.
2. Execute o script de setup para criar automaticamente a base filtrada de 10 espécies:
```bash
python setup_dataset.py
```
Isso criará a pasta `data/raw/` com as 10 espécies balanceadas em `train`, `test` e `valid`.
*(As pastas `archive/` e `data/` estão no `.gitignore` para manter o repositório leve e rápido).*

---

## 2. Frente 1: Pré-processamento e Espaços de Cores (Théo)

O módulo `src/preprocessing.py` fornece as seguintes funções para a equipe:

- `load_image(path, target_size=(224, 224))`: Carrega e redimensiona a imagem com OpenCV.
- `apply_gaussian_blur(image, ksize=(5, 5), sigma_x=1.0)`: Suavização linear contra ruído gaussiano.
- `apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)`: Filtro bilateral com preservação de bordas da plumagem e bico.
- `convert_color_spaces(image_bgr)`: Decompõe nos espaços BGR, RGB, HSV, CIELAB e escala de cinza.
- `enhance_contrast_clahe(image)`: Realce adaptativo no canal de luminância L* do CIELAB.
- `preprocess_pipeline(image_path_or_array)`: Pipeline unificado pronto para consumo pelos módulos de segmentação e extração de características.

### Como testar o módulo de pré-processamento:
Execute:
```bash
python test_theo_pipeline.py
```
O script gerará uma imagem comparativa de alta resolução em `reports/demonstracao_pre_processamento.png` ilustrando todos os canais e filtros.

---

## 3. Próximos Passos
- **Guilherme:** Implementar `src/segmentation.py` (Limiarização Otsu e Morfologia Matemática sobre os canais HSV/Lab gerados).
- **Douglas:** Implementar `src/features.py` (Momentos de Hu, LBP, Histogramas) e `src/classifier.py` (SVM/Random Forest).
