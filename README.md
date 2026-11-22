# Birdpedia — Identificação de espécies de aves

Projeto da disciplina de Processamento Digital de Imagens (PDI): a partir de uma foto, o sistema separa a ave do fundo, extrai características de forma, cor e textura e diz qual é a espécie.

## Equipe

- **Théo do Rego Barros Serrano**: pré-processamento, filtragem e espaços de cor
- **Guilherme da Rocha Lima Barros**: segmentação e morfologia matemática
- **Douglas Wesley**: características, classificação e avaliação

## Contexto

Identificar a espécie de uma ave numa foto é uma tarefa difícil até para quem não é especialista. Muitas espécies têm cores parecidas, e as fotos variam muito em fundo, iluminação e pose. Ferramentas que fazem isso automaticamente ajudam observadores de aves, projetos de ciência cidadã e o monitoramento de biodiversidade. Nosso objetivo foi montar esse sistema usando técnicas clássicas de PDI vistas na disciplina, sem redes neurais na classificação.

## Base de dados

Usamos o dataset público do Kaggle [NOME DO DATASET](LINK), que tem 20 espécies. Escolhemos 10 delas:

- African Crowned Crane
- African Emerald Cuckoo
- African Firefinch
- Albatross
- Alexandrine Parakeet
- Alpine Chough
- American Avocet
- American Flamingo
- American Goldfinch
- American Kestrel

| Conjunto | Imagens |
|---|---|
| Treino | 1.523 (entre 133 e 179 por espécie) |
| Validação | 50 (5 por espécie) |
| Teste | 50 (5 por espécie) |

Todas as imagens são JPG coloridas de 224×224 pixels. Para a avaliação final juntamos validação e teste num único conjunto de 100 imagens.

## Pipeline

| Etapa | O que é feito | Arquivo |
|---|---|---|
| 1. Aquisição | Leitura das imagens e montagem da base com as 10 espécies | `setup_dataset.py`, `src/preprocessing.py` |
| 2. Pré-processamento | Filtro bilateral (suaviza sem borrar as bordas) e CLAHE no canal L* (realce de contraste) | `src/preprocessing.py` |
| 3. Espaços de cor | Conversão para HSV, CIELAB e tons de cinza | `src/preprocessing.py` |
| 4. Segmentação | Limiarização de Otsu no canal que melhor separa ave e fundo, morfologia matemática (abertura, fechamento, maior componente e preenchimento de buracos) e refinamento com GrabCut | `src/segmentation.py` |
| 5. Características | 292 valores calculados só dentro da máscara da ave: forma (14, momentos de Hu e descritores geométricos), cor (262, histograma H×S e média/desvio em L\*a\*b\*) e textura (16, LBP e GLCM) | `src/features.py` |
| 6. Classificação | Transformação de Hellinger nos histogramas, padronização e SVM ou regressão logística, escolhidos por validação cruzada | `src/classifier.py` |

A interface `app.py` mostra todas essas etapas para uma imagem, do pré-processamento até a espécie prevista.

## Como executar

### 1. Instalar as dependências

É preciso ter Python 3.10 ou superior.

```bash
pip install -r requirements.txt
```

### 2. Preparar a base de dados

Baixe o dataset do Kaggle (link acima), extraia a pasta `archive/` na raiz do projeto e rode:

```bash
python setup_dataset.py
```

Isso cria `data/raw/` com as 10 espécies divididas em `train`, `valid` e `test`. As pastas `archive/` e `data/` estão no `.gitignore`.

### 3. Extrair as características

```bash
python build_features.py
python build_features.py --method otsu
```

O primeiro comando usa a segmentação com GrabCut, que é a padrão, e gera `data/processed/features_grabcut_v2.npz`. O segundo usa só Otsu + morfologia e gera `data/processed/features_otsu_v2.npz`. Cada um também salva um `.json` com tempo de execução, contagem de imagens e checagens de qualidade.

### 4. Treinar e avaliar o classificador

```bash
python -m src.classifier
```

Treina os modelos com as características do GrabCut, escolhe o melhor pela validação cruzada e avalia nas 100 imagens de validação + teste. Tudo vai para `reports/classifier/`: o modelo (`model.joblib`), a matriz de confusão, as métricas por espécie e um `report.md` com o resumo.

Para treinar com as características do Otsu, numa pasta separada:

```bash
python -m src.classifier --features data/processed/features_otsu_v2.npz --out reports/classifier_otsu --models svm logreg rf
```

### 5. Avaliar a segmentação

```bash
python eval_segmentation.py --split test --n 5 --method grabcut --out reports/segmentation/iou_test_grabcut.csv
```

Compara as máscaras com uma referência gerada pelo rembg e calcula IoU e Dice por espécie. A tabela é salva em CSV, já com a média. Troque `--method grabcut` por `--method otsu` para avaliar só o Otsu.

### 6. Procurar imagens duplicadas

```bash
python scripts/check_duplicates.py
```

Compara cada imagem de validação e teste com todas as de treino. Os pares quase idênticos vão para `reports/duplicatas.csv`, e o script também mostra a acurácia sem essas imagens.

### 7. Rodar o teste sem segmentação

```bash
python scripts/baseline_sem_mascara.py
python -m src.classifier --features data/processed/features_sem_mascara_v2.npz --out reports/classifier_sem_mascara --models svm logreg --no-ablation
```

Calcula as mesmas características usando a imagem inteira, sem separar a ave do fundo. Serve para medir o quanto a segmentação ajuda.

### 8. Gerar as figuras de demonstração

```bash
python test_theo_pipeline.py
python test_guilherme_segmentation.py --stats 30
```

O primeiro gera `reports/demonstracao_pre_processamento.png`, com os filtros e os espaços de cor. O segundo gera `reports/demonstracao_segmentacao.png`, com uma ave de cada espécie passando por todas as etapas da segmentação, e mostra estatísticas da área das máscaras.

### 9. Abrir a interface

```bash
streamlit run app.py
```

Abre no navegador em `http://localhost:8501`. Dá para escolher uma imagem da base ou enviar uma foto e acompanhar cada etapa até a classificação. É preciso ter rodado o passo 4 antes, porque a interface usa o `reports/classifier/model.joblib`.

### 10. Classificar uma imagem pelo terminal

```bash
python -m src.classifier --predict caminho/da/imagem.jpg
```

Mostra a espécie prevista, a confiança e as espécies mais prováveis.

## Resultados principais

| Métrica | Sem segmentação | Otsu | GrabCut |
|---|---|---|---|
| F1 macro na validação cruzada (treino) | 0,7267 | 0,7768 | **0,8512** |
| Acurácia em validação + teste (100 imagens) | 0,84 | 0,84 | **0,93** |
| F1 macro em validação + teste (100 imagens) | 0,8380 | 0,8395 | **0,9302** |
| IoU médio da segmentação no teste | — | 0,6642 | **0,7859** |

Cada coluna mostra o modelo que teve o melhor F1 na validação cruzada: SVM sem segmentação e com Otsu, regressão logística com GrabCut. O modelo final, usado na interface, é o do GrabCut.

As tabelas completas estão em [docs/RESULTADOS.md](docs/RESULTADOS.md): resultados por modelo, ablação das características, tempo por imagem e erros por espécie.

## Limitações

- **Conjunto de teste pequeno:** são só 100 imagens (10 por espécie), então cada erro muda a acurácia em 1 ponto. Por isso escolhemos o modelo pela validação cruzada nas 1.523 imagens de treino.
- **Imagens duplicadas:** encontramos 3 imagens de validação/teste praticamente iguais a imagens de treino. Elas não foram removidas; sem elas, a acurácia fica em 92,8%.
- **Referência da segmentação:** o IoU e o Dice foram medidos contra máscaras geradas automaticamente pelo rembg, não desenhadas à mão. Elas servem para comparar Otsu e GrabCut, mas também têm erros.
- **Poucas espécies:** usamos só 10 das 20 espécies do dataset. Com mais espécies parecidas entre si, o problema fica mais difícil.
