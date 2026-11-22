# Birdpedia — Resumo dos resultados

Todos os números foram tirados dos arquivos gerados em `data/processed/` e `reports/`. A fonte de cada tabela está indicada abaixo dela.

## 1. Base de dados

| Item | Valor |
|---|---|
| Espécies | 10 |
| Total de imagens | 1623 |
| Treino | 1523 (de 133 a 179 por espécie) |
| Validação (valid) | 50 (5 por espécie) |
| Teste (test) | 50 (5 por espécie) |
| Holdout usado na avaliação final | 100 (valid + test, 10 por espécie) |
| Formato | JPEG, colorida (3 canais) |
| Resolução | 224 × 224 (todas as 1623 imagens) |

| Espécie | Treino | Valid | Test |
|---|---|---|---|
| AFRICAN CROWNED CRANE | 137 | 5 | 5 |
| AFRICAN EMERALD CUCKOO | 154 | 5 | 5 |
| AFRICAN FIREFINCH | 137 | 5 | 5 |
| ALBATROSS | 133 | 5 | 5 |
| ALEXANDRINE PARAKEET | 165 | 5 | 5 |
| ALPINE CHOUGH | 166 | 5 | 5 |
| AMERICAN AVOCET | 179 | 5 | 5 |
| AMERICAN FLAMINGO | 164 | 5 | 5 |
| AMERICAN GOLDFINCH | 133 | 5 | 5 |
| AMERICAN KESTREL | 155 | 5 | 5 |

Fonte: `data/processed/features_grabcut_v2.json` (campo `counts`). O formato e a resolução foram conferidos lendo todas as imagens de `data/raw/`.

## 2. Segmentação: Otsu × GrabCut

IoU e Dice médios contra a máscara de referência (pseudo-gabarito gerado pelo rembg/u2net):

| Conjunto | Imagens | Otsu IoU | Otsu Dice | GrabCut IoU | GrabCut Dice |
|---|---|---|---|---|---|
| Treino | 200 (20 por espécie) | 0,6674 | 0,7783 | **0,7727** | **0,8591** |
| Teste | 50 (5 por espécie) | 0,6642 | 0,7690 | **0,7859** | **0,8679** |

Tempo por imagem na extração completa (1623 imagens, 8 processos em paralelo):

| Método | Tempo total | Tempo por imagem |
|---|---|---|
| Otsu + morfologia | 45,4 s | 0,028 s |
| Otsu + morfologia + GrabCut | 180,4 s | 0,111 s |

Esse tempo cobre o pipeline inteiro de cada imagem: pré-processamento, segmentação e extração das características. Como os 8 processos rodam em paralelo, ele é o tempo total dividido pelo número de imagens.

Fontes: `reports/segmentation/iou_{train,test}_{otsu,grabcut}.csv` (inclui a tabela por espécie) e `data/processed/features_{otsu,grabcut}_v2.json` (`elapsed_seconds`, `seconds_per_image`).

## 3. Classificação

F1 macro na validação cruzada estratificada de 5 folds sobre o treino (1523 imagens), média ± desvio-padrão:

| Modelo | Máscara Otsu | Máscara GrabCut | Sem máscara |
|---|---|---|---|
| SVM (Hellinger + RBF) | **0,7768 ± 0,0122** | 0,8469 ± 0,0162 | **0,7267 ± 0,0118** |
| Regressão logística (Hellinger) | 0,7638 ± 0,0252 | **0,8512 ± 0,0191** | 0,7172 ± 0,0120 |
| Random Forest | 0,7588 ± 0,0204 | 0,8426 ± 0,0111 | — |

"Sem máscara" é o baseline: as mesmas 292 características calculadas sobre a imagem inteira, sem segmentar a ave. O Random Forest não foi rodado nesse caso.

O modelo selecionado em cada caso é o de maior F1 na CV (em negrito). A avaliação dele no holdout (100 imagens) foi:

| Máscara | Modelo selecionado | Acurácia | F1 macro | Kappa |
|---|---|---|---|---|
| Otsu | SVM | 0,8400 | 0,8395 | 0,8222 |
| GrabCut | Regressão logística | **0,9300** | **0,9302** | **0,9222** |
| Sem máscara | SVM | 0,8400 | 0,8380 | 0,8222 |

No holdout, Otsu e sem máscara empatam em 0,84 de acurácia, mas na CV, que usa 1523 imagens, o Otsu fica 5 pontos acima (0,7768 contra 0,7267).

O modelo usado no app (`reports/classifier/model.joblib`) é o GrabCut + regressão logística. A rodada com os três modelos reproduziu os mesmos valores e as mesmas predições no holdout.

Fontes: `reports/classifier_otsu/`, `reports/classifier_grabcut_3modelos/`, `reports/classifier_sem_mascara/` e `reports/classifier/` (`model_comparison.csv`, `report.md`, `metrics.json`).

## 4. Ablação das características (máscara GrabCut)

A ablação usa o **SVM** (Hellinger + RBF) com validação cruzada de 5 folds no treino:

| Características | Nº de características | F1 macro (CV) |
|---|---|---|
| Forma | 14 | 0,2895 ± 0,0284 |
| Textura | 16 | 0,4617 ± 0,0127 |
| Forma + textura | 30 | 0,5889 ± 0,0063 |
| Cor | 262 | 0,7577 ± 0,0090 |
| Forma + cor | 276 | 0,7908 ± 0,0146 |
| Cor + textura | 278 | 0,8309 ± 0,0154 |
| **Todas** | **292** | **0,8469 ± 0,0162** |

Fonte: `reports/classifier_grabcut_3modelos/ablation.csv`.

## 5. Limitações

**Imagens duplicadas entre treino e holdout.** Foram encontrados 3 pares quase idênticos, com correlação acima de 0,97 entre miniaturas 16×16 em tons de cinza:

| Holdout | Treino | Correlação |
|---|---|---|
| test/AFRICAN CROWNED CRANE/2.jpg | train/AFRICAN CROWNED CRANE/074.jpg | 0,9998 |
| valid/AFRICAN CROWNED CRANE/4.jpg | train/AFRICAN CROWNED CRANE/131.jpg | 0,9990 |
| valid/ALBATROSS/3.jpg | train/ALBATROSS/052.jpg | 0,9854 |

As três foram classificadas corretamente. Sem elas, a acurácia no holdout fica assim:

| Modelo | Holdout completo (100) | Sem as duplicatas (97) |
|---|---|---|
| GrabCut + regressão logística (app) | 0,9300 | 0,9278 |
| Otsu + SVM | 0,8400 | 0,8351 |

As imagens não foram removidas do dataset. Fontes: `reports/duplicatas.csv` e `scripts/check_duplicates.py`.

**Holdout pequeno.** São só 100 imagens, 10 por espécie, então cada erro vale 1 ponto percentual de acurácia. Por isso a validação cruzada nas 1523 imagens de treino é a métrica usada para escolher o modelo.

**Referência de segmentação.** O IoU e o Dice são medidos contra máscaras geradas automaticamente pelo rembg (u2net), não contra anotação manual. A referência também erra e foi usada só como régua de comparação entre Otsu e GrabCut, numa amostra de 250 imagens (200 de treino e 50 de teste).

**Espécies mais confundidas.** O modelo do app (GrabCut + regressão logística) erra 7 das 100 imagens do holdout:

| Espécie real | Predita como | Erros |
|---|---|---|
| AFRICAN EMERALD CUCKOO | AMERICAN KESTREL | 2 |
| AMERICAN AVOCET | ALBATROSS | 2 |
| ALBATROSS | AFRICAN CROWNED CRANE | 1 |
| AMERICAN KESTREL | ALBATROSS | 1 |
| AMERICAN KESTREL | AMERICAN AVOCET | 1 |

As espécies com menor F1 no holdout são:

| Espécie | F1 no holdout |
|---|---|
| AMERICAN KESTREL | 0,80 |
| ALBATROSS | 0,82 |
| AMERICAN AVOCET | 0,84 |
| AFRICAN EMERALD CUCKOO | 0,89 |

As outras seis espécies têm F1 de 0,95 ou mais. Fontes: `reports/classifier/holdout_predictions.csv` e `reports/classifier_grabcut_3modelos/metrics_per_class.csv`.
