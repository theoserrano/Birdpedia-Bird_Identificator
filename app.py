import random
from pathlib import Path

import altair as alt
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from skimage.feature import local_binary_pattern

from src import segmentation as seg
from src.classifier import DEFAULT_MODEL
from src.features import (
    COLOR_FEATURE_NAMES,
    FEATURE_GROUPS,
    H_BINS,
    LBP_BINS,
    LBP_P,
    LBP_R,
    S_BINS,
    SHAPE_FEATURE_NAMES,
    TEXTURE_FEATURE_NAMES,
    extract_features,
)
from src.preprocessing import apply_bilateral_filter, preprocess_pipeline

DATA_DIR = Path("data/raw")
BAR_COLOR = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CHANNEL_NAMES = {
    "s": "S (saturação, HSV)",
    "a": "a* (verde ↔ vermelho, Lab)",
    "b": "b* (azul ↔ amarelo, Lab)",
    "l": "L* (luminosidade, Lab)",
    "d": "D (distância de cor ao fundo)",
}

st.set_page_config(page_title="Birdpedia", page_icon="🐦", layout="wide")


def pretty(species: str) -> str:
    return species.title()


@st.cache_resource
def load_model(path: str):
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact.get("metadata", {})
    return artifact, {}


@st.cache_data
def list_dataset(split: str):
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return {}
    return {sp.name: sorted(str(p) for p in sp.glob("*.jpg"))
            for sp in sorted(split_dir.iterdir()) if sp.is_dir()}


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def overlay_contour(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    out = img_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def labeled_result(img_bgr: np.ndarray, mask: np.ndarray, label: str, scale: int = 3) -> np.ndarray:
    big = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    big = overlay_contour(big, big_mask, (0, 220, 255))
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, tht), base = cv2.getTextSize(label, font, fs, th)
    cv2.rectangle(big, (0, 0), (tw + 20, tht + base + 16), (20, 20, 20), -1)
    cv2.putText(big, label, (10, tht + 8), font, fs, (255, 255, 255), th, cv2.LINE_AA)
    return to_rgb(big)


def bar_chart(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
              horizontal: bool = False, fmt: str = ".3f", height: int = 260):
    if horizontal:
        enc = dict(x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format="%")),
                   y=alt.Y(f"{y}:N", title=None, sort="-x"))
    else:
        enc = dict(x=alt.X(f"{y}:N", title=y_title, sort=None),
                   y=alt.Y(f"{x}:Q", title=x_title))
    chart = (alt.Chart(df)
             .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4)
             .encode(**enc, tooltip=[alt.Tooltip(f"{y}:N", title=y_title),
                                     alt.Tooltip(f"{x}:Q", title=x_title, format=fmt)])
             .properties(height=height))
    st.altair_chart(chart, width="stretch")


st.title("🐦 Birdpedia — identificação de aves passo a passo")
st.caption("Processamento Digital de Imagens · pré-processamento → segmentação → "
           "extração de características → classificação")

model_path = Path(DEFAULT_MODEL)
if not model_path.exists():
    st.error("Modelo treinado não encontrado. Rode antes:\n\n"
             "`python build_features.py` e depois `python -m src.classifier`")
    st.stop()
pipeline, metadata = load_model(str(model_path))
method = metadata.get("method", "otsu")
MODEL_LABELS = {"logreg": "Regressão logística + Hellinger", "svm": "SVM (RBF) + Hellinger",
                "rf": "Random Forest"}
model_label = MODEL_LABELS.get(metadata.get("model_name"), str(metadata.get("model_name", "?")))
indices = metadata.get("feature_indices")

with st.sidebar:
    st.header("Imagem de entrada")
    source = st.radio("Origem", ["Dataset (imagens de teste)", "Enviar uma imagem"])
    true_species = None
    image_bgr = None
    if source.startswith("Dataset"):
        split = st.selectbox("Conjunto", ["test", "valid", "train"],
                             help="test e valid não foram usados no treino do classificador.")
        dataset = list_dataset(split)
        if not dataset:
            st.error(f"{DATA_DIR / split} não encontrado. Rode `python setup_dataset.py`.")
            st.stop()
        species = st.selectbox("Espécie", list(dataset), format_func=pretty)
        files = dataset[species]
        key = f"idx_{split}_{species}"
        if key not in st.session_state:
            st.session_state[key] = 0
        if st.button("🎲 Sortear outra imagem", width="stretch"):
            st.session_state[key] = random.randrange(len(files))
        idx = st.selectbox("Arquivo", range(len(files)), index=st.session_state[key],
                           format_func=lambda i: Path(files[i]).name)
        st.session_state[key] = idx
        image_bgr = cv2.imread(files[idx], cv2.IMREAD_COLOR)
        true_species = species
    else:
        up = st.file_uploader("Foto de uma ave (jpg/png)", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if up is None:
            st.info("Envie uma imagem para começar.")
            st.stop()
        image_bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Não foi possível ler a imagem.")
            st.stop()

    st.divider()
    st.caption(f"Classificador: **{model_label}** · "
               f"segmentação usada no treino: **{method}** · "
               f"{len(metadata.get('class_names', []))} espécies")
    st.caption("O app só reconhece as 10 espécies do treino; qualquer outra ave será "
               "classificada como a mais parecida entre elas.")

mask, steps = seg.get_bird_mask(image_bgr, return_steps=True, method=method)
pre = steps["preprocessed"]
spaces = pre["color_spaces"]
filtered = pre["filtered_bgr"]

st.header("1 · Pré-processamento")
st.markdown("A imagem é redimensionada para **224×224**, suavizada com **filtro bilateral** "
            "(remove ruído preservando as bordas da plumagem e do bico) e tem o contraste "
            "realçado com **CLAHE** no canal L* do Lab (equalização de histograma adaptativa "
            "por blocos de 8×8, limite de recorte 2,0).")
c1, c2, c3 = st.columns(3)
c1.image(to_rgb(pre["original_bgr"]), caption="Original (224×224)", width="stretch")
c2.image(to_rgb(apply_bilateral_filter(pre["original_bgr"])),
         caption="Filtro bilateral (d=9, σ=75)", width="stretch")
c3.image(to_rgb(filtered), caption="Bilateral + CLAHE", width="stretch")

st.subheader("Espaços de cor")
st.markdown("A imagem filtrada é decomposta em **HSV**, **CIELAB** e **escala de cinza**; "
            "cada canal destaca uma propriedade diferente da ave.")
chan_cols = st.columns(7)
for col, (k, name) in zip(chan_cols, [("h", "H (matiz)"), ("s", "S (saturação)"),
                                      ("v", "V (brilho)"), ("l", "L* (luminosidade)"),
                                      ("a", "a* (verde↔vermelho)"), ("b", "b* (azul↔amarelo)"),
                                      ("gray", "Cinza")]):
    col.image(spaces[k], caption=name, width="stretch", clamp=True)

use_grabcut = method == "grabcut"
st.header("2 · Segmentação: Otsu + morfologia matemática"
          + (" + GrabCut" if use_grabcut else ""))
st.markdown("Cada canal candidato é limiarizado com **Otsu** (limiar que maximiza a variância "
            "entre as classes). A polaridade é corrigida para que a ave fique branca, usando "
            "a ideia de que a ave ocupa mais o **centro** que a **moldura** da foto. "
            "O canal escolhido é o de maior pontuação: **separabilidade η × (cobertura do "
            "centro − cobertura da borda)**.")

weights = seg.center_weight_map(spaces["gray"].shape)
cand_spaces = dict(spaces)
cand_spaces["d"] = seg.background_distance_channel(spaces["lab"])
chosen = steps["channel"]
cand_cols = st.columns(len(seg.DEFAULT_CANDIDATES))
for col, k in zip(cand_cols, seg.DEFAULT_CANDIDATES):
    r = seg.score_channel(cand_spaces[k], 5, weights)
    mark = " ✅ escolhido" if k == chosen else ""
    col.markdown(f"**{CHANNEL_NAMES[k]}**{mark}")
    col.image(cand_spaces[k], width="stretch", clamp=True)
    col.image(r["mask"], width="stretch", clamp=True)
    col.caption(f"limiar {r['threshold']:.0f} · η {r['separability']:.2f}"
                f"{' · invertido' if r['inverted'] else ''}  \n"
                f"pontuação **{r['score']:.3f}**")

st.subheader("Morfologia matemática")
st.markdown("Sobre a máscara do canal escolhido, com elemento estruturante **elíptico**: "
            "**abertura** 5×5 remove ruídos pequenos do fundo; **fechamento** 11×11 une partes "
            "do corpo; mantém-se o **componente conexo** de maior área ponderada pelo centro; "
            "e os **buracos internos** são preenchidos (flood fill a partir da borda).")
morph = [("otsu_polarity", "Otsu + polaridade"), ("opened", "Abertura (5×5)"),
         ("closed", "Fechamento (11×11)"), ("largest", "Componente central"),
         ("filled", "Buracos preenchidos")]
m_cols = st.columns(len(morph))
for col, (k, name) in zip(m_cols, morph):
    col.image(steps[k], caption=name, width="stretch", clamp=True)

if use_grabcut:
    st.subheader("Refinamento com GrabCut")
    st.markdown("O GrabCut modela as cores da ave e do fundo com **misturas de Gaussianas "
                "(GMM)** e separa os dois por **corte mínimo em grafo**, o que une partes de "
                "cores diferentes que um único limiar de Otsu não separa. É inicializado com a "
                "moldura da imagem como fundo certo e o interior como provável ave; o "
                "resultado passa pela **mesma morfologia** acima.")
    gc = steps["grabcut"]
    used = steps["method"]
    if used == "grabcut+otsu":
        st.info("O GrabCut por retângulo colapsou (área < 10%); foi refeito ancorado no "
                "núcleo da máscara de Otsu (erosão 15×15) como ave certa.")
    elif used == "otsu":
        st.warning("O GrabCut falhou nas duas tentativas; a máscara final é a de "
                   "Otsu + morfologia.")
    g1, g2, g3, g4 = st.columns(4)
    g1.image(steps["otsu_mask"], caption="Máscara Otsu + morfologia", width="stretch", clamp=True)
    if gc:
        seeds = gc["seeds"]
        seed_img = np.full(seeds.shape, 110, np.uint8)
        seed_img[seeds == 0] = 0
        seed_img[seeds == 1] = 255
        g2.image(seed_img, caption="Sementes: fundo certo (preto), provável ave (cinza), "
                 "ave certa (branco)", width="stretch", clamp=True)
        g3.image(gc["raw"], caption="Saída do GrabCut", width="stretch", clamp=True)
        g4.image(gc["filled"], caption="GrabCut + morfologia", width="stretch", clamp=True)

st.subheader("Resultado da segmentação")
r1, r2, r3 = st.columns(3)
r1.image(mask, caption=f"Máscara final · área {steps['area_fraction']:.0%} da imagem",
         width="stretch", clamp=True)
r2.image(to_rgb(overlay_contour(filtered, mask)), caption="Contorno sobre a imagem",
         width="stretch")
r3.image(to_rgb(seg.apply_mask(filtered, mask)), caption="Ave segmentada",
         width="stretch")

feat = extract_features(image_bgr, mask=mask)
vector = feat["vector"]
meta = feat["meta"]
st.header("3 · Extração de características")
st.markdown(f"Da região da ave sai um vetor com **{len(vector)} características** em três "
            f"grupos: **forma** ({len(SHAPE_FEATURE_NAMES)}), **cor** "
            f"({len(COLOR_FEATURE_NAMES)}) e **textura** ({len(TEXTURE_FEATURE_NAMES)}).")
if meta.get("mask_fallback"):
    st.warning("A máscara ficou com área implausível; as características foram extraídas da "
               "imagem inteira.")

shape_v = vector[slice(*FEATURE_GROUPS["shape"])]
color_v = vector[slice(*FEATURE_GROUPS["color"])]
texture_v = vector[slice(*FEATURE_GROUPS["texture"])]

t_shape, t_color, t_texture = st.tabs(["Forma", "Cor", "Textura"])
with t_shape:
    a, b = st.columns([1, 2])
    contour_img = np.zeros((*mask.shape, 3), np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [cv2.convexHull(c)], -1, (90, 90, 90), 1)
        cv2.drawContours(contour_img, [c], -1, (255, 255, 255), 1)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (214, 120, 42), 1)
    a.image(to_rgb(contour_img), caption="Contorno (branco), fecho convexo (cinza) e "
            "retângulo envolvente (azul)", width="stretch")
    labels = [f"Momento de Hu φ{i + 1} (−sinal·log₁₀|φ|)" for i in range(7)] + [
        "Área (fração da imagem)", "Perímetro normalizado", "Circularidade 4πA/P²",
        "Extensão (área / retângulo)", "Razão de aspecto (L/A)", "Solidez (área / fecho convexo)",
        "Excentricidade"]
    b.markdown("Momentos de **Hu** (invariantes a translação, escala e rotação) e descritores "
               "geométricos do contorno.")
    b.dataframe(pd.DataFrame({"Descritor": labels, "Valor": shape_v}),
                hide_index=True, width="stretch",
                column_config={"Valor": st.column_config.NumberColumn(format="%.4f")})

with t_color:
    a, b = st.columns([3, 2])
    hist = color_v[:H_BINS * S_BINS].reshape(H_BINS, S_BINS)
    hs = pd.DataFrame([{"Matiz (H)": f"{int(i * 180 / H_BINS)}–{int((i + 1) * 180 / H_BINS)}",
                        "Saturação (S)": f"{int(j * 256 / S_BINS)}–{int((j + 1) * 256 / S_BINS)}",
                        "hi": i, "sj": j, "Fração dos pixels": hist[i, j]}
                       for i in range(H_BINS) for j in range(S_BINS)])
    heat = (alt.Chart(hs).mark_rect(stroke="white", strokeWidth=0.5)
            .encode(x=alt.X("sj:O", title="Saturação (S) →", axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("hi:O", title="Matiz (H) →", axis=alt.Axis(labels=False, ticks=False)),
                    color=alt.Color("Fração dos pixels:Q",
                                    scale=alt.Scale(range=SEQUENTIAL), legend=alt.Legend(format="%")),
                    tooltip=["Matiz (H)", "Saturação (S)",
                             alt.Tooltip("Fração dos pixels:Q", format=".1%")])
            .properties(height=360))
    a.markdown(f"**Histograma 2D H×S** ({H_BINS}×{S_BINS} bins) só dos pixels da ave: "
               "quais combinações de cor e saturação a plumagem tem.")
    a.altair_chart(heat, width="stretch")
    lab_stats = color_v[H_BINS * S_BINS:]
    b.markdown("**Estatísticas no CIELAB** (média e desvio padrão de cada canal, na escala "
               "0–255 do OpenCV).")
    b.dataframe(pd.DataFrame({"Canal": ["L*", "a*", "b*"],
                              "Média": lab_stats[0::2], "Desvio padrão": lab_stats[1::2]}),
                hide_index=True, width="stretch",
                column_config={"Média": st.column_config.NumberColumn(format="%.1f"),
                               "Desvio padrão": st.column_config.NumberColumn(format="%.1f")})
    b.image(to_rgb(seg.apply_mask(filtered, mask)), caption="Região analisada",
            width="stretch")

with t_texture:
    a, b = st.columns([1, 2])
    lbp = local_binary_pattern(spaces["gray"], P=LBP_P, R=LBP_R, method="uniform")
    lbp_img = (lbp * (255.0 / max(1, LBP_BINS - 1))).astype(np.uint8)
    lbp_img[mask == 0] = 0
    a.image(lbp_img, caption=f"Mapa LBP uniforme (P={LBP_P}, R={LBP_R}) na ave",
            width="stretch", clamp=True)
    with b:
        st.markdown("**LBP** (Local Binary Patterns): compara cada pixel com seus vizinhos "
                    "e conta os padrões locais (bordas, cantos, regiões lisas). "
                    "**GLCM**: estatísticas da matriz de coocorrência de níveis de cinza.")
        lbp_df = pd.DataFrame({"Padrão": [f"P{i}" if i < LBP_BINS - 1 else "não uniforme"
                                          for i in range(LBP_BINS)],
                               "Fração": texture_v[:LBP_BINS]})
        bar_chart(lbp_df, "Fração", "Padrão", "Fração dos pixels", "Padrão LBP", height=220)
        glcm_names = ["Contraste", "Dissimilaridade", "Homogeneidade", "Energia",
                      "Correlação", "ASM (segundo momento angular)"]
        st.dataframe(pd.DataFrame({"Propriedade GLCM": glcm_names,
                                   "Valor": texture_v[LBP_BINS:]}),
                     hide_index=True, width="stretch",
                     column_config={"Valor": st.column_config.NumberColumn(format="%.4f")})

st.header("4 · Classificação")
x = vector if indices is None else vector[np.asarray(indices, dtype=int)]
x = x.reshape(1, -1)
classes = [str(c) for c in pipeline.steps[-1][1].classes_]
proba = pipeline.predict_proba(x)[0]
order = np.argsort(proba)[::-1]
pred = classes[order[0]]
conf = float(proba[order[0]])

st.markdown(f"O vetor de características alimenta o classificador "
            f"**{model_label}** treinado com "
            f"{metadata.get('n_train', '?')} imagens das 10 espécies."
            + (" Antes, os histogramas de cor (H×S) e de textura (LBP) passam pela "
               "**transformação de Hellinger** (raiz quadrada de cada bin): assim a distância "
               "entre vetores passa a medir a semelhança entre distribuições, e os bins "
               "pequenos, que distinguem detalhes da plumagem, deixam de ser ofuscados "
               "pelos grandes." if metadata.get("hellinger_columns") else ""))
a, b = st.columns([3, 2])
a.image(labeled_result(filtered, mask, f"{pretty(pred)} ({conf:.0%})"),
        width="stretch")
with b:
    st.metric("Espécie prevista", pretty(pred), f"confiança {conf:.0%}", delta_color="off")
    if true_species is not None:
        if true_species == pred:
            st.success(f"✔ Acertou — espécie real: {pretty(true_species)}")
        else:
            st.error(f"✘ Errou — espécie real: {pretty(true_species)}")
    prob_df = pd.DataFrame({"Espécie": [pretty(classes[i]) for i in order],
                            "Probabilidade": proba[order]})
    st.markdown("**Probabilidade por espécie**")
    bar_chart(prob_df, "Probabilidade", "Espécie", "Probabilidade", "Espécie",
              horizontal=True, fmt=".1%", height=320)
