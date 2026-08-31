import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import base64
from wordcloud import WordCloud
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

st.markdown("""
<style>
[data-testid="stFileUploaderDropzone"] {
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    gap: 0.5rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    order: 2;
}
[data-testid="stFileUploaderDropzone"] button {
    order: 1;
}
[data-testid="stFileUploaderDropzoneInstructions"] span:first-of-type {
    display: none;
}
[data-testid="stFileUploaderDropzoneInstructions"] div::before {
    content: "Seret file xlsx di sini, atau klik untuk pilih file";
    font-weight: 500;
    display: block;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

GREEN = "#09A750"       
BLUE_BPJS = "#232E7A"   
GREEN_DARK = "#007A3B"  
GREEN_LIGHT = "#E6F5EE" 
GREEN_TINT = "#F2F9F6" 
RED = "#C0392B"         
RED_LIGHT = "#FBEAE8"  
INK = "#1C2321"        
MUTED = "#5B6B64"       
BORDER = "#DCE6E0"            

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Lora:wght@600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    color-scheme: light dark;
}}

/* Overall canvas: faint tinted background instead of stark white,
   reads more like an institutional portal than a raw app shell. */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(180deg, {GREEN_TINT} 0%, var(--background-color) 320px);
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
.block-container {{
    padding-top: 1.6rem;
    max-width: 1200px;
}}

.app-header {{
    padding: 0 0 1.1rem 0;
    border-bottom: 2px solid {GREEN};
    margin-bottom: 1.6rem;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
}}
.app-header h1 {{
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 1.65rem;
    color: {GREEN};
    margin: 0;
    letter-spacing: -0.01em;
}}
.app-header p {{
    color: {MUTED};
    font-size: 0.88rem;
    margin: 0.25rem 0 0 0;
}}
.app-header .tag {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {BLUE_BPJS};
    border: 1px solid {BLUE_BPJS};
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    white-space: nowrap;
}}

.section-eyebrow {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {GREEN};
    margin-bottom: 0.9rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {BORDER};
}}
.section-eyebrow::before {{
    content: "";
    width: 4px;
    height: 14px;
    background: {GREEN};
    border-radius: 2px;
    display: inline-block;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--secondary-background-color);
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    box-shadow: 0 1px 2px rgba(0,67,42,0.04), 0 1px 12px rgba(0,67,42,0.03);
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    padding: 1.35rem 1.4rem 1.1rem 1.4rem;
}}

div[data-testid="stMetric"] {{
    background: transparent;
}}
div[data-testid="stMetricLabel"] {{
    color: {MUTED};
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}}
div[data-testid="stMetricValue"] {{
    color: {GREEN};
    font-weight: 800;
    font-size: 2.05rem;
}}

/* Bigger, hero-style metrics reserved for the home page */
.home-metric div[data-testid="stMetricValue"] {{
    font-size: 2.6rem;
}}
.home-metric div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    padding: 1.7rem 1.5rem 1.4rem 1.5rem;
}}

/* Card title text (used above pyplot / dataframe blocks) */
.card-title {{
    font-weight: 700;
    font-size: 0.95rem;
    color: {INK};
    margin-bottom: 0.5rem;
}}

.pill-positif {{
    background: {GREEN_LIGHT};
    border: 1px solid {GREEN};
    color: {GREEN_DARK};
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    font-weight: 700;
    font-size: 1.15rem;
    text-align: center;
}}
.pill-negatif {{
    background: {RED_LIGHT};
    border: 1px solid {RED};
    color: {RED};
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    font-weight: 700;
    font-size: 1.15rem;
    text-align: center;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {GREEN_DARK} 0%, {GREEN_DARK} 100%);
    border-right: none;
}}
section[data-testid="stSidebar"] * {{
    color: #F2F7F4;
}}
section[data-testid="stSidebar"] .sidebar-brand {{
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.18);
    margin-bottom: 1.1rem;
    line-height: 1.2;
}}
section[data-testid="stSidebar"] .sidebar-brand .brand-title {{
    font-weight: 800;
    font-size: 1.05rem;
    color: #FFFFFF;
    letter-spacing: 0.01em;
}}
section[data-testid="stSidebar"] .sidebar-brand .brand-sub {{
    font-weight: 800;
    font-size: 1.05rem;
    color: #FFFFFF;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}}
section[data-testid="stSidebar"] .sidebar-nav-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin: 0 0 0.5rem 0.1rem;
}}

/* Radio nav items styled as sidebar buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 0.15rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent;
    border-radius: 8px;
    padding: 0.55rem 0.7rem !important;
    font-weight: 500;
    font-size: 0.9rem;
    transition: background 0.15s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.08);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
    background: rgba(255,255,255,0.14);
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.18);
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: rgba(255,255,255,0.55) !important;
}}

div.stButton > button {{
    background: {GREEN};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}}
div.stButton > button:hover {{
    background: {GREEN_DARK};
    color: white;
}}

/* Text area border tint */
div[data-testid="stTextArea"] textarea {{
    border-color: {BORDER} !important;
}}
div[data-testid="stTextArea"] textarea:focus {{
    border-color: {GREEN} !important;
    box-shadow: 0 0 0 1px {GREEN} !important;
}}

/* Dataframe header tint */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
</style>
""", unsafe_allow_html=True)

MPL_PALETTE = {"Positif": GREEN, "Negatif": RED}


def set_mpl_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    })

set_mpl_style()

def header(kicker="Analisis Sentimen", subtitle="Ulasan pengguna aplikasi Mobile JKN pada Google Play Store Periode Januari 2025 - Januari 2026"):
    try:
        with open("logo bpjs.png", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        img_html = f'<img src="data:image/png;base64,{encoded_image}" style="height: 45px; width: auto; object-fit: contain;">'
    except:
        img_html = ''

    st.markdown(
        f"""
        <div style="text-align: left; margin-bottom: 10px;">
            {img_html}
        </div>
        <div class="app-header" style="display: flex; align-items: flex-end; justify-content: space-between; padding-top: 0.2rem;">
            <div>
                <h1 style="margin: 0;">Dashboard Analisis Sentimen Mobile JKN</h1>
                <p style="margin: 4px 0 0 0;">{subtitle}</p>
            </div>
            <span class="tag" style="margin-bottom: 5px;">{kicker}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def eyebrow(text):
    st.markdown(f'<div class="section-eyebrow">{text}</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model_dan_vectorizer():
    model = joblib.load('naive_bayes_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer


@st.cache_data
def load_dataset():
    df = pd.read_excel('df_labeled.xlsx')
    return df


@st.cache_data
def hitung_evaluasi(_model, _vectorizer, df):
    X = _vectorizer.transform(df['teks_bersih'].astype(str))
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred_test = _model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred_test)
    presisi = precision_score(y_test, y_pred_test, pos_label='Positif')
    recall = recall_score(y_test, y_pred_test, pos_label='Positif')
    f1 = f1_score(y_test, y_pred_test, pos_label='Positif')
    cm = confusion_matrix(y_test, y_pred_test, labels=['Negatif', 'Positif'])

    y_pred_train = _model.predict(X_train)
    akurasi_train = accuracy_score(y_train, y_pred_train)
    presisi_train = precision_score(y_train, y_pred_train, pos_label='Positif')
    recall_train = recall_score(y_train, y_pred_train, pos_label='Positif')
    f1_train = f1_score(y_train, y_pred_train, pos_label='Positif')
    cm_train = confusion_matrix(y_train, y_pred_train, labels=['Negatif', 'Positif'])

    return {
        'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1, 'cm': cm,
        'akurasi_train': akurasi_train, 'presisi_train': presisi_train,
        'recall_train': recall_train, 'f1_train': f1_train, 'cm_train': cm_train,
        'n_train': X_train.shape[0], 'n_test': X_test.shape[0]
    }

model, vectorizer = load_model_dan_vectorizer()
df = load_dataset()
eval_hasil = hitung_evaluasi(model, vectorizer, df)

with st.sidebar:
    try:
        with open("logo bpjs putih.png", "rb") as logo_file:
            logo_bpjs_putih_b64 = base64.b64encode(logo_file.read()).decode()
        logo_bpjs_putih_html = f'<img src="data:image/png;base64,{logo_bpjs_putih_b64}" style="height: 80px; width: auto; margin-bottom: 12px;">'
    except:
        logo_bpjs_putih_html = ''

    st.markdown(
        f"""
        <div class="sidebar-brand">
            {logo_bpjs_putih_html}
            <div class="brand-title">Mobile JKN</div>
            <div class="brand-sub">Dashboard Analisis Sentimen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="sidebar-nav-label">Navigasi</div>', unsafe_allow_html=True)
    halaman = st.radio(
        "Navigasi",
        ["Halaman Utama", "Visualisasi Dataset", "Visualisasi TF-IDF Bigram",
         "Evaluasi Model", "Prediksi Sentimen"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Analisis Sentimen Mobile JKN, 2026")

# HALAMAN 1 — UTAMA

if halaman == "Halaman Utama":
    header("Ringkasan")

    total_ulasan = len(df)
    total_positif = int((df['label'] == 'Positif').sum())
    total_negatif = int((df['label'] == 'Negatif').sum())

    eyebrow("Ringkasan Data — Hasil Pelabelan")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.metric("Total Ulasan", f"{total_ulasan:,}".replace(",", "."))
    with c2, st.container(border=True):
        st.metric("Label Positif", f"{total_positif:,}".replace(",", "."))
    with c3, st.container(border=True):
        st.metric("Label Negatif", f"{total_negatif:,}".replace(",", "."))

    eyebrow("Performa Model — Data Training")
    c4, c5, c6, c7 = st.columns(4)
    with c4, st.container(border=True):
        st.metric("Akurasi", f"{eval_hasil['akurasi_train']*100:.2f}%")
    with c5, st.container(border=True):
        st.metric("Presisi", f"{eval_hasil['presisi_train']*100:.2f}%")
    with c6, st.container(border=True):
        st.metric("Recall", f"{eval_hasil['recall_train']*100:.2f}%")
    with c7, st.container(border=True):
        st.metric("F1-Score", f"{eval_hasil['f1_train']*100:.2f}%")

    eyebrow("Performa Model — Data Testing")
    c8, c9, c10, c11 = st.columns(4)
    with c8, st.container(border=True):
        st.metric("Akurasi", f"{eval_hasil['akurasi']*100:.2f}%")
    with c9, st.container(border=True):
        st.metric("Presisi", f"{eval_hasil['presisi']*100:.2f}%")
    with c10, st.container(border=True):
        st.metric("Recall", f"{eval_hasil['recall']*100:.2f}%")
    with c11, st.container(border=True):
        st.metric("F1-Score", f"{eval_hasil['f1']*100:.2f}%")
        
# HALAMAN 2 — VISUALISASI DATASET

elif halaman == "Visualisasi Dataset":
    header("Dataset")

    col1, col2 = st.columns(2)

    with col1, st.container(border=True):
        eyebrow("Distribusi Sentimen")
        label_counts = df['label'].value_counts()
        colors = [MPL_PALETTE.get(lbl, MUTED) for lbl in label_counts.index]
        fig1, ax1 = plt.subplots(figsize=(5, 4.2))
        wedges, texts, autotexts = ax1.pie(
            label_counts, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 10},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax1.axis('equal')
        st.pyplot(fig1, use_container_width=True)

    with col2, st.container(border=True):
        eyebrow("Jumlah Ulasan per Label")
        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        bars = ax2.bar(label_counts.index, label_counts.values,
                        color=[MPL_PALETTE.get(lbl, MUTED) for lbl in label_counts.index],
                        width=0.5)
        ax2.set_ylabel("Jumlah Ulasan")
        ax2.bar_label(bars, fmt='{:,.0f}', padding=4, fontsize=9)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
        ax2.grid(axis="y", color=BORDER, linewidth=0.8)
        ax2.set_axisbelow(True)
        st.pyplot(fig2, use_container_width=True)

    with st.container(border=True):
        eyebrow("Info Split Data")
        n_train = eval_hasil['n_train']
        n_test = eval_hasil['n_test']
        total = n_train + n_test
        c1, c2 = st.columns(2)
        c1.metric("Data Training", f"{n_train:,}".replace(",", ".") + f" ({n_train/total*100:.0f}%)")
        c2.metric("Data Testing", f"{n_test:,}".replace(",", ".") + f" ({n_test/total*100:.0f}%)")

# HALAMAN 3 — VISUALISASI TF-IDF BIGRAM

elif halaman == "Visualisasi TF-IDF Bigram":
    header("TF-IDF Bigram")

    @st.cache_data
    def hitung_top_bigram(df):
        df_pos = df[df['label'] == 'Positif']
        df_neg = df[df['label'] == 'Negatif']

        X_pos = vectorizer.transform(df_pos['teks_bersih'].astype(str))
        X_neg = vectorizer.transform(df_neg['teks_bersih'].astype(str))

        feature_names = vectorizer.get_feature_names_out()
        mean_pos = np.asarray(X_pos.mean(axis=0)).flatten()
        mean_neg = np.asarray(X_neg.mean(axis=0)).flatten()

        top_pos_idx = mean_pos.argsort()[::-1][:20]
        top_neg_idx = mean_neg.argsort()[::-1][:20]

        top_pos = [(feature_names[i], mean_pos[i]) for i in top_pos_idx]
        top_neg = [(feature_names[i], mean_neg[i]) for i in top_neg_idx]

        return top_pos, top_neg

    top_pos, top_neg = hitung_top_bigram(df)

    col1, col2 = st.columns(2)

    with col1, st.container(border=True):
        eyebrow("Top 20 Bigram — Kelas Positif")
        bigram_pos, skor_pos = zip(*top_pos)
        fig3, ax3 = plt.subplots(figsize=(6, 7))
        ax3.barh(bigram_pos[::-1], skor_pos[::-1], color=GREEN)
        ax3.set_xlabel("Bobot TF-IDF Rata-rata")
        ax3.grid(axis="x", color=BORDER, linewidth=0.8)
        ax3.set_axisbelow(True)
        st.pyplot(fig3, use_container_width=True)

    with col2, st.container(border=True):
        eyebrow("Top 20 Bigram — Kelas Negatif")
        bigram_neg, skor_neg = zip(*top_neg)
        fig4, ax4 = plt.subplots(figsize=(6, 7))
        ax4.barh(bigram_neg[::-1], skor_neg[::-1], color=RED)
        ax4.set_xlabel("Bobot TF-IDF Rata-rata")
        ax4.grid(axis="x", color=BORDER, linewidth=0.8)
        ax4.set_axisbelow(True)
        st.pyplot(fig4, use_container_width=True)

    col3, col4 = st.columns(2)

    def get_bigram_freq(teks_series, ngram_range=(2, 2)):
        teks_list = [str(t) for t in teks_series if str(t).strip()]
        if not teks_list:
            return None
        try:
            cv = CountVectorizer(ngram_range=ngram_range)
            freq_matrix = cv.fit_transform(teks_list)
            return dict(zip(cv.get_feature_names_out(), freq_matrix.sum(axis=0).A1))
        except ValueError:
            return None

    freq_pos = get_bigram_freq(df[df['label'] == 'Positif']['teks_bersih'])
    freq_neg = get_bigram_freq(df[df['label'] == 'Negatif']['teks_bersih'])

    with col3, st.container(border=True):
        eyebrow("Word Cloud — Positif")
        if freq_pos:
            wc_pos = WordCloud(width=600, height=340, background_color=None, colormap='Greens').generate_from_frequencies(freq_pos)
            fig5, ax5 = plt.subplots()
            ax5.imshow(wc_pos, interpolation='bilinear')
            ax5.axis('off')
            st.pyplot(fig5, use_container_width=True)
        else:
            st.caption("Data tidak cukup untuk membentuk Bigram Positif.")

    with col4, st.container(border=True):
        eyebrow("Word Cloud — Negatif")
        if freq_neg:
            wc_neg = WordCloud(width=600, height=340, background_color=None, colormap='Reds').generate_from_frequencies(freq_neg)
            fig6, ax6 = plt.subplots()
            ax6.imshow(wc_neg, interpolation='bilinear')
            ax6.axis('off')
            st.pyplot(fig6, use_container_width=True)
        else:
            st.caption("Data tidak cukup untuk membentuk Bigram Negatif.")

# HALAMAN 4 — EVALUASI MODEL

elif halaman == "Evaluasi Model":
    header("Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1, st.container(border=True):
        eyebrow("Confusion Matrix — Data Training")
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        green_cmap = sns.light_palette(GREEN, as_cmap=True)
        sns.heatmap(eval_hasil['cm_train'], annot=True, fmt='d', cmap=green_cmap,
                    xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'],
                    linewidths=1, linecolor="white", cbar=False,
                    annot_kws={"fontsize": 12, "fontweight": "bold"}, ax=ax7)
        ax7.set_xlabel("Prediksi")
        ax7.set_ylabel("Aktual")
        st.pyplot(fig7, use_container_width=True)

    with col2, st.container(border=True):
        eyebrow("Confusion Matrix — Data Testing")
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        sns.heatmap(eval_hasil['cm'], annot=True, fmt='d', cmap=green_cmap,
                    xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'],
                    linewidths=1, linecolor="white", cbar=False,
                    annot_kws={"fontsize": 12, "fontweight": "bold"}, ax=ax8)
        ax8.set_xlabel("Prediksi")
        ax8.set_ylabel("Aktual")
        st.pyplot(fig8, use_container_width=True)
        
    eyebrow("Metrik — Data Training")
    c7, c8, c9, c10 = st.columns(4)
    with c7, st.container(border=True):
        st.metric("Akurasi", f"{eval_hasil['akurasi_train']*100:.2f}%")
    with c8, st.container(border=True):
        st.metric("Presisi", f"{eval_hasil['presisi_train']*100:.2f}%")
    with c9, st.container(border=True):
        st.metric("Recall", f"{eval_hasil['recall_train']*100:.2f}%")
    with c10, st.container(border=True):
        st.metric("F1-Score", f"{eval_hasil['f1_train']*100:.2f}%")    

    eyebrow("Metrik — Data Testing")
    c3, c4, c5, c6 = st.columns(4)
    with c3, st.container(border=True):
        st.metric("Akurasi", f"{eval_hasil['akurasi']*100:.2f}%")
    with c4, st.container(border=True):
        st.metric("Presisi", f"{eval_hasil['presisi']*100:.2f}%")
    with c5, st.container(border=True):
        st.metric("Recall", f"{eval_hasil['recall']*100:.2f}%")
    with c6, st.container(border=True):
        st.metric("F1-Score", f"{eval_hasil['f1']*100:.2f}%")

# HALAMAN 5 — PREDIKSI SENTIMEN

elif halaman == "Prediksi Sentimen":
    header("Prediksi & Auto-Labeling", subtitle="Unggah kumpulan ulasan baru untuk diprediksi sentimennya secara otomatis")

    if 'df_hasil_prediksi' not in st.session_state:
        st.session_state.df_hasil_prediksi = None
    if 'nama_file_upload' not in st.session_state:
        st.session_state.nama_file_upload = None

    @st.cache_data(show_spinner=False)
    def proses_labeling(df_mentah):
        df_mentah = df_mentah.copy()
        df_mentah["teks_bersih"] = df_mentah["ulasan"].apply(preprocessing)
        
        KOREKSI_SENTIMEN_POSITIF = {
            'baik': 2, 'sangat': 2, 'mudah': 3, 'gampang': 2, 'membantu': 5, 'cepat': 2, 'ramah': 3, 'lengkap': 2,
            'sehat': 2, 'bagus': 4, 'keren': 3,'aman': 2, 'mantap': 5, 'memuaskan': 3, 'memudahkan': 3, 'mempermudah': 3,
            'sederhana': 2,'lumayan': 2, 'sukses': 3, 'pertahankan' : 3,'terbantu': 4, 'senang': 4, 'pelayanannya': 2,
        }
        KOREKSI_SENTIMEN_NEGATIF = {
            'perbaiki': -1, 'memperbaiki': -1, 'kadang': -3,'kecewa': -4, 'payah': -4, 'dipersulit': -4, 'persulit': -4,
            'menyulitkan': -4, 'mempersulit': -4, 'sulit': -3, 'kendala': -3, 'gangguan': -3, 'bermasalah': -3,
            'dongo': -4, 'namun': -1, 'susah': -3, 'lambat': -3, 'pusing': -4, 'urgent': -2, 'mengesalkan': -3, 'biasa': -1,
        }
        KATA_DINETRALKAN = {
            'aplikasi': 0, 'periksa': 0, 'keluarga': 0, 'anggota': 0,'login': 0, 'logout': 0, 'sering': 0, 'akses': 0,
            'setelah': 0,'masuk': 0, 'buka': 0, 'keluar': 0, 'suka': 0, 'semoga': 0, 'informasi': 0, 'terima': 0, 'kasih': 0,
            'permudah': 0, 'lancar': 0, 'layan': 0,
        }
        koreksi_skor = {**KOREKSI_SENTIMEN_POSITIF, **KOREKSI_SENTIMEN_NEGATIF, **KATA_DINETRALKAN}
        KATA_NEGASI = {'tidak', 'bukan', 'belum', 'tanpa', 'kurang', 'jangan'}
        JARAK_NEGASI = 2
        FRASA_NEGATIF_TETAP = {('tidak', 'bisa'): -3.0, ('tidak', 'dapat'): -3.0, ('belum', 'bisa'): -3.0, ('tidak', 'ada'): -2.0}
        inset_lexicon = {} 

        def hitung_skor_inset(teks):
            if pd.isna(teks) or str(teks).strip() == '': return 0
            kata_kata = str(teks).strip().split()
            n = len(kata_kata)
            consumed = set()
            total = 0
            KATA_KONTRAS = {'malah', 'padahal', 'tapi', 'tetapi', 'giliran'}

            for i in range(n - 1):
                pair = (kata_kata[i], kata_kata[i + 1])
                if pair in FRASA_NEGATIF_TETAP and i not in consumed and (i + 1) not in consumed:
                    total += int(FRASA_NEGATIF_TETAP[pair])
                    consumed.add(i); consumed.add(i + 1)

            i = 0
            while i < n:
                if i in consumed:
                    i += 1; continue
                kata = kata_kata[i]
                if kata in KATA_NEGASI or kata in KATA_KONTRAS:
                    i += 1; continue
                
                skor_kata = koreksi_skor.get(kata, inset_lexicon.get(kata, 0))
                if skor_kata != 0:
                    window_before = [kata_kata[j] for j in range(max(0, i - JARAK_NEGASI), i) if j not in consumed]
                    ada_negasi = any(w in KATA_NEGASI for w in window_before)
                    ada_kontras = any(w in KATA_KONTRAS for w in window_before)
                    if ada_negasi or ada_kontras:
                        total += int(-1 * skor_kata) if skor_kata > 0 else int(skor_kata)
                    else:
                        total += int(skor_kata)
                i += 1
            return total

        def hybrid_labeling(row):
            if row['rating'] in [1, 2]: return 'Negatif'
            skor = row['skor_inset']
            if skor > 0: return 'Positif'
            elif skor < 0: return 'Negatif'
            else: return None

        df_mentah['skor_inset'] = df_mentah['teks_bersih'].apply(hitung_skor_inset)
        df_mentah['Label_Lexicon'] = df_mentah.apply(hybrid_labeling, axis=1)
        
        df_mentah = df_mentah[df_mentah['Label_Lexicon'].notna()].reset_index(drop=True)

        return df_mentah

    @st.cache_data(show_spinner=False)
    def proses_analisis(df_label):
        df_label = df_label.copy()
        if len(df_label) > 0:
            vec = vectorizer.transform(df_label["teks_bersih"])
            df_label["Prediksi_ML"] = model.predict(vec)
        return df_label

    with st.container(border=True):
        eyebrow("Unggah Ulasan Baru")
        berkas = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")
        st.caption("Maksimal ukuran file 10 MB dan wajib ada kolom 'ulasan' dan 'rating'")

    if berkas is not None:
        df_baru = pd.read_excel(berkas)
        st.success(f"{berkas.name} — {len(df_baru)} baris terbaca")

        if st.session_state.nama_file_upload != berkas.name:
            st.session_state.df_hasil_prediksi = None
            st.session_state.nama_file_upload = berkas.name
            
        if "ulasan" not in df_baru.columns or "rating" not in df_baru.columns:
            st.error("Berkas error: Pastikan ada kolom 'ulasan' dan 'rating'.")
        else:

            if st.session_state.df_hasil_prediksi is None:
                loader_html = """
                <style>
                .custom-loader-wrapper {
                    display: flex;
                    align-items: center;
                    justify-content: flex-start;
                    gap: 12px;
                    margin: 15px 0 25px 0;
                    padding-left: 5px;
                    box-sizing: border-box;
                }
                .custom-spinner {
                    width: 22px;
                    height: 22px;
                    min-width: 22px;
                    min-height: 22px;
                    border: 3px solid #E6F5EE;
                    border-top: 3px solid #09A750;
                    border-radius: 50%;
                    box-sizing: border-box;
                    animation: spin 0.8s linear infinite;
                }
                @keyframes spin { 
                    0% { transform: rotate(0deg); } 
                    100% { transform: rotate(360deg); } 
                }
                .carousel-container {
                    height: 26px;
                    overflow: hidden;
                    position: relative;
                    display: flex;
                    align-items: flex-start;
                    box-sizing: border-box;
                }
                .carousel-text {
                    display: flex;
                    flex-direction: column;
                    font-family: 'Plus Jakarta Sans', sans-serif;
                    font-size: 0.95rem;
                    font-weight: 600;
                    color: var(--text-color);
                    animation: swipeUpText 222s cubic-bezier(0.65, 0, 0.35, 1) forwards;
                }
                .carousel-text span {
                    height: 26px;
                    line-height: 26px;
                    display: block;
                    white-space: nowrap;
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    border: none;
                }
                @keyframes swipeUpText {
                    0%   { transform: translateY(0); }
                    42%  { transform: translateY(0); }       
                    43%  { transform: translateY(-26px); }   
                    82%  { transform: translateY(-26px); }   
                    83%  { transform: translateY(-52px); }   
                    100% { transform: translateY(-52px); }   
                }
                </style>
                
                <div class="custom-loader-wrapper">
                    <div class="custom-spinner"></div>
                    <div class="carousel-container">
                        <div class="carousel-text">
                            <span>Sedang memproses data...</span>
                            <span>Sedang melakukan pelabelan sentimen...</span>
                            <span>Sedang melakukan analisis sentimen...</span>
                        </div>
                    </div>
                </div>
                """
                
                loader_placeholder = st.empty()
                loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
                
                df_label = proses_labeling(df_baru)
                df_final = proses_analisis(df_label)
                
                loader_placeholder.empty()
                st.session_state.df_hasil_prediksi = df_final

    if st.session_state.df_hasil_prediksi is not None:
        df_final = st.session_state.df_hasil_prediksi
        total_berhasil = len(df_final)

        if total_berhasil > 0:
            akurasi_baru = accuracy_score(df_final["Label_Lexicon"], df_final["Prediksi_ML"])
            presisi_baru = precision_score(df_final["Label_Lexicon"], df_final["Prediksi_ML"], pos_label='Positif')
            recall_baru = recall_score(df_final["Label_Lexicon"], df_final["Prediksi_ML"], pos_label='Positif')
            f1_baru = f1_score(df_final["Label_Lexicon"], df_final["Prediksi_ML"], pos_label='Positif')

            eyebrow("Metrik Data Baru")
            c1, c2, c3, c4 = st.columns(4)
            with c1, st.container(border=True):
                st.metric("Akurasi", f"{akurasi_baru*100:.2f}%")
            with c2, st.container(border=True):
                st.metric("Presisi", f"{presisi_baru*100:.2f}%")
            with c3, st.container(border=True):
                st.metric("Recall", f"{recall_baru*100:.2f}%")
            with c4, st.container(border=True):
                st.metric("F1-Score", f"{f1_baru*100:.2f}%")

            cm_baru = confusion_matrix(df_final["Label_Lexicon"], df_final["Prediksi_ML"], labels=["Negatif", "Positif"])
            tn_baru, fp_baru, fn_baru, tp_baru = cm_baru.ravel()

            with st.expander("Rincian Confusion Matrix"):
                st.markdown(
                    f'''
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                        <div style="padding:10px 12px;background:var(--secondary-background-color);border-radius:8px;">
                            <span style="color:var(--text-color);font-size:0.8rem;">True Positive (TP)</span><br>
                            <span style="font-size:1.1rem;font-weight:700;">{tp_baru}</span>
                        </div>
                        <div style="padding:10px 12px;background:var(--secondary-background-color);border-radius:8px;">
                            <span style="color:var(--text-color);font-size:0.8rem;">True Negative (TN)</span><br>
                            <span style="font-size:1.1rem;font-weight:700;">{tn_baru}</span>
                        </div>
                        <div style="padding:10px 12px;background:var(--secondary-background-color);border-radius:8px;">
                            <span style="color:var(--text-color);font-size:0.8rem;">False Positive (FP)</span><br>
                            <span style="font-size:1.1rem;font-weight:700;">{fp_baru}</span>
                        </div>
                        <div style="padding:10px 12px;background:var(--secondary-background-color);border-radius:8px;">
                            <span style="color:var(--text-color);font-size:0.8rem;">False Negative (FN)</span><br>
                            <span style="font-size:1.1rem;font-weight:700;">{fn_baru}</span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            n_positif = int((df_final["Prediksi_ML"] == "Positif").sum())
            n_negatif = int((df_final["Prediksi_ML"] == "Negatif").sum())
            pct_pos = (n_positif / total_berhasil * 100)
            pct_neg = (n_negatif / total_berhasil * 100)

            eyebrow("Distribusi Hasil Prediksi")
            c1_dist, c2_dist = st.columns(2)
            c1_dist.markdown(f'<div class="pill-positif">Positif&nbsp;&nbsp;{n_positif} ({pct_pos:.1f}%)</div>', unsafe_allow_html=True)
            c2_dist.markdown(f'<div class="pill-negatif">Negatif&nbsp;&nbsp;{n_negatif} ({pct_neg:.1f}%)</div>', unsafe_allow_html=True)

            with st.expander("Pratinjau hasil — 10 baris pertama"):
                preview_df = df_final[["ulasan", "Prediksi_ML"]].head(10).rename(columns={"ulasan": "Ulasan", "Prediksi_ML": "Prediksi"})
                st.dataframe(preview_df, use_container_width=True, hide_index=True)

            c5, c6 = st.columns(2)
            
            from sklearn.feature_extraction.text import CountVectorizer
            
            def get_bigram_freq(teks_series):
                teks_list = [str(t) for t in teks_series if str(t).strip()]
                if not teks_list: return None
                try:
                    cv = CountVectorizer(ngram_range=(2, 2))
                    freq_matrix = cv.fit_transform(teks_list)
                    return dict(zip(cv.get_feature_names_out(), freq_matrix.sum(axis=0).A1))
                except ValueError:
                    return None

            freq_pos = get_bigram_freq(df_final.loc[df_final["Prediksi_ML"] == "Positif", "teks_bersih"])
            freq_neg = get_bigram_freq(df_final.loc[df_final["Prediksi_ML"] == "Negatif", "teks_bersih"])

            with c5, st.container(border=True):
                eyebrow("Word Cloud — Positif")
                if freq_pos:
                    wc_pos = WordCloud(width=500, height=300, background_color=None, colormap='Greens').generate_from_frequencies(freq_pos)
                    figA, axA = plt.subplots(figsize=(5, 3))
                    axA.imshow(wc_pos, interpolation='bilinear')
                    axA.axis('off')
                    st.pyplot(figA, use_container_width=True)
                else:
                    st.caption("Data tidak cukup untuk membentuk Bigram Positif.")

            with c6, st.container(border=True):
                eyebrow("Word Cloud — Negatif")
                if freq_neg:
                    wc_neg = WordCloud(width=500, height=300, background_color=None, colormap='Reds').generate_from_frequencies(freq_neg)
                    figB, axB = plt.subplots(figsize=(5, 3))
                    axB.imshow(wc_neg, interpolation='bilinear')
                    axB.axis('off')
                    st.pyplot(figB, use_container_width=True)
                else:
                    st.caption("Data tidak cukup untuk membentuk Bigram Negatif.")

            
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_final[["ulasan", "Prediksi_ML"]].rename(
                    columns={"ulasan": "Ulasan", "Prediksi_ML": "Prediksi"}
                ).to_excel(writer, index=False, sheet_name="Hasil Prediksi")

            col_left, col_right = st.columns([4, 1])
            with col_right:
                st.download_button(
                    label="Unduh Hasil Prediksi", data=buffer.getvalue(),
                    file_name="hasil_prediksi_baru.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.warning("Semua teks di dalam file bernilai Netral/Kosong menurut Lexicon, sehingga tidak bisa diproses.")
