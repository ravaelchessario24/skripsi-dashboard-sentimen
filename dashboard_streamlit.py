import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import base64
import io
from datetime import datetime
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)
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
/* ===== Halaman Utama: KPI card ===== */
.kpi-label {{
    color: {MUTED};
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
}}
.kpi-value {{
    font-weight: 800;
    font-size: 1.9rem;
    color: {GREEN};
    line-height: 1.15;
}}
.kpi-sub {{
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 0.4rem;
}}
.kpi-sub.green {{ color: {GREEN_DARK}; }}
.kpi-sub.red {{ color: {RED}; }}
.kpi-sub.neutral {{ color: {MUTED}; font-weight: 600; }}
.mm-label {{
    color: {MUTED};
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}}
.mm-value {{
    font-weight: 800;
    font-size: 1.7rem;
    color: {GREEN};
}}
.legend-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.55rem;
    color: {INK};
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.legend-dot.green {{ background: {GREEN}; }}
.legend-dot.red {{ background: {RED}; }}
.legend-count {{ color: {MUTED}; font-weight: 500; margin-left: auto; }}
/* ===== Insight Bisnis ===== */
.insight-card {{
    border-left: 4px solid {GREEN};
    background: {GREEN_TINT};
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 8px;
    font-size: 0.92rem;
    color: {INK};
    line-height: 1.5;
}}
.insight-card.warn {{
    border-left-color: {RED};
    background: {RED_LIGHT};
}}
/* ===== end tambahan ===== */
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
div[data-testid="stTextArea"] textarea {{
    border-color: {BORDER} !important;
}}
div[data-testid="stTextArea"] textarea:focus {{
    border-color: {GREEN} !important;
    box-shadow: 0 0 0 1px {GREEN} !important;
}}
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
</style>
""", unsafe_allow_html=True)

MPL_PALETTE = {"Positif": GREEN, "Negatif": RED}

KATEGORI_ISU = {
    "Login & Akun": ["login", "akun", "otp", "daftar", "masuk"],
    "Error & Bug": ["error", "bug", "gagal", "force", "crash", "eror"],
    "Performa & Kecepatan": ["lambat", "lemot", "lama", "loading", "lelet"],
    "Sulit Digunakan": ["sulit", "susah", "ribet", "bingung", "rumit"],
    "Layanan & Fitur": ["layanan", "fitur", "antrian", "jadwal", "pelayanan"],
}

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

def kategorikan_bigram(freq_dict, kategori_map):
    hasil = {k: 0 for k in kategori_map}
    hasil["Lainnya"] = 0
    for bigram, jumlah in freq_dict.items():
        matched = False
        for kategori, keywords in kategori_map.items():
            if any(kw in bigram for kw in keywords):
                hasil[kategori] += int(jumlah)
                matched = True
                break
        if not matched:
            hasil["Lainnya"] += int(jumlah)
    return hasil

def generate_laporan_pdf(nama_file, total, akurasi, presisi, recall, f1, n_pos, n_neg, freq_pos, freq_neg):
    buffer_pdf = io.BytesIO()
    with PdfPages(buffer_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        teks = (
            "Laporan Ringkas Prediksi Sentimen\n"
            "Mobile JKN - BPJS Kesehatan\n\n"
            f"Berkas: {nama_file}\n"
            f"Waktu: {datetime.now().strftime('%d %B %Y %H:%M')}\n\n"
            f"Jumlah Data: {total:,}\n".replace(",", ".") +
            f"Distribusi: Positif {n_pos:,} | Negatif {n_neg:,}\n\n".replace(",", ".") +
            f"Akurasi: {akurasi*100:.2f}%\n"
            f"Presisi: {presisi*100:.2f}%\n"
            f"Recall: {recall*100:.2f}%\n"
            f"F1-Score: {f1*100:.2f}%\n"
        )
        ax.text(0.05, 0.95, teks, va='top', fontsize=12, family='sans-serif')
        pdf.savefig(fig)
        plt.close(fig)

        if freq_pos or freq_neg:
            fig2, axes2 = plt.subplots(1, 2, figsize=(11.69, 5))
            if freq_pos:
                wc_pos = WordCloud(width=600, height=350, background_color="white", colormap="Greens").generate_from_frequencies(freq_pos)
                axes2[0].imshow(wc_pos)
                axes2[0].axis('off')
                axes2[0].set_title("Word Cloud Positif")
            else:
                axes2[0].axis('off')
            if freq_neg:
                wc_neg = WordCloud(width=600, height=350, background_color="white", colormap="Reds").generate_from_frequencies(freq_neg)
                axes2[1].imshow(wc_neg)
                axes2[1].axis('off')
                axes2[1].set_title("Word Cloud Negatif")
            else:
                axes2[1].axis('off')
            pdf.savefig(fig2)
            plt.close(fig2)
    buffer_pdf.seek(0)
    return buffer_pdf.getvalue()

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

    classes_list = list(_model.classes_)
    idx_pos = classes_list.index('Positif')
    proba_test = _model.predict_proba(X_test)
    y_test_bin = (y_test == 'Positif').astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, proba_test[:, idx_pos])
    roc_auc = auc(fpr, tpr)

    return {
        'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1, 'cm': cm,
        'akurasi_train': akurasi_train, 'presisi_train': presisi_train,
        'recall_train': recall_train, 'f1_train': f1_train, 'cm_train': cm_train,
        'n_train': X_train.shape[0], 'n_test': X_test.shape[0],
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
    }

@st.cache_data
def hitung_tren_bulanan(df):
    df_tanggal = df.copy()
    df_tanggal['tanggal'] = pd.to_datetime(df_tanggal['tanggal'], errors='coerce')
    df_tanggal = df_tanggal.dropna(subset=['tanggal'])
    if len(df_tanggal) == 0:
        return pd.DataFrame(), df_tanggal
    df_tanggal['bulan'] = df_tanggal['tanggal'].dt.to_period('M')
    tren = (
        df_tanggal.groupby(['bulan', 'label']).size()
        .unstack(fill_value=0)
        .reindex(columns=['Negatif', 'Positif'], fill_value=0)
        .sort_index()
    )
    return tren, df_tanggal

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
         "Evaluasi Model", "Insight Bisnis", "Prediksi Sentimen"],
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
    pct_pos_home = total_positif / total_ulasan * 100
    pct_neg_home = total_negatif / total_ulasan * 100

    tren, df_tanggal = hitung_tren_bulanan(df)
    n_bulan = tren.shape[0] if not tren.empty else 0

    # ---------- KPI ROW ----------
    eyebrow("Ringkasan Utama")
    k1, k2, k3 = st.columns(3)
    with k1, st.container(border=True):
        st.markdown(
            f"""<div class="kpi-label">Total Ulasan</div>
            <div class="kpi-value">{total_ulasan:,}</div>
            <div class="kpi-sub neutral">Data {n_bulan} bulan</div>""".replace(",", "."),
            unsafe_allow_html=True,
        )
    with k2, st.container(border=True):
        st.markdown(
            f"""<div class="kpi-label">Positif</div>
            <div class="kpi-value">{total_positif:,}</div>
            <div class="kpi-sub green">{pct_pos_home:.1f}% dari total</div>""".replace(",", "."),
            unsafe_allow_html=True,
        )
    with k3, st.container(border=True):
        st.markdown(
            f"""<div class="kpi-label">Negatif</div>
            <div class="kpi-value">{total_negatif:,}</div>
            <div class="kpi-sub red">{pct_neg_home:.1f}% dari total</div>""".replace(",", "."),
            unsafe_allow_html=True,
        )

    # ---------- TREN & KOMPOSISI ----------
    eyebrow("Tren & Komposisi Sentimen")
    col_tren, col_donut = st.columns([2, 1])

    with col_tren, st.container(border=True):
        st.markdown('<div class="card-title">Tren Sentimen per Bulan</div>', unsafe_allow_html=True)
        if n_bulan > 0:
            x_labels = [p.strftime('%b %y') for p in tren.index]
            fig_tren, ax_tren = plt.subplots(figsize=(8, 3.8))
            ax_tren.plot(x_labels, tren['Positif'], marker='o', markersize=5,
                         linewidth=2.2, color=GREEN, label='Positif', zorder=3)
            ax_tren.plot(x_labels, tren['Negatif'], marker='o', markersize=5,
                         linewidth=2.2, color=RED, label='Negatif', zorder=3)
            ax_tren.set_ylim(bottom=0)
            ax_tren.set_ylabel("Jumlah Ulasan")
            ax_tren.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", "."))
            )
            ax_tren.grid(axis="y", color=BORDER, linewidth=0.8, zorder=0)
            ax_tren.set_axisbelow(True)
            ax_tren.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
            plt.xticks(rotation=40, ha='right', fontsize=8)
            fig_tren.tight_layout()
            st.pyplot(fig_tren, use_container_width=True)
        else:
            st.caption("Kolom 'tanggal' tidak dapat dibaca sebagai tanggal — cek formatnya di file sumber.")

    with col_donut, st.container(border=True):
        st.markdown('<div class="card-title">Perbandingan Sentimen</div>', unsafe_allow_html=True)
        fig_donut, ax_donut = plt.subplots(figsize=(3.6, 3.6))
        ax_donut.pie(
            [total_positif, total_negatif],
            colors=[GREEN, RED],
            startangle=90,
            wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 3},
        )
        label_dominan = "Positif" if pct_pos_home >= pct_neg_home else "Negatif"
        pct_dominan = max(pct_pos_home, pct_neg_home)
        warna_dominan = GREEN_DARK if label_dominan == "Positif" else RED
        ax_donut.text(0, 0.08, f"{pct_dominan:.1f}%", ha='center', va='center',
                      fontsize=19, fontweight='bold', color=warna_dominan)
        ax_donut.text(0, -0.14, label_dominan, ha='center', va='center',
                      fontsize=10, color=MUTED, fontweight='medium')
        ax_donut.axis('equal')
        st.pyplot(fig_donut, use_container_width=True)
        st.markdown(
            f"""<div class="legend-row"><span class="legend-dot green"></span>Positif
                <span class="legend-count">{total_positif:,} ({pct_pos_home:.1f}%)</span></div>
            <div class="legend-row"><span class="legend-dot red"></span>Negatif
                <span class="legend-count">{total_negatif:,} ({pct_neg_home:.1f}%)</span></div>""".replace(",", "."),
            unsafe_allow_html=True,
        )

    # ---------- PERFORMA MODEL ----------
    eyebrow("Performa Model (Data Testing)")
    m1, m2, m3, m4 = st.columns(4)
    metrik_home = [
        ("Akurasi", eval_hasil['akurasi']),
        ("Presisi", eval_hasil['presisi']),
        ("Recall", eval_hasil['recall']),
        ("F1-Score", eval_hasil['f1']),
    ]
    for kolom_home, (label_m, val_m) in zip([m1, m2, m3, m4], metrik_home):
        with kolom_home, st.container(border=True):
            st.markdown(
                f"""<div class="mm-label">{label_m}</div>
                <div class="mm-value">{val_m*100:.2f}%</div>""",
                unsafe_allow_html=True,
            )

# HALAMAN 2 — VISUALISASI DATASET
elif halaman == "Visualisasi Dataset":
    header("Dataset")

    tren_dataset, df_tanggal_dataset = hitung_tren_bulanan(df)
    if not df_tanggal_dataset.empty:
        tgl_min = df_tanggal_dataset['tanggal'].min().date()
        tgl_max = df_tanggal_dataset['tanggal'].max().date()
        rentang = st.date_input(
            "Filter Periode Ulasan", value=(tgl_min, tgl_max),
            min_value=tgl_min, max_value=tgl_max
        )
        if isinstance(rentang, tuple) and len(rentang) == 2:
            mulai, akhir = rentang
            df_view = df_tanggal_dataset[
                (df_tanggal_dataset['tanggal'].dt.date >= mulai) &
                (df_tanggal_dataset['tanggal'].dt.date <= akhir)
            ]
