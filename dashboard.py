"""
Dashboard Analisis Sentimen Ulasan AdaKami
Tesis MSI — Google Play Store Periode Januari–Desember 2025
Jalankan: streamlit run dashboard.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimen AdaKami 2025",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* Shell */
.main { background: #f1f5f9; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: none !important;
    min-width: 230px !important;
    max-width: 230px !important;
}

/* Sidebar — force ALL text to be visible on dark bg */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *:not(button) {
    color: #94a3b8 !important;
}

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .25rem !important; }
[data-testid="stSidebar"] .stButton { margin: 0 8px !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 0 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
    transition: all .15s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button > div { width: 100% !important; justify-content: flex-start !important; }
[data-testid="stSidebar"] .stButton > button p { text-align: left !important; font-size: .85rem !important; }
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
/* Active nav button: semua menu memakai st.button, yang aktif type="primary" */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(59,130,246,.18) !important;
    font-weight: 600 !important;
    cursor: default !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── Card borders (st.container border=True override) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 4px 12px rgba(0,0,0,.04) !important;
    padding: 8px 4px !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: white;
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    border-left: 4px solid var(--ac, #3b82f6);
    height: 100%;
}
.kpi-label {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #94a3b8; margin-bottom: 4px;
}
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
.kpi-badge {
    display: inline-block; margin-top: 5px;
    padding: 2px 9px; border-radius: 20px;
    font-size: .68rem; font-weight: 700;
}
.kpi-sub { font-size: .78rem; color: #64748b; margin-top: 3px; }

/* ── Section titles ── */
.sec-h { font-size: 1rem; font-weight: 700; color: #0f172a; margin: 0 0 4px; }
.sec-s { font-size: .8rem; color: #64748b; margin: 0 0 14px; line-height: 1.5; }

/* ── Page banner ── */
.banner {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
    border-radius: 14px; padding: 20px 26px; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 18px;
}
.banner-icon { font-size: 2rem; flex-shrink: 0; }
.banner-title { font-size: 1.15rem; font-weight: 800; color: white; margin: 0; line-height: 1.3; }
.banner-rq { font-size: .8rem; color: #93c5fd; margin: 5px 0 0; font-style: italic; line-height: 1.5; }

/* ── Info boxes ── */
.box-info { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
             padding:12px 16px; font-size:.83rem; color:#1e40af; line-height:1.6; }
.box-warn { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
             padding:12px 16px; font-size:.83rem; color:#9a3412; line-height:1.6; }
.box-ok   { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
             padding:12px 16px; font-size:.83rem; color:#15803d; line-height:1.6; }

/* ── Research info table ── */
.info-table { width:100%; border-collapse:collapse; font-size:.84rem; }
.info-table tr { border-bottom: 1px solid #f1f5f9; }
.info-table tr:last-child { border-bottom: none; }
.info-table td { padding: 7px 4px; vertical-align: top; }
.info-table td:first-child { color: #94a3b8; font-weight: 600; width: 120px; white-space: nowrap; }
.info-table td:last-child  { color: #334155; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9; padding: 5px; border-radius: 10px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important; padding: 7px 16px !important;
    font-size: .84rem !important; font-weight: 600 !important;
    color: #64748b !important; background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #3b82f6 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.08) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ── Nav cards (beranda) ── */
.nav-card {
    background: white; border-radius: 12px; padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    border-top: 3px solid var(--ac2, #3b82f6);
    height: 100%;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── General text ── */
h1 { font-size:1.55rem !important; font-weight:800 !important; color:#0f172a !important; }
h2 { font-size:1.2rem  !important; font-weight:700 !important; color:#0f172a !important; }
h3 { font-size:1rem   !important; font-weight:700 !important; color:#1e293b !important; }
p  { font-size:.88rem; color:#334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

C = dict(
    Positif="#22c55e", Negatif="#f43f5e", Netral="#f59e0b",
    NBC="#3b82f6", SVM="#8b5cf6", RF="#10b981", Baseline="#94a3b8",
)

BULAN_S = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
           7:"Jul",8:"Agt",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
BULAN_F = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
           7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}

METRICS = {
    "NBC":      dict(Acc=.8221, Pre=.6386, Rec=.7021, F1=.6416, FP=.90, FN=.81, FNet=.21),
    "SVM":      dict(Acc=.8164, Pre=.6344, Rec=.6786, F1=.6299, FP=.89, FN=.83, FNet=.17),
    "RF":       dict(Acc=.8240, Pre=.6166, Rec=.6272, F1=.6134, FP=.90, FN=.78, FNet=.16),
    "Baseline": dict(Acc=.6485, Pre=.2162, Rec=.3333, F1=.2623, FP=.79, FN=.00, FNet=.00),
}
CM = {
    "NBC": np.array([[3631,302,335],[104,1676,305],[61,64,103]]),
    "SVM": np.array([[3575,252,441],[92,1712,281],[66,76,86]]),
    "RF":  np.array([[3824,234,210],[329,1543,213],[97,75,56]]),
}

# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "ulasan_dengan_prediksi.csv"))
    df["bulan"] = df["bulan"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def monthly_pivot(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [range(1,13), ["Positif","Negatif","Netral"]],
        names=["bulan","sentimen_prediksi"],
    )
    cnt = df.groupby(["bulan","sentimen_prediksi"]).size().reindex(idx, fill_value=0).reset_index(name="n")
    tot = df.groupby("bulan").size().reset_index(name="total")
    cnt = cnt.merge(tot, on="bulan")
    cnt["pct"] = (cnt["n"] / cnt["total"] * 100).round(2)
    cnt["bs"] = pd.Categorical(cnt["bulan"].map(BULAN_S), list(BULAN_S.values()), ordered=True)
    cnt["bf"] = pd.Categorical(cnt["bulan"].map(BULAN_F), list(BULAN_F.values()), ordered=True)
    return cnt.sort_values("bs")

@st.cache_data(show_spinner=False)
def word_freq(b0: int, b1: int) -> Counter:
    neg = _df[(_df["sentimen_prediksi"]=="Negatif")&(_df["bulan"]>=b0)&(_df["bulan"]<=b1)]
    c = Counter()
    for t in neg["teks_bersih"].dropna():
        c.update(str(t).split())
    return c

with st.spinner("Memuat data…"):
    _df = load_data()
    _mp = monthly_pivot(_df)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
T = "plotly_white"
FONT = dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#334155")

def fmt_chart(fig, h=420):
    fig.update_layout(
        template=T, height=h, font=FONT, separators=",.",
        margin=dict(l=16,r=16,t=48,b=16),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig

def ribu(n):
    return f"{int(n):,}".replace(",", ".")

def des(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")

def kpi(label, value, badge="", badge_bg="#dbeafe", badge_fg="#1e40af", sub="", accent="#3b82f6"):
    bdg = (f'<span class="kpi-badge" style="background:{badge_bg};color:{badge_fg};">'
           f'{badge}</span>') if badge else ""
    st.markdown(f"""
    <div class="kpi-card" style="--ac:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {bdg}
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(title, sub=""):
    st.markdown(f'<p class="sec-h">{title}</p>'
                f'{"<p class=sec-s>"+sub+"</p>" if sub else ""}',
                unsafe_allow_html=True)

def banner(title, rq=""):
    rq_h = f'<p class="banner-rq">"{rq}"</p>' if rq else ""
    st.markdown(f"""
    <div class="banner">
        <div><p class="banner-title">{title}</p>{rq_h}</div>
    </div>""", unsafe_allow_html=True)

def box(text, kind="info"):
    cls = {"info":"box-info","warn":"box-warn","ok":"box-ok"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE — navigation
# ─────────────────────────────────────────────────────────────────
PAGES = [
    "Beranda",
    "Distribusi Sentimen",
    "Tren Temporal",
    "Perbandingan Algoritma",
    "Topik Dominan Negatif",
]
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:22px 16px 14px;">
        <div style="font-size:.97rem;font-weight:800;color:#f1f5f9;line-height:1.3;">
            Sentimen AdaKami</div>
        <div style="font-size:.73rem;color:#64748b;margin-top:3px;">
            Google Play Store, 2025</div>
    </div>
    <div style="border-top:1px solid #1e293b;margin:0 16px 10px;"></div>
    <div style="padding:0 4px 2px;">
        <div style="font-size:.66rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.1em;color:#475569;padding:0 12px 8px;">Menu</div>
    </div>
    """, unsafe_allow_html=True)

    for name in PAGES:
        is_active = (st.session_state.page == name)
        # Active page uses "primary" kind so CSS can target it differently
        label = name
        if st.button(label, key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary") and not is_active:
            st.session_state.page = name
            st.rerun()

    n_pos = (_df["sentimen_prediksi"]=="Positif").sum()
    n_neg = (_df["sentimen_prediksi"]=="Negatif").sum()
    total = len(_df)

    st.markdown(f"""
    <div style="border-top:1px solid #1e293b;margin:12px 16px 0;"></div>
    <div style="padding:12px 16px 20px;font-size:.8rem;line-height:2.2;">
        <div style="font-size:.66rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.1em;color:#475569;margin-bottom:6px;">Dataset</div>
        <div style="color:#64748b;"><b style="color:#cbd5e1;">{ribu(total)}</b> ulasan</div>
        <div style="color:#64748b;"><b style="color:#4ade80;">{des(n_pos/total*100)}%</b> Positif</div>
        <div style="color:#64748b;"><b style="color:#f87171;">{des(n_neg/total*100)}%</b> Negatif</div>
        <div style="border-top:1px solid #1e293b;margin:8px 0;"></div>
        <div style="color:#64748b;">Model: <b style="color:#93c5fd;">NBC</b></div>
        <div style="color:#64748b;">F1-macro: <b style="color:#93c5fd;">0,6416</b></div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ═════════════════════════════════════════════════════════════════
# BERANDA
# ═════════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
    <h1 style="margin:0 0 4px;">Dashboard Analisis Sentimen</h1>
    <p style="color:#64748b;font-size:.9rem;margin:0 0 1.4rem;">
        Ulasan Pengguna Aplikasi <b>AdaKami</b> di Google Play Store,
        Periode Januari sampai Desember 2025
    </p>""", unsafe_allow_html=True)

    total = len(_df)
    n_pos = (_df["sentimen_prediksi"]=="Positif").sum()
    n_neg = (_df["sentimen_prediksi"]=="Negatif").sum()
    n_net = (_df["sentimen_prediksi"]=="Netral").sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Ulasan", ribu(total), "32.905 ulasan", "#dbeafe","#1e40af",
                  "Januari sampai Desember 2025, setelah preprocessing", "#3b82f6")
    with c2: kpi("Positif", f"{des(n_pos/total*100)}%", "Prediksi NBC","#dcfce7","#15803d",
                  f"{ribu(n_pos)} ulasan", "#22c55e")
    with c3: kpi("Negatif", f"{des(n_neg/total*100)}%", "Prediksi NBC","#fee2e2","#dc2626",
                  f"{ribu(n_neg)} ulasan", "#f43f5e")
    with c4: kpi("F1-macro Terbaik","0,6416","Model Terbaik","#ede9fe","#6d28d9",
                  "NBC. Dibandingkan baseline ZeroR, F1-macro NBC naik 37,9 poin persentase", "#8b5cf6")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2], gap="medium")

    # ── Donut chart ──
    with col_l:
        with st.container(border=True):
            sec("Distribusi Sentimen Keseluruhan",
                "Prediksi model NBC pada 32.905 ulasan terproses")
            preds = _df["sentimen_prediksi"].value_counts().reindex(["Positif","Negatif","Netral"])
            fig_d = go.Figure(go.Pie(
                labels=preds.index.tolist(), values=preds.values.tolist(), hole=0.58,
                marker=dict(colors=[C["Positif"],C["Negatif"],C["Netral"]],
                            line=dict(color="white",width=3)),
                textinfo="percent+label", textfont_size=12.5, sort=False,
                hovertemplate="<b>%{label}</b><br>%{value:,} ulasan (%{percent})<extra></extra>",
                pull=[.03,.03,.03],
            ))
            fig_d.add_annotation(
                text=f"<b>{ribu(total)}</b><br><span style='font-size:11px'>ulasan</span>",
                x=.5, y=.5, showarrow=False,
                font=dict(size=14, family="Plus Jakarta Sans", color="#0f172a"),
            )
            fig_d.update_layout(template=T, height=310, separators=",.",
                margin=dict(l=10,r=10,t=10,b=10), font=FONT,
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h",x=.5,xanchor="center",y=-0.08,
                            bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Tentang penelitian ──
    with col_r:
        with st.container(border=True):
            sec("Tentang Penelitian")
            st.markdown("""
<table class="info-table">
  <tr><td>Judul</td><td>Analisis Sentimen Ulasan AdaKami</td></tr>
  <tr><td>Sumber data</td><td>Google Play Store</td></tr>
  <tr><td>Algoritma</td><td>NBC, SVM, Random Forest</td></tr>
  <tr><td>Fitur</td><td>TF-IDF (10.000 fitur)</td></tr>
  <tr><td>Imbalance</td><td>SMOTE (ImbPipeline)</td></tr>
  <tr><td>Validasi</td><td>5-fold Stratified K-Fold</td></tr>
  <tr><td>Data uji</td><td>6.581 ulasan (pembagian 80:20)</td></tr>
</table>
<div style="border-top:1px solid #f1f5f9;margin:12px 0 10px;"></div>
<div style="font-size:.82rem;color:#475569;">
  <b style="color:#0f172a;">Rumusan Masalah</b>
  <ol style="margin:7px 0 0;padding-left:16px;line-height:1.5;color:#64748b;">
    <li style="margin-bottom:6px;">Algoritma manakah yang menunjukkan kinerja paling baik antara Naive Bayes Classifier (NBC), Support Vector Machine (SVM), dan Random Forest dalam mengklasifikasikan sentimen ulasan pengguna AdaKami di Google Play Store periode Januari hingga Desember 2025?</li>
    <li style="margin-bottom:6px;">Bagaimana pola distribusi sentimen ulasan pengguna AdaKami selama periode Januari hingga Desember 2025?</li>
    <li>Dimensi kualitas layanan elektronik (e-SERVQUAL) mana saja yang terlihat pada kata dominan ulasan pengguna AdaKami bersentimen negatif selama periode Januari hingga Desember 2025?</li>
  </ol>
</div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Nav cards ──
    sec("Jelajahi Dashboard")
    n1,n2,n3,n4 = st.columns(4)
    nav = [
        ("Distribusi Sentimen","Rumusan Masalah 2","Proporsi dan perbandingan label dengan prediksi","#22c55e","#dcfce7","#15803d"),
        ("Tren Temporal","Rumusan Masalah 2","Fluktuasi sentimen per bulan dan tabel data","#3b82f6","#dbeafe","#1e40af"),
        ("Perbandingan Algoritma","Rumusan Masalah 1","NBC, SVM, dan RF: metrik dan confusion matrix","#8b5cf6","#ede9fe","#6d28d9"),
        ("Topik Dominan Negatif","Rumusan Masalah 3","Word cloud, frekuensi kata, dan pencarian contoh ulasan","#f43f5e","#fee2e2","#dc2626"),
    ]
    for col_, (title,rq,desc,ac,bg,fg) in zip([n1,n2,n3,n4], nav):
        with col_:
            st.markdown(f"""
<div class="nav-card" style="--ac2:{ac}">
  <div style="font-size:.88rem;font-weight:700;color:#0f172a;margin-bottom:5px;">{title}</div>
  <span style="background:{bg};color:{fg};font-size:.68rem;font-weight:700;
               padding:2px 9px;border-radius:20px;">{rq}</span>
  <div style="font-size:.78rem;color:#64748b;margin-top:8px;line-height:1.5;">{desc}</div>
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# DISTRIBUSI SENTIMEN (RQ2)
# ═════════════════════════════════════════════════════════════════
elif page == "Distribusi Sentimen":
    banner("Distribusi Sentimen Ulasan AdaKami",
           "Bagaimana pola distribusi sentimen ulasan pengguna AdaKami selama periode Januari hingga Desember 2025?")

    total = len(_df)
    pred  = _df["sentimen_prediksi"].value_counts().reindex(["Positif","Negatif","Netral"])
    label = _df["sentimen"].value_counts().reindex(["Positif","Negatif","Netral"])

    c1,c2,c3 = st.columns(3)
    with c1: kpi("Positif (Prediksi)", ribu(pred['Positif']), "Rating 4-5★","#dcfce7","#15803d",
                  f"{des(pred['Positif']/total*100)}% dari total", "#22c55e")
    with c2: kpi("Negatif (Prediksi)", ribu(pred['Negatif']), "Rating 1-2★","#fee2e2","#dc2626",
                  f"{des(pred['Negatif']/total*100)}% dari total", "#f43f5e")
    with c3: kpi("Netral (Prediksi)", ribu(pred['Netral']), "Rating 3★","#fef3c7","#b45309",
                  f"{des(pred['Netral']/total*100)}% dari total", "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="medium")

    def donut(data, center_label, title, sub):
        with st.container(border=True):
            sec(title, sub)
            fig = go.Figure(go.Pie(
                labels=data.index.tolist(), values=data.values.tolist(), hole=0.55,
                marker=dict(colors=[C["Positif"],C["Negatif"],C["Netral"]],
                            line=dict(color="white",width=3)),
                textinfo="percent+label", textfont_size=12, sort=False,
                hovertemplate="<b>%{label}</b><br>%{value:,} ulasan (%{percent})<extra></extra>",
                pull=[.03,.03,.03],
            ))
            fig.add_annotation(text=center_label, x=.5, y=.5, showarrow=False,
                font=dict(size=12, family="Plus Jakarta Sans", color="#64748b"))
            fig.update_layout(template=T, height=320, separators=",.",
                margin=dict(l=10,r=10,t=10,b=10), font=FONT,
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h",x=.5,xanchor="center",y=-0.1,
                            bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

    with col_l:
        donut(pred, "Prediksi\nNBC", "Hasil Prediksi Model NBC",
              "Distribusi dari 32.905 ulasan yang diklasifikasikan model")
    with col_r:
        donut(label, "Label\nRating", "Label Berbasis Rating",
              "Distribusi dari rating bintang pengguna sebagai acuan")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        sec("Perbandingan Label vs Prediksi",
            "Selisih antara distribusi label asli dan hasil prediksi model NBC")
        rows = []
        for k in ["Positif","Negatif","Netral"]:
            d = pred[k]-label[k]
            rows.append({
                "Kelas": k,
                "Label (Rating)": ribu(label[k]),
                "% Label": f"{des(label[k]/total*100)}%",
                "Prediksi NBC": ribu(pred[k]),
                "% Prediksi": f"{des(pred[k]/total*100)}%",
                "Selisih (n)": f"+{ribu(d)}" if d>=0 else f"-{ribu(-d)}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        box("Proporsi kelas Netral meningkat dari <b>3,5%</b> (label berbasis rating pada 32.905 ulasan "
            "yang dimodelkan) menjadi <b>11,6%</b> (prediksi model pada korpus yang sama), yaitu sekitar "
            "3,3 kali lipat. Peningkatan ini diduga berasal dari SMOTE. Dugaan ini belum diuji, sebab "
            "penelitian ini tidak melatih model pembanding tanpa SMOTE. Karena itu angka 11,6% Netral "
            "dibaca dengan hati-hati.", "warn")

# ═════════════════════════════════════════════════════════════════
# TREN TEMPORAL (RQ2)
# ═════════════════════════════════════════════════════════════════
elif page == "Tren Temporal":
    banner("Tren Sentimen Temporal, Januari sampai Desember 2025",
           "Bagaimana pola distribusi sentimen ulasan pengguna AdaKami selama periode Januari hingga Desember 2025?")

    c1,c2,c3 = st.columns(3)
    with c1: kpi("Positif Tertinggi","70,0%","Puncak Positif","#dcfce7","#15803d",
                  "April 2025","#22c55e")
    with c2: kpi("Negatif Tertinggi","36,7%","Puncak Negatif","#fee2e2","#dc2626",
                  "Oktober 2025","#f43f5e")
    with c3: kpi("Rentang Fluktuasi Negatif","14,9","Poin Persentase","#fef3c7","#b45309",
                  "21,8% (April) hingga 36,7% (Oktober)","#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        sec("Grafik Tren Sentimen", "Gunakan filter di bawah untuk menyesuaikan tampilan")

        col_f1,col_f2,col_f3 = st.columns([3,2,2])
        with col_f1:
            sel = st.multiselect("Kelas sentimen:", ["Positif","Negatif","Netral"],
                                  default=["Positif","Negatif","Netral"], key="t_kelas")
        with col_f2:
            view = st.radio("Tampilkan:", ["Proporsi (%)","Volume (n)"],
                             horizontal=True, key="t_view")
        with col_f3:
            annot = st.checkbox("Anotasi puncak", value=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not sel:
            st.warning("Pilih minimal satu kelas sentimen.")
        else:
            mp_f = _mp[_mp["sentimen_prediksi"].isin(sel)]

            if view == "Proporsi (%)":
                fig = px.line(mp_f, x="bs", y="pct", color="sentimen_prediksi",
                              markers=True, color_discrete_map=C,
                              labels={"pct":"Proporsi (%)","bs":"Bulan","sentimen_prediksi":"Sentimen"},
                              category_orders={"bs":list(BULAN_S.values())},
                              title="Tren Proporsi Sentimen per Bulan (Januari sampai Desember 2025)")
                fig.update_traces(line_width=2.5, marker_size=9,
                                  marker_line_width=1.5, marker_line_color="white",
                                  hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra>%{fullData.name}</extra>")
                fig.update_yaxes(range=[0,80], ticksuffix="%")
                if annot and "Positif" in sel:
                    fig.add_annotation(x="Apr", y=70.0,
                        text="<b>70,0%</b><br>Puncak Positif",
                        showarrow=True, arrowhead=2, arrowcolor=C["Positif"], arrowwidth=2,
                        ax=36, ay=-44, bgcolor="white",
                        font=dict(size=11,color=C["Positif"]),
                        bordercolor=C["Positif"], borderwidth=1.5, borderpad=5)
                if annot and "Negatif" in sel:
                    fig.add_annotation(x="Okt", y=36.7,
                        text="<b>36,7%</b><br>Puncak Negatif",
                        showarrow=True, arrowhead=2, arrowcolor=C["Negatif"], arrowwidth=2,
                        ax=-44, ay=-44, bgcolor="white",
                        font=dict(size=11,color=C["Negatif"]),
                        bordercolor=C["Negatif"], borderwidth=1.5, borderpad=5)
            else:
                vol = (_df[_df["sentimen_prediksi"].isin(sel)]
                       .groupby(["bulan","sentimen_prediksi"]).size().reset_index(name="n")
                       .assign(bs=lambda d: pd.Categorical(
                           d["bulan"].map(BULAN_S), list(BULAN_S.values()), ordered=True))
                       .sort_values("bs"))
                fig = px.bar(vol, x="bs", y="n", color="sentimen_prediksi", barmode="stack",
                             color_discrete_map=C,
                             labels={"n":"Jumlah Ulasan","bs":"Bulan","sentimen_prediksi":"Sentimen"},
                             category_orders={"bs":list(BULAN_S.values())},
                             title="Volume Ulasan per Bulan (Januari sampai Desember 2025)")
                fig.update_traces(hovertemplate="%{y:,}<extra>%{fullData.name}</extra>")

            st.plotly_chart(fmt_chart(fig, 440), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Tabel data lengkap per bulan"):
        pv = _mp.pivot(index="bf", columns="sentimen_prediksi", values="pct")
        pv = pv.reindex(list(BULAN_F.values()))
        vol_map = {BULAN_F[r["bulan"]]: r["total"]
                   for _, r in _mp.drop_duplicates("bulan").iterrows()}
        pv.insert(0, "Total Ulasan", pd.Series(vol_map))
        pv.columns.name = None; pv.index.name = "Bulan"
        disp = pv.copy()
        for c_ in disp.columns:
            disp[c_] = disp[c_].apply(
                lambda x: ribu(x) if c_=="Total Ulasan" else f"{des(x)}%"
            )
        st.dataframe(disp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    box("Puncak sentimen positif pada April 2025: proporsi sentimen positif mencapai angka tertinggi "
        "sepanjang tahun sebesar <b>70,0%</b> (2.554 dari 3.648 ulasan), disertai proporsi sentimen negatif "
        "terendah sebesar 21,8%. Peningkatan sentimen negatif pada Juli sampai Oktober: proporsi sentimen "
        "negatif naik dari 31,6% (Juli) menjadi <b>36,7%</b> (Oktober), angka tertinggi sepanjang tahun, "
        "dengan penurunan tipis pada September menjadi 34,9%. Pada Juli sampai Oktober, proporsi sentimen "
        "positif turun hingga 49,2% (Oktober), satu-satunya bulan dengan sentimen positif di bawah 50%. "
        "Penyebab kenaikan pada Juli sampai Oktober 2025 itu tidak dapat dipastikan dari data ulasan.", "info")

# ═════════════════════════════════════════════════════════════════
# PERBANDINGAN ALGORITMA (RQ1)
# ═════════════════════════════════════════════════════════════════
elif page == "Perbandingan Algoritma":
    banner("Perbandingan Algoritma NBC, SVM, dan RF",
           "Algoritma manakah yang menunjukkan kinerja paling baik antara Naive Bayes Classifier (NBC), Support Vector Machine (SVM), dan Random Forest dalam mengklasifikasikan sentimen ulasan pengguna AdaKami di Google Play Store periode Januari hingga Desember 2025?")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("NBC: F1-macro","0,6416","Peringkat 1","#dbeafe","#1e40af",
                  "Accuracy 0,8221, Recall 0,7021","#3b82f6")
    with c2: kpi("SVM: F1-macro","0,6299","Peringkat 2","#ede9fe","#6d28d9",
                  "0,0117 di bawah NBC","#8b5cf6")
    with c3: kpi("RF: F1-macro","0,6134","Peringkat 3","#d1fae5","#065f46",
                  "0,0282 di bawah NBC","#10b981")
    with c4: kpi("Baseline F1-macro","0,2623","ZeroR","#f1f5f9","#475569",
                  "F1-macro NBC naik 37,9 poin persentase","#94a3b8")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  Metrik Perbandingan  ",
        "  F1-Score per Kelas  ",
        "  Confusion Matrix  ",
    ])

    # ── Tab 1: Metrik ──
    with tab1:
        with st.container(border=True):
            sec("Empat Metrik Evaluasi pada Data Uji 6.581 Ulasan",
                "Diukur pada data uji yang ikut memuat ulasan duplikat dari data latih")
            mdl_ord = ["Baseline","RF","SVM","NBC"]
            rows_m = []
            for m in mdl_ord:
                for met,key in [("Accuracy","Acc"),("Precision","Pre"),
                                 ("Recall","Rec"),("F1-macro","F1")]:
                    rows_m.append({"Model":m,"Metrik":met,"Nilai":METRICS[m][key]})
            fig_b = px.bar(pd.DataFrame(rows_m), x="Metrik", y="Nilai",
                           color="Model", barmode="group", color_discrete_map=C,
                           text_auto=".3f",
                           category_orders={"Model":mdl_ord,
                               "Metrik":["Accuracy","Precision","Recall","F1-macro"]},
                           title="Perbandingan Metrik NBC, SVM, RF, dan Baseline")
            fig_b.update_traces(textfont_size=10, textangle=0, textposition="outside",
                                cliponaxis=False,
                                hovertemplate="<b>%{x}</b>: %{y:.4f}<extra>%{fullData.name}</extra>")
            fig_b.update_yaxes(range=[0,1.12], tickformat=".2f")
            st.plotly_chart(fmt_chart(fig_b, 440), use_container_width=True)

            tbl = pd.DataFrame([{
                "Model":m, "Accuracy":des(METRICS[m]['Acc'],4),
                "Precision":des(METRICS[m]['Pre'],4),
                "Recall":des(METRICS[m]['Rec'],4),
                "F1-macro":des(METRICS[m]['F1'],4),
                "Peringkat":["1","2","3","4"][i],
            } for i,m in enumerate(["NBC","SVM","RF","Baseline"])])
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            box("Berdasarkan metrik utama F1-macro, model terbaik adalah NBC dengan alpha=0,5, yang mencatat "
                "nilai F1-macro <b>0,6416</b>, diikuti oleh SVM (0,6299) dan RF (0,6134). RF mencatat akurasi "
                "tertinggi, yaitu 0,8240, sedikit lebih tinggi daripada NBC dengan 0,8221. Namun kinerja RF "
                "pada kelas Netral (F1=0,16) jauh di bawah NBC (F1=0,21). Uji Wilcoxon signed-rank dua sisi "
                "pada skor F1-macro per fold NBC dan SVM menghasilkan statistik W=0 dan p=0,0625, di atas "
                "tingkat signifikansi alpha=0,05 yang dipakai penelitian ini. Dengan begitu uji ini tidak dapat "
                "memutuskan mana yang lebih baik, dan hasilnya bukan bukti bahwa kinerja NBC dan SVM sama.", "ok")

    # ── Tab 2: F1 per kelas ──
    with tab2:
        with st.container(border=True):
            sec("F1-Score per Kelas Sentimen",
                "Seberapa baik masing-masing model mengenali tiap kelas sentimen")
            f1r = [{"Model":m,"Kelas":k,"F1":v}
                   for m in ["NBC","SVM","RF"]
                   for k,v in [("Positif",METRICS[m]["FP"]),
                                ("Negatif",METRICS[m]["FN"]),
                                ("Netral", METRICS[m]["FNet"])]]
            fig_f = px.bar(pd.DataFrame(f1r), x="Kelas", y="F1",
                           color="Model", barmode="group", color_discrete_map=C,
                           text_auto=".2f",
                           category_orders={"Kelas":["Positif","Negatif","Netral"]},
                           title="F1-Score per Kelas: NBC, SVM, dan RF")
            fig_f.update_traces(textfont_size=11, textangle=0, textposition="outside",
                                cliponaxis=False,
                                hovertemplate="<b>%{x}</b>: %{y:.2f}<extra>%{fullData.name}</extra>")
            fig_f.update_yaxes(range=[0,1.18])
            st.plotly_chart(fmt_chart(fig_f, 400), use_container_width=True)

            f1_tbl = pd.DataFrame({
                "Kelas (Support)":["Positif (4.268)","Negatif (2.085)","Netral (228)"],
                "NBC":["0,90","0,81","0,21"],
                "SVM":["0,89","0,83","0,17"],
                "RF": ["0,90","0,78","0,16"],
                "Catatan":[
                    "Kelas Positif mencakup 64,9% data uji",
                    "SVM sedikit lebih unggul",
                    "228 sampel uji, 3,5% dari total data uji",
                ],
            })
            st.dataframe(f1_tbl, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
            box("Penyebab utama F1-macro yang relatif rendah adalah F1 kelas Netral yang hanya 0,16 sampai "
                "0,21 pada ketiga model. Kelas itu hanya 3,5% dari 32.905 ulasan yang dimodelkan, sehingga "
                "data uji hanya memuat 228 sampel Netral, dan jumlah sekecil itu menyulitkan model "
                "menghasilkan klasifikasi yang stabil meskipun SMOTE telah diterapkan pada data latih. "
                "Selain kelangkaan data tersebut, noise label yang menumpuk pada rating 3 ikut menekan "
                "F1-macro.", "warn")

    # ── Tab 3: Confusion Matrix ──
    with tab3:
        with st.container(border=True):
            sec("Confusion Matrix pada Data Uji 6.581 Ulasan",
                "Distribusi prediksi dibandingkan label berbasis rating per kelas sentimen")
            col_s1,col_s2 = st.columns([2,2])
            with col_s1:
                mdl_sel = st.selectbox("Model:", ["NBC","SVM","RF"], key="cm_mdl")
            with col_s2:
                cm_mode = st.radio("Tampilkan nilai:", ["Jumlah (n)","Proporsi (%)"],
                                    horizontal=True, key="cm_mode")

            cm = CM[mdl_sel].astype(float)
            cm_d = cm/cm.sum(axis=1,keepdims=True)*100 if "Proporsi" in cm_mode else cm
            lbl = ["Positif","Negatif","Netral"]
            text = [[f"{des(cm_d[i][j])}%" if "Proporsi" in cm_mode
                     else ribu(cm[i][j]) for j in range(3)] for i in range(3)]

            fig_cm = go.Figure(go.Heatmap(
                z=cm_d,
                x=[f"Prediksi {l}" for l in lbl],
                y=[f"Label {l}" for l in lbl],
                colorscale=[[0,"#f0f9ff"],[0.5,"#60a5fa"],[1,"#1e3a8a"]],
                showscale=True, text=text, texttemplate="%{text}",
                textfont=dict(size=15, family="Plus Jakarta Sans"),
                hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix {mdl_sel}",
                template=T, height=370, separators=",.",
                margin=dict(l=16,r=16,t=50,b=16), font=FONT,
                paper_bgcolor="white", xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.caption(f"Accuracy {des(METRICS[mdl_sel]['Acc'],4)}, "
                       f"F1-macro {des(METRICS[mdl_sel]['F1'],4)}. "
                       f"Data uji: Positif 4.268, Negatif 2.085, Netral 228")
            ck1,ck2,ck3 = st.columns(3)
            with ck1: kpi("F1 Positif",des(METRICS[mdl_sel]['FP'],2),"Support: 4.268",
                           "#dcfce7","#15803d","","#22c55e")
            with ck2: kpi("F1 Negatif",des(METRICS[mdl_sel]['FN'],2),"Support: 2.085",
                           "#fee2e2","#dc2626","","#f43f5e")
            with ck3: kpi("F1 Netral",des(METRICS[mdl_sel]['FNet'],2),"Support: 228",
                           "#fef3c7","#b45309","","#f59e0b")

# ═════════════════════════════════════════════════════════════════
# TOPIK DOMINAN NEGATIF (Business Understanding)
# ═════════════════════════════════════════════════════════════════
elif page == "Topik Dominan Negatif":
    banner("Topik Dominan pada Ulasan Negatif AdaKami",
           "Dimensi kualitas layanan elektronik (e-SERVQUAL) mana saja yang terlihat pada kata dominan ulasan pengguna AdaKami bersentimen negatif selama periode Januari hingga Desember 2025?")

    # ── Filter bar ──
    with st.container(border=True):
        st.markdown('<p class="sec-h">Filter</p>', unsafe_allow_html=True)
        col_f1,col_f2,col_f3,col_f4 = st.columns([2,2,1,2])
        with col_f1:
            b_s_lbl = st.selectbox("Dari bulan:", list(BULAN_F.values()), index=0)
        with col_f2:
            b_e_lbl = st.selectbox("Sampai bulan:", list(BULAN_F.values()), index=11)
        with col_f3:
            top_n = st.selectbox("Jumlah kata teratas:", [5,10,20,30,50], index=2)
        with col_f4:
            keyword = st.text_input("Cari kata kunci:", value="pinjam",
                                    placeholder="iklan, tolak, data, bayar")

    bs_ = [k for k,v in BULAN_F.items() if v==b_s_lbl][0]
    be_ = [k for k,v in BULAN_F.items() if v==b_e_lbl][0]
    if bs_ > be_:
        st.error("Bulan mulai tidak boleh melebihi bulan akhir.")
        st.stop()

    with st.spinner("Menghitung frekuensi kata…"):
        ctr = word_freq(bs_, be_)
    top_w = dict(ctr.most_common(top_n))

    df_neg = _df[(_df["sentimen_prediksi"]=="Negatif") &
                 (_df["bulan"]>=bs_) & (_df["bulan"]<=be_)]
    n_neg = len(df_neg)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    top1 = list(top_w.items())[0] if top_w else ("-",0)
    with c1: kpi("Ulasan Negatif",ribu(n_neg),"Negatif","#fee2e2","#dc2626",
                  f"{b_s_lbl} sampai {b_e_lbl}","#f43f5e")
    with c2: kpi("% dari Total",f"{des(n_neg/len(_df)*100)}%","Periode dipilih","#fef3c7","#b45309",
                  "dari 32.905 ulasan","#f59e0b")
    with c3: kpi("Kata Teratas",top1[0].upper(),"#1 Kata","#fee2e2","#dc2626",
                  f"Frekuensi: {ribu(top1[1])}","#f43f5e")

    st.markdown("<br>", unsafe_allow_html=True)
    col_wc, col_bar = st.columns([1.4, 1], gap="medium")

    # ── Word cloud ──
    with col_wc:
        with st.container(border=True):
            sec(f"Word Cloud {top_n} Kata Teratas",
                f"Periode: {b_s_lbl} sampai {b_e_lbl}, {ribu(n_neg)} ulasan negatif")
            if top_w:
                wc = WordCloud(
                    width=760, height=400, background_color="white",
                    colormap="Reds", max_words=top_n, prefer_horizontal=.85,
                    min_font_size=10, max_font_size=96, collocations=False, margin=8,
                ).generate_from_frequencies(top_w)
                fig_wc, ax = plt.subplots(figsize=(7.6,4))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                fig_wc.patch.set_facecolor("white"); plt.tight_layout(pad=0)
                st.pyplot(fig_wc); plt.close(fig_wc)
            else:
                st.info("Tidak ada data untuk periode yang dipilih.")

    # ── Bar chart ──
    with col_bar:
        with st.container(border=True):
            sec(f"Frekuensi {top_n} Kata Teratas")
            if top_w:
                fd = pd.DataFrame(list(reversed(list(top_w.items()))),
                                  columns=["Kata","Frekuensi"])
                fig_freq = px.bar(fd, x="Frekuensi", y="Kata", orientation="h",
                                  color="Frekuensi",
                                  color_continuous_scale=[[0,"#fca5a5"],[.4,"#f87171"],[1,"#dc2626"]])
                fig_freq.update_traces(
                    hovertemplate="<b>%{y}</b><br>%{x:,} kemunculan<extra></extra>",
                    marker_line_width=0)
                fig_freq.update_layout(
                    template=T, height=max(340,top_n*22+50), separators=",.",
                    margin=dict(l=8,r=24,t=10,b=8), font=FONT,
                    coloraxis_showscale=False,
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#f1f5f9",title=""),
                    yaxis=dict(gridcolor="#f1f5f9",title=""),
                )
                st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_tbl, col_ex = st.columns([1, 1.6], gap="medium")

    # ── Tabel frekuensi ──
    with col_tbl:
        with st.container(border=True):
            sec("Tabel Frekuensi Detail")
            if top_w:
                ft = pd.DataFrame(list(top_w.items()), columns=["Kata (Stem)","Frekuensi"])
                ft.index = ft.index+1
                ft["% Ulasan Negatif"] = (ft["Frekuensi"]/n_neg*100).apply(lambda x: f"{des(x,2)}%")
                ft["Frekuensi"] = ft["Frekuensi"].apply(ribu)
                st.dataframe(ft, use_container_width=True)

    # ── Contoh ulasan ──
    with col_ex:
        with st.container(border=True):
            sec("Contoh Ulasan Negatif",
                f"Menampilkan ulasan yang mengandung kata: '{keyword}'")
            if keyword.strip():
                mask = df_neg["teks_bersih"].fillna("").str.contains(
                    keyword.strip().lower(), case=False, regex=False)
                hits = df_neg[mask].copy()
                if len(hits):
                    hits["Bulan"] = hits["bulan"].map(BULAN_F)
                    disp = (hits[["teks_ulasan","Bulan","rating"]]
                            .head(8).reset_index(drop=True))
                    disp.index += 1
                    disp = disp.rename(columns={
                        "teks_ulasan":"Teks Ulasan","rating":"Rating ★"})
                    st.dataframe(disp, use_container_width=True)
                    st.caption(f"8 dari {ribu(mask.sum())} ulasan, "
                               f"kata '{keyword}' di {des(mask.sum()/n_neg*100)}% "
                               f"ulasan negatif periode ini")
                else:
                    st.info(f"Tidak ditemukan ulasan negatif yang mengandung '{keyword}'.")
            else:
                st.info("Masukkan kata kunci di kolom filter di atas.")

    st.markdown("<br>", unsafe_allow_html=True)
    box("Lima kata dominan Januari sampai Desember 2025: <b>pinjam (3.657)</b>, <b>iklan (2.055)</b>, "
        "<b>data (1.886)</b>, <b>bayar (1.775)</b>, dan <b>tolak (1.276)</b>. Pada lima kata dominan "
        "ulasan negatif terlihat tiga dimensi, yaitu fulfillment pada kata pinjam, tolak, dan sebagian "
        "iklan, privacy pada kata data, dan efficiency pada kata bayar. Empat dimensi lainnya tidak "
        "terlihat, yaitu system availability, responsiveness, compensation, dan contact. Iklan di "
        "aplikasi lain, penagihan, dan besarnya bunga juga muncul pada kata dominan, tetapi berada di "
        "luar jangkauan kedua skala.", "info")
