import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import CLUSTER_NAMES, FEATURES, prepare_features

st.set_page_config(
    page_title="Segmentasi Pelanggan UMKM",
    layout="wide"
)

DATA_PATH = PROJECT_ROOT / "data" / "data_pelanggan_clean.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata.json"

SEGMENT_DETAILS = {
    0: {
        "summary": "Pelanggan loyal dengan frekuensi transaksi dan nilai belanja tinggi.",
        "recommendation": (
            "Pertahankan loyalitas dengan akses awal menu baru atau bonus produk. "
            "Tidak perlu mengandalkan diskon besar."
        )
    },
    1: {
        "summary": "Pelanggan tidak aktif dengan frekuensi transaksi rendah.",
        "recommendation": (
            "Kirim voucher kunjungan kembali dengan masa berlaku terbatas."
        )
    },
    2: {
        "summary": "Pelanggan aktif yang cenderung menggunakan promo.",
        "recommendation": (
            "Tawarkan bundel kopi dan roti dengan promo terarah atau minimum belanja."
        )
    },
    3: {
        "summary": (
            "Nilai belanja tinggi, tetapi frekuensi transaksi relatif rendah."
        ),
        "recommendation": (
            "Tawarkan bonus produk pelengkap pada kunjungan berikutnya untuk "
            "mendorong pembelian lebih rutin."
        )
    }
}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_segment_profiles():
    customers = pd.read_csv(DATA_PATH)
    model = load_model()
    customers["cluster"] = model.predict(customers[FEATURES])

    profiles = (
        customers.groupby("cluster", as_index=False)
        .agg(
            jumlah_pelanggan=("cluster", "size"),
            median_transaksi=("jumlah_transaksi_6bln", "median"),
            median_belanja=("total_belanja_rp", "median"),
            median_hari_terakhir=("hari_sejak_transaksi_terakhir", "median"),
            median_promo=("persen_pakai_promo", "median"),
            median_rating=("rating_rata", "median")
        )
        .set_index("cluster")
        .reindex(sorted(CLUSTER_NAMES))
        .rename_axis("cluster")
        .reset_index()
    )
    profiles["segmen"] = profiles["cluster"].map(CLUSTER_NAMES)
    profiles["porsi_persen"] = (
        profiles["jumlah_pelanggan"] / len(customers) * 100
    )
    return profiles


@st.cache_data(show_spinner=False)
def load_category_options():
    customers = pd.read_csv(DATA_PATH)
    return {
        column: sorted(customers[column].dropna().astype(str).unique())
        for column in ("kecamatan", "channel_utama", "metode_bayar", "kategori_favorit")
    }


def format_count(value):
    return f"{int(value):,}".replace(",", ".")


def format_rupiah(value):
    return f"Rp {value:,.0f}".replace(",", ".")


st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fa;
        color: #17212f;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .page-header {
        margin-bottom: 2rem;
    }

    .eyebrow {
        color: #52657a;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
    }

    .page-header h1 {
        color: #17212f;
        font-size: 2.25rem;
        font-weight: 650;
        letter-spacing: -0.035em;
        margin: 0 0 0.5rem;
    }

    .page-header p {
        color: #64748b;
        font-size: 1rem;
        margin: 0;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #e2e8f0;
        border-radius: 12px;
    }

    [data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    [data-testid="stWidgetLabel"] p {
        color: #344256;
        font-size: 0.88rem;
        font-weight: 500;
    }

    [data-testid="stNumberInputContainer"] {
        background: #f8fafc !important;
        border-color: #d9e2ec !important;
    }

    [data-testid="stNumberInputField"] {
        background: #f8fafc !important;
        color: #17212f;
    }

    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
        background: #f8fafc !important;
        color: #344256;
    }

    [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stNumberInputStepUp"]:hover {
        background: #edf2f7 !important;
    }

    [data-testid="stFormSubmitButton"] button {
        background: #1f4b7a !important;
        border: 1px solid #1f4b7a !important;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 600;
        min-height: 2.8rem;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #183c63 !important;
        border-color: #183c63 !important;
    }

    [data-testid="stFormSubmitButton"] button:disabled {
        background: #1f4b7a !important;
        color: #ffffff !important;
        opacity: 0.72;
    }

    .panel-heading {
        color: #17212f;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .result-heading {
        margin-bottom: 1.25rem;
        text-align: center;
    }

    .panel-description {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-bottom: 1.25rem;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #17212f;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <header class="page-header">
        <div class="eyebrow">Toko roti & kopi | Bandung</div>
        <h1>Segmentasi pelanggan</h1>
        <p>
            Kenali pola belanja anggota dan arahkan penawaran yang lebih sesuai
            untuk setiap segmen.
        </p>
    </header>
    """,
    unsafe_allow_html=True
)

profiles = load_segment_profiles()
category_options = load_category_options()
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
profile_by_cluster = profiles.set_index("cluster")

prediction_tab, profile_tab = st.tabs(
    ["Prediksi pelanggan", "Profil segmen"]
)

with prediction_tab:
    input_column, info_column = st.columns([1.55, 1], gap="large")

    with input_column:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-heading">Data satu pelanggan</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="panel-description">Isi data aktivitas pelanggan selama enam bulan terakhir.</div>',
                unsafe_allow_html=True
            )

            with st.form("form_pelanggan"):
                first_column, second_column = st.columns(2, gap="medium")

                with first_column:
                    umur = st.number_input(
                        "Umur pelanggan (tahun)",
                        min_value=1,
                        max_value=100,
                        value=27
                    )
                    transaksi = st.number_input(
                        "Transaksi dalam 6 bulan",
                        min_value=0,
                        value=10
                    )
                    belanja = st.number_input(
                        "Total belanja 6 bulan (Rp)",
                        min_value=0,
                        value=500000,
                        step=100
                    )

                with second_column:
                    recency = st.number_input(
                        "Hari sejak transaksi terakhir",
                        min_value=0,
                        value=10
                    )
                    promo = st.number_input(
                        "Transaksi memakai promo (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=20.0
                    )
                    rating = st.number_input(
                        "Rating rata-rata",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.1
                    )

                first_category_column, second_category_column = st.columns(2)
                with first_category_column:
                    kecamatan = st.selectbox(
                        "Kecamatan",
                        options=category_options["kecamatan"]
                    )
                    metode_bayar = st.selectbox(
                        "Metode pembayaran",
                        options=category_options["metode_bayar"]
                    )
                with second_category_column:
                    channel_utama = st.selectbox(
                        "Channel utama",
                        options=category_options["channel_utama"]
                    )
                    kategori_favorit = st.selectbox(
                        "Kategori favorit",
                        options=category_options["kategori_favorit"]
                    )

                submit = st.form_submit_button(
                    "Lihat segmen dan rekomendasi",
                    use_container_width=True,
                    type="primary"
                )

    with info_column:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-heading">Tentang analisis</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="panel-description">Model K-Means memakai enam fitur numerik dan empat fitur kategorikal: umur, transaksi, belanja, hari sejak transaksi, penggunaan promo, rating, kecamatan, channel, metode pembayaran, dan kategori favorit.</div>',
                unsafe_allow_html=True
            )
            model_column, score_column = st.columns(2)
            with model_column:
                st.metric("Jumlah segmen", metadata["n_clusters"])
            with score_column:
                st.metric(
                    "Silhouette score",
                    f'{metadata["silhouette_score"]:.4f}'
                )
            st.caption(
                "Skor silhouette tercatat untuk model yang digunakan; "
                "nilai lebih tinggi menunjukkan pemisahan cluster yang lebih jelas."
            )

    result_container = st.empty()

    if submit:
        data = {
            "umur": umur,
            "jumlah_transaksi_6bln": transaksi,
            "total_belanja_rp": belanja,
            "hari_sejak_transaksi_terakhir": recency,
            "persen_pakai_promo": promo,
            "rating_rata": rating,
            "kecamatan": kecamatan,
            "channel_utama": channel_utama,
            "metode_bayar": metode_bayar,
            "kategori_favorit": kategori_favorit,
        }

        try:
            with st.spinner("Menganalisis pelanggan..."):
                features = prepare_features(data)
                cluster = int(load_model().predict(features)[0])
                profile = profile_by_cluster.loc[cluster]

            with result_container.container(border=True):
                st.markdown(
                    '<div class="panel-heading result-heading">Hasil untuk pelanggan ini</div>',
                    unsafe_allow_html=True
                )
                result_column, cluster_column = st.columns(
                    [2.5, 1],
                    gap="large"
                )
                with result_column:
                    st.markdown("**Segmen pelanggan**")
                    st.subheader(CLUSTER_NAMES[cluster])
                with cluster_column:
                    st.markdown("**Cluster**")
                    st.subheader(str(cluster))

                st.markdown(
                    f'**Karakteristik:** {SEGMENT_DETAILS[cluster]["summary"]}'
                )
                st.markdown("**Profil tipikal segmen (median)**")
                metric_columns = st.columns(3)
                metric_columns[0].metric(
                    "Transaksi / 6 bulan",
                    format_count(profile["median_transaksi"])
                )
                metric_columns[1].metric(
                    "Total belanja",
                    format_rupiah(profile["median_belanja"])
                )
                metric_columns[2].metric(
                    "Hari sejak transaksi",
                    f'{profile["median_hari_terakhir"]:.0f} hari'
                )
                st.markdown("**Rekomendasi penawaran**")
                st.write(SEGMENT_DETAILS[cluster]["recommendation"])
        except ValueError as error:
            with result_container.container():
                st.error(f"Data pelanggan tidak valid: {error}")

with profile_tab:
    st.markdown(
        '<div class="panel-heading">Profil anggota berdasarkan hasil clustering</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel-description">Median membantu menggambarkan anggota tipikal dan tidak mudah terpengaruh nilai ekstrem.</div>',
        unsafe_allow_html=True
    )

    chart_column, overview_column = st.columns([1.5, 1], gap="large")
    with chart_column:
        st.markdown("**Sebaran anggota per segmen**")
        chart_data = profiles.set_index("segmen")["jumlah_pelanggan"]
        st.bar_chart(chart_data, horizontal=True, height=280)

    with overview_column:
        with st.container(border=True):
            st.metric(
                "Anggota dianalisis",
                format_count(profiles["jumlah_pelanggan"].sum())
            )
            st.metric(
                "Jumlah segmen",
                metadata["n_clusters"]
            )
            st.caption(
                "Model dibangun dari data pelanggan enam bulan terakhir. "
                "Fitur numerik yang kosong diisi median lalu distandardisasi; "
                "fitur kategorikal diisi modus lalu di-one-hot encoding."
            )

    st.markdown("### Profil dan rekomendasi tiap segmen")
    for start in range(0, len(profiles), 2):
        segment_columns = st.columns(2, gap="large")
        for column, (_, profile) in zip(
            segment_columns,
            profiles.iloc[start:start + 2].iterrows()
        ):
            cluster = int(profile["cluster"])
            with column:
                with st.container(border=True):
                    st.caption(
                        f'CLUSTER {cluster} | {profile["porsi_persen"]:.1f}% anggota'
                    )
                    st.subheader(profile["segmen"])
                    st.markdown(
                        f'**Jumlah anggota:** '
                        f'{format_count(profile["jumlah_pelanggan"])}'
                    )
                    metric_columns = st.columns(2)
                    metric_columns[0].metric(
                        "Median transaksi",
                        format_count(profile["median_transaksi"])
                    )
                    metric_columns[1].metric(
                        "Median belanja",
                        format_rupiah(profile["median_belanja"])
                    )
                    metric_columns[0].metric(
                        "Median hari sejak transaksi",
                        f'{profile["median_hari_terakhir"]:.0f} hari'
                    )
                    metric_columns[1].metric(
                        "Median penggunaan promo",
                        f'{profile["median_promo"]:.0f}%'
                    )
                    st.markdown("**Rekomendasi penawaran**")
                    st.write(SEGMENT_DETAILS[cluster]["recommendation"])

    st.caption(
        "Sumber profil: data_pelanggan_clean.csv. "
        "Angka profil dihitung ulang menggunakan model yang sama dengan prediksi."
    )
