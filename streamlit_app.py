"""Dashboard Analitik DJPb: capaian realisasi anggaran satker (versi interaktif).

Cara menjalankan dari folder proyek:
    streamlit run streamlit_app.py

Tab Prediksi membaca model Orange (.pkcls) di folder models/ dan
membutuhkan paket orange3:
    uv add orange3        (atau: pip install orange3)
"""

import pickle
import re
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

try:
    from Orange.data import Domain, Table
except Exception as exc:  # orange3 belum terpasang atau gagal dimuat
    Domain = Table = None
    ORANGE_ERROR = exc
else:
    ORANGE_ERROR = None

st.set_page_config(
    page_title="Dashboard Analitik DJPb",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# Konfigurasi
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "02_realisasi_anggaran_klasifikasi.csv"
MODEL_DIR = BASE_DIR / "models"

# Isi dengan nama Anda/tim agar tampil di bagian bawah halaman.
PEMBUAT = ""

TARGET = "realisasi_tercapai_95persen"
YA, TIDAK = "Ya", "Tidak"

# Palet mengikuti tema pink di .streamlit/config.toml
PINK_TUA = "#DB2777"
PINK_MUDA = "#F9A8D4"
PINK_GELAP = "#9D174D"
ABU = "#9CA3AF"
TEKS = "#262730"
SKALA_STATUS = alt.Scale(domain=[TIDAK, YA], range=[PINK_MUDA, PINK_TUA])
SKALA_IKPA = alt.Scale(
    domain=["Kurang", "Cukup", "Baik", "Sangat Baik"],
    range=["#FCE7F3", "#F9A8D4", "#EC4899", "#9D174D"],
)
RGB_MUDA = np.array([251, 207, 232])  # #FBCFE8
RGB_TUA = np.array([157, 23, 77])  # #9D174D

# Ambang resmi IKPA menurut PER-5/PB/2024 (dirangkum dari artikel DJPb)
IKPA_SANGAT_BAIK = 95.0
DEVIASI_MAKS = 5.0
SUMBER_IKPA = (
    "https://djpb.kemenkeu.go.id/kppn/metro/id/data-publikasi/artikel/"
    "3521-ikpa-rapor-kinerja-pelaksanaan-anggaran-satuan-kerja-2.html"
)
SUMBER_DEVIASI = (
    "https://djpb.kemenkeu.go.id/kppn/medan1/id/data-publikasi/artikel/"
    "2934-optimalisasi-peran-kppn-sebagai-financial-advisor-fa-dalam-"
    "meningkatkan-kualitas-perencanaan-pelaksanaan-anggaran-satuan-kerja.html"
)
ACUAN = {
    "skor_ikpa": (IKPA_SANGAT_BAIK, "IKPA Sangat Baik (≥95)"),
    "deviasi_rpd_persen": (DEVIASI_MAKS, "Ambang deviasi 5%"),
}

LABEL = {
    "kode_satker": "Kode satker",
    "nama_kementerian": "Kementerian",
    "provinsi": "Provinsi",
    "tipe_satker": "Tipe satker",
    "jenis_belanja_utama": "Jenis belanja utama",
    "pagu_miliar": "Pagu (miliar Rp)",
    "jumlah_pegawai": "Jumlah pegawai",
    "jumlah_spm": "Jumlah SPM",
    "revisi_dipa": "Jumlah revisi DIPA",
    "realisasi_tw1_persen": "Realisasi TW I (%)",
    "realisasi_tw2_persen": "Realisasi TW II (%)",
    "realisasi_tw3_persen": "Realisasi TW III (%)",
    "deviasi_rpd_persen": "Deviasi RPD (%)",
    "skor_ikpa": "Nilai IKPA",
    TARGET: "Realisasi ≥95%",
    "kategori_ikpa": "Kategori IKPA",
}

# Label untuk dipakai di tengah kalimat, beserta satuannya
LABEL_KALIMAT = {
    "skor_ikpa": ("nilai IKPA", ""),
    "deviasi_rpd_persen": ("deviasi RPD", "%"),
    "jumlah_spm": ("jumlah SPM", ""),
    "revisi_dipa": ("jumlah revisi DIPA", " kali"),
    "pagu_miliar": ("pagu", " miliar"),
    "jumlah_pegawai": ("jumlah pegawai", ""),
}

KETERANGAN = {
    "kode_satker": "Kode unik satuan kerja (satker).",
    "nama_kementerian": "Kementerian/lembaga induk satker.",
    "provinsi": "Provinsi lokasi satker.",
    "tipe_satker": "Kantor Pusat, Kantor Daerah, Dekonsentrasi, atau Tugas Pembantuan.",
    "jenis_belanja_utama": "Jenis belanja dengan porsi terbesar di satker.",
    "pagu_miliar": "Pagu anggaran satker dalam miliar rupiah.",
    "jumlah_pegawai": "Jumlah pegawai satker.",
    "jumlah_spm": "Jumlah Surat Perintah Membayar (SPM) yang diajukan.",
    "revisi_dipa": "Berapa kali DIPA direvisi.",
    "realisasi_tw1_persen": "Persentase realisasi anggaran pada TW I.",
    "realisasi_tw2_persen": "Persentase realisasi anggaran pada TW II.",
    "realisasi_tw3_persen": "Persentase realisasi anggaran pada TW III.",
    "deviasi_rpd_persen": (
        "Selisih realisasi terhadap Rencana Penarikan Dana (RPD) di Halaman III DIPA. "
        "Makin kecil makin baik."
    ),
    "skor_ikpa": "Nilai Indikator Kinerja Pelaksanaan Anggaran (IKPA), skala 0–100.",
    TARGET: "Target model: apakah realisasi anggaran mencapai minimal 95% (Ya/Tidak).",
    "kategori_ikpa": "Kolom turunan: Sangat Baik ≥95; Baik 89–<95; Cukup 70–<89; Kurang <70.",
}

KOORDINAT_PROVINSI = {
    "Aceh": [96.0, 4.0],
    "Bali": [115.2, -8.4],
    "DKI Jakarta": [106.8, -6.2],
    "Jawa Barat": [107.6, -6.9],
    "Jawa Tengah": [110.1, -7.3],
    "Jawa Timur": [112.2, -7.5],
    "Kalimantan Timur": [116.5, 0.5],
    "Lampung": [105.2, -4.9],
    "NTB": [117.5, -8.6],
    "NTT": [121.8, -9.7],
    "Papua": [138.1, -4.5],
    "Riau": [101.8, 0.5],
    "Sulawesi Selatan": [120.1, -3.7],
    "Sumatera Barat": [100.4, -0.7],
    "Sumatera Utara": [99.0, 2.1],
}

KOLOM_WAJIB = [
    "kode_satker", "nama_kementerian", "provinsi", "tipe_satker",
    "jenis_belanja_utama", "pagu_miliar", "jumlah_pegawai", "jumlah_spm",
    "revisi_dipa", "realisasi_tw1_persen", "realisasi_tw2_persen",
    "realisasi_tw3_persen", "deviasi_rpd_persen", "skor_ikpa", TARGET,
]
NUMERIK = [
    "skor_ikpa", "deviasi_rpd_persen", "realisasi_tw1_persen",
    "realisasi_tw2_persen", "realisasi_tw3_persen", "pagu_miliar",
    "jumlah_pegawai", "jumlah_spm", "revisi_dipa",
]
PILIHAN_WARNA = [TARGET, "kategori_ikpa", "tipe_satker", "jenis_belanja_utama"]
DIMENSI = {
    "Tipe satker": "tipe_satker",
    "Jenis belanja": "jenis_belanja_utama",
    "Kategori IKPA": "kategori_ikpa",
    "Kementerian": "nama_kementerian",
    "Provinsi": "provinsi",
}
METRIK_PETA = {
    "% realisasi ≥95%": "persen_ya",
    "Jumlah satker": "jumlah",
    "Rata-rata nilai IKPA": "rata_ikpa",
    "Rata-rata deviasi RPD": "rata_deviasi",
}
TEKS_KOSONG = "Tidak ada satker yang cocok dengan filter. Longgarkan filter di sidebar."
ISI_MANUAL = "(isi manual)"

KONFIG_KOLOM = {
    "kode_satker": st.column_config.TextColumn("Kode satker"),
    "nama_kementerian": st.column_config.TextColumn("Kementerian"),
    "provinsi": st.column_config.TextColumn("Provinsi"),
    "tipe_satker": st.column_config.TextColumn("Tipe satker"),
    "jenis_belanja_utama": st.column_config.TextColumn("Jenis belanja utama"),
    "pagu_miliar": st.column_config.NumberColumn("Pagu (miliar Rp)", format="%.2f"),
    "jumlah_pegawai": st.column_config.NumberColumn("Jumlah pegawai", format="%d"),
    "jumlah_spm": st.column_config.NumberColumn("Jumlah SPM", format="%d"),
    "revisi_dipa": st.column_config.NumberColumn("Revisi DIPA", format="%d"),
    "realisasi_tw1_persen": st.column_config.NumberColumn("Realisasi TW I (%)", format="%.2f"),
    "realisasi_tw2_persen": st.column_config.NumberColumn("Realisasi TW II (%)", format="%.2f"),
    "realisasi_tw3_persen": st.column_config.ProgressColumn(
        "Realisasi TW III (%)", min_value=0, max_value=100, format="%.1f"
    ),
    "deviasi_rpd_persen": st.column_config.NumberColumn("Deviasi RPD (%)", format="%.2f"),
    "skor_ikpa": st.column_config.ProgressColumn(
        "Nilai IKPA", min_value=0, max_value=100, format="%.2f"
    ),
    TARGET: st.column_config.TextColumn("Realisasi ≥95%"),
    "kategori_ikpa": st.column_config.TextColumn("Kategori IKPA"),
}


# =============================================================================
# Utilitas format
# =============================================================================
def angka(nilai, desimal=1):
    """Format angka gaya Indonesia, misalnya 1.234,5."""
    if nilai is None or pd.isna(nilai):
        return "–"
    teks = f"{nilai:,.{desimal}f}"
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")


def persen(nilai, desimal=1):
    return f"{angka(nilai, desimal)}%"


def selisih(nilai, desimal=1, satuan=""):
    """Teks delta untuk st.metric, lengkap dengan tanda + atau -."""
    nilai = round(float(nilai), desimal) + 0.0  # + 0.0 menghapus -0.0
    tanda = "+" if nilai >= 0 else ""
    return f"{tanda}{angka(nilai, desimal)}{satuan}"


def label(kolom):
    if kolom is None:
        return "Seragam"
    return LABEL.get(kolom, str(kolom).replace("_", " ").capitalize())


def label_fitur(nama):
    """Ubah nama fitur hasil one-hot Orange, mis. 'tipe_satker=Kantor Pusat'."""
    if "=" in nama:
        kolom, nilai = nama.split("=", 1)
        return f"{label(kolom)}: {nilai}"
    return label(nama)


def kategori_ikpa(nilai):
    if nilai >= IKPA_SANGAT_BAIK:
        return "Sangat Baik"
    if nilai >= 89:
        return "Baik"
    if nilai >= 70:
        return "Cukup"
    return "Kurang"


# =============================================================================
# Data
# =============================================================================
@st.cache_data(show_spinner="Memuat data…")
def muat_data(path: str, mtime: float) -> pd.DataFrame:
    """mtime ikut di-hash agar cache diperbarui ketika file CSV berubah."""
    data = pd.read_csv(path)
    kurang = [k for k in KOLOM_WAJIB if k not in data.columns]
    if kurang:
        raise ValueError("kolom berikut tidak ditemukan: " + ", ".join(kurang))
    data[TARGET] = data[TARGET].astype(str).str.strip().str.capitalize()
    data["kategori_ikpa"] = pd.cut(
        data["skor_ikpa"],
        bins=[-np.inf, 70, 89, IKPA_SANGAT_BAIK, np.inf],
        right=False,
        labels=["Kurang", "Cukup", "Baik", "Sangat Baik"],
    ).astype(str)
    return data


def nilai_default_filter(data):
    ikpa = data["skor_ikpa"]
    dev = data["deviasi_rpd_persen"]
    return {
        "f_prov": [],
        "f_kem": [],
        "f_tipe": [],
        "f_belanja": [],
        "f_ikpa": (float(np.floor(ikpa.min())), float(np.ceil(ikpa.max()))),
        "f_dev": (float(np.floor(dev.min())), float(np.ceil(dev.max()))),
    }


def atur_ulang_filter(default):
    for kunci, nilai in default.items():
        st.session_state[kunci] = nilai


def saring(data, prov, kem, tipe, belanja, rentang_ikpa, rentang_dev):
    mask = data["skor_ikpa"].between(*rentang_ikpa) & data["deviasi_rpd_persen"].between(
        *rentang_dev
    )
    for kolom, pilihan in [
        ("provinsi", prov),
        ("nama_kementerian", kem),
        ("tipe_satker", tipe),
        ("jenis_belanja_utama", belanja),
    ]:
        if pilihan:
            mask &= data[kolom].isin(pilihan)
    return data[mask]


def rasio_ya(data):
    return (data[TARGET] == YA).mean() * 100


def susun_temuan(d):
    """Temuan otomatis yang dihitung ulang setiap kali filter berubah."""
    n = len(d)
    ya = int((d[TARGET] == YA).sum())
    hasil = [f"**{ya} dari {n} satker ({persen(ya / n * 100)})** mencapai realisasi ≥95%."]

    dalam = d[d["deviasi_rpd_persen"] <= DEVIASI_MAKS]
    luar = d[d["deviasi_rpd_persen"] > DEVIASI_MAKS]
    if len(dalam) >= 5 and len(luar) >= 5:
        hasil.append(
            f"Satker dengan deviasi RPD ≤5% mencapai target **{persen(rasio_ya(dalam))}** "
            f"({len(dalam)} satker), dibanding **{persen(rasio_ya(luar))}** "
            f"pada satker dengan deviasi di atas 5% ({len(luar)} satker)."
        )

    per_tipe = d.groupby("tipe_satker")[TARGET].apply(lambda s: (s == YA).mean() * 100)
    if len(per_tipe) >= 2:
        hasil.append(
            f"Capaian tertinggi ada di tipe **{per_tipe.idxmax()}** ({persen(per_tipe.max())}); "
            f"terendah di **{per_tipe.idxmin()}** ({persen(per_tipe.min())})."
        )

    per_prov = d.assign(ya=d[TARGET].eq(YA)).groupby("provinsi")["ya"].agg(["size", "mean"])
    per_prov = per_prov[per_prov["size"] >= 10]
    if len(per_prov) >= 2:
        atas, bawah = per_prov["mean"].idxmax(), per_prov["mean"].idxmin()
        hasil.append(
            f"Provinsi dengan capaian tertinggi: **{atas}** "
            f"({persen(per_prov.loc[atas, 'mean'] * 100)}); terendah: **{bawah}** "
            f"({persen(per_prov.loc[bawah, 'mean'] * 100)}). "
            "Hanya provinsi dengan minimal 10 satker yang dibandingkan."
        )

    kategori = d["kategori_ikpa"].value_counts()
    if not kategori.empty:
        sangat_baik = (d["skor_ikpa"] >= IKPA_SANGAT_BAIK).mean() * 100
        hasil.append(
            f"Kategori IKPA terbanyak: **{kategori.idxmax()}** "
            f"({persen(kategori.max() / n * 100)} satker); "
            f"{persen(sangat_baik)} satker sudah masuk kategori Sangat Baik (≥95)."
        )
    return hasil


# =============================================================================
# Grafik
# =============================================================================
def garis_acuan(kolom, sumbu):
    """Garis putus-putus untuk ambang resmi (IKPA 95 dan deviasi 5%)."""
    if kolom not in ACUAN:
        return []
    nilai, teks = ACUAN[kolom]
    data = pd.DataFrame({kolom: [nilai], "teks": [teks]})
    garis = alt.Chart(data).mark_rule(color=PINK_GELAP, strokeDash=[6, 4], strokeWidth=1.5)
    tulisan = alt.Chart(data).mark_text(
        align="left", fontSize=11, fontWeight="bold", color=PINK_GELAP
    )
    if sumbu == "x":
        garis = garis.encode(x=alt.X(f"{kolom}:Q"))
        tulisan = tulisan.mark_text(
            align="left", baseline="top", dx=5, dy=5, fontSize=11, color=PINK_GELAP
        ).encode(x=alt.X(f"{kolom}:Q"), y=alt.value(0), text="teks:N")
    else:
        garis = garis.encode(y=alt.Y(f"{kolom}:Q"))
        tulisan = tulisan.mark_text(
            align="left", baseline="bottom", dx=5, dy=-4, fontSize=11, color=PINK_GELAP
        ).encode(y=alt.Y(f"{kolom}:Q"), x=alt.value(0), text="teks:N")
    return [garis, tulisan]


def grafik_donut(d):
    hitung = d[TARGET].value_counts().reindex([TIDAK, YA], fill_value=0)
    data = hitung.rename_axis("status").reset_index(name="jumlah")
    total = int(data["jumlah"].sum())
    data["teks"] = (data["jumlah"] / max(total, 1) * 100).map(persen)
    dasar = alt.Chart(data).encode(
        theta=alt.Theta("jumlah:Q", stack=True),
        order=alt.Order("status:N", sort="descending"),
        color=alt.Color(
            "status:N", scale=SKALA_STATUS, title="Realisasi ≥95%",
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("jumlah:Q", title="Jumlah satker"),
            alt.Tooltip("teks:N", title="Persentase"),
        ],
    )
    cincin = dasar.mark_arc(innerRadius=70, outerRadius=115, stroke="white", strokeWidth=2)
    teks = dasar.mark_text(radius=138, fontSize=14, fontWeight="bold").encode(
        text="teks:N", color=alt.value(TEKS)
    )
    tengah = (
        alt.Chart(pd.DataFrame({"t": [f"{total} satker"]}))
        .mark_text(fontSize=15, fontWeight="bold", color=TEKS)
        .encode(text="t:N")
    )
    return (cincin + teks + tengah).properties(height=320)


def grafik_kelompok(d, kolom):
    """Batang 100% bertumpuk: proporsi Ya/Tidak per kelompok."""
    ringkas = (
        d.assign(ya=d[TARGET].eq(YA))
        .groupby(kolom, observed=True)["ya"]
        .agg(jumlah="size", ya="sum")
        .reset_index()
    )
    ringkas[kolom] = ringkas[kolom].astype(str)
    ringkas["tidak"] = ringkas["jumlah"] - ringkas["ya"]
    ringkas["persen_ya"] = ringkas["ya"] / ringkas["jumlah"] * 100
    urutan = ringkas.sort_values("persen_ya", ascending=False)[kolom].tolist()

    panjang = ringkas.melt(
        id_vars=[kolom, "jumlah"], value_vars=["ya", "tidak"], var_name="status", value_name="n"
    )
    panjang["status"] = panjang["status"].map({"ya": YA, "tidak": TIDAK})
    panjang["persen"] = panjang["n"] / panjang["jumlah"] * 100
    panjang["urut"] = panjang["status"].map({YA: 0, TIDAK: 1})
    panjang["teks"] = panjang["persen"].map(persen)

    rata = rasio_ya(d)
    batang = alt.Chart(panjang).mark_bar(cornerRadius=2).encode(
        y=alt.Y(f"{kolom}:N", sort=urutan, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(
            "persen:Q", stack="zero", scale=alt.Scale(domain=[0, 100]),
            title="Persentase satker (%)",
        ),
        color=alt.Color(
            "status:N", scale=SKALA_STATUS, title="Realisasi ≥95%",
            legend=alt.Legend(orient="bottom"),
        ),
        order=alt.Order("urut:Q"),
        tooltip=[
            alt.Tooltip(f"{kolom}:N", title=label(kolom)),
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("n:Q", title="Jumlah satker"),
            alt.Tooltip("teks:N", title="Persentase"),
        ],
    )
    acuan = pd.DataFrame({"persen": [rata], "teks": [f"Rata-rata {persen(rata)}"]})
    garis = alt.Chart(acuan).mark_rule(color=TEKS, strokeDash=[4, 4]).encode(x="persen:Q")
    tulisan = (
        alt.Chart(acuan)
        .mark_text(align="left", baseline="top", dx=4, dy=2, fontSize=11, color=TEKS)
        .encode(x="persen:Q", y=alt.value(0), text="teks:N")
    )
    tinggi = max(230, 40 * len(urutan) + 130)  # termasuk ruang sumbu dan legenda
    return (batang + garis + tulisan).properties(height=tinggi)


def grafik_triwulan(d):
    kolom_tw = {
        "realisasi_tw1_persen": "TW I",
        "realisasi_tw2_persen": "TW II",
        "realisasi_tw3_persen": "TW III",
    }
    rata = (
        d.groupby(TARGET)[list(kolom_tw)]
        .mean()
        .reset_index()
        .melt(id_vars=TARGET, var_name="triwulan", value_name="rata")
    )
    rata["triwulan"] = rata["triwulan"].map(kolom_tw)
    rata["teks"] = rata["rata"].map(persen)
    garis = alt.Chart(rata).mark_line(
        point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3
    ).encode(
        x=alt.X(
            "triwulan:N", sort=list(kolom_tw.values()), title=None,
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("rata:Q", title="Rata-rata realisasi (%)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            f"{TARGET}:N", scale=SKALA_STATUS, title="Realisasi ≥95%",
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip(f"{TARGET}:N", title="Status"),
            alt.Tooltip("triwulan:N", title="Triwulan"),
            alt.Tooltip("teks:N", title="Rata-rata realisasi"),
        ],
    )
    teks = garis.mark_text(dy=-14, fontSize=12, fontWeight="bold").encode(text="teks:N")
    return (garis + teks).properties(height=300)


def grafik_sebaran(d, x, y, ukuran, warna):
    pilih = alt.selection_interval(name="pilih", encodings=["x", "y"])
    if warna == TARGET:
        enc_warna = alt.Color(f"{TARGET}:N", scale=SKALA_STATUS, title=label(TARGET))
    elif warna == "kategori_ikpa":
        enc_warna = alt.Color("kategori_ikpa:N", scale=SKALA_IKPA, title=label(warna))
    else:
        enc_warna = alt.Color(f"{warna}:N", title=label(warna))
    enc_ukuran = (
        alt.Size(f"{ukuran}:Q", title=label(ukuran), scale=alt.Scale(range=[15, 650]))
        if ukuran
        else alt.value(70)
    )
    tooltip = [
        alt.Tooltip("kode_satker:N", title="Kode satker"),
        alt.Tooltip("nama_kementerian:N", title="Kementerian"),
        alt.Tooltip("provinsi:N", title="Provinsi"),
        alt.Tooltip("tipe_satker:N", title="Tipe satker"),
    ]
    for kolom in dict.fromkeys([x, y, ukuran, "skor_ikpa", "deviasi_rpd_persen"]):
        if kolom:
            tooltip.append(alt.Tooltip(f"{kolom}:Q", title=label(kolom), format=",.2f"))
    tooltip.append(alt.Tooltip(f"{TARGET}:N", title=label(TARGET)))

    titik = (
        alt.Chart(d)
        .mark_circle(stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(f"{x}:Q", title=label(x), scale=alt.Scale(zero=False, nice=True)),
            y=alt.Y(f"{y}:Q", title=label(y), scale=alt.Scale(zero=False, nice=True)),
            size=enc_ukuran,
            color=enc_warna,
            opacity=alt.when(pilih).then(alt.value(0.85)).otherwise(alt.value(0.15)),
            tooltip=tooltip,
        )
        .add_params(pilih)
    )
    lapisan = [titik] + garis_acuan(x, "x") + garis_acuan(y, "y")
    return alt.layer(*lapisan).properties(height=480)


def ambil_pilihan(d, event, x, y):
    """Ambil satker di dalam kotak seleksi grafik (None jika belum memilih)."""
    try:
        seleksi = event.selection.get("pilih") or {}
    except AttributeError:
        seleksi = ((event or {}).get("selection") or {}).get("pilih") or {}
    if not seleksi:
        return None
    mask = pd.Series(True, index=d.index)
    dipakai = False
    for kolom, rentang in dict(seleksi).items():
        if kolom in (x, y) and isinstance(rentang, (list, tuple)) and len(rentang) == 2:
            bawah, atas = sorted(float(v) for v in rentang)
            mask &= d[kolom].between(bawah, atas)
            dipakai = True
    return d[mask] if dipakai else None


def grafik_kotak(d, kolom):
    return (
        alt.Chart(d)
        .mark_boxplot(size=26, extent=1.5, median={"color": TEKS})
        .encode(
            y=alt.Y(f"{TARGET}:N", title=None, sort=[YA, TIDAK]),
            x=alt.X(f"{kolom}:Q", title=label(kolom), scale=alt.Scale(zero=False)),
            color=alt.Color(f"{TARGET}:N", scale=SKALA_STATUS, legend=None),
        )
        .properties(height=150, title=f"{label(kolom)} per status realisasi")
    )


def agregat_provinsi(d):
    ringkas = (
        d.assign(ya=d[TARGET].eq(YA))
        .groupby("provinsi")
        .agg(
            jumlah=("kode_satker", "size"),
            persen_ya=("ya", "mean"),
            rata_ikpa=("skor_ikpa", "mean"),
            rata_deviasi=("deviasi_rpd_persen", "mean"),
        )
        .reset_index()
    )
    ringkas["persen_ya"] = ringkas["persen_ya"] * 100
    koordinat = ringkas["provinsi"].map(KOORDINAT_PROVINSI)
    ringkas["lon"] = koordinat.map(lambda k: k[0] if isinstance(k, list) else np.nan)
    ringkas["lat"] = koordinat.map(lambda k: k[1] if isinstance(k, list) else np.nan)
    tanpa_koordinat = ringkas.loc[ringkas["lon"].isna(), "provinsi"].tolist()
    return ringkas.dropna(subset=["lon", "lat"]).reset_index(drop=True), tanpa_koordinat


def warna_gradasi(nilai):
    v = np.asarray(nilai, dtype=float)
    rentang = np.nanmax(v) - np.nanmin(v)
    t = (v - np.nanmin(v)) / rentang if rentang > 0 else np.full_like(v, 0.6)
    rgb = RGB_MUDA + np.outer(t, RGB_TUA - RGB_MUDA)
    return [[int(r), int(g), int(b), 230] for r, g, b in rgb]


def peta_provinsi(peta, kolom_metrik):
    peta = peta.copy()
    peta["warna"] = warna_gradasi(peta[kolom_metrik])
    peta["radius"] = 9 + 15 * np.sqrt(peta["jumlah"] / peta["jumlah"].max())
    peta["teks_ya"] = peta["persen_ya"].map(persen)
    peta["teks_ikpa"] = peta["rata_ikpa"].map(lambda v: angka(v, 2))
    peta["teks_dev"] = peta["rata_deviasi"].map(persen)
    lapisan = pdk.Layer(
        "ScatterplotLayer",
        id="provinsi",
        data=peta,
        get_position="[lon, lat]",
        get_fill_color="warna",
        get_line_color=[157, 23, 77, 255],
        get_radius="radius",
        radius_units="pixels",
        line_width_min_pixels=1.5,
        stroked=True,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 110],
    )
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=-2.4, longitude=117.9, zoom=4.1, pitch=0),
        layers=[lapisan],
        tooltip={
            "html": (
                "<b>{provinsi}</b><br/>Jumlah satker: {jumlah}<br/>"
                "Realisasi ≥95%: {teks_ya}<br/>Rata-rata nilai IKPA: {teks_ikpa}<br/>"
                "Rata-rata deviasi RPD: {teks_dev}"
            ),
            "style": {"backgroundColor": "white", "color": TEKS, "fontSize": "12px"},
        },
    )


# =============================================================================
# Model (Orange)
# =============================================================================
def daftar_model():
    if not MODEL_DIR.is_dir():
        return []
    return sorted(p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".pkcls", ".pkl"})


@st.cache_resource(show_spinner="Memuat model…")
def muat_model(path: str, mtime: float):
    """Buka model dari folder models/. Hanya untuk file tepercaya: pickle bisa
    menjalankan kode saat dibuka, jadi jangan pernah membuka pickle unggahan."""
    with warnings.catch_warnings(record=True) as tangkapan:
        warnings.simplefilter("always")
        with open(path, "rb") as berkas:
            model = pickle.load(berkas)
    pesan = sorted({str(w.message).splitlines()[0][:220] for w in tangkapan})
    return model, pesan


def variabel_masukan(domain):
    """Kolom mentah yang dibutuhkan model.

    Jika data dipra-proses di Orange sebelum dilatih (mis. widget Preprocess untuk one-hot
    dan normalisasi), atribut domain model berupa variabel turunan seperti
    'tipe_satker=Kantor Pusat'. Telusuri compute_value sampai ke variabel aslinya, lalu
    biarkan Orange sendiri yang menjalankan ulang pra-prosesnya saat prediksi."""
    hasil, sudah = [], set()
    for var in domain.attributes:
        akar = var
        for _ in range(12):
            sumber = getattr(getattr(akar, "compute_value", None), "variable", None)
            if sumber is None:
                break
            akar = sumber
        if akar.name not in sudah:
            sudah.add(akar.name)
            hasil.append(akar)
    return hasil


def struktur_model(model):
    domain = getattr(model, "original_domain", None)
    if domain is None:
        domain = model.domain
    return variabel_masukan(domain), domain.class_var


def nama_model(model):
    nama = re.sub(r"\s*\(\d+\)$", "", str(getattr(model, "name", "") or type(model).__name__))
    return nama.title() if nama.islower() else nama


def catatan_versi(model):
    skl = getattr(model, "skl_model", None)
    versi_simpan = getattr(skl, "_sklearn_version", None)
    try:
        import sklearn

        versi_kini = sklearn.__version__
    except Exception:
        versi_kini = None
    if versi_simpan and versi_kini and versi_simpan != versi_kini:
        return (
            f"Model disimpan dengan scikit-learn {versi_simpan}, sedangkan lingkungan ini "
            f"memakai {versi_kini}. Biasanya model tetap bisa dipakai, tetapi untuk "
            "konsistensi simpan ulang model dari Orange dengan versi yang sama."
        )
    return None


def prediksi(model, fitur, kelas, data):
    """Bangun tabel Orange sesuai domain asli model, lalu prediksi."""
    X = np.full((len(data), len(fitur)), np.nan)
    for j, var in enumerate(fitur):
        if var.name not in data.columns:
            continue
        kolom = data[var.name]
        if var.is_discrete:
            indeks = {nilai: i for i, nilai in enumerate(var.values)}
            X[:, j] = kolom.astype(str).map(indeks).astype(float).to_numpy()
        else:
            X[:, j] = pd.to_numeric(kolom, errors="coerce").to_numpy(dtype=float)
    tabel = Table.from_numpy(Domain(fitur), X)
    if kelas.is_discrete:
        nilai, peluang = model(tabel, model.ValueProbs)
        hasil = pd.DataFrame(
            {"prediksi": [kelas.values[int(i)] for i in nilai]}, index=data.index
        )
        for j, nama in enumerate(kelas.values):
            hasil[f"peluang_{nama}"] = peluang[:, j]
    else:
        hasil = pd.DataFrame({"prediksi": np.asarray(model(tabel), dtype=float)}, index=data.index)
    return hasil


def kode_data_latih(model):
    """Kode satker pada data latih yang ikut tersimpan di file model Orange (jika ada)."""
    for nilai in vars(model).values():
        if type(nilai).__name__ != "Table":
            continue
        try:
            return set(map(str, nilai.get_column("kode_satker")))
        except Exception:
            continue
    return set()


def koefisien_model(model, kelas):
    """Koefisien model linear (mis. Logistic Regression) beserta kelas positifnya."""
    skl = getattr(model, "skl_model", None)
    koef = getattr(skl, "coef_", None)
    if koef is None:
        return None, None
    koef = np.asarray(koef, dtype=float)
    nama = [a.name for a in model.domain.attributes]
    if koef.ndim == 2:
        if koef.shape[0] != 1:
            return None, None
        koef = koef[0]
    if koef.shape[0] != len(nama):
        return None, None
    kelas_positif = None
    if kelas is not None and kelas.is_discrete:
        kelas_skl = getattr(skl, "classes_", None)
        indeks = int(kelas_skl[-1]) if kelas_skl is not None else len(kelas.values) - 1
        kelas_positif = kelas.values[indeks]
    tabel = pd.DataFrame({"fitur": [label_fitur(n) for n in nama], "koef": koef})
    return tabel, kelas_positif


def ada_normalisasi(model):
    """True jika Orange menormalisasi fitur sebelum melatih model."""
    for var in model.domain.attributes:
        nilai = getattr(var, "compute_value", None)
        for _ in range(4):
            if nilai is None:
                break
            if type(nilai).__name__ == "Normalizer":
                return True
            dalam = getattr(nilai, "variable", None)
            nilai = getattr(dalam, "compute_value", None)
    return False


def grafik_koefisien(tabel, kelas_positif, dinormalisasi):
    naik = f"Menaikkan peluang {kelas_positif}" if kelas_positif else "Menaikkan prediksi"
    turun = f"Menurunkan peluang {kelas_positif}" if kelas_positif else "Menurunkan prediksi"
    data = tabel.assign(arah=np.where(tabel["koef"] >= 0, naik, turun))
    urutan = data.sort_values("koef", ascending=False)["fitur"].tolist()
    judul = "Koefisien (fitur dinormalisasi)" if dinormalisasi else "Koefisien"
    return (
        alt.Chart(data)
        .mark_bar(cornerRadius=2)
        .encode(
            y=alt.Y("fitur:N", sort=urutan, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("koef:Q", title=judul),
            color=alt.Color(
                "arah:N", scale=alt.Scale(domain=[naik, turun], range=[PINK_TUA, ABU]),
                title=None, legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("fitur:N", title="Fitur"),
                alt.Tooltip("koef:Q", title="Koefisien", format="+.3f"),
            ],
        )
        .properties(height=32 * len(data) + 50)
    )


def kunci_input(var):
    return f"inp_{var.name}"


def nilai_awal(var, data):
    if var.is_discrete:
        if var.name in data.columns:
            modus = data[var.name].astype(str).mode()
            if not modus.empty and modus.iloc[0] in var.values:
                return modus.iloc[0]
        return var.values[0]
    if var.name in data.columns and pd.api.types.is_numeric_dtype(data[var.name]):
        median = data[var.name].median()
        if pd.api.types.is_integer_dtype(data[var.name]):
            return int(round(median))
        return round(float(median), 2)
    return 0.0


def isi_otomatis(data, fitur):
    """Callback: salin nilai satker terpilih ke input prediksi."""
    kode = st.session_state.get("prefill_satker")
    if not kode or kode == ISI_MANUAL:
        return
    baris = data.loc[data["kode_satker"] == kode]
    if baris.empty:
        return
    baris = baris.iloc[0]
    for var in fitur:
        if var.name not in baris.index:
            continue
        nilai = baris[var.name]
        if var.is_discrete:
            if str(nilai) in var.values:
                st.session_state[kunci_input(var)] = str(nilai)
        elif pd.api.types.is_integer_dtype(data[var.name]):
            st.session_state[kunci_input(var)] = int(nilai)
        else:
            st.session_state[kunci_input(var)] = round(float(nilai), 2)


def widget_input(var, data):
    kunci = kunci_input(var)
    st.session_state.setdefault(kunci, nilai_awal(var, data))
    teks_label = label(var.name)
    if var.is_discrete:
        pilihan = list(var.values)
        if st.session_state[kunci] not in pilihan:
            st.session_state[kunci] = pilihan[0]
        if len(pilihan) <= 5:
            st.segmented_control(teks_label, pilihan, key=kunci, required=True)
        else:
            st.selectbox(teks_label, pilihan, key=kunci)
        return
    if var.name in data.columns and pd.api.types.is_numeric_dtype(data[var.name]):
        kolom = data[var.name]
        bilangan_bulat = pd.api.types.is_integer_dtype(kolom)
        bawah, atas = kolom.min(), kolom.max()
        if bilangan_bulat:
            bawah, atas, langkah = int(bawah), int(atas), 1
            st.session_state[kunci] = int(np.clip(st.session_state[kunci], bawah, atas))
        else:
            bawah, atas, langkah = float(np.floor(bawah)), float(np.ceil(atas)), 0.01
            st.session_state[kunci] = float(np.clip(st.session_state[kunci], bawah, atas))
        st.slider(
            teks_label, bawah, atas, step=langkah, key=kunci,
            help=f"Rentang di data: {angka(kolom.min(), 2)} sampai {angka(kolom.max(), 2)}",
        )
    else:
        st.number_input(teks_label, key=kunci, step=0.1)


def rentang_simulasi(var, data, nilai_kini):
    if var.name in data.columns and pd.api.types.is_numeric_dtype(data[var.name]):
        bawah, atas = float(data[var.name].min()), float(data[var.name].max())
    else:
        bawah, atas = min(0.0, nilai_kini), max(1.0, nilai_kini * 2)
    bawah, atas = min(bawah, nilai_kini), max(atas, nilai_kini)
    if var.name in data.columns and pd.api.types.is_integer_dtype(data[var.name]):
        return np.arange(int(np.floor(bawah)), int(np.ceil(atas)) + 1)
    return np.linspace(bawah, atas, 61)


def catatan_ketentuan(nilai_input):
    catatan = []
    dev = nilai_input.get("deviasi_rpd_persen")
    if dev is not None and dev > DEVIASI_MAKS:
        catatan.append(
            f"Deviasi RPD {persen(dev, 2)} masih di atas ambang 5%. Rata-rata deviasi "
            "bulanan paling tinggi 5% diperlukan agar nilai indikator deviasi Halaman III "
            "DIPA maksimal."
        )
    ikpa = nilai_input.get("skor_ikpa")
    if ikpa is not None and ikpa < IKPA_SANGAT_BAIK:
        catatan.append(
            f"Nilai IKPA {angka(ikpa, 2)} masuk kategori **{kategori_ikpa(ikpa)}**; "
            "kategori Sangat Baik dimulai dari 95."
        )
    return catatan


def tampilkan_prediksi(df, dff, berkas_model):
    if len(berkas_model) > 1:
        berkas = st.selectbox("Model", berkas_model, format_func=lambda p: p.name, key="pilih_model")
    else:
        berkas = berkas_model[0]
    try:
        model, peringatan = muat_model(str(berkas), berkas.stat().st_mtime)
        fitur, kelas = struktur_model(model)
    except Exception as exc:
        st.error(f"Model **{berkas.name}** gagal dibuka: {exc}", icon=":material/error:")
        st.caption(
            "Penyebab yang umum: versi orange3/scikit-learn berbeda jauh dari versi saat "
            "model disimpan. Simpan ulang model dari Orange lalu coba lagi."
        )
        return
    if kelas is None or not fitur:
        st.warning("Model ini tidak memiliki variabel target atau fitur yang bisa dibaca.")
        return
    tak_terlacak = [v.name for v in fitur if getattr(v, "compute_value", None) is not None]
    if tak_terlacak:
        st.warning(
            "Fitur turunan berikut tidak bisa dilacak ke kolom mentah, jadi prediksi butuh "
            "kolom dengan nama persis sama: " + ", ".join(tak_terlacak)
        )
    klasifikasi = kelas.is_discrete
    positif = (YA if YA in kelas.values else kelas.values[-1]) if klasifikasi else None

    with st.container(border=True):
        a, b, c = st.columns(3)
        a.markdown(f"**Model**  \n{nama_model(model)}")
        b.markdown(f"**Jenis**  \n{'Klasifikasi' if klasifikasi else 'Regresi'}")
        c.markdown(f"**Target**  \n{label(kelas.name)}")
        st.caption("Fitur yang dipakai model: " + ", ".join(label(v.name) for v in fitur) + ".")
        versi = catatan_versi(model)
        if versi:
            st.caption(f"⚠️ {versi}")

    kiri, kanan = st.columns([1, 1.35], gap="large")
    with kiri:
        st.markdown("##### Kondisi satker")
        if "kode_satker" in df.columns:
            st.selectbox(
                "Isi otomatis dari data satker (opsional)",
                [ISI_MANUAL] + df["kode_satker"].astype(str).tolist(),
                key="prefill_satker",
                on_change=isi_otomatis,
                args=(df, fitur),
            )
            kode = st.session_state.get("prefill_satker", ISI_MANUAL)
            if kode != ISI_MANUAL and klasifikasi and kelas.name in df.columns:
                aktual = df.loc[df["kode_satker"] == kode, kelas.name]
                if not aktual.empty:
                    st.caption(
                        f"Status aktual {kode}: **{aktual.iloc[0]}**. Geser input untuk "
                        "mencoba skenario lain."
                    )
        for var in fitur:
            widget_input(var, df)

    nilai_input = {v.name: st.session_state[kunci_input(v)] for v in fitur}
    try:
        hasil = prediksi(model, fitur, kelas, pd.DataFrame([nilai_input]))
    except Exception as exc:
        kanan.error(f"Prediksi gagal: {exc}", icon=":material/error:")
        return

    with kanan:
        with st.container(border=True):
            if klasifikasi:
                peluang = float(hasil.loc[0, f"peluang_{positif}"])
                judul = (
                    "Peluang realisasi ≥95%" if kelas.name == TARGET
                    else f"Peluang {label(kelas.name)} = {positif}"
                )
                st.metric(judul, persen(peluang * 100))
                st.progress(min(max(peluang, 0.0), 1.0))
                if hasil.loc[0, "prediksi"] == positif:
                    st.success(
                        f"Prediksi model: **{positif}**. Dengan kondisi ini satker diperkirakan "
                        "mencapai target.", icon=":material/check_circle:",
                    )
                else:
                    st.warning(
                        f"Prediksi model: **{hasil.loc[0, 'prediksi']}**. Dengan kondisi ini "
                        "satker diperkirakan belum mencapai target.", icon=":material/info:",
                    )
            else:
                st.metric(f"Prediksi {label(kelas.name)}", angka(float(hasil.loc[0, "prediksi"]), 2))
            catatan = catatan_ketentuan(nilai_input)
            if catatan:
                st.markdown("**Catatan berdasarkan ketentuan IKPA**")
                for teks in catatan:
                    st.markdown(f"- {teks}")
            st.caption("Prediksi bersifat probabilistik: pakai sebagai bahan diskusi, bukan keputusan final.")

        kontinu = [v for v in fitur if v.is_continuous]
        if kontinu:
            nama_kontinu = [v.name for v in kontinu]
            awal = nama_kontinu.index("deviasi_rpd_persen") if "deviasi_rpd_persen" in nama_kontinu else 0
            dipilih = st.selectbox(
                "Simulasi: bagaimana jika nilai ini berubah?", nama_kontinu, index=awal,
                format_func=label, key="var_simulasi",
            )
            var_sim = kontinu[nama_kontinu.index(dipilih)]
            kini = float(nilai_input[var_sim.name])
            grid = rentang_simulasi(var_sim, df, kini)
            data_sim = pd.DataFrame([nilai_input] * len(grid))
            data_sim[var_sim.name] = grid
            hasil_sim = prediksi(model, fitur, kelas, data_sim)
            kolom_y = f"peluang_{positif}" if klasifikasi else "prediksi"
            skala = 100 if klasifikasi else 1
            judul_y = "Peluang realisasi ≥95% (%)" if klasifikasi else label(kelas.name)
            data_sim["y"] = hasil_sim[kolom_y].to_numpy() * skala
            y_kini = float(hasil[kolom_y].iloc[0]) * skala
            garis = alt.Chart(data_sim).mark_line(color=PINK_TUA, strokeWidth=3).encode(
                x=alt.X(f"{var_sim.name}:Q", title=label(var_sim.name)),
                y=alt.Y(
                    "y:Q", title=judul_y,
                    scale=alt.Scale(domain=[0, 100]) if klasifikasi else alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(f"{var_sim.name}:Q", title=label(var_sim.name), format=",.2f"),
                    alt.Tooltip("y:Q", title=judul_y, format=",.1f"),
                ],
            )
            titik = alt.Chart(
                pd.DataFrame({var_sim.name: [kini], "y": [y_kini], "t": ["Kondisi saat ini"]})
            ).mark_point(size=170, filled=True, color=PINK_GELAP).encode(
                x=f"{var_sim.name}:Q", y="y:Q", tooltip=[alt.Tooltip("t:N", title="Titik")]
            )
            lapisan = [garis, titik] + garis_acuan(var_sim.name, "x")
            st.altair_chart(alt.layer(*lapisan).properties(height=280), width="stretch")
            if var_sim.name in ACUAN and klasifikasi:
                acuan, _ = ACUAN[var_sim.name]
                skenario = dict(nilai_input, **{var_sim.name: acuan})
                y_acuan = float(
                    prediksi(model, fitur, kelas, pd.DataFrame([skenario]))[kolom_y].iloc[0]
                ) * skala
                nama, satuan = LABEL_KALIMAT.get(var_sim.name, (label(var_sim.name), ""))
                st.caption(
                    f"Jika {nama} diubah dari {angka(kini, 2)}{satuan} menjadi "
                    f"{angka(acuan, 2)}{satuan} (input lain tetap), peluang realisasi ≥95% "
                    f"berubah dari **{persen(y_kini)}** menjadi **{persen(y_acuan)}**."
                )

    tabel_koef, kelas_positif = koefisien_model(model, kelas)
    if tabel_koef is not None:
        st.markdown("##### Faktor yang paling berpengaruh menurut model")
        dinormalisasi = ada_normalisasi(model)
        st.altair_chart(
            grafik_koefisien(tabel_koef, kelas_positif, dinormalisasi), width="stretch"
        )
        st.caption(
            ("Fitur dinormalisasi oleh Orange sebelum model dilatih, jadi panjang batang bisa "
             "dibandingkan antarfitur. " if dinormalisasi else "")
            + "Koefisien menunjukkan arah keterkaitan di dalam model, bukan sebab-akibat."
        )

    st.divider()
    st.markdown("##### Prediksi massal")
    sumber = st.segmented_control(
        "Sumber data", ["Satker sesuai filter", "Unggah file CSV"],
        default="Satker sesuai filter", key="sumber_massal", required=True,
    )
    data_massal = None
    if sumber == "Unggah file CSV":
        unggahan = st.file_uploader(
            "Unggah CSV", type=["csv"], key="unggah_csv",
            help="Kolom yang dibutuhkan: " + ", ".join(v.name for v in fitur),
        )
        if unggahan is None:
            st.caption("Kolom yang dibutuhkan: " + ", ".join(v.name for v in fitur) + ".")
        else:
            try:
                data_massal = pd.read_csv(unggahan)
            except Exception as exc:
                st.error(f"File CSV tidak bisa dibaca: {exc}")
    else:
        data_massal = dff
    if data_massal is None:
        return
    if data_massal.empty:
        st.info(TEKS_KOSONG)
        return
    hilang = [v.name for v in fitur if v.name not in data_massal.columns]
    if hilang:
        st.error("Kolom berikut tidak ada di data: " + ", ".join(hilang))
        return
    for var in fitur:
        if var.is_discrete:
            asing = data_massal[var.name].notna() & ~data_massal[var.name].astype(str).isin(var.values)
            if asing.any():
                st.warning(
                    f"{int(asing.sum())} baris punya nilai {label(var.name)} yang tidak dikenal "
                    "model; nilainya diperlakukan sebagai kosong."
                )
    hasil_massal = prediksi(model, fitur, kelas, data_massal.reset_index(drop=True))
    gabungan = pd.concat([data_massal.reset_index(drop=True), hasil_massal], axis=1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baris diprediksi", angka(len(gabungan), 0), border=True)
    konfig = dict(KONFIG_KOLOM)
    if klasifikasi:
        m2.metric(
            f"Diprediksi {positif}", persen((gabungan["prediksi"] == positif).mean() * 100),
            border=True,
        )
        gabungan[f"peluang_{positif}"] = gabungan[f"peluang_{positif}"] * 100
        konfig[f"peluang_{positif}"] = st.column_config.ProgressColumn(
            f"Peluang {positif} (%)", min_value=0, max_value=100, format="%.1f"
        )
        konfig["prediksi"] = st.column_config.TextColumn("Prediksi")
        if kelas.name in gabungan.columns:
            aktual = gabungan[kelas.name].astype(str).str.strip().str.capitalize()
            akurasi = (aktual == gabungan["prediksi"]).mean() * 100
            mayoritas = aktual.value_counts()
            baseline = mayoritas.iloc[0] / len(aktual) * 100
            m3.metric(
                "Sesuai status aktual", persen(akurasi), border=True,
                delta=selisih(akurasi - baseline, 1, " poin vs baseline"),
                help="Persentase prediksi yang sama dengan status aktual.",
            )
            m4.metric(
                "Baseline", persen(baseline), border=True,
                help=f"Akurasi jika selalu menebak kelas terbanyak ('{mayoritas.index[0]}'). "
                "Model yang berguna harus jelas lebih tinggi dari angka ini.",
            )
            kode_latih = kode_data_latih(model)
            tumpang = 0.0
            if kode_latih and "kode_satker" in gabungan.columns:
                tumpang = gabungan["kode_satker"].astype(str).isin(kode_latih).mean() * 100
            if tumpang > 0:
                st.caption(
                    f"⚠️ {persen(tumpang, 0)} satker di sini juga dipakai untuk melatih model, "
                    "jadi angka kesesuaian ini cenderung terlalu optimistis. Untuk menilai "
                    "performa secara adil, pakai hasil Test and Score (validasi silang) di Orange."
                )
            else:
                st.caption(
                    "Jika data ini juga dipakai untuk melatih model, angka kesesuaian cenderung "
                    "terlalu optimistis. Untuk menilai performa secara adil, pakai hasil Test and "
                    "Score (validasi silang) di Orange."
                )
            with st.expander("Matriks kebingungan (aktual vs prediksi)"):
                st.dataframe(pd.crosstab(aktual.rename("Aktual"), gabungan["prediksi"].rename("Prediksi")))
        kolom_tampil = [k for k in ["kode_satker", "nama_kementerian", "tipe_satker"] if k in gabungan.columns]
        kolom_tampil += [k for k in [kelas.name, "prediksi", f"peluang_{positif}"] if k in gabungan.columns]
        urut = gabungan.sort_values(f"peluang_{positif}")
    else:
        gabungan = gabungan.rename(columns={"prediksi": f"prediksi_{kelas.name}"})
        kolom_tampil = [k for k in ["kode_satker", kelas.name, f"prediksi_{kelas.name}"] if k in gabungan.columns]
        urut = gabungan
    st.dataframe(urut[kolom_tampil], hide_index=True, column_config=konfig, height=320)
    st.download_button(
        "Unduh hasil prediksi (CSV)",
        data=gabungan.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# =============================================================================
# Halaman
# =============================================================================
if not DATA_PATH.exists():
    st.error(
        f"File data tidak ditemukan: {DATA_PATH.relative_to(BASE_DIR)}. Pastikan folder "
        "data berada di samping streamlit_app.py.", icon=":material/error:",
    )
    st.stop()
try:
    df = muat_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Data gagal dimuat: {exc}", icon=":material/error:")
    st.stop()

default = nilai_default_filter(df)
for kunci, nilai in default.items():
    st.session_state.setdefault(kunci, nilai)
for kunci in ("f_ikpa", "f_dev"):  # jaga agar nilai slider tetap di dalam rentang data
    bawah, atas = default[kunci]
    lo, hi = st.session_state[kunci]
    st.session_state[kunci] = (min(max(lo, bawah), atas), max(min(hi, atas), bawah))

with st.sidebar:
    st.header("Filter data")
    st.caption("Biarkan kosong untuk menampilkan semua.")
    f_prov = st.multiselect(
        "Provinsi", sorted(df["provinsi"].unique()), key="f_prov", placeholder="Semua provinsi"
    )
    f_kem = st.multiselect(
        "Kementerian", sorted(df["nama_kementerian"].unique()), key="f_kem",
        placeholder="Semua kementerian",
    )
    f_tipe = st.multiselect(
        "Tipe satker", sorted(df["tipe_satker"].unique()), key="f_tipe",
        placeholder="Semua tipe satker",
    )
    f_belanja = st.multiselect(
        "Jenis belanja utama", sorted(df["jenis_belanja_utama"].unique()), key="f_belanja",
        placeholder="Semua jenis belanja",
    )
    f_ikpa = st.slider(
        "Nilai IKPA", *default["f_ikpa"], key="f_ikpa", step=0.5,
        help="Kategori: Sangat Baik ≥95 · Baik 89–<95 · Cukup 70–<89 · Kurang <70",
    )
    f_dev = st.slider(
        "Deviasi RPD (%)", *default["f_dev"], key="f_dev", step=0.5,
        help="Ambang untuk nilai maksimal indikator deviasi Halaman III DIPA: 5%",
    )
    st.button(
        "Atur ulang filter", icon=":material/restart_alt:", on_click=atur_ulang_filter,
        args=(default,),
    )

dff = saring(df, f_prov, f_kem, f_tipe, f_belanja, f_ikpa, f_dev)
tersaring = len(dff) != len(df)

with st.sidebar:
    st.caption(f"Menampilkan **{angka(len(dff), 0)}** dari {angka(len(df), 0)} satker.")
    with st.expander("Cara memakai", icon=":material/help:"):
        st.markdown(
            "1. Atur **filter** di atas; semua angka dan grafik ikut berubah.\n"
            "2. Di tab **Eksplorasi**, seret di grafik untuk memilih sekelompok satker.\n"
            "3. Di tab **Peta sebaran**, klik lingkaran provinsi untuk melihat satkernya.\n"
            "4. Di tab **Prediksi**, geser nilai input dan lihat peluangnya berubah seketika."
        )

st.title("📊 Dashboard Analitik DJPb")
st.caption(
    "Pantau capaian realisasi anggaran satker, telusuri faktor yang berkaitan, dan coba "
    "prediksi dengan model Orange."
)

# --- Kartu ringkasan --------------------------------------------------------
kolom_kpi = st.columns(5)
kolom_kpi[0].metric(
    "Satker ditampilkan", angka(len(dff), 0), border=True,
    help=f"Dari total {angka(len(df), 0)} satker di data.",
)
if dff.empty:
    for kolom, judul in zip(
        kolom_kpi[1:],
        ["Realisasi ≥95%", "Rata-rata nilai IKPA", "Rata-rata deviasi RPD", "Deviasi ≤5%"],
    ):
        kolom.metric(judul, "–", border=True)
else:
    def delta(kini, semua, desimal=1, satuan=" poin"):
        return selisih(kini - semua, desimal, satuan) if tersaring else None

    dev_ok = (dff["deviasi_rpd_persen"] <= DEVIASI_MAKS).mean() * 100
    dev_ok_semua = (df["deviasi_rpd_persen"] <= DEVIASI_MAKS).mean() * 100
    kolom_kpi[1].metric(
        "Realisasi ≥95%", persen(rasio_ya(dff)), delta=delta(rasio_ya(dff), rasio_ya(df)),
        border=True,
        help="Persentase satker yang realisasi anggarannya mencapai minimal 95%. Delta "
        "dibandingkan dengan seluruh satker.",
    )
    kolom_kpi[2].metric(
        "Rata-rata nilai IKPA", angka(dff["skor_ikpa"].mean(), 2),
        delta=delta(dff["skor_ikpa"].mean(), df["skor_ikpa"].mean(), 2, ""), border=True,
        help="Kategori IKPA: Sangat Baik ≥95; Baik 89–<95; Cukup 70–<89; Kurang <70.",
    )
    kolom_kpi[3].metric(
        "Rata-rata deviasi RPD", persen(dff["deviasi_rpd_persen"].mean()),
        delta=delta(dff["deviasi_rpd_persen"].mean(), df["deviasi_rpd_persen"].mean()),
        delta_color="inverse", border=True,
        help="Selisih realisasi terhadap Rencana Penarikan Dana (Halaman III DIPA). "
        "Makin kecil makin baik.",
    )
    kolom_kpi[4].metric(
        "Deviasi ≤5%", persen(dev_ok), delta=delta(dev_ok, dev_ok_semua),
        border=True,
        help="Persentase satker dengan deviasi RPD paling tinggi 5%, yaitu ambang untuk "
        "nilai maksimal indikator deviasi Halaman III DIPA (PER-5/PB/2024).",
    )

tab_ringkasan, tab_eksplorasi, tab_peta, tab_prediksi, tab_data = st.tabs(
    [
        ":material/dashboard: Ringkasan",
        ":material/scatter_plot: Eksplorasi",
        ":material/map: Peta sebaran",
        ":material/psychology: Prediksi",
        ":material/table_view: Data",
    ]
)

# --- Tab Ringkasan ----------------------------------------------------------
with tab_ringkasan:
    if dff.empty:
        st.info(TEKS_KOSONG)
    else:
        with st.container(border=True):
            st.markdown("#### 💡 Temuan otomatis")
            st.markdown("\n".join(f"- {t}" for t in susun_temuan(dff)))
            st.caption(
                "Dihitung ulang setiap kali filter berubah. Temuan menunjukkan keterkaitan, "
                "bukan sebab-akibat."
            )
        kiri, kanan = st.columns([1, 1.6], gap="large")
        with kiri:
            st.subheader("Proporsi capaian")
            st.altair_chart(grafik_donut(dff), width="stretch")
        with kanan:
            st.subheader("Capaian per kelompok")
            dimensi = st.segmented_control(
                "Kelompokkan menurut", list(DIMENSI), default="Tipe satker",
                key="dimensi_ringkasan", required=True,
            )
            st.altair_chart(grafik_kelompok(dff, DIMENSI[dimensi]), width="stretch")
        st.subheader("Rata-rata realisasi per triwulan")
        st.altair_chart(grafik_triwulan(dff), width="stretch")
        tw3 = dff.groupby(TARGET)["realisasi_tw3_persen"].mean()
        if {YA, TIDAK} <= set(tw3.index):
            st.caption(
                f"Pada TW III, rata-rata realisasi satker yang akhirnya mencapai ≥95% adalah "
                f"{persen(tw3[YA])}, dibanding {persen(tw3[TIDAK])} pada satker yang tidak "
                "mencapai target."
            )

# --- Tab Eksplorasi ---------------------------------------------------------
with tab_eksplorasi:
    st.subheader("Hubungan antarvariabel")
    st.caption(
        "Setiap gelembung mewakili satu satker. Seret (klik dan tahan) di area grafik untuk "
        "memilih sekelompok satker; klik dua kali di area kosong untuk membatalkan."
    )
    p1, p2, p3, p4 = st.columns(4)
    sumbu_x = p1.selectbox(
        "Sumbu X", NUMERIK, index=NUMERIK.index("skor_ikpa"), format_func=label, key="eks_x"
    )
    sumbu_y = p2.selectbox(
        "Sumbu Y", NUMERIK, index=NUMERIK.index("deviasi_rpd_persen"), format_func=label,
        key="eks_y",
    )
    ukuran = p3.selectbox(
        "Ukuran gelembung", [None] + NUMERIK, index=1 + NUMERIK.index("pagu_miliar"),
        format_func=label, key="eks_ukuran",
    )
    warna = p4.selectbox("Warna", PILIHAN_WARNA, format_func=label, key="eks_warna")
    if dff.empty:
        st.info(TEKS_KOSONG)
    else:
        kejadian = st.altair_chart(
            grafik_sebaran(dff, sumbu_x, sumbu_y, ukuran, warna),
            width="stretch", on_select="rerun", selection_mode="pilih", key="grafik_sebaran",
        )
        terpilih = ambil_pilihan(dff, kejadian, sumbu_x, sumbu_y)
        if terpilih is None:
            st.caption("Belum ada area yang dipilih.")
        elif terpilih.empty:
            st.info("Tidak ada satker di area yang dipilih.")
        else:
            with st.container(border=True):
                st.markdown(
                    f"**{len(terpilih)} satker terpilih** · realisasi ≥95%: "
                    f"**{persen(rasio_ya(terpilih))}** · rata-rata nilai IKPA "
                    f"{angka(terpilih['skor_ikpa'].mean(), 2)} · rata-rata deviasi RPD "
                    f"{persen(terpilih['deviasi_rpd_persen'].mean())}"
                )
                kolom_pilih = list(dict.fromkeys(
                    ["kode_satker", "nama_kementerian", "provinsi", "tipe_satker",
                     sumbu_x, sumbu_y] + ([ukuran] if ukuran else []) + [TARGET]
                ))
                st.dataframe(
                    terpilih[kolom_pilih], hide_index=True, column_config=KONFIG_KOLOM,
                    height=260,
                )
        st.subheader("Perbandingan per status")
        b1, b2 = st.columns(2, gap="large")
        b1.altair_chart(grafik_kotak(dff, sumbu_x), width="stretch")
        b2.altair_chart(grafik_kotak(dff, sumbu_y), width="stretch")

# --- Tab Peta ---------------------------------------------------------------
with tab_peta:
    st.subheader("Sebaran satker per provinsi")
    metrik = st.segmented_control(
        "Warna lingkaran menunjukkan", list(METRIK_PETA), default="% realisasi ≥95%",
        key="metrik_peta", required=True,
    )
    if dff.empty:
        st.info(TEKS_KOSONG)
    else:
        kolom_metrik = METRIK_PETA[metrik]
        peta, tanpa_koordinat = agregat_provinsi(dff)
        if tanpa_koordinat:
            st.warning(
                "Provinsi berikut belum punya koordinat sehingga tidak tampil di peta: "
                + ", ".join(tanpa_koordinat)
                + ". Tambahkan ke KOORDINAT_PROVINSI."
            )
        if not peta.empty:
            kejadian_peta = st.pydeck_chart(
                peta_provinsi(peta, kolom_metrik), height=520, on_select="rerun",
                selection_mode="single-object", key="peta_provinsi",
            )
            fmt = (lambda v: angka(v, 0)) if kolom_metrik == "jumlah" else (
                (lambda v: angka(v, 2)) if kolom_metrik == "rata_ikpa" else persen
            )
            st.markdown(
                "<div style='display:flex;align-items:center;gap:.6rem;font-size:.85rem;"
                "margin:.25rem 0 .75rem'>"
                f"<span>{fmt(peta[kolom_metrik].min())}</span>"
                "<div style='flex:1;max-width:320px;height:10px;border-radius:5px;"
                "background:linear-gradient(90deg,#FBCFE8,#9D174D)'></div>"
                f"<span>{fmt(peta[kolom_metrik].max())}</span>"
                f"<span style='color:#6B7280'>· {metrik} (ukuran lingkaran = jumlah satker)</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            try:
                terklik = kejadian_peta.selection.get("objects", {}).get("provinsi", [])
            except AttributeError:
                terklik = []
            if terklik:
                prov = terklik[0].get("provinsi")
                satker_prov = dff[dff["provinsi"] == prov]
                st.markdown(
                    f"**{prov}**: {len(satker_prov)} satker · realisasi ≥95%: "
                    f"**{persen(rasio_ya(satker_prov))}**"
                )
                st.dataframe(
                    satker_prov[["kode_satker", "nama_kementerian", "tipe_satker", "skor_ikpa",
                                 "deviasi_rpd_persen", TARGET]],
                    hide_index=True, column_config=KONFIG_KOLOM, height=250,
                )
            else:
                st.caption("Klik lingkaran provinsi untuk melihat daftar satkernya.")
            st.dataframe(
                peta[["provinsi", "jumlah", "persen_ya", "rata_ikpa", "rata_deviasi"]]
                .sort_values(kolom_metrik, ascending=False),
                hide_index=True,
                column_config={
                    "provinsi": st.column_config.TextColumn("Provinsi"),
                    "jumlah": st.column_config.NumberColumn("Jumlah satker", format="%d"),
                    "persen_ya": st.column_config.ProgressColumn(
                        "Realisasi ≥95% (%)", min_value=0, max_value=100, format="%.1f"
                    ),
                    "rata_ikpa": st.column_config.NumberColumn("Rata-rata nilai IKPA", format="%.2f"),
                    "rata_deviasi": st.column_config.NumberColumn(
                        "Rata-rata deviasi RPD (%)", format="%.2f"
                    ),
                },
            )

# --- Tab Prediksi -----------------------------------------------------------
with tab_prediksi:
    st.subheader("Prediksi capaian realisasi")
    berkas_model = daftar_model()
    if ORANGE_ERROR is not None:
        st.error(
            "Paket **orange3** belum terpasang, jadi model Orange (.pkcls) belum bisa dibuka.",
            icon=":material/error:",
        )
        st.markdown("Jalankan perintah berikut di terminal, lalu muat ulang halaman ini:")
        st.code("uv add orange3\n# atau: pip install orange3", language="bash")
        with st.expander("Detail teknis"):
            st.code(repr(ORANGE_ERROR))
    elif not berkas_model:
        st.info(
            "Belum ada model di folder models/. Simpan model dari Orange (widget Save Model) "
            "dengan ekstensi .pkcls ke folder tersebut."
        )
    else:
        tampilkan_prediksi(df, dff, berkas_model)

# --- Tab Data ---------------------------------------------------------------
with tab_data:
    st.subheader("Data satker")
    cari = st.text_input(
        "Cari", placeholder="Ketik kode satker, kementerian, atau provinsi…", key="cari",
        label_visibility="collapsed",
    )
    tampil = dff
    if cari:
        cocok = dff[["kode_satker", "nama_kementerian", "provinsi"]].apply(
            lambda s: s.astype(str).str.contains(cari, case=False, regex=False)
        )
        tampil = dff[cocok.any(axis=1)]
    st.caption(f"{angka(len(tampil), 0)} baris sesuai filter dan pencarian.")
    st.dataframe(tampil, hide_index=True, column_config=KONFIG_KOLOM, height=420)
    st.download_button(
        "Unduh data (CSV)",
        data=tampil.to_csv(index=False).encode("utf-8"),
        file_name="data_satker_terfilter.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    with st.expander("Kamus data dan ambang batas", icon=":material/menu_book:"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Kolom": list(KETERANGAN),
                    "Label di dashboard": [label(k) for k in KETERANGAN],
                    "Keterangan": list(KETERANGAN.values()),
                }
            ),
            hide_index=True,
        )
        st.markdown(
            "- **Kategori nilai IKPA**: Sangat Baik ≥95; Baik 89 sampai <95; Cukup 70 sampai "
            f"<89; Kurang <70. [Sumber: artikel KPPN Metro, DJPb]({SUMBER_IKPA})\n"
            "- **Ambang deviasi Halaman III DIPA**: rata-rata deviasi bulanan paling tinggi 5% "
            "untuk memperoleh nilai indikator maksimal (PER-5/PB/2024). "
            f"[Sumber: artikel KPPN Medan I, DJPb]({SUMBER_DEVIASI})"
        )

st.divider()
st.caption(
    "Sumber data: data/02_realisasi_anggaran_klasifikasi.csv · Dibangun dengan Streamlit, "
    "Altair, pydeck, dan Orange Data Mining"
    + (f" · Dibuat oleh {PEMBUAT}" if PEMBUAT else "")
)
