# 📊 Dashboard Analitik DJPb

Dashboard interaktif berbasis Streamlit untuk memantau capaian realisasi anggaran satuan kerja (satker), menelusuri faktor yang berkaitan, dan menyimulasikan prediksi dengan model Logistic Regression yang dilatih di Orange Data Mining.

## Fitur

| Tab | Isi |
|---|---|
| **Ringkasan** | Kartu KPI, temuan otomatis, proporsi capaian, capaian per kelompok, dan rata-rata realisasi TW I–III |
| **Eksplorasi** | Grafik gelembung interaktif (sumbu, ukuran, dan warna bisa dipilih) dengan garis acuan IKPA 95 dan deviasi 5%. Seret di grafik untuk memilih satker. |
| **Peta sebaran** | Peta per provinsi. Warna menunjukkan metrik pilihan; klik lingkaran untuk melihat daftar satker. |
| **Prediksi** | Peluang realisasi ≥95% dari model Orange, simulasi "bagaimana jika", faktor yang paling berpengaruh, dan prediksi massal (unduh CSV) |
| **Data** | Tabel data dengan pencarian, unduhan CSV, dan kamus data |

Semua tab mengikuti filter di sidebar: provinsi, kementerian, tipe satker, jenis belanja, rentang nilai IKPA, dan deviasi RPD.

## Cara menjalankan

**GitHub Codespaces:** buka repositori ini di Codespaces. Dependensi terpasang otomatis dan aplikasi berjalan di port 8501.

**Komputer sendiri** (dengan [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Atau dengan pip:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Struktur folder

```
├── streamlit_app.py           aplikasi dashboard
├── data/
│   ├── 02_realisasi_anggaran_klasifikasi.csv   data utama (500 satker)
│   └── 06_spm_processing_klasifikasi.csv
├── models/
│   └── Best_model.pkcls       model Orange (Logistic Regression)
├── .streamlit/config.toml     tema warna
├── pyproject.toml, uv.lock, requirements.txt
└── .devcontainer/             konfigurasi Codespaces
```

## Data dan model

- **Target:** `realisasi_tercapai_95persen` (Ya/Tidak), yaitu apakah realisasi anggaran satker mencapai minimal 95%.
- **Fitur model:** jumlah SPM, jumlah revisi DIPA, deviasi RPD, nilai IKPA, dan tipe satker.
- **Model:** Logistic Regression dari Orange Data Mining dengan pra-proses imputasi, normalisasi, dan one-hot. Aplikasi menjalankan ulang pra-proses Orange secara otomatis, jadi input cukup berupa kolom mentah.

**Keterbatasan:** pada data latih, prediksi model sesuai dengan status aktual pada 62,2% satker, sedikit di atas baseline 58,6% (selalu menebak "Tidak"). Gunakan prediksi sebagai bahan simulasi dan diskusi, bukan keputusan final. Untuk evaluasi yang adil, gunakan hasil Test and Score (validasi silang) di Orange.

## Ambang acuan

- Kategori nilai IKPA: Sangat Baik ≥95; Baik 89 sampai <95; Cukup 70 sampai <89; Kurang <70 ([artikel KPPN Metro, DJPb](https://djpb.kemenkeu.go.id/kppn/metro/id/data-publikasi/artikel/3521-ikpa-rapor-kinerja-pelaksanaan-anggaran-satuan-kerja-2.html)).
- Ambang rata-rata deviasi bulanan Halaman III DIPA untuk nilai maksimal: 5% (PER-5/PB/2024; [artikel KPPN Medan I, DJPb](https://djpb.kemenkeu.go.id/kppn/medan1/id/data-publikasi/artikel/2934-optimalisasi-peran-kppn-sebagai-financial-advisor-fa-dalam-meningkatkan-kualitas-perencanaan-pelaksanaan-anggaran-satuan-kerja.html)).

## Teknologi

Streamlit · pandas · Altair · pydeck · Orange Data Mining
