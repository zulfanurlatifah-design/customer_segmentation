# Segmentasi Pelanggan UMKM (Toko Roti & Kopi, Bandung) · Tim HIHUUU

**Kasus:** 05 · Segmentasi Pelanggan UMKM (tantangan) · **Aplikasi:** [customersegmentationgit-vz52sekm4mhuw3segtclxq.streamlit.app](https://customersegmentationgit-vz52sekm4mhuw3segtclxq.streamlit.app/) · **Video:** (https://youtu.be/mCeGdorV_u0)

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Zulfa Nurlatifah | 2401346 |  EDA dan pembersihan data; preprocessing, pemilihan k, dan evaluasi (baseline, pembagian train/validation/test); laporan; pemeriksaan akhir notebook dan daftar periksa. |
| Luthfi Bunga Nurmalasari | 2400903 | profil segmen dan rekomendasi bisnis; pipeline produksi (src/features.py, train_model.py, artifacts/); API FastAPI; aplikasi Streamlit dan deployment; README; video. |

## Masalah dan pengguna

Pengguna aplikasi adalah pemilik toko roti dan kopi di Bandung yang punya sekitar 10.000 anggota, tetapi selama ini mengirim promo yang sama ke semua orang. Model K-Means mengelompokkan anggota ke empat segmen berdasarkan perilaku belanja enam bulan terakhir, sehingga pemilik toko bisa memutuskan penawaran apa untuk pelanggan setia, pelanggan yang mulai tidak aktif, pelanggan yang responsif terhadap promo, dan pelanggan yang sekali belanja nilainya besar. Jika segmen keliru, promo bisa salah sasaran dan biaya pemasaran terbuang.

## Data dan pembersihan

Data mentah: 10.150 baris dan 12 kolom (data anggota sampai Juni 2026). Setelah dibersihkan: **10.000 pelanggan unik**. `id_pelanggan` dan `tanggal_daftar` tidak dipakai sebagai fitur model.

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris identik | 90 | Dihapus dengan `drop_duplicates()` |
| ID pelanggan berulang dengan isi berbeda | 60 | Digabung per ID: median untuk kolom numerik, modus untuk kategori, tanggal daftar paling awal; hasilnya 10.000 ID unik |
| Penulisan `kecamatan` tidak seragam (huruf besar/kecil, spasi, `Bdg Wetan`, `Buah Batu`) | 254 | Spasi dibuang, huruf dijadikan kecil, lalu dipetakan menjadi `bandung wetan` dan `buahbatu`; hasilnya 10 kecamatan |
| `umur` di luar rentang 1–100 tahun | 55 (27 di bawah 1, 28 di atas 100) | Dijadikan nilai kosong |
| `umur` kosong | 208 | Diisi median saat pemodelan (255 kosong di data bersih, termasuk yang tadi dijadikan kosong) |
| `rating_rata` kosong | 812 | Diisi median saat pemodelan (800 kosong di data bersih) |
| Nilai tidak valid lain (transaksi, belanja, atau hari negatif; promo di luar 0–100; rating di luar 1–5) | 0 | Diperiksa; nilai seperti itu akan dijadikan kosong bila ada |

Data bersih disimpan di `data/data_pelanggan_clean.csv`. Kolom kategori (`kecamatan`, `channel_utama`, `metode_bayar`, `kategori_favorit`) yang kosong diisi modus saat pemodelan.

## Hasil

**Model:** K-Means dengan k = 4 (`random_state=42`, `n_init=10`). Fitur numerik (umur, jumlah transaksi, total belanja, hari sejak transaksi terakhir, persen promo, rating) diimputasi median lalu distandardisasi. Fitur kategorikal (kecamatan, channel, metode bayar, kategori favorit) diimputasi modus lalu di-*one-hot encoding*, sehingga model memakai 28 kolom. Model dilatih pada seluruh 10.000 pelanggan bersih.

| Model | silhouette *validation* | silhouette *test* |
|---|---|---|
| *Baseline*: K-Means, k = 4 | tidak dihitung | tidak dihitung |
| K-Means final (model yang sama dengan *baseline*) | tidak dihitung | tidak dihitung |

Pembagian *train/validation/test* belum diterapkan, sehingga silhouette hanya tersedia untuk seluruh data: **0,2499** (sampel 3.000 titik, `random_state=42`). Skor silhouette untuk k = 2 sampai 10:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| silhouette | 0,2082 | 0,2326 | **0,2499** | 0,1951 | 0,1414 | 0,1007 | 0,1048 | 0,0923 | 0,0885 |

k = 4 dipilih karena memberi silhouette tertinggi. Nilai 0,25 menunjukkan pemisahan antarsegmen yang masih terbatas, jadi hasilnya dipakai untuk menyusun hipotesis pemasaran, bukan bukti bahwa segmen benar-benar terpisah tajam.

*Threshold*: tidak berlaku, karena clustering tidak memakai ambang keputusan. Segmen ditentukan dari pusat klaster terdekat.

**Profil dan rekomendasi segmen** (angka = median):

| Segmen | Jumlah | Transaksi | Belanja 6 bln | Hari sejak transaksi | Promo | Rating | Rekomendasi |
|---|---|---|---|---|---|---|---|
| Pelanggan Loyal dan Aktif | 2.850 (28,5%) | 28 | Rp 1.347.650 | 5 | 20% | 4,4 | Akses awal menu baru atau bonus produk, tanpa mengandalkan diskon besar |
| Pelanggan Tidak Aktif | 2.917 (29,2%) | 3 | Rp 99.200 | 110 | 34% | 3,6 | Voucher kunjungan kembali dengan masa berlaku terbatas |
| Pelanggan Sensitif terhadap Promo | 2.533 (25,3%) | 11 | Rp 336.600 | 12 | 81% | 4,4 | Bundel kopi dan roti dengan promo terarah atau minimum belanja |
| Pelanggan Bernilai Belanja Tinggi | 1.700 (17,0%) | 4 | Rp 974.850 | 28 | 13% | 4,4 | Bonus produk pelengkap pada kunjungan berikutnya untuk mendorong pembelian lebih rutin |

Proyek ini tidak menghasilkan *feature importance*; interpretasi dilakukan dengan membandingkan profil median tiap segmen.

## Cara menjalankan

Jalankan dari folder utama proyek (yang berisi `requirements.txt`, `app/`, dan `src/`).

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload       # API: GET /health, POST /predict (dokumentasi di /docs)
streamlit run app/streamlit_app.py  # aplikasi
python train_model.py               # melatih ulang model, menulis artifacts/model.joblib dan metadata.json
```

Aplikasi Streamlit memuat model langsung dan tidak memanggil API, sehingga deployment tidak butuh FastAPI yang berjalan terpisah. Aplikasi yang sudah di-deploy ada di tautan di bagian atas.

Contoh permintaan ke API:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "umur": 27, "jumlah_transaksi_6bln": 29, "total_belanja_rp": 1362600,
  "hari_sejak_transaksi_terakhir": 5, "persen_pakai_promo": 20, "rating_rata": 4.4,
  "kecamatan": "coblong", "channel_utama": "toko", "metode_bayar": "qris", "kategori_favorit": "kopi"
}'
```

Notebook: `02_data_cleaning.ipynb` meminta unggahan file data lewat Google Colab; `modeling_KMeans_pelanggan.ipynb` membaca `data/data_pelanggan_clean.csv` dari folder proyek.

## Struktur repositori

```
notebooks/        01_eda.ipynb, 02_data_cleaning.ipynb, modeling_KMeans_pelanggan.ipynb
data/             data_pelanggan_clean.csv (data bersih)
src/features.py   daftar fitur, nama segmen, validasi dan penyusunan masukan; dipakai saat pelatihan dan saat prediksi
train_model.py    melatih model final dan menyimpan artifacts
artifacts/        model.joblib + metadata.json
app/              main.py (API FastAPI) dan streamlit_app.py (antarmuka Streamlit)
prediction.log    catatan prediksi dari API
```

## Keterbatasan

- Silhouette 0,2499 menunjukkan pemisahan antarsegmen yang belum kuat. Tidak ada label kebenaran, sehingga kualitas segmen tidak bisa dinilai dengan akurasi atau metrik klasifikasi.
- Model dilatih dan dinilai pada data yang sama (10.000 pelanggan), tanpa pembagian *train/validation/test* dan tanpa perbandingan dengan algoritma clustering lain.
- Data hanya satu potret enam bulan dan belum tentu mewakili UMKM lain, jadi perubahan perilaku pelanggan dari waktu ke waktu tidak terlihat.
- K-Means sensitif terhadap nilai ekstrem, dan *one-hot encoding* membuat hasil bergantung pada kategori yang ada di data. Jika model dilatih ulang, nomor klaster bisa berubah. `train_model.py` mengurutkan ulang nomor klaster berdasarkan profilnya, tetapi pemetaan nama segmen tetap perlu diperiksa.
- Aplikasi hanya membatasi rentang dasar (misalnya tidak boleh negatif, promo 0–100%, rating 1–5). Nilai yang sangat jauh di luar data latih tetap ditempatkan ke segmen terdekat tanpa peringatan.
- Sebelum dipakai untuk keputusan bisnis, hasil sebaiknya divalidasi bersama pemilik toko dan diuji lewat kampanye kecil. Yang perlu dipantau: proporsi dan profil tiap segmen, kelengkapan data, kategori baru, dan respons pelanggan terhadap promo.

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [ ] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [ ] *Baseline* wajib dilaporkan dan dibandingkan
- [ ] Data *test* hanya dibuka satu kali, di akhir
- [ ] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [ ] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
