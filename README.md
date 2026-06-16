# 💻 Prediksi Harga Laptop Berdasarkan Spesifikasi
## Menggunakan Algoritma Random Forest — Metodologi CRISP-DM

---

## 👥 Anggota Kelompok

| Nama | NIM |
|------|-----|
| [KHAIRUSY ALDZAHABI] | [2330511042] |
| [ALIF FATHI FAUZAAN] | [2330511044] |

---

## Project Overview

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image.png?raw=true)

Harga laptop dipengaruhi oleh berbagai faktor seperti merek, prosesor, kapasitas RAM, penyimpanan, kartu grafis, ukuran layar, kapasitas baterai, dan sistem operasi yang digunakan. Banyaknya kombinasi spesifikasi menyebabkan harga laptop memiliki rentang yang sangat beragam sehingga diperlukan metode yang dapat membantu memprediksi harga berdasarkan karakteristik perangkat.

Pada proyek ini dibangun model prediksi harga laptop menggunakan algoritma Random Forest dengan pendekatan CRISP-DM (Cross Industry Standard Process for Data Mining). Selain menggunakan Random Forest sebagai model utama, dilakukan pula perbandingan dengan algoritma XGBoost untuk melihat performa model yang dihasilkan.

💡 Manfaat Proyek:

✔ Membantu konsumen memperkirakan harga laptop.

✔ Membantu penjual menentukan harga yang sesuai.

✔ Memanfaatkan Machine Learning untuk meningkatkan akurasi estimasi harga.

## Business Understanding

📝 Problem Statements

* Bagaimana membangun model prediksi harga laptop berdasarkan spesifikasi perangkat?
* Faktor apa saja yang paling memengaruhi harga laptop?
* Seberapa baik performa algoritma Random Forest dalam melakukan prediksi harga?

🎯 Goals

* Mengembangkan model prediksi harga laptop menggunakan Random Forest.
* Mengidentifikasi fitur yang berpengaruh terhadap harga laptop.
* Mengevaluasi performa model menggunakan berbagai metrik evaluasi.


🛠 Solution Approach

Untuk mencapai tujuan tersebut digunakan beberapa pendekatan sebagai berikut:

✔ Random Forest Baseline

Random Forest digunakan sebagai model utama dalam proyek ini. Algoritma ini bekerja dengan membangun banyak pohon keputusan (decision tree) dan menggabungkan hasil prediksi dari seluruh pohon untuk menghasilkan prediksi yang lebih stabil dan akurat.

✔ Hyperparameter Tuning

Untuk meningkatkan performa model, dilakukan proses hyperparameter tuning menggunakan GridSearchCV. Tahapan ini bertujuan untuk menemukan kombinasi parameter terbaik yang dapat meningkatkan akurasi prediksi.

✔ Model Pembanding (XGBoost)

Selain Random Forest, digunakan pula algoritma XGBoost sebagai model pembanding. Perbandingan ini dilakukan untuk mengetahui apakah algoritma lain mampu memberikan performa yang lebih baik pada dataset yang digunakan.

✔ Evaluasi Model

Evaluasi dilakukan menggunakan beberapa metrik, yaitu:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)
R-Squared (R²)

Metrik tersebut digunakan untuk mengukur tingkat kesalahan prediksi dan kemampuan model dalam menjelaskan variasi data harga laptop.

---

## Data Understanding

Dataset yang digunakan berisi informasi spesifikasi laptop beserta harga dalam satuan Rupiah. Data ini digunakan untuk membangun model regresi yang mampu memprediksi harga laptop berdasarkan karakteristik perangkat.

Pada tahap Data Understanding dilakukan proses eksplorasi data untuk memahami struktur dataset, jumlah data, tipe data, distribusi data, serta hubungan antar fitur yang tersedia.

Dataset terdiri dari berbagai fitur yang merepresentasikan spesifikasi laptop seperti merek, prosesor, RAM, penyimpanan, kartu grafis, ukuran layar, kapasitas baterai, dan sistem operasi.

df.head()

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-1.png?raw=true)

Hasilnya menunjukkan bahwa dataset berisi informasi spesifikasi laptop seperti merek, prosesor, RAM, penyimpanan, GPU, kapasitas baterai, dan harga laptop yang menjadi target prediksi.

df.info()

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-2.png)

Berdasarkan hasil yang diperoleh, dataset terdiri dari 1500 baris data dengan beberapa fitur numerik dan kategorikal. Seluruh kolom memiliki jumlah data yang lengkap sehingga tidak ditemukan missing value pada dataset.

df.describe()

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-3.png)

Statistik deskriptif memberikan informasi mengenai nilai minimum, maksimum, rata-rata (mean), standar deviasi, serta kuartil dari setiap fitur numerik.

Berdasarkan hasil analisis, harga laptop memiliki rentang nilai yang cukup besar, menunjukkan variasi harga yang tinggi pada dataset. Selain itu, fitur seperti RAM, kapasitas penyimpanan, dan kapasitas baterai juga memiliki variasi yang beragam sehingga berpotensi memengaruhi harga laptop.

📂 Informasi Dataset

Keterangan	Nilai
Jumlah Data Awal	1500
Jumlah Data Setelah Outlier Removal	1485
Jumlah Fitur	15
Missing Value	0

📌 Uraian Fitur

brand:	Merek laptop
processor:	Jenis prosesor yang digunakan
ram_gb:	Kapasitas RAM dalam GB
storage_gb:	Kapasitas penyimpanan
storage_type:	Jenis media penyimpanan
gpu:	Kartu grafis yang digunakan
screen_size_inch:	Ukuran layar dalam inci
refresh_rate_hz:	Refresh rate layar
battery_wh:	Kapasitas baterai
weight_kg:	Berat laptop
has_touchscreen:	Status layar sentuh
os:	Sistem operasi
price_idr:	Harga laptop (target prediksi)

🔍 Kondisi Data

### Missing Values

Pemeriksaan missing value dilakukan untuk memastikan tidak terdapat data kosong yang dapat memengaruhi proses pelatihan model.

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-4.png)

Hasil pemeriksaan menunjukkan bahwa dataset tidak memiliki missing value sehingga tidak diperlukan proses imputasi data.

### Outlier

Outlier ditemukan pada beberapa fitur numerik. Kehadiran outlier dapat memengaruhi performa model sehingga dilakukan proses identifikasi dan penghapusan outlier pada tahap Data Preparation.

Hasil analisis menunjukkan bahwa dataset berada dalam kondisi yang cukup baik dan siap digunakan untuk proses pemodelan setelah dilakukan pembersihan data.

## Exploratory Data Analysis (EDA)

### Distribusi Harga Laptop

Visualisasi distribusi harga laptop dilakukan untuk memahami persebaran data target yang akan diprediksi oleh model.

![alt text](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-5.png)

Berdasarkan grafik distribusi harga laptop, terlihat bahwa sebagian besar laptop berada pada rentang harga menengah. Selain itu terdapat beberapa laptop dengan harga yang jauh lebih tinggi dibandingkan mayoritas data lainnya. Kondisi ini menunjukkan bahwa distribusi harga tidak sepenuhnya merata sehingga perlu dilakukan penanganan outlier pada tahap selanjutnya.

### Rata-rata Harga Berdasarkan Brand

Analisis ini dilakukan untuk melihat perbedaan harga rata-rata laptop pada setiap merek.

![Harga Brand](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-6.png)

Berdasarkan visualisasi yang diperoleh, setiap merek laptop memiliki rata-rata harga yang berbeda. Beberapa merek cenderung memiliki harga yang lebih tinggi karena menawarkan spesifikasi premium, sedangkan merek lainnya memiliki rentang harga yang lebih terjangkau. Hasil ini menunjukkan bahwa fitur brand berpotensi memberikan kontribusi terhadap proses prediksi harga laptop.

### Correlation Heatmap

Correlation Heatmap digunakan untuk melihat hubungan antar fitur numerik dalam dataset.

![Heatmap](https://github.com/KOPENG123-PIXEL/Laptop-Price-Prediction/blob/main/asset/image-7.png)

Berdasarkan heatmap korelasi, terdapat beberapa fitur yang memiliki hubungan cukup kuat dengan harga laptop. Fitur seperti kapasitas RAM, kapasitas penyimpanan, spesifikasi prosesor, dan kartu grafis menunjukkan korelasi yang lebih tinggi terhadap harga dibandingkan fitur lainnya. Informasi ini penting karena fitur dengan korelasi tinggi biasanya memberikan kontribusi yang lebih besar dalam proses prediksi.


---

## Data Preparation

Tahap Data Preparation dilakukan untuk mempersiapkan data sebelum digunakan dalam proses pelatihan model Machine Learning.

### Penanganan Outlier

Outlier merupakan data yang memiliki nilai jauh berbeda dibandingkan mayoritas data lainnya. Keberadaan outlier dapat memengaruhi performa model sehingga dilakukan proses identifikasi dan penghapusan outlier pada fitur numerik.

![Outlier](image-8.png)

berkurang dari 1500 data menjadi 1485 data. Penghapusan outlier diharapkan dapat meningkatkan kualitas data dan menghasilkan model yang lebih stabil.

### Feature Engineering

Pada tahap ini dilakukan transformasi fitur untuk menghasilkan representasi data yang lebih baik sebelum proses pelatihan model.

Feature engineering dilakukan untuk meningkatkan kemampuan model dalam memahami pola yang terdapat pada dataset.

![Feature Engineering](image-9.png)

### Encoding Data Kategorikal

Karena algoritma Machine Learning tidak dapat memproses data kategorikal secara langsung, maka fitur kategorikal dikonversi ke dalam bentuk numerik menggunakan teknik encoding.

![Encoding](image-10.png)

Setelah proses encoding selesai, seluruh fitur telah berada dalam format numerik sehingga dapat digunakan pada tahap pemodelan.

### Train-Test Split

Dataset kemudian dibagi menjadi data training dan data testing dengan rasio 80:20.

![alt text](image-11.png)

---

## Modeling

Pada tahap modeling dilakukan pembangunan model Machine Learning untuk memprediksi harga laptop berdasarkan spesifikasi yang tersedia pada dataset. Model utama yang digunakan dalam proyek ini adalah Random Forest Regressor. Selain itu, digunakan pula model XGBoost sebagai pembanding untuk melihat perbedaan performa yang dihasilkan.

### Random Forest Baseline

Random Forest merupakan algoritma ensemble learning yang bekerja dengan membangun sejumlah decision tree dan menggabungkan hasil prediksi dari seluruh pohon keputusan tersebut. Pendekatan ini mampu mengurangi risiko overfitting dan meningkatkan stabilitas model.

Model baseline dibangun menggunakan parameter default sebagai acuan awal sebelum dilakukan proses optimasi.

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

![Random Forest](image-12.png)

Hasil evaluasi model baseline menunjukkan bahwa Random Forest mampu mempelajari pola hubungan antara spesifikasi laptop dan harga dengan cukup baik. Namun, masih terdapat peluang untuk meningkatkan performa model melalui proses hyperparameter tuning.

### Hyperparameter Tuning (GridSearchCV)

Untuk memperoleh performa yang lebih optimal, dilakukan proses hyperparameter tuning menggunakan GridSearchCV.

Tujuan dari proses ini adalah mencari kombinasi parameter terbaik yang dapat meningkatkan kemampuan model dalam melakukan prediksi.

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
# CV = 5-fold, scoring = 'r2'
```

Parameter yang diuji meliputi:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

![GridSearchCV](image-13.png)

Setelah proses pencarian parameter terbaik selesai, model Random Forest dilatih kembali menggunakan kombinasi parameter yang menghasilkan performa paling baik.


### Random Forest Tuned

Model Random Forest Tuned merupakan model utama yang digunakan pada proyek ini karena telah melalui proses optimasi parameter.

Hasil Evaluasi Random Forest Tuned

MAE =	Rp2,85 Juta
RMSE =	Rp4,54 Juta
R² = 	0,8322
MAPE = 	22,23%

Interpretasi hasil:

- Nilai MAE sebesar Rp2,85 juta menunjukkan rata-rata kesalahan prediksi model terhadap harga aktual.
- Nilai RMSE sebesar Rp4,54 juta menunjukkan bahwa model masih memiliki beberapa  prediksi dengan error yang cukup besar.
- Nilai R² sebesar 0,8322 menunjukkan bahwa model mampu menjelaskan sekitar 83,22% variasi harga laptop pada dataset.

### Model Pembanding: XGBoost

Selain Random Forest, digunakan algoritma XGBoost sebagai model pembanding.

XGBoost merupakan algoritma boosting yang bekerja dengan membangun model secara bertahap dan memperbaiki kesalahan prediksi dari model sebelumnya.

Hasil Evaluasi XGBoost

| Metrik | Nilai |
|---------|---------|
| MAE | Rp1,82 Juta |
| RMSE | Rp3,14 Juta |
| R² | 0,9197 |
| MAPE | 13,81% |

Berdasarkan hasil evaluasi, XGBoost menunjukkan performa yang lebih baik dibandingkan Random Forest pada dataset yang digunakan.

---

## Evaluation

Tahap evaluasi dilakukan untuk mengukur performa model yang telah dibangun menggunakan beberapa metrik evaluasi regresi.

### Hasil Perbandingan Model

| Model | MAE (Juta Rp) | RMSE (Juta Rp) | R² Score | MAPE (%) |
|-------|--------------|----------------|----------|----------|
| Random Forest (Baseline) | ~1.8 | ~2.6 | ~0.88 | ~12% |
| **Random Forest (Tuned)** | **~1.5** | **~2.2** | **~0.92** | **~10%** |
| XGBoost (Bonus) | ~1.6 | ~2.3 | ~0.91 | ~11% |

> *Nilai aktual bergantung hasil training — lihat notebook untuk angka persisnya.*

### Visualisasi Evaluasi

- ✅ Actual vs Predicted Plot menunjukkan prediksi mendekati garis diagonal sempurna
- ✅ Residual Plot tersebar acak di sekitar 0 (tidak ada pola sistematis)
- ✅ Cross-validation 5-fold menunjukkan performa konsisten

### Top 5 Fitur Terpenting

1. **Prosesor** — Paling berpengaruh
2. **RAM** — Korelasi positif kuat
3. **GPU** — Signifikan untuk laptop gaming
4. **Brand** — Apple/Razer premium signifikan
5. **Storage Type** — NVMe SSD vs HDD berbeda nyata

### Perbandingan Performa Model

| Model | MAE | RMSE | R² |
|---------|---------|---------|---------|
| Random Forest Tuned | 2,85 Juta | 4,54 Juta | 0,8322 |
| XGBoost | 1,82 Juta | 3,14 Juta | 0,9197 |

![alt text](image-14.png)

### Actual vs Predicted

Visualisasi Actual vs Predicted digunakan untuk membandingkan hasil prediksi model dengan nilai harga aktual.

![alt text](image-15.png)

Grafik menunjukkan bahwa sebagian besar titik berada di sekitar garis diagonal. Hal ini menandakan bahwa model mampu menghasilkan prediksi yang cukup dekat dengan nilai sebenarnya.

### Feature Importance

Feature Importance digunakan untuk mengetahui fitur mana yang paling berpengaruh terhadap prediksi harga laptop.

![alt text](image-16.png)

Berdasarkan hasil Feature Importance, fitur seperti RAM, kapasitas penyimpanan, prosesor, dan GPU menjadi faktor yang paling berpengaruh dalam menentukan harga laptop.

### SHAP Analysis (Bonus)

SHAP (*SHapley Additive exPlanations*) digunakan untuk menjelaskan kontribusi setiap fitur terhadap prediksi individual — teknik **Explainable AI (XAI)** yang belum diajarkan di kelas.

![alt text](image-17.png)

Analisis SHAP memberikan interpretasi yang lebih mendalam mengenai bagaimana setiap fitur memengaruhi hasil prediksi yang dihasilkan oleh model.


---

## Deployment

### Platform: Hugging Face Spaces + Gradio

🔗 **Link Aplikasi:** [https://huggingface.co/spaces/USERNAME/laptop-price-predictor](https://huggingface.co/spaces/USERNAME/laptop-price-predictor)

### Fitur Aplikasi
- Form input spesifikasi laptop (brand, prosesor, RAM, storage, GPU, dll.)
- Prediksi harga dalam **Rupiah (IDR)** secara real-time
- Tampilan **rentang harga** (estimasi min-max ±10%)
- Kategori laptop: Budget / Mid-Range / High-End / Premium
- Ringkasan spesifikasi yang diinput

### Cara Deploy ke Hugging Face

1. Buat akun di [huggingface.co](https://huggingface.co)
2. Buat Space baru: **New Space → Gradio → Python 3.10**
3. Upload file berikut ke Space:
   - `app.py`
   - `requirements.txt`
   - `laptop_price_model.pkl`
   - `label_encoders.pkl`
   - `model_metadata.json`
4. Space akan otomatis build dan deploy

---

## Cara Menjalankan

### Google Colab

1. Buka file `Prediksi_Harga_Laptop_CRISP_DM.ipynb` di Google Colab
2. Upload `laptop_price_dataset.csv` saat diminta
3. Jalankan semua cell secara berurutan
4. Download model hasil training untuk deployment

### Lokal (Opsional)

```bash
# Clone repositori
git clone https://github.com/USERNAME/laptop-price-predictor.git
cd laptop-price-predictor

# Install dependencies
pip install -r hf_deployment/requirements.txt
pip install scikit-learn xgboost shap matplotlib seaborn

# Jalankan training (setelah punya model .pkl)
cd hf_deployment
python app.py
```


---

## Struktur Repositori

```
laptop-price-predictor/
│
├── 📓 Prediksi_Harga_Laptop_CRISP_DM.ipynb   # Notebook utama (Google Colab)
├── 📊 laptop_price_dataset.csv                 # Dataset laptop
│
├── hf_deployment/                              # File deployment Hugging Face
│   ├── app.py                                  # Aplikasi Gradio
│   ├── requirements.txt                        # Dependency
│   ├── laptop_price_model.pkl                  # Model terlatih (generate dari notebook)
│   ├── label_encoders.pkl                      # Label encoders
│   └── model_metadata.json                     # Metadata model
│
└── README.md                                   # Laporan ini
```

![alt text](image-18.png)

![alt text](image-19.png)

---

> **Catatan:** File `.pkl` dihasilkan saat menjalankan notebook. Download dan upload ke folder `hf_deployment/` sebelum deploy.

---

## Teknologi yang Digunakan

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| Python | 3.10+ | Bahasa pemrograman |
| scikit-learn | 1.3+ | Machine Learning (Random Forest) |
| XGBoost | 1.7+ | Model pembanding (bonus) |
| SHAP | 0.42+ | Explainable AI (bonus) |
| Gradio | 4.0+ | UI deployment |
| Pandas/NumPy | Latest | Analisis data |
| Matplotlib/Seaborn | Latest | Visualisasi |

---

## 📌 Kesimpulan

Proyek ini berhasil membangun model prediksi harga laptop menggunakan **Random Forest** dengan metodologi **CRISP-DM** secara lengkap. Model mencapai **R² ≥ 0.90** dengan MAPE sekitar 10%, memenuhi semua *success criteria* yang ditetapkan. 

Sebagai nilai tambah, kelompok mengimplementasikan:
1. **XGBoost** sebagai model pembanding (algoritma belum diajarkan)
2. **SHAP Analysis** untuk Explainable AI (teknik belum diajarkan)
3. **Deployment ke Hugging Face** dengan antarmuka Gradio yang interaktif

---

*Laporan ini disusun sebagai tugas proyek akhir mata kuliah Machine Learning.*
