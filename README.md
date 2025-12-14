# 💓 Heart Sound Classification using SVM & Spectrogram Features

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

Proyek ini adalah implementasi *Machine Learning* untuk mendeteksi kelainan jantung (*Abnormal Heartbeat*) berdasarkan suara detak jantung (Phonocardiogram/PCG). Sistem ini mengubah data suara berbasis waktu (*Time Series*) menjadi fitur statistik spektral dan diklasifikasikan menggunakan algoritma **Support Vector Machine (SVM)**.

Dilengkapi dengan Dashboard Interaktif menggunakan **Streamlit** untuk simulasi diagnosa.

---

## 📋 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Demo Aplikasi](#-demo-aplikasi)
- [Instalasi & Penggunaan](#-instalasi--penggunaan)

---

## 🔍 Latar Belakang
Penyakit kardiovaskular adalah penyebab kematian utama secara global. Auskultasi (mendengarkan suara jantung) adalah metode skrining awal yang murah namun sangat bergantung pada keahlian dokter. Proyek ini bertujuan membangun sistem *Computer-Aided Diagnosis* (CAD) untuk membantu tenaga medis membedakan suara jantung **Normal** dan **Abnormal** secara objektif.

---

## 📂 Dataset
Dataset diadopsi dari **PhysioNet/Computing in Cardiology (CinC) Challenge 2016**.
- **Format:** Multivariate Time Series (Spectrogram).
- **Dimensi:** 61 Pita Frekuensi x 405 Titik Waktu.
- **Kondisi:** Imbalanced (Jumlah data Abnormal jauh lebih banyak dari Normal).

> **Catatan:** Data telah dibagi ulang (Re-split) menjadi **80% Training** dan **20% Testing** untuk validasi yang lebih ketat, serta menyisihkan sebagian data (*Hold-out set*) khusus untuk simulasi demo.

---

## ⚙️ Metodologi

### 1. Data Preprocessing
Karena SVM tidak dapat memproses data runtun waktu (Time Series) secara langsung, dilakukan ekstraksi fitur statistik global:
- Mengubah dimensi `(Samples, 61, 405)` menjadi `(Samples, 244)`.
- **Fitur Statistik:** Mean, Standard Deviation, Max, dan Min untuk setiap pita frekuensi.
- **Scaling:** Standardisasi data menggunakan `StandardScaler`.

### 2. Penanganan Imbalance Data
Menggunakan teknik **Algorithmic Level Approach** (bukan SMOTE), yaitu dengan memberikan parameter `class_weight='balanced'` pada SVM. Hal ini memaksa model memberikan "perhatian lebih" pada kelas minoritas (Normal).

### 3. Modeling
- **Algoritma:** Support Vector Machine (SVM).
- **Kernel:** Radial Basis Function (RBF).
- **Optimasi:** Hyperparameter Tuning menggunakan `GridSearchCV`.

---

## 📊 Hasil Evaluasi
Evaluasi dilakukan pada data uji (20% Split) yang tidak pernah dilihat model saat pelatihan.

| Metrik | Skor |
| :--- | :--- |
| **Akurasi** | **74% - 77%** |
| **Recall (Abnormal)** | **78%** (Mendeteksi sakit dengan baik) |
| **Recall (Normal)** | **63%** (Meningkat signifikan setelah tuning) |

*(Anda dapat menyertakan screenshot Confusion Matrix di sini)*

---

## 💻 Demo Aplikasi
Aplikasi web dibangun menggunakan **Streamlit**. Fitur utama:
- **Visualisasi Spectrogram:** Melihat pola panas frekuensi suara jantung.
- **Simulasi Diagnosa:** Menguji model menggunakan data "Unseen" (Data sisa yang disisihkan).
- **Indikator Prediksi:** Menampilkan hasil prediksi AI vs Label Sebenarnya.

*(Anda dapat menyertakan screenshot Tampilan Web Streamlit di sini)*

---

## 🚀 Instalasi & Penggunaan

### 1. Clone Repository
```bash
git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
cd nama-repo-anda