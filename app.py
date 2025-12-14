import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Suara Jantung",
    page_icon="💓",
    layout="wide"
)

st.title("💓 Klasifikasi Suara Jantung (PCG)")
st.markdown("Sistem deteksi dini kelainan jantung menggunakan **SVM (Support Vector Machine)** berbasis Spectrogram.")
st.markdown("---")

# --- FUNGSI BANTUAN ---

# 1. Fungsi Ekstraksi Fitur (Sama persis dengan di Notebook)
def extract_features_single(spectrogram_data):
    # Input: Array 2D (61, 405)
    # Kita ubah statistik sepanjang waktu (axis 1)
    mean_val = np.mean(spectrogram_data, axis=1)
    std_val = np.std(spectrogram_data, axis=1)
    max_val = np.max(spectrogram_data, axis=1)
    min_val = np.min(spectrogram_data, axis=1)
    
    # Gabung jadi satu baris vektor
    features = np.concatenate([mean_val, std_val, max_val, min_val])
    return features.reshape(1, -1) # Bentuk jadi (1 baris, n fitur)

# 2. Fungsi Visualisasi Spectrogram
def plot_spectrogram(data, title):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(data, aspect='auto', origin='lower', cmap='inferno')
    ax.set_title(title)
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Frekuensi")
    plt.colorbar(img, ax=ax, label="Intensitas")
    return fig

# --- LOAD MODEL & DATA ---

@st.cache_resource # Agar tidak loading ulang tiap klik
def load_all_files():
    try:
        # 1. Load Model SVM
        with open('model_jantung_svm.pkl', 'rb') as f:
            model_packet = pickle.load(f)
            
        # 2. Load Data Demo (Rahasia)
        with open('data_demo_streamlit.pkl', 'rb') as f:
            demo_data = pickle.load(f)
            
        return model_packet, demo_data
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None, None

# Eksekusi Load
packet, data_demo = load_all_files()

if packet is not None:
    model = packet['model']
    scaler = packet['scaler']
    encoder = packet['encoder']
    
    X_demo_raw = data_demo['X_raw']
    y_demo_label = data_demo['y_label']
    
    st.success("✅ Model & Data berhasil dimuat! Siap digunakan.")
else:
    st.stop() # Berhenti jika file tidak ketemu
    
# --- SIDEBAR NAVIGASI ---
st.sidebar.header("Panel Kontrol")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Beranda", "🩺 Simulasi Diagnosa"])

# Tampilkan Info Model di Sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Model Info:")
st.sidebar.text(f"Algoritma: SVM (RBF)")
st.sidebar.text(f"Imbalance Handler: Class Weight")

# --- HALAMAN UTAMA ---

if menu == "🏠 Beranda":
    st.subheader("Tentang Proyek")
    st.info("""
    Aplikasi ini dirancang untuk membantu tenaga medis melakukan skrining awal 
    penyakit jantung melalui suara detak jantung (Phonocardiogram).
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Statistik Data")
        st.write(f"Total Data Latih: **204 Pasien**")
        st.write(f"Total Data Uji: **51 Pasien**")
    with col2:
        st.markdown("### 🎯 Performa Model")
        st.write("Akurasi Test: **74% - 77%**") 
        st.write("Fokus Utama: **Recall (Sensitivitas)**")

elif menu == "🩺 Simulasi Diagnosa":
    st.subheader("Simulasi Diagnosa Pasien")
    st.write("Pilih salah satu Sampel Data Uji (Data ini belum pernah dilihat model sebelumnya).")
    
    # 1. Pilih ID Pasien
    total_demo = len(X_demo_raw)
    pilihan_index = st.slider("Pilih Nomor Sampel Pasien:", 0, total_demo-1, 0)
    
    if st.button("🔍 Analisis Suara Jantung"):
        # Ambil data sesuai pilihan
        sample_data = X_demo_raw[pilihan_index]
        real_label = y_demo_label[pilihan_index]
        
        # Tampilkan Visualisasi
        st.write("### 1. Visualisasi Sinyal (Spectrogram)")
        fig = plot_spectrogram(sample_data, f"Spectrogram Pasien #{pilihan_index}")
        st.pyplot(fig)
        
        # PROSES PREDIKSI (Backend Logic)
        # a. Ekstraksi Fitur
        features = extract_features_single(sample_data)
        # b. Scaling (Pakai Scaler yg sudah disimpan)
        features_scaled = scaler.transform(features)
        # c. Prediksi (Pakai Model SVM)
        pred_code = model.predict(features_scaled)[0]
        pred_label = encoder.inverse_transform([pred_code])[0] # Ubah 0/1 jadi Teks
        
        # TAMPILKAN HASIL
        st.write("### 2. Hasil Diagnosa AI")
        
        col_hasil, col_asli = st.columns(2)
        
        with col_hasil:
            if pred_label == 'Normal':
                st.success(f"**PREDIKSI: {pred_label.upper()}** ✅")
            else:
                st.error(f"**PREDIKSI: {pred_label.upper()}** ⚠️")
                
        with col_asli:
            st.info(f"**Kenyataan (Dokter): {real_label}**")
            
        # Cek Benar/Salah
        if pred_label.lower() == str(real_label).lower():
            st.toast("Prediksi Benar! 🎉")
        else:
            st.warning("Prediksi Meleset (Missclassification).")
            
