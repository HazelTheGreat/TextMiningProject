import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_processor import load_and_preprocess_data
from lstm_model import build_and_train_model
from config import MAX_LEN, EPOCHS, BATCH_SIZE

# Konfigurasi UI Streamlit
st.set_page_config(page_title="LSTM Sentiment Analysis", page_icon="🧠", layout="centered")
st.title("🧠 LSTM Text Sentiment Analysis")
st.write("Aplikasi ini menggunakan arsitektur **modular (multi-file)** agar rapi dan profesional.")

@st.cache_resource(show_spinner="Mengunduh data, memproses teks, dan melatih model LSTM... (Tunggu sebentar!)")
def setup_pipeline():
    # 1. Ambil dan proses data dari data_processor.py
    X, y, tokenizer, encoder = load_and_preprocess_data()
    
    # 2. Latih model menggunakan lstm_model.py
    num_classes = len(np.unique(y))
    model = build_and_train_model(X, y, num_classes, EPOCHS, BATCH_SIZE)
    
    return model, tokenizer, encoder

try:
    # Eksekusi pipeline (hanya berjalan sekali berkat cache)
    model, tokenizer, encoder = setup_pipeline()
    st.success("✅ Model LSTM berhasil dilatih dan siap digunakan!")
    
    # --- UI Analisis Sentimen ---
    st.markdown("### Uji Sentimen Teks Anda")
    user_input = st.text_area("Masukkan teks ulasan atau opini di sini:", placeholder="Contoh: Produk ini sangat membantu dan UI-nya keren!")
    
    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            # Preprocess input dari user
            seq = tokenizer.texts_to_sequences([user_input])
            pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            
            # Prediksi menggunakan model LSTM
            prediction = model.predict(pad_seq)
            
            # Menerjemahkan hasil prediksi kembali ke label teks
            if len(encoder.classes_) > 2:
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
            else:
                predicted_class = 1 if prediction[0][0] > 0.5 else 0
                confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                
            sentiment_label = encoder.inverse_transform([predicted_class])[0]
            
            # Menampilkan Hasil
            st.markdown("#### Hasil Analisis:")
            st.info(f"**Sentimen:** {sentiment_label}")
            st.progress(float(confidence))
            st.write(f"Tingkat Kepercayaan (Confidence): {confidence:.2%}")

except Exception as e:
    st.error(f"Terjadi kesalahan pada pipeline aplikasi: {e}")