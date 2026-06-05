import sys
import os

# Memaksa Streamlit untuk membaca file di direktori saat ini
sys.path.append(os.path.abspath('.')) 

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocessing import preprocess_texts, load_tokenizer

st.set_page_config(page_title="LSTM Sentiment Analysis", page_icon="🧠", layout="centered")

@st.cache_resource
def load_ml_components():
    model = load_model('models/sentiment_lstm.h5')
    tokenizer = load_tokenizer('models/tokenizer.pickle')
    with open('models/label_encoder.pickle', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le

try:
    model, tokenizer, le = load_ml_components()
    st.sidebar.success("Model LSTM berhasil dimuat!")
except Exception as e:
    st.error(f"Model tidak ditemukan atau error: {e}")
    st.stop()

st.title("🧠 LSTM Text Sentiment Analysis")
st.markdown("Aplikasi ini menggunakan Deep Learning (Long Short-Term Memory) untuk mendeteksi sentimen dari sebuah teks.")

user_input = st.text_area("📝 Masukkan teks yang ingin dianalisis:", height=150, placeholder="Contoh: Saya sangat puas dengan pelayanan yang diberikan!")

if st.button("Analisis Sentimen", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner('Menganalisis menggunakan LSTM...'):
            padded_seq = preprocess_texts([user_input], tokenizer)
            prediction = model.predict(padded_seq)
            
            if len(le.classes_) > 2:
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
            else:
                predicted_class = 1 if prediction[0][0] > 0.5 else 0
                confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                
            sentiment = le.inverse_transform([predicted_class])[0]
            
            st.success("Analisis Selesai!")
            
            # FIX: Mengubah sentiment menjadi string (str) sebelum di-uppercase
            st.metric(label="Prediksi Sentimen", value=str(sentiment).upper())
            st.progress(float(confidence), text=f"Confidence Score: {confidence:.2%}")
