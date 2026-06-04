import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="LSTM Sentiment Analysis", page_icon="🧠", layout="centered")

st.title("🧠 LSTM Text Sentiment Analysis")
st.write("Aplikasi ini dilatih menggunakan metode vibe coding dengan model Long Short-Term Memory (LSTM).")

# Parameter Global
MAX_WORDS = 5000
MAX_LEN = 100

@st.cache_resource(show_spinner="Mengunduh dataset dan melatih model LSTM... (Ini mungkin memakan waktu beberapa saat)")
def train_model():
    # 1. Load Dataset (Menggunakan URL raw dari GitHub)
    url = "https://raw.githubusercontent.com/rzyunanda/Text-Mining-Session-12/main/data.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None, None, None
    
    # Asumsi: Kolom pertama adalah teks, kolom kedua adalah sentimen/label
    text_col = df.columns[0]
    label_col = df.columns[1]
    
    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).values
    labels = df[label_col].values
    
    # 2. Preprocessing Labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    num_classes = len(np.unique(encoded_labels))
    
    # 3. Preprocessing Texts (Tokenisasi & Padding)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Build LSTM Model
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    
    # 5. Train Model (Sangat basic untuk keperluan demo web)
    model.fit(padded_sequences, encoded_labels, epochs=3, batch_size=32, verbose=0)
    
    return model, tokenizer, encoder

# Eksekusi fungsi training
model, tokenizer, encoder = train_model()

if model:
    st.success("✅ Model berhasil dilatih dan siap digunakan!")
    
    # UI untuk Prediksi
    st.markdown("### Uji Sentimen Teks Anda")
    user_input = st.text_area("Masukkan teks di sini:", placeholder="Ketik sesuatu...")
    
    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            # Preprocess input
            seq = tokenizer.texts_to_sequences([user_input])
            pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            
            # Predict
            prediction = model.predict(pad_seq)
            
            # Decode Prediksi
            if len(encoder.classes_) > 2:
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
            else:
                predicted_class = 1 if prediction[0][0] > 0.5 else 0
                confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                
            sentiment_label = encoder.inverse_transform([predicted_class])[0]
            
            st.markdown("#### Hasil Analisis:")
            st.info(f"**Sentimen:** {sentiment_label}")
            st.progress(float(confidence))
            st.write(f"Confidence: {confidence:.2%}")