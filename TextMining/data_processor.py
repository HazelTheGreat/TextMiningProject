import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_LEN, DATA_URL

def load_and_preprocess_data():
    """Mengunduh dataset dan melakukan preprocessing teks."""
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        raise Exception(f"Gagal memuat dataset: {e}")
    
    # Asumsi kolom pertama: teks, kolom kedua: label
    text_col = df.columns[0]
    label_col = df.columns[1]
    
    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).values
    labels = df[label_col].values
    
    # Encode label teks menjadi angka
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Tokenisasi dan Padding teks
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return padded_sequences, encoded_labels, tokenizer, encoder