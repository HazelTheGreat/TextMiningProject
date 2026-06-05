import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 5000
MAX_LEN = 100

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    return text

def fit_and_save_tokenizer(texts, save_path='models/tokenizer.pickle'):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    with open(save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

def load_tokenizer(load_path='models/tokenizer.pickle'):
    with open(load_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def preprocess_texts(texts, tokenizer=None):
    cleaned_texts = [clean_text(t) for t in texts]
    if tokenizer is None:
        tokenizer = load_tokenizer()
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded
