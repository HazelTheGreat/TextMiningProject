import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from preprocessing import clean_text, fit_and_save_tokenizer, preprocess_texts, MAX_WORDS, MAX_LEN

# Buat folder models
os.makedirs('models', exist_ok=True)

print("Loading data...")
url = "https://raw.githubusercontent.com/rzyunanda/Text-Mining-Session-12/main/data.csv"
df = pd.read_csv(url)

texts = df.iloc[:, 0].astype(str).tolist()
labels = df.iloc[:, 1].tolist()

print("Preprocessing data...")
cleaned_texts = [clean_text(t) for t in texts]
tokenizer = fit_and_save_tokenizer(cleaned_texts)
X = preprocess_texts(cleaned_texts, tokenizer)

le = LabelEncoder()
y = le.fit_transform(labels)
with open('models/label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building LSTM model...")
num_classes = len(set(y))

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.5)
])

if num_classes > 2:
    model.add(Dense(num_classes, activation='softmax'))
    loss_fn = 'sparse_categorical_crossentropy'
else:
    model.add(Dense(1, activation='sigmoid'))
    loss_fn = 'binary_crossentropy'

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

print("Saving model...")
model.save('models/sentiment_lstm.h5')
print("Selesai! Model berhasil disimpan.")
