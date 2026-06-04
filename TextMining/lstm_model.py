from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from config import MAX_WORDS, MAX_LEN

def build_and_train_model(X_train, y_train, num_classes, epochs, batch_size):
    """Membangun arsitektur LSTM dan melatih model."""
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        # Pilih fungsi aktivasi output berdasarkan jumlah class
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Pilih loss function berdasarkan jumlah class
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    
    # Proses Training
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model