import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import kerastuner
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L2, L1L2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Suppress warnings
tf.get_logger().setLevel('ERROR')
print("TensorFlow version:", tf.__version__)

# F1 Score Callback
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        val_pred = (self.model.predict(val_data) > 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_pred, average='macro')
        logs['val_f1_score'] = val_f1
        print(f' — val_f1_score: {val_f1:.4f}')

# Define model
def build_lstm_model(input_shape, output_dim):
    model = Sequential()
    # LSTM 1
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), input_shape=input_shape))
    model.add(Dropout(0.3))
    # LSTM 2
    model.add(LSTM(128, return_sequences=True, recurrent_regularizer=L2(l2=1e-5)))
    model.add(Dropout(0.3))
    # LSTM 3
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define training and evaluation function
def train_and_evaluate(df):
    targets = ['is_recession', 'recession_in_1q', 'recession_in_2q', 'recession_in_4q']
    X = df.drop(targets, axis=1).values
    y = df[targets].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=targets))
    
    def create_sequences(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(targets[i+seq_length])
        return np.array(X), np.array(y)

    # Sequence length = number of months of history
    seq_len = 12
    X, y = create_sequences(scaled_features, y, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((seq_len, X_train.shape[2]), len(targets))

    checkpoint_callback = ModelCheckpoint(
        'model_checkpoint.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=1
    )

    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.2, 
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1 Score: {f1}')
    
    return model, history

# Load data
def load_data(file_path: str):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    df_final = load_data('data/processed/final_data.csv')

    model, history = train_and_evaluate(df_final)

    # Save model
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model.save(f'models/lstm_model_{date}.keras')

    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'models/lstm_model_history_{date}.csv', index=False)

    print("Model and history saved successfully!")


