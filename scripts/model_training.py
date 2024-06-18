import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from ncps.tf import LTC

# Suppress warnings
tf.get_logger().setLevel('ERROR')

# Load preprocessed data
def load_data(train_file: str, test_file: str):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

# Preprocess data for the model
def preprocess_data(train_df, test_df, features, targets):
    scaler = StandardScaler()

    X_train = train_df[features]
    y_train = train_df[targets]
    X_test = test_df[features]
    y_test = test_df[targets]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM/LTC which expects 3D input [samples, timesteps, features]
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Split targets into separate arrays
    y_train_recession_1m = y_train['recession_1m'].values.reshape(-1, 1, 1)
    y_train_recession_3m = y_train['recession_3m'].values.reshape(-1, 1, 1)
    y_train_phase = pd.get_dummies(y_train['phase']).values.reshape(-1, 1, len(pd.get_dummies(y_train['phase']).columns))

    y_test_recession_1m = y_test['recession_1m'].values.reshape(-1, 1, 1)
    y_test_recession_3m = y_test['recession_3m'].values.reshape(-1, 1, 1)
    y_test_phase = pd.get_dummies(y_test['phase']).values.reshape(-1, 1, len(pd.get_dummies(y_test['phase']).columns))

    return X_train_scaled, (y_train_recession_1m, y_train_recession_3m, y_train_phase), X_test_scaled, (y_test_recession_1m, y_test_recession_3m, y_test_phase)

# Build the LTC model
def build_ltc_model(input_dim, num_classes):
    input_layer = Input(shape=(None, input_dim))
    lstm_layer = LTC(units=128, return_sequences=True)(input_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)
    lstm_layer = LTC(units=64, return_sequences=True)(lstm_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)
    lstm_layer = LTC(units=32, return_sequences=True)(lstm_layer)
    lstm_layer = TimeDistributed(Dense(16, activation='relu'))(lstm_layer)
    
    # Output layers
    output_recession_1m = TimeDistributed(Dense(1, activation='sigmoid'), name='recession_1m')(lstm_layer)
    output_recession_3m = TimeDistributed(Dense(1, activation='sigmoid'), name='recession_3m')(lstm_layer)
    output_phase = TimeDistributed(Dense(num_classes, activation='softmax'), name='phase')(lstm_layer)

    model = Model(inputs=input_layer, outputs=[output_recession_1m, output_recession_3m, output_phase])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss={'recession_1m': 'binary_crossentropy',
                        'recession_3m': 'binary_crossentropy',
                        'phase': 'categorical_crossentropy'},
                  metrics={'recession_1m': 'mean_absolute_error',
                           'recession_3m': 'mean_absolute_error',
                           'phase': 'accuracy'})
    
    return model

# Callback to reduce learning rate on plateau
class CustomReduceLROnPlateau(Callback):
    def __init__(self, factor=0.1, patience=10, min_lr=1e-6):
        super(CustomReduceLROnPlateau, self).__init__()
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = float('inf')
        self.lr = float(self.model.optimizer.learning_rate.numpy())

    def on_epoch_end(self, epoch, logs=None):
        current = float(logs.get("val_loss"))
        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.lr > self.min_lr:
                    new_lr = max(self.lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    self.lr = new_lr
                    print(f"\nReducing learning rate to {new_lr}")
                    self.wait = 0
                    self.model.set_weights(self.best_weights)

# Train the model
def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    reduce_lr = CustomReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    
    history = model.fit(X_train, 
                        {'recession_1m': y_train[0], 'recession_3m': y_train[1], 'phase': y_train[2]},
                        epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                        callbacks=[reduce_lr, tf.keras.callbacks.EarlyStopping(patience=10)])
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_recession_1m = (y_pred[0].flatten() > 0.5).astype(int)
    y_pred_recession_3m = (y_pred[1].flatten() > 0.5).astype(int)
    y_pred_phase = np.argmax(y_pred[2].reshape(-1, y_pred[2].shape[-1]), axis=-1)

    y_test_recession_1m = y_test[0].flatten()
    y_test_recession_3m = y_test[1].flatten()
    y_test_phase = np.argmax(y_test[2].reshape(-1, y_test[2].shape[-1]), axis=-1)

    accuracy_recession_1m = accuracy_score(y_test_recession_1m, y_pred_recession_1m)
    accuracy_recession_3m = accuracy_score(y_test_recession_3m, y_pred_recession_3m)
    accuracy_phase = accuracy_score(y_test_phase, y_pred_phase)

    f1_recession_1m = f1_score(y_test_recession_1m, y_pred_recession_1m, average='weighted')
    f1_recession_3m = f1_score(y_test_recession_3m, y_pred_recession_3m, average='weighted')
    f1_phase = f1_score(y_test_phase, y_pred_phase, average='weighted')

    # Add labels parameter to specify all class labels
    report_phase = classification_report(y_test_phase, y_pred_phase, zero_division=1)

    return (accuracy_recession_1m, accuracy_recession_3m, accuracy_phase), (f1_recession_1m, f1_recession_3m, f1_phase), report_phase

# Save the model
def save_model(model, filepath):
    model.save(filepath)

if __name__ == "__main__":
    # Define file paths
    train_file = os.path.join('data', 'processed', 'train_data.csv')
    test_file = os.path.join('data', 'processed', 'test_data.csv')

    # Load data
    train_df, test_df = load_data(train_file, test_file)

    # Define features and targets
    features = ['unrate', 'unrate_1m_pct', 'unrate_3m_pct', 'cpi', 'cpi_1m_pct', 'interest_rate', 'yield_curve', 
                'adj_close', 'building_permits', 'consumer_confidence', 'industrial_production', 
                'industrial_production_1m_pct', 'corporate_profits', 'corporate_profits_q_pct', 
                'consumer_debt', 'consumer_debt_q_pct']
    targets = ['recession_1m', 'recession_3m', 'phase']

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, features, targets)

    # Build model
    input_dim = X_train.shape[2]
    num_classes = len(pd.get_dummies(train_df['phase']).columns)
    print(f"Number of classes in phase: {num_classes}")
    model = build_ltc_model(input_dim, num_classes)

    # Train model
    history = train_model(model, X_train, y_train, epochs=100, batch_size=3)

    # Evaluate model
    accuracies, f1_scores, report_phase = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy - Recession 1M: {accuracies[0]}, Recession 3M: {accuracies[1]}, Phase: {accuracies[2]}")
    print(f"F1 Score - Recession 1M: {f1_scores[0]}, Recession 3M: {f1_scores[1]}, Phase: {f1_scores[2]}")
    print(f"Classification Report - Phase:\n{report_phase}")

    # Save the model
    save_model(model, 'ltc_model.h5')
