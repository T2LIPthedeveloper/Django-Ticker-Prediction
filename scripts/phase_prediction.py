import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
import glob
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
tf.get_logger().setLevel('ERROR')

def plot_metrics(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.show()

def generate_X_y(df):
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
    return X, y

def evaluate_model_performance(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1 Score: {f1}')
    
    return accuracy, recall, precision, f1, y_pred

def plot_recession_predictions(y_true, y_pred, title='Recession Predictions vs Actual Values'):
    plt.figure(figsize=(12, 8))

    quarters = ['Now', 'In 1 Quarter', 'In 2 Quarters', 'In 4 Quarters']
    for i, quarter in enumerate(quarters):
        plt.subplot(4, 1, i+1)
        plt.plot(y_true[:, i], label=f'Actual Recession {quarter}', color='blue')
        plt.plot(y_pred[:, i], label=f'Predicted Recession {quarter}', color='red')
        plt.legend()
        plt.title(f'{title} - {quarter}')
        plt.xlabel('Time')
        plt.ylabel('Recession Indicator')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recession prediction using LSTM or LTC model.')
    parser.add_argument('--model', type=str, required=True, choices=['LSTM', 'LTC'],
                        help='Specify the model type: LSTM or LTC')
    args = parser.parse_args()

    # Load data
    df_final = pd.read_csv('data/processed/final_data.csv')

    # Search for most recent model in the 'models' directory with format '{model_type}_model_{date}.keras'
    model_files = glob.glob(f'models/{args.model.lower()}_model_*.keras')
    if not model_files:
        raise FileNotFoundError(f"No model files found for model type {args.model}")
    
    # Sort by date and get most recent
    model_file = sorted(model_files, key=os.path.getmtime)[-1]
    
    # Load most recent model history with format '{model_type}_model_history_{date}.csv'
    history_files = glob.glob(f'models/{args.model.lower()}_model_history_*.csv')

    if not history_files:
        raise FileNotFoundError(f"No history files found for model type {args.model}")
    
    # Sort by date and get most recent
    history_file = sorted(history_files, key=os.path.getmtime)[-1]

    # Evaluate model performance
    model = load_model(model_file)
    history = pd.read_csv(history_file)
    plot_metrics(history)

    # Generate X and y
    X, y = generate_X_y(df_final)

    accuracy, recall, precision, f1, y_pred = evaluate_model_performance(model, X, y)
    print(f"Metrics for full dataset:\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nF1 Score: {f1}")

    # Plot predictions vs actual values
    plot_recession_predictions(y, y_pred)
