import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from ncps.tf import LTC, LTCCell

# Suppress warnings
tf.get_logger().setLevel('ERROR')

# Load preprocessed data
def load_data(file_path: str):
    return pd.read_csv(file_path)

# Preprocess data for the model
def preprocess_data(df, features):
    scaler = StandardScaler()

    X = df[features]
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM/LTC which expects 3D input [samples, timesteps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_scaled

# Load the trained model
def load_trained_model(model_path):
    custom_objects = {'LTC': LTC, 'LTCCell': LTCCell}
    return keras.models.load_model(model_path, custom_objects=custom_objects)

# Predict macroeconomic outputs
def predict_outputs(model, X):
    y_pred = model.predict(X)
    
    y_pred_recession_1m = (y_pred[0].flatten() > 0.5).astype(int)
    y_pred_recession_3m = (y_pred[1].flatten() > 0.5).astype(int)
    y_pred_phase = np.argmax(y_pred[2].reshape(-1, y_pred[2].shape[-1]), axis=-1)
    
    return y_pred_recession_1m, y_pred_recession_3m, y_pred_phase

# Main function for prediction
def main(input_file, model_file):
    features = ['unrate', 'unrate_1m_pct', 'unrate_3m_pct', 'cpi', 'cpi_1m_pct', 'interest_rate', 'yield_curve', 
                'adj_close', 'building_permits', 'consumer_confidence', 'industrial_production', 
                'industrial_production_1m_pct', 'corporate_profits', 'corporate_profits_q_pct', 
                'consumer_debt', 'consumer_debt_q_pct']

    # Load data
    df = load_data(input_file)

    # Preprocess data
    X = preprocess_data(df, features)

    # Load model
    model = load_trained_model(model_file)

    # Predict outputs
    y_pred_recession_1m, y_pred_recession_3m, y_pred_phase = predict_outputs(model, X)

    # Add predictions to the dataframe and save to CSV
    df['predicted_recession_1m'] = y_pred_recession_1m
    df['predicted_recession_3m'] = y_pred_recession_3m
    df['predicted_phase'] = y_pred_phase
    df.to_csv('predicted_outputs.csv', index=False)
    print(f"Predictions saved to predicted_outputs.csv")

if __name__ == "__main__":
    input_file = os.path.join('data', 'processed', 'all_data.csv')
    model_file = 'ltc_model.h5'
    main(input_file, model_file)
