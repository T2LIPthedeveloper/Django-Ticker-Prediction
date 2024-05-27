import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN
from data_collection.data_fetcher import get_yahoo_finance_data, get_fred_data

# TODO: Implement Liquid Time Constant model for macro-economic phase prediction

class LTCCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LTCCell, self).__init__()
        self.units = units
        self.state_size = units
        self.kernel = None
        self.recurrent_kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.kernel)
        output = tf.tanh(h + tf.matmul(prev_output, self.recurrent_kernel))
        return output, [output]

def build_ltc_model(input_shape):
    model = Sequential()
    model.add(RNN(LTCCell(64), input_shape=input_shape))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_data():
    stock_data = get_yahoo_finance_data('AAPL')['Close'].values
    gdp_data = get_fred_data('GDP').values
    min_len = min(len(stock_data), len(gdp_data))
    stock_data = stock_data[-min_len:]
    gdp_data = gdp_data[-min_len:]
    combined_data = np.column_stack((stock_data, gdp_data))
    return combined_data

def predict_macro_phase(quarters_ahead):
    data = preprocess_data()
    input_shape = (data.shape[1], 1)
    model = build_ltc_model((None, input_shape[0]))
    
    X = data[:-quarters_ahead]
    y = data[quarters_ahead:, 0]  # Assuming we are predicting stock data as macro phase indicator
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    
    model.fit(X, y, epochs=10, verbose=0)
    
    prediction = model.predict(data[-quarters_ahead:].reshape((1, data.shape[1], 1)))
    return f"Predicted macro-economic phase for {quarters_ahead} quarter(s) ahead: {prediction[0][0]}"
