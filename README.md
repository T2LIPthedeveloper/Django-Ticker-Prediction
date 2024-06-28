# Macroeconomic Analysis of the United States

Using data from FRED (Federal Reserve Economic Data) and Yahoo! Finance to analyse the macroeconomic phase of the United States in the current time. The data is collected from the year 2000 to 2020. The data is collected from the following sources:
- [FRED](https://fred.stlouisfed.org/)
- [Yahoo! Finance](https://finance.yahoo.com/)
- [Bureau of Economic Analysis](https://www.bea.gov/)

An LSTM (Long Short Term Memory) model is used to predict the macroeconomic phase of the United States as a baseline model. We would like to compare it to a newer architecture called LTCNN (Liquid Time-Constant Neural Network) which is a more adaptable model architecture, capable of changing activation functions and other parameters dynamically during training and testing for increased robustness and accuracy.

We want to predict the macroeconomic phase (trough, peak, recession and expansion) of the current time and forecast the phase for the next 1, 2, and 4 quarters respectively.

## Requirements

To install the required packages in your virtual environment, run the following command:

```bash
pip install -r requirements.txt
```

Create a .env file in the root directory and add the following environment variables:

```bash
FRED_API_KEY=YOUR_FRED_API_KEY
```

Additionally, you will need to modify the keras-ncp library within your virtual environment to include the following changes:

```python
# keras-ncp/tf/cfc_cell.py
# keras-ncp/tf/ltc_cell.py
# keras-ncp/tf/mm_rnn.py
# keras-ncp/ltc_cell.py

# Replace tf.keras.Layers.AbstractRNNCell with tf.keras.layers.Layer
# For example:
class LTCCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCCell, self).__init__(**kwargs)
        self.units = units
```

## Running the Code

Ensure that you have a valid FRED API key in your .env file and a data directory containing processed, raw and interim subfolders. Run the following command to preprocess the data:

```bash
python scripts/data_handling.py
```

This will populate your data directory with the necessary data files as well as extra logging files detailing the different states of the data collected.

To train the LSTM model, run the following command:

```bash
python scripts/model_training_LSTM.py
```

To train the LTCNN model, run the following command:

```bash
python scripts/model_training_LTC.py
```

These will generate the necessary model files and logs in the models directory.

## Prediction and Forecasting

To predict the macroeconomic phase of the current time and forecast the phase for the next 1, 2, and 4 quarters respectively, run the following command:

```bash
python scripts/phase_prediction.py --model LSTM
```


For initial data collection and preprocessing 

