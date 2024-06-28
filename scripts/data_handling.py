import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from functools import reduce
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load environment variables
load_dotenv()
fred_api_key = os.getenv("FRED_API")

# Fred instance
fred = Fred(
    api_key=fred_api_key
)

# Set up Yahoo Finance API
yf.pdr_override()

# List of NBER recession periods from peak to trough
recessions = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

# Dictionary of features to work with for recession prediction
fred_features = {
    'com_ind_loans': "TOTCI", # Commercial and Industrial Loans
    'unemp': 'UNRATE', # Unemployment Rate
    'cpi': 'CPIAUCSL', # Consumer Price Index (Urban Consumers)
    'con_sentiment_pre': 'UMCSENT1', # Consumer Sentiment (Pre-1978)
    'con_sentiment_post': 'UMCSENT', #Consumer Sentiment (Post-1978)
    'payroll': 'PAYEMS', # All Employees, Total Non-Farm Payroll
    'effective_rate': 'DFF', # Fed Fund Effective Rate
    'M1': 'M1SL', # Monetary Aggregate 1
    'M2': 'M2SL', # Monetary Aggregate 2
    'treasury_10y': 'GS10', # 10-Year Treasury Yield (Constant Maturity)
    'treasury_5y': 'GS5', # 5-Year Treasury Yield (Constant Maturity)
    'treasury_1y': 'GS1' # 1-Year Treasury Yield (Constant Maturity)
}

yf_features = {
    'sp500': '^GSPC' # S&P 500 Index Data
}

# start and end dates for recessions

#TODO: Complete functions to get information from FRED, Yahoo Finance, and NBER

def get_fred_data(tickers):
    start_date = datetime.datetime(1900, 1, 1)
    end_date = datetime.datetime.now()
    fred_data = {}
    for name, ticker in tickers.items():
        df = fred.get_series(ticker, start_date, end_date)
        # Also save to data/raw folder for logging purposes
        df.to_csv(f'data/raw/{name}_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv')
        fred_data[name] = df
    return fred_data

fred_data = get_fred_data(fred_features)

def get_yfinance_data(tickers):
    yf_data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start="1900-01-01")
        df = df['Adj Close']
        # Also save to data/raw folder for logging purposes
        df.to_csv(f'data/raw/{name}_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv')
        yf_data[name] = df
    return yf_data

yf_data = get_yfinance_data(yf_features)

def fill_nan_null(data):
    """
    Fill NaN values in data
    """
    for name in data.keys():
        data[name] = data[name].ffill().bfill()
        if data[name].isna().any():
            n_count+=1
            print(f"{name} has empty values.")

def get_df_from_dict(data, series_list):
    for name, series in data.items():
        temp_df = pd.DataFrame({'date': series.index, name: series.values})
        temp_df.date = pd.to_datetime(temp_df.date)
        print(temp_df.head())
        series_list.append(temp_df)

    return series_list

def create_lagged_features(df, lags):
    new_columns = []
    lagged_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        for lag in lags:
            lag_col = f'{col}_lag_{lag}'
            lagged_df[lag_col] = df[col].shift(lag)
            new_columns.append(lag_col)
    return lagged_df

def create_pct_change_features(df, periods):
    new_columns = []
    pct_change_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if 'lag' not in col:  # Avoid creating pct_change for lagged features
            for period in periods:
                pct_change_col = f'{col}_pct_change_{period}'
                pct_change_df[pct_change_col] = df[col].pct_change(period)
                new_columns.append(pct_change_col)
    return pct_change_df

def create_moving_average_features(df, windows):
    new_columns = []
    ma_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if 'lag' not in col:  # Avoid creating moving averages for lagged features
            for window in windows:
                ma_col = f'{col}_ma_{window}'
                ma_df[ma_col] = df[col].rolling(window=window).mean()
                new_columns.append(ma_col)
    return ma_df

def add_recession_indicators(df, recession_periods, shift_periods=[3, 6, 12]):
    # Convert recession dates to datetime
    recession_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in recession_periods]

    def is_in_recession(date):
        for start, end in recession_periods:
            if start <= date <= end:
                return 1.0
        return 0.0

    # Add initial recession indicator
    recession_df = pd.DataFrame(index=df.index)
    recession_df['is_recession'] = df.index.to_series().apply(is_in_recession)

    # Add shifted recession indicators
    new_columns = ['is_recession']
    for shift in shift_periods:
        col_name = f'recession_in_{shift//3}q'
        recession_df[col_name] = recession_df['is_recession'].shift(-shift).fillna(0).astype(int)
        new_columns.append(col_name)

    return recession_df[new_columns]

def preprocess_features(df, recession_periods):
    lags = [1, 3, 6, 12]
    periods = [1, 3, 6, 12]
    windows = [3, 6, 12]
    
    lagged_df = create_lagged_features(df, lags)
    print(df.columns)
    pct_change_df = create_pct_change_features(df, periods)
    ma_df = create_moving_average_features(df, windows)
    recession_df = add_recession_indicators(df, recession_periods)
    
    # Combine the original df with the new features
    df_combined = df.join(lagged_df, rsuffix='_lag')
    df_combined = df_combined.join(pct_change_df, rsuffix='_pct_change')
    df_combined = df_combined.join(ma_df, rsuffix='_ma')
    df_combined = df_combined.join(recession_df, rsuffix='_recession')
    
    df_combined = df_combined.ffill().bfill()  # Fill remaining NaN values after feature creation
    
    return df_combined

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Function to test feature importance and select relevant features
def test_feature_importance(df):
    # Assuming 'is_recession' and 'recession_in_1q', 'recession_in_2q', 'recession_in_4q' are targets
    targets = ['is_recession', 'recession_in_1q', 'recession_in_2q', 'recession_in_4q']
    features = df.drop(targets, axis=1)  # Exclude target columns

    # Use a classifier (e.g., RandomForestClassifier) to assess feature importance
    clf = RandomForestClassifier(random_state=42)
    clf.fit(features.values, df['is_recession'].values)  # Example using 'is_recession' as target

    # Select features based on importance (e.g., using SelectFromModel)
    sfm = SelectFromModel(clf, threshold='median', max_features=50)
    sfm.fit(features.values, df['is_recession'].values)  # Fit SelectFromModel

    # Get selected features and transform dataframe
    selected_features = features.columns[sfm.get_support()]
    df_selected = df[targets].join(pd.DataFrame(data=df[selected_features], index=df.index))

    # Calculate VIF for selected features
    vif_data = calculate_vif(df_selected.drop(targets, axis=1))

    # Remove features with high VIF
    high_vif_threshold = 10
    features_to_keep = vif_data[vif_data["VIF"] <= high_vif_threshold]["feature"].tolist()
    df_final = df[targets].join(df_selected[features_to_keep])

    return df_final

def main():
    """
    Main function to collect data
    """
    # if anything is 1 then the data was collected successfully
    fred_data = get_fred_data(fred_features)
    yf_data = get_yfinance_data(yf_features)

    if all([len(data) > 1 for data in fred_data.values()]) and all([len(data) > 1 for data in yf_data.values()]):
        print("Data collection successful")
    else:
        raise ValueError("Data collection failed")
    
    # Combine consumer_sentiment columns for FRED data
    fred_data['con_sentiment'] = pd.concat([fred_data['con_sentiment_pre'], fred_data['con_sentiment_post']]).sort_index().loc[~pd.concat([fred_data['con_sentiment_pre'], fred_data['con_sentiment_post']]).index.duplicated(keep='first')]
    del fred_data['con_sentiment_pre']
    del fred_data['con_sentiment_post']

    # Data resampling to start of month
    fred_data['com_ind_loans'] = fred_data['com_ind_loans'].resample('MS').mean()
    fred_data['effective_rate'] = fred_data['effective_rate'].resample('MS').mean()

    # Fill NaN values
    fill_nan_null(fred_data)
    fill_nan_null(yf_data)

    idx_series = []
    idx_series = get_df_from_dict(fred_data, idx_series)
    idx_series = get_df_from_dict(yf_data, idx_series)

    # Merge all dataframes
    df = reduce(lambda x, y: pd.merge(x, y, how='outer', on='date'), idx_series)
    df = df.ffill().bfill()
    df = df[df['date'] >= '1959-01-01']
    df.set_index('date', inplace=True)

    # Drop features due to multicollinearity (explored in EDA)
    df.drop(['treasury_1y', 'treasury_10y', 'M1', ], axis=1, inplace=True)

    # Store full raw data
    df.to_csv('data/processed/all_data.csv')

    # Process dataframe
    df_processed = preprocess_features(df, recessions)
    # Store for logging purposes
    df_processed.to_csv('data/interim/all_data_processed.csv')
    # Test feature importance and select relevant features
    df_final = test_feature_importance(df_processed)
    # Store final processed data
    df_final.to_csv('data/processed/final_data.csv')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    pass



