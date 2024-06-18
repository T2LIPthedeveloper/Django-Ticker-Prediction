import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import warnings
warnings.filterwarnings("ignore")

# Indicators used:
# - Unemployment Rate
# - Inflation Rate (CPI)
# - Interest Rate (Federal Funds Rate)
# - Yield Curve (10Y-2Y)
# - Stock Market (S&P 500)
# - Building Permits
# - Consumer Confidence Index
# - Industrial Production Index
# - Corporate Profits
# - Consumer Debt

# Recessions start from peaks and end at troughs.
# The following dates are from the NBER.
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

# Convert str dates in the recessions list to datetime objects
recessions = [(datetime.datetime.strptime(start, '%Y-%m-%d'), datetime.datetime.strptime(end, '%Y-%m-%d')) for start, end in recessions]


def get_using_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

def preprocess_and_resample(df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'M') -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.resample(freq).mean()
    df[value_col] = df[value_col].ffill()
    df.reset_index(inplace=True)
    return df

def preprocess_unemp(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_and_resample(df, 'date', 'unrate')
    df['unrate_1m_pct'] = df['unrate'].pct_change()
    df['unrate_3m_pct'] = df['unrate'].pct_change(3)
    return df

def preprocess_cpi(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_and_resample(df, 'date', 'cpi')
    df['cpi_1m_pct'] = df['cpi'].pct_change()
    return df

def preprocess_interest(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_and_resample(df, 'date', 'interest_rate')

def preprocess_yield_curve(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_and_resample(df, 'date', 'yield_curve')

def preprocess_stock_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['date', 'adj_close']]
    return preprocess_and_resample(df, 'date', 'adj_close')

def preprocess_building_permits(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_and_resample(df, 'date', 'building_permits')

def preprocess_consumer_confidence(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_and_resample(df, 'date', 'consumer_confidence')

def preprocess_industrial_production(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_and_resample(df, 'date', 'industrial_production')
    df['industrial_production_1m_pct'] = df['industrial_production'].pct_change()
    return df

def preprocess_corporate_profits(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_and_resample(df, 'date', 'corporate_profits', freq='M')
    df['corporate_profits_q_pct'] = df['corporate_profits'].pct_change()
    return df

def preprocess_consumer_debt(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_and_resample(df, 'date', 'consumer_debt', freq='M')
    df['consumer_debt_q_pct'] = df['consumer_debt'].pct_change()
    return df

def preprocess_recessions(df: pd.DataFrame, recessions: list) -> pd.DataFrame:
    """
    Indicates expansion or recession periods in the DataFrame.
    Features:
    - phase: 0 for expansion, 1 for recession
    - recession_1m: 1 if a recession starts in the next month, 0 otherwise
    - recession_3m: 1 if a recession starts in the next 3 months, 0 otherwise
    """

    df['phase'] = 0
    df['recession_1m'] = 0
    df['recession_3m'] = 0

    for start, end in recessions:
        df.loc[(df['date'] >= start) & (df['date'] <= end), 'phase'] = 1
        df.loc[(df['date'] == (end + datetime.timedelta(days=30)), 'recession_1m')] = 1
        df.loc[(df['date'] == (end + datetime.timedelta(days=90)), 'recession_3m')] = 1

    return df

def scale_features(df: pd.DataFrame, columns: list, scaler=StandardScaler()) -> pd.DataFrame:
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled

def get_all_time_series_from_dir(directory: str) -> dict:
    files = os.listdir(directory)
    time_series = {}
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            time_series[file] = df
    return time_series

def main() -> None:
    time_series = get_all_time_series_from_dir('data/raw')

    unemp = preprocess_unemp(time_series['unemployment.csv'])
    cpi = preprocess_cpi(time_series['cpi_data.csv'])
    interest = preprocess_interest(time_series['fed_interest.csv'])
    yield_curve = preprocess_yield_curve(time_series['yield_curve.csv'])
    stock_market = preprocess_stock_market(time_series['sp500_yf.csv'])
    building_permits = preprocess_building_permits(time_series['building_permits.csv'])
    consumer_confidence = preprocess_consumer_confidence(time_series['consumer_confidence.csv'])
    industrial_production = preprocess_industrial_production(time_series['industrial_production.csv'])
    corporate_profits = preprocess_corporate_profits(time_series['corporate_profits.csv'])
    consumer_debt = preprocess_consumer_debt(time_series['consumer_debt.csv'])

    df = unemp.merge(cpi, on='date', how='outer')
    df = df.merge(interest, on='date', how='outer')
    df = df.merge(yield_curve, on='date', how='outer')
    df = df.merge(stock_market, on='date', how='outer')
    df = df.merge(building_permits, on='date', how='outer')
    df = df.merge(consumer_confidence, on='date', how='outer')
    df = df.merge(industrial_production, on='date', how='outer')
    df = df.merge(corporate_profits, on='date', how='outer')
    df = df.merge(consumer_debt, on='date', how='outer')

    columns_to_scale = ['unrate', 'cpi', 'interest_rate', 'yield_curve', 'adj_close', 'building_permits', 'consumer_confidence', 'industrial_production', 'corporate_profits', 'consumer_debt']
    df = scale_features(df, columns_to_scale)

    df = preprocess_recessions(df, recessions)

    df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '2023-10-01')]

    df['date'] = df['date'].dt.strftime('%Y-%m-01')

    df = df.ffill().bfill()

    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values remain in the DataFrame.")
    
    df.to_csv('data/processed/all_data.csv', index=False)

    train_df = df[df['date'] < '2021-01-01']
    test_df = df[df['date'] >= '2021-01-01']

    train_df.date = pd.to_datetime(train_df.date)
    test_df.date = pd.to_datetime(test_df.date)
                              
    train_df.to_csv('data/processed/train_data.csv', index=False)
    test_df.to_csv('data/processed/test_data.csv', index=False)


if __name__ == '__main__':
    main()
