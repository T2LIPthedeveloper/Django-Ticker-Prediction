import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import FunctionTransformer
import datetime

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

def get_using_dates():
    return pd.DataFrame(
        data=pd.date_range(
            '1965-01-01',
            datetime.datetime.today().strftime("%Y-%m-%d"),
            freq='MS'
            ).strftime("%Y-%m-%d").tolist(), 
        columns=['date']
        )

def preprocess_unemp():
    # Load data
    unemp = pd.read_csv('data/raw/unemployment.csv')
    # Reset index after start date
    unemp = unemp[unemp['date'] >= '1965-01-01'].reset_index(drop=True)
    # What we want is the percentage change in unemployment rate over the past 12 months
    unemp['un_1y_pct'] = 0.
    for i in range(12, len(unemp)):
        change = (unemp.at[i, 'unrate'] - unemp.at[i - 12, 'unrate'])
        unemp.at[i, '12_mon'] = change
    # Merge with using dates
    unemp = unemp[unemp['date'] >= '1968-01-01'].reset_index(drop=True)
    unemp = unemp.merge(get_using_dates(), how='right', on='date')
    # Fill in missing values
    unemp['un_1y_pct'] = unemp['un_1y_pct'].fillna(method='bfill').fillna(method='ffill')
    unemp['unrate'] = unemp['unrate'].fillna(method='bfill').fillna(method='ffill')

    # remove 12 mon column
    unemp = unemp.drop(columns=['12_mon'])
    # Return table
    return unemp


def preprocess_cpi():
    '''
    Preprocess the Consumer Price Index data to get inflation rate
    Convert from CPI amounts to year-over-year inflation rate
    Formula: (CPI[t] - CPI[t-12]) / CPI[t-12]

    Get monthly, quarterly, and yearly inflation rates
    '''
    latest_date = datetime.datetime.today() - datetime.timedelta(days=1)
    ld_str = latest_date.strftime("%Y-%m-%d")
    latest_month = datetime.datetime(2024, datetime.datetime.today().month - 1, 1)
    lm_str = latest_month.strftime("%Y-%m-%d")

    cpi = pd.read_csv('data/raw/cpi_data.csv')
    cpi = cpi[cpi['date'] >= '1965-01-01'].reset_index(drop=True)

    # Annual inflation rate
    cpi['cpi_1y_inf'] = cpi['cpi'].pct_change(periods=12)
    # Quarterly inflation rate
    cpi['cpi_1q_inf'] = cpi['cpi'].pct_change(periods=3)

    # Monthly inflation rate
    cpi['cpi_1m_inf'] = cpi['cpi'].pct_change(periods=1)

    # Merge with using dates
    cpi = cpi.merge(get_using_dates(), how='right', on='date')
    # Fill in missing values
    cpi['cpi_1y_inf'] = cpi['cpi_1y_inf'].fillna(method='bfill').fillna(method='ffill')
    cpi['cpi_1q_inf'] = cpi['cpi_1q_inf'].fillna(method='bfill').fillna(method='ffill')
    cpi['cpi_1m_inf'] = cpi['cpi_1m_inf'].fillna(method='bfill').fillna(method='ffill')

    return cpi

def preprocess_interest_rate():
    '''
    Preprocess the Federal Funds Rate data
    '''
    interest = pd.read_csv('data/raw/fed_interest.csv')
    interest = interest[interest['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    interest['fed_1y_ema'] = interest['interest_rate'].ewm(span=12).mean()

    # Calculate the 12-month percentage change
    interest['fed_1y_pct'] = interest['fed_1y_ema'].pct_change(periods=12)

    # Merge with using dates
    interest = interest.merge(get_using_dates(), how='right', on='date')
    # Fill in missing values
    interest['fed_1y_ema'] = interest['fed_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    interest['fed_1y_pct'] = interest['fed_1y_pct'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original interest rate column
    interest = interest.drop(columns=['interest_rate'])

    return interest

def preprocess_yield_curve():
    '''
    Preprocess the Yield Curve data
    (10 year - 2 year - 1 year)
    '''
    # Yield curve data currently provides spread between 10-year and 2-year
    # We want to use this to predict recessions
    yield_curve = pd.read_csv('data/raw/yield_curve.csv')
    yield_curve = yield_curve[yield_curve['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    yield_curve['yc_1y_ema'] = yield_curve['yield_curve'].ewm(span=12).mean()

    # Calculate the 12-month percentage change
    yield_curve['yc_1y_pct'] = yield_curve['yc_1y_ema'].pct_change(periods=12)

    # Merge with using dates
    yield_curve = yield_curve.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    yield_curve['yc_1y_ema'] = yield_curve['yc_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    yield_curve['yc_1y_pct'] = yield_curve['yc_1y_pct'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original yield curve column
    yield_curve = yield_curve.drop(columns=['yield_curve'])

    return yield_curve

def preprocess_stock_market():
    '''
    Preprocess the S&P 500 data
    Sources of S&P 500 data:
    - Yahoo Finance
    '''
    # Preprocess the S&P 500 data on Yahoo Finance
    # Yahoo Finance data starts from 1960 so remove data before 1960
    sp500_yh = pd.read_csv('data/raw/sp500_yf.csv')
    sp500_yh = sp500_yh[sp500_yh['date'] >= '1965-01-01'].reset_index(drop=True)

    # YF parameters: open, close, high, low, adj_close, volume
    sp500_yh['open'] = sp500_yh['open'].fillna(sp500_yh['adj_close'])

    sp500_yh['sp_pct_1m'] = sp500_yh['adj_close'].pct_change(periods=1)
    sp500_yh['sp_pct_3m'] = sp500_yh['adj_close'].pct_change(periods=3)
    sp500_yh['sp_pct_1y'] = sp500_yh['adj_close'].pct_change(periods=12)

    # Merge with using dates
    sp500_yh = sp500_yh.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    sp500_yh['sp_pct_1m'] = sp500_yh['sp_pct_1m'].fillna(method='bfill').fillna(method='ffill')
    sp500_yh['sp_pct_3m'] = sp500_yh['sp_pct_3m'].fillna(method='bfill').fillna(method='ffill')
    sp500_yh['sp_pct_1y'] = sp500_yh['sp_pct_1y'].fillna(method='bfill').fillna(method='ffill')

    return sp500_yh

def preprocess_building_permits():
    '''
    Preprocess the Building Permits data from FRED
    '''
    # Building permits usually indicate future construction activity, which is a leading indicator of economic activity.
    # We want to use this to predict recessions

    bp = pd.read_csv('data/raw/building_permits.csv')
    bp = bp[bp['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    bp['bp_1y_ema'] = bp['building_permits'].ewm(span=12).mean()

    bp['bp_pct_1m'] = bp['building_permits'].pct_change(periods=1)
    bp['bp_pct_3m'] = bp['building_permits'].pct_change(periods=3)
    bp['bp_pct_1y'] = bp['building_permits'].pct_change(periods=12)

    # Merge with using dates
    bp = bp.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    bp['bp_1y_ema'] = bp['bp_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    bp['bp_pct_1m'] = bp['bp_pct_1m'].fillna(method='bfill').fillna(method='ffill')
    bp['bp_pct_1y'] = bp['bp_pct_1y'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original building permits column
    bp = bp.drop(columns=['building_permits'])

    return bp

def preprocess_consumer_confidence():
    '''
    Preprocess the Consumer Confidence Index data from FRED
    '''

    cc = pd.read_csv('data/raw/consumer_confidence.csv')
    cc = cc[cc['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    cc['cc_1y_ema'] = cc['consumer_confidence'].ewm(span=12).mean()

    # Calculate the 12-month percentage change
    cc['cc_1y_pct'] = cc['cc_1y_ema'].pct_change(periods=12)

    # Merge with using dates
    cc = cc.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    cc['cc_1y_ema'] = cc['cc_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    cc['cc_1y_pct'] = cc['cc_1y_pct'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original consumer confidence column
    cc = cc.drop(columns=['consumer_confidence'])

    return cc

def preprocess_industrial_production():
    '''
    Preprocess the Industrial Production Index data from FRED
    IPI can be used to predict recessions since it is a leading indicator of economic activity.
    '''

    ip = pd.read_csv('data/raw/industrial_production.csv')
    ip = ip[ip['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    ip['ip_1y_ema'] = ip['industrial_production'].ewm(span=12).mean()

    # Calculate the 12-month percentage change
    ip['ip_1y_pct'] = ip['ip_1y_ema'].pct_change(periods=12)

    # Merge with using dates
    ip = ip.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    ip['ip_1y_ema'] = ip['ip_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    ip['ip_1y_pct'] = ip['ip_1y_pct'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original industrial production column
    ip = ip.drop(columns=['industrial_production'])

    return ip

def preprocess_corporate_profits():
    '''
    Preprocess the Corporate Profits data from FRED
    '''

    cp = pd.read_csv('data/raw/corporate_profits.csv')
    cp = cp[cp['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    cp['cp_1y_ema'] = cp['corporate_profits'].ewm(span=12).mean()

    cp['cp_pct_1m'] = cp['corporate_profits'].pct_change(periods=1)
    cp['cp_pct_3m'] = cp['corporate_profits'].pct_change(periods=3)
    cp['cp_pct_1y'] = cp['corporate_profits'].pct_change(periods=12)

    # Merge with using dates
    cp = cp.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    cp['cp_1y_ema'] = cp['cp_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    cp['cp_pct_1m'] = cp['cp_pct_1m'].fillna(method='bfill').fillna(method='ffill')
    cp['cp_pct_3m'] = cp['cp_pct_3m'].fillna(method='bfill').fillna(method='ffill')
    cp['cp_pct_1y'] = cp['cp_pct_1y'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original corporate profits column
    cp = cp.drop(columns=['corporate_profits'])

    return cp

def preprocess_consumer_debt():
    '''
    Preprocess the Consumer Debt data from FRED
    '''

    cd = pd.read_csv('data/raw/consumer_debt.csv')
    cd = cd[cd['date'] >= '1965-01-01'].reset_index(drop=True)

    # Calculate the 12-month EMA
    cd['cd_1y_ema'] = cd['consumer_debt'].ewm(span=12).mean()

    cd['cd_pct_1m'] = cd['consumer_debt'].pct_change(periods=1)
    cd['cd_pct_3m'] = cd['consumer_debt'].pct_change(periods=3)
    cd['cd_pct_1y'] = cd['consumer_debt'].pct_change(periods=12)
    
    # Merge with using dates
    cd = cd.merge(get_using_dates(), how='right', on='date')

    # Fill in missing values
    cd['cd_1y_ema'] = cd['cd_1y_ema'].fillna(method='bfill').fillna(method='ffill')
    cd['cd_pct_1m'] = cd['cd_pct_1m'].fillna(method='bfill').fillna(method='ffill')
    cd['cd_pct_3m'] = cd['cd_pct_3m'].fillna(method='bfill').fillna(method='ffill')
    cd['cd_pct_1y'] = cd['cd_pct_1y'].fillna(method='bfill').fillna(method='ffill')

    # Remove the original consumer debt column
    cd = cd.drop(columns=['consumer_debt'])

    return cd

# Recession prediction metrics
def years_to_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_til_list = []
    for start, end in recessions:
        days_til_list.append((datetime.datetime.strptime(start, "%Y-%m-%d") - date).days)
    try:
        min_days = min(d for d in days_til_list if d >= 0)
    except ValueError:
        if date <= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_til_list.index(min_days)
    
    if (min_days_index != 0) and (date <= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 365

def years_since_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_since_list = []
    for start, end in recessions:
        days_since_list.append((date - datetime.datetime.strptime(end, "%Y-%m-%d")).days)
    try:
        min_days = min(d for d in days_since_list if d >= 0)
    except ValueError:
        if date >= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_since_list.index(min_days)

    if (min_days_index != 0) and (date >= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 365

def months_to_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_til_list = []
    for start, end in recessions:
        days_til_list.append((datetime.datetime.strptime(start, "%Y-%m-%d") - date).days)
    try:
        min_days = min(d for d in days_til_list if d >= 0)
    except ValueError:
        if date <= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_til_list.index(min_days)

    if (min_days_index != 0) and (date <= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 30
    
def qts_to_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_til_list = []
    for start, end in recessions:
        days_til_list.append((datetime.datetime.strptime(start, "%Y-%m-%d") - date).days)
    try:
        min_days = min(d for d in days_til_list if d >= 0)
    except ValueError:
        if date <= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_til_list.index(min_days)

    if (min_days_index != 0) and (date <= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 90
    
def qts_since_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_since_list = []
    for start, end in recessions:
        days_since_list.append((date - datetime.datetime.strptime(end, "%Y-%m-%d")).days)
    try:
        min_days = min(d for d in days_since_list if d >= 0)
    except ValueError:
        if date >= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_since_list.index(min_days)

    if (min_days_index != 0) and (date >= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 90
    
def months_since_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    days_since_list = []
    for start, end in recessions:
        days_since_list.append((date - datetime.datetime.strptime(end, "%Y-%m-%d")).days)
    try:
        min_days = min(d for d in days_since_list if d >= 0)
    except ValueError:
        if date >= datetime.datetime.strptime(recessions[-1][1], "%Y-%m-%d"):
            return 0
        return float('NaN')
    min_days_index = days_since_list.index(min_days)

    if (min_days_index != 0) and (date >= datetime.datetime.strptime(recessions[min_days_index - 1][1], "%Y-%m-%d")):
        return 0
    else:
        return min_days / 30
    
def in_recession(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    for start, end in recessions:
        # Check if date is between start and end of recession
        if (date >= datetime.datetime.strptime(start, "%Y-%m-%d")) and (date <= datetime.datetime.strptime(end, "%Y-%m-%d")):
            return 1
    return 0

def recession_next_period(time_period: str, date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    if time_period == 'year':
        return years_to_recession(date)
    elif time_period == 'quarter':
        return qts_to_recession(date)
    elif time_period == 'month':
        return months_to_recession(date)
    else:
        return float('NaN')

def recession_previous_period(time_period: str, date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    if in_recession(date):
        return 0
    
    if time_period == 'year':
        return years_since_recession(date)
    elif time_period == 'quarter':
        return qts_since_recession(date)
    elif time_period == 'month':
        return months_since_recession(date)
    else:
        return float('NaN')

# Returns 1 if a recession is predicted to occur in the next time period else 0
# Time periods: year, quarter, month (1y, 3m, 1m)
def recession_next_period(time_period: str, date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    
    # calculate closest time period to recession
    if in_recession(date):
        return 1
    
    if time_period == 'year':
        if years_to_recession(date) < 1:
            return 1
        return 0
    elif time_period == 'quarter':
        if qts_to_recession(date) < 1:
            return 1
        return 0
    elif time_period == 'month':
        if months_to_recession(date) < 1:
            return 1
        return 0
    else:
        raise ValueError("Invalid time period")

# Returns tables for predictions for next year, quarter, month
# Collates all data in processed folder as total_data
def preprocess_data():
    # For each column from each table, preprocess the data and name columns according to the df name
    # Merge all tables into one table
    # Save table to data/processed/total_data.csv
    
    # Preprocess unemployment data
    unemp = preprocess_unemp()
    # Preprocess CPI data
    cpi = preprocess_cpi()
    # Preprocess interest rate data
    interest = preprocess_interest_rate()
    # Preprocess yield curve data
    yield_curve = preprocess_yield_curve()
    # Preprocess stock market data
    stock_market = preprocess_stock_market()
    # Preprocess building permits data
    building_permits = preprocess_building_permits()
    # Preprocess consumer confidence data
    consumer_confidence = preprocess_consumer_confidence()
    # Preprocess industrial production data
    industrial_production = preprocess_industrial_production()
    # Preprocess corporate profits data
    corporate_profits = preprocess_corporate_profits()
    # Preprocess consumer debt data
    consumer_debt = preprocess_consumer_debt()

    # Merge all into one table
    total_data = unemp.merge(cpi, how='inner', on='date')
    total_data = total_data.merge(interest, how='inner', on='date')
    total_data = total_data.merge(yield_curve, how='inner', on='date')
    total_data = total_data.merge(stock_market, how='left', on='date')
    total_data = total_data.merge(building_permits, how='left', on='date')
    total_data = total_data.merge(consumer_confidence, how='left', on='date')
    total_data = total_data.merge(industrial_production, how='left', on='date')
    total_data = total_data.merge(corporate_profits, how='left', on='date')
    total_data = total_data.merge(consumer_debt, how='inner', on='date')

    total_data['years_to_recession'] = total_data['date'].apply(years_to_recession)
    total_data['years_since_recession'] = total_data['date'].apply(years_since_recession)
    total_data['qts_to_recession'] = total_data['date'].apply(qts_to_recession)
    total_data['in_recession'] = total_data['date'].apply(in_recession)
    total_data['recession_next_year'] = total_data['date'].apply(lambda x: recession_next_period('year', x))
    total_data['recession_next_quarter'] = total_data['date'].apply(lambda x: recession_next_period('quarter', x))
    total_data.fillna(method='bfill', inplace=True)
    total_data.fillna(method='ffill', inplace=True)
    total_data.to_csv('data/processed/total_data.csv')
    return total_data

if __name__ == "__main__":
    preprocess_data()
    pass
