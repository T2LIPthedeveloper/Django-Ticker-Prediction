import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
fred_api_key = os.getenv("FRED_API")

# Fred instance
fred = Fred(
    api_key=fred_api_key
)

# Set up Yahoo Finance API
yf.pdr_override()

# Define indicators to collect from FRED
indicators = {
    'GDP': 'GDP',
    'Unemployment Rate': 'UNRATE',
    'Inflation Rate (CPI)': 'CPIAUCSL',
    'Interest Rate (Federal Funds Rate)': 'FEDFUNDS',
    'Yield Curve (10Y-2Y)': 'T10Y2Y',
    'Stock Market (S&P 500)': 'SP500',
    'Building Permits': 'PERMIT',
    'Consumer Confidence Index': 'UMCSENT',
    'Industrial Production Index': 'INDPRO',
    'Corporate Profits': 'CP',
    'Consumer Debt': 'TDSP'
}

# start and end dates for recessions

#TODO: Complete functions to get information from FRED, Yahoo Finance, and NBER

def save_to_csv(data, filename):
    """
    Save the data to a CSV file
    """
    data.to_csv(filename, index=False)


def get_unemployment_table():
    """
    Get the unemployment rate from FRED
    """
    unemp_table = fred.get_series('UNRATE')
    unemp_table = pd.DataFrame(unemp_table).reset_index()
    unemp_table.columns = ['date', 'unrate']
    unemp_table.date = unemp_table.date.astype(str)
    save_to_csv(unemp_table, 'data/raw/unemployment.csv')
    return 1 if unemp_table.shape[0] > 0 else 0

def get_cpi_table():
    """
    Get CPI data from FRED
    """
    infl_table = fred.get_series('CPIAUCSL')
    infl_table = pd.DataFrame(infl_table).reset_index()
    infl_table.columns = ['date', 'cpi']
    infl_table.date = infl_table.date.astype(str)
    save_to_csv(infl_table, 'data/raw/cpi_data.csv')
    return 1 if infl_table.shape[0] > 0 else 0
    pass

def get_interest_rate_table():
    """
    Get the interest rate from FRED
    """
    interest_table = fred.get_series('FEDFUNDS')
    interest_table = pd.DataFrame(interest_table).reset_index()
    interest_table.columns = ['date', 'interest_rate']
    interest_table.date = interest_table.date.astype(str)
    save_to_csv(interest_table, 'data/raw/fed_interest.csv')
    return 1 if interest_table.shape[0] > 0 else 0
    pass

def get_yield_curve_table():
    """
    Get the yield curve from FRED
    """
    yield_table = fred.get_series('T10Y2Y')
    yield_table = pd.DataFrame(yield_table).reset_index()
    yield_table.columns = ['date', 'yield_curve']
    yield_table.date = yield_table.date.astype(str)
    save_to_csv(yield_table, 'data/raw/yield_curve.csv')
    return 1 if yield_table.shape[0] > 0 else 0
    pass

# def get_stock_market_table():
#     """
#     Get the stock market from FRED
#     """
#     stock_table = fred.get_series('SP500')
#     stock_table = pd.DataFrame(stock_table).reset_index()
#     stock_table.columns = ['date', 'sp500']
#     stock_table.date = stock_table.date.astype(str)
#     save_to_csv(stock_table, 'data/raw/sp500.csv')
#     return 1 if stock_table.shape[0] > 0 else 0
#     pass

def get_building_permits_table():
    """
    Get the building permits from FRED
    """
    bp_table = fred.get_series('PERMIT')
    bp_table = pd.DataFrame(bp_table).reset_index()
    bp_table.columns = ['date', 'building_permits']
    bp_table.date = bp_table.date.astype(str)
    save_to_csv(bp_table, 'data/raw/building_permits.csv')
    return 1 if bp_table.shape[0] > 0 else 0
    pass

def get_consumer_confidence_table():
    """
    Get the consumer confidence index from FRED
    """
    cc_table = fred.get_series('UMCSENT')
    cc_table = pd.DataFrame(cc_table).reset_index()
    cc_table.columns = ['date', 'consumer_confidence']
    cc_table.date = cc_table.date.astype(str)
    save_to_csv(cc_table, 'data/raw/consumer_confidence.csv')
    return 1 if cc_table.shape[0] > 0 else 0
    pass

def get_industrial_production_table():
    """
    Get the industrial production index from FRED
    """
    ip_table = fred.get_series('INDPRO')
    ip_table = pd.DataFrame(ip_table).reset_index()
    ip_table.columns = ['date', 'industrial_production']
    ip_table.date = ip_table.date.astype(str)
    save_to_csv(ip_table, 'data/raw/industrial_production.csv')
    return 1 if ip_table.shape[0] > 0 else 0
    pass

def get_corporate_profits_table():
    """
    Get the corporate profits from FRED
    """
    cp_table = fred.get_series('CP')
    cp_table = pd.DataFrame(cp_table).reset_index()
    cp_table.columns = ['date', 'corporate_profits']
    cp_table.date = cp_table.date.astype(str)
    save_to_csv(cp_table, 'data/raw/corporate_profits.csv')
    return 1 if cp_table.shape[0] > 0 else 0
    pass

def get_consumer_debt_table():
    """
    Get the consumer debt from FRED
    """
    cd_table = fred.get_series('TDSP')
    cd_table = pd.DataFrame(cd_table).reset_index()
    cd_table.columns = ['date', 'consumer_debt']
    cd_table.date = cd_table.date.astype(str)
    save_to_csv(cd_table, 'data/raw/consumer_debt.csv')
    return 1 if cd_table.shape[0] > 0 else 0
    pass

def get_stock_data():
    """
    Get the stock data from Yahoo Finance
    """
    sp500 = yf.download('^GSPC', start='1960-01-01')
    sp500 = sp500.reset_index()
    sp500.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    sp500.date = sp500.date.astype(str)
    save_to_csv(sp500, 'data/raw/sp500_yf.csv')
    return 1 if sp500.shape[0] > 0 else 0
    pass

# def get_recession_data():
#     """
#     Get the recession data from NBER
#     """
#     rec_table = pd.DataFrame(recessions, columns=['start_date', 'end_date'])
#     save_to_csv(rec_table, 'data/raw/recessions.csv')
#     return 1 if rec_table.shape[0] > 0 else 0
#     pass

def main():
    """
    Main function to collect data
    """
    # if anything is 1 then the data was collected successfully
    print("Unemployment data collected") if get_unemployment_table() else (print("Unemployment data not collected") and exit(1))
    print("CPI data collected") if get_cpi_table() else (print("Inflation data not collected") and exit(1))
    print("Interest rate data collected") if get_interest_rate_table() else (print("Interest rate data not collected") and exit(1))
    print("Yield curve data collected") if get_yield_curve_table() else (print("Yield curve data not collected") and exit(1))
    # print("Stock market data collected") if get_stock_market_table() else (print("Stock market data not collected") and exit(1))
    print("Building permits data collected") if get_building_permits_table() else (print("Building permits data not collected") and exit(1))
    print("Consumer confidence data collected") if get_consumer_confidence_table() else (print("Consumer confidence data not collected") and exit(1))
    print("Industrial production data collected") if get_industrial_production_table() else (print("Industrial production data not collected") and exit(1))
    print("Corporate profits data collected") if get_corporate_profits_table() else (print("Corporate profits data not collected") and exit(1))
    print("Consumer debt data collected") if get_consumer_debt_table() else (print("Consumer debt data not collected") and exit(1))
    print("Stock data collected") if get_stock_data() else (print("Stock data not collected") and exit(1))
    # print("Recession data collected") if get_recession_data() else (print("Recession data not collected") and exit(1))
    pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    pass



