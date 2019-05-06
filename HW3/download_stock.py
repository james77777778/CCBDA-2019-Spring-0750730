import json
import ticker

'''
import os
if 'HW3' not in os.getcwd():
    os.chdir('HW3')
'''
stock_names = [
    'INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB',
    'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS'
    ]
stocks_dict = {}
for name in stock_names:
    tic = ticker.Ticker(name)
    df_tic = tic.history(period="2y")
    df_tic = df_tic[["Close"]]
    df_tic.dropna(inplace=True)
    stocks_dict[name] = df_tic.values.squeeze().tolist()

last_day = str(df_tic.index[-1].date())
with open('{}_stocks.json'.format(last_day), 'w') as f:
    json.dump(stocks_dict, f)
with open('stocks.json', 'w') as f:
    json.dump(stocks_dict, f)
