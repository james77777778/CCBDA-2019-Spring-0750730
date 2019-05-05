import json
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import TestStockDataset
from model import Stock_LSTM


# params
input_size = 20
hidden_size = 256
if torch.cuda.is_available():
    device = torch.device("cuda:0")

# load data
with open('stocks.json', 'r') as f:
    stocks_dict = json.load(f)
stock_names = [
    'INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN',
    'NFLX', 'FB', 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK',
    'TSLA', 'MSFT', 'SNPS']
all_data = []
for name in stock_names:
    all_data.append(stocks_dict[name])
all_data = np.array(all_data)

predict_dataset = TestStockDataset(all_data, 0)
predict_loader = DataLoader(
        predict_dataset, batch_size=1, shuffle=False)

pretrain = torch.load("models/best_model[{}].pt".format("INTC"))
model = Stock_LSTM(input_size, hidden_size).to(device)
model.load_state_dict(pretrain['model_state_dict'])

# predict
h_state = torch.zeros(2, 1, hidden_size).to(device)
c_state = torch.zeros(2, 1, hidden_size).to(device)
for i, data in enumerate(predict_loader):
    inputs = data['Sequence']
    inputs = inputs.to(device)
    pred, (h_state, c_state) = model(inputs, h_state, c_state)
    pred = pred[:, 29, :].squeeze()
    h_state = Variable(h_state.data)
    c_state = Variable(c_state.data)
    bs = 1

print("last day: {}".format(inputs.cpu().numpy()))
print("predict: {}".format(pred.cpu().detach().numpy()))
