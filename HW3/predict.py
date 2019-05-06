import json
import pickle
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
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
labels = []
for i, name in enumerate(stock_names):
    predict_dataset = TestStockDataset(all_data, 0, scalers=scalers)
    predict_loader = DataLoader(
            predict_dataset, batch_size=1, shuffle=False)

    pretrain = torch.load("models/best_model[{}].pt".format(name))
    model = Stock_LSTM(input_size, hidden_size).to(device)
    model.load_state_dict(pretrain['model_state_dict'])
    model.eval()
    # predict
    h_state = torch.zeros(2, 1, hidden_size).to(device)
    c_state = torch.zeros(2, 1, hidden_size).to(device)
    with torch.no_grad():
        for j, data in enumerate(predict_loader):
            inputs = data['Sequence']
            inputs = inputs.to(device)
            pred, (h_state, c_state) = model(inputs, h_state, c_state)
            pred = pred[:, 29, :].squeeze()
            h_state = Variable(h_state.data)
            c_state = Variable(c_state.data)
    scaler = predict_dataset.scalers[i]
    last_value = inputs.cpu().numpy()
    last_value = last_value[0, 29, i]
    pred = pred.cpu().numpy()
    last_value = scaler.inverse_transform(last_value.reshape(-1, 1))
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    label = -1
    rchange = np.abs(pred-last_value)/last_value
    if rchange < 0.02:
        label = 1
    else:
        label = 0 if pred > last_value else 2
    print("last day: {}".format(last_value))
    print("predict: {}".format(pred))
    print("label: {}".format(label))
    labels.append(label)
with open("result.txt", "w") as f:
    for n in labels:
        f.write("{}\n".format(n))
