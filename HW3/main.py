import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from data import StockDataset
from model import Stock_LSTM


'''
import os
if 'HW3' not in os.getcwd():
    os.chdir('HW3')
'''

# load data
with open('stocks.pkl', 'rb') as f:
    stocks_dict = pickle.load(f)
stock_names = [
    'INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN',
    'NFLX', 'FB', 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK',
    'TSLA', 'MSFT', 'SNPS']
all_data = []
for name in stock_names:
    all_data.append(stocks_dict[name])
all_data = np.array(all_data)

# split train, valid
master = StockDataset(all_data, 0)
n = len(master)
n_valid = int(n * 0.1)
n_train = n - n_valid
idx = list(range(n))
train_idx = idx[:n_train]
valid_idx = idx[n_train:(n_train + n_valid)]
train_dataset = Subset(master, train_idx)
valid_dataset = Subset(master, valid_idx)

# dataloader
batch_size = 71
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(
    valid_dataset, batch_size=1, shuffle=False)

# train
if torch.cuda.is_available():
    device = torch.device("cuda:0")
net = Stock_LSTM().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoc in range(100):
    running_loss = 0.0
    pred_plot = np.empty([len(train_loader.dataset)])
    h_state = torch.zeros(2, batch_size, 128).to(device)
    c_state = torch.zeros(2, batch_size, 128).to(device)
    for i, data in enumerate(train_loader):
        inputs, labels = data['Sequence'], data['Target']
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        pred, (h_state, c_state) = net(inputs, h_state, c_state)
        pred = pred[:, 29, :].squeeze()
        h_state = Variable(h_state.data)
        c_state = Variable(c_state.data)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 0:    # print every 2 mini-batches
            print('[{}, {}] loss: {:.7f} pred: {:.4f} label: {:.4f}'.format(
                epoc + 1, i + 1, running_loss / 2, pred.data[0], labels[0]))
            running_loss = 0.0
        bs = batch_size
        pred_plot[i*bs:i*bs+bs] = pred.cpu().detach().numpy()
print('Finished Training')

# plot
plt.plot(
    np.arange(0.0, n_train, 1.0),
    [data["Target"] for data in train_dataset],
    'r-')
plt.plot(np.arange(0.0, n_train, 1.0), pred_plot, 'b-')
plt.draw()
