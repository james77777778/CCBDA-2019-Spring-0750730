import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from data import StockDataset
from model import Stock_LSTM


'''
if 'HW3' not in os.getcwd():
    os.chdir('HW3')
'''
# make dir
folder = ["models", "results"]
for f in folder:
    if not os.path.exists(f):
        os.makedirs(f)

# params
batch_size = 60
input_size = 20
hidden_size = 256
nepoch = 700

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

for i, name in enumerate(stock_names):
    # split train, valid
    master = StockDataset(all_data, i)
    n = len(master)
    n_valid = int(n * 0.1)
    n_train = n - n_valid
    drop_train = n_train - int(n_train/batch_size)*batch_size
    idx = list(range(n))
    train_idx = idx[drop_train:n_train]
    valid_idx = idx[n_train:(n_train + n_valid)]
    train_dataset = Subset(master, train_idx)
    valid_dataset = Subset(master, valid_idx)

    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False)

    # train
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    net = Stock_LSTM(input_size, hidden_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)

    all_loss = []
    best_loss = [100, -1]
    best_plot = 0
    for epoc in range(nepoch):
        running_loss = 0.0
        epoch_loss = 0.0
        pred_plot = np.empty([len(train_loader.dataset)])
        h_state = torch.zeros(2, batch_size, hidden_size).to(device)
        c_state = torch.zeros(2, batch_size, hidden_size).to(device)
        for i, data in enumerate(train_loader):
            inputs, labels = data['Sequence'], data['Target']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred, (h_state, c_state) = net(inputs, h_state, c_state)
            pred = pred[:, 29, :].squeeze()
            h_state = Variable(h_state.data)
            c_state = Variable(c_state.data)
            loss = criterion(pred, labels)
            epoch_loss += loss.item()*inputs.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 0:    # print every 2 mini-batches
                print('[{}, {}] loss: {:.7f}'.format(
                    epoc+1, i+1, running_loss/2))
                running_loss = 0.0
            bs = batch_size
            pred_plot[i*bs:i*bs+bs] = pred.cpu().detach().numpy()
        epoch_loss /= len(train_dataset)
        if epoch_loss < best_loss[0]:
            best_loss[0] = epoch_loss
            best_loss[1] = epoc
            best_plot = pred_plot
            torch.save({
                "name": name,
                "epoch": epoc,
                "model_state_dict": net.state_dict()},
                "models/best_model[{}].pt".format(name)
            )
        all_loss.append(epoch_loss)
    print('Finished Training, best loss= {:.7f}, i= {}'.format(
        best_loss[0], best_loss[1]))

    # plot
    plt.figure()
    plt.plot(
        np.arange(drop_train, n_train, 1.0),
        [data["Target"] for data in train_dataset],
        'r-')
    plt.plot(np.arange(drop_train, n_train, 1.0), best_plot, 'b-')
    plt.savefig("results/result[{}].png".format(name))
    plt.figure()
    plt.plot(all_loss)
    plt.savefig("results/loss[{}].png".format(name))
