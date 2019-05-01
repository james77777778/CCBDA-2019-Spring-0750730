import torch
import torch.nn as nn


class Stock_LSTM(nn.Module):
    def __init__(self, in_size=20, hid_size=256):
        super(Stock_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_size, hidden_size=hid_size, num_layers=2,
            batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hid_size, 256),
            nn.Linear(256, 1),
            )

    def forward(self, x, h_state, c_state):
        r_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), (h_state, c_state)
