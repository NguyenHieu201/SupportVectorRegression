import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, params):
        super(LSTM, self).__init__()
        
        self.l = params['l']
        self.p = params['p']

        self.l1 = nn.LSTM(1, 128, batch_first=True)
        self.l2 = nn.LSTM(128, 64, batch_first=True)
        self.l3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, self.p)
        

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.l)
        
        x, _ = self.l1(x)
        x, _ = self.l2(x)
        
        x = self.l3(x)
        x = F.relu(x)
        
        x = self.output(x[:,-1,:])

        return x.reshape(batch_size, -1, self.p)