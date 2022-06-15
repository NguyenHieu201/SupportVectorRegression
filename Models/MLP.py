import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, params):
        super(MLP, self).__init__()
        
        self.l = params['l']
        self.p = params['p']

        self.l1 = nn.Linear(self.l, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 16)
        
        self.output = nn.Linear(16, self.p)
        

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(-1, self.l)
        
        x = self.l1(x)
        x = F.relu(x)
        
        x = self.l2(x)
        x = F.relu(x)
        
        x = self.l3(x)
        x = F.relu(x)
        
        x = self.output(x)

        return x.reshape(batch_size, -1, self.p)