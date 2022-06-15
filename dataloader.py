import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data, l, p):
        self.l = l
        self.p = p
        
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(data)
        
        
    def reverse_normalize(self, x):
        x = x.reshape(-1,1).to('cpu')
        x = self.scaler.inverse_transform(x)

        x = torch.tensor(x).to(dtype=torch.float32)
        return x
        

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.l].reshape(1,-1)
        y = self.data[idx+self.l:idx+self.l+self.p].squeeze()
        return x, y
    

    def __len__(self):
        return len(self.data) - self.l - self.p + 1
    
def collate_fn(batch):
    x ,y = zip(*batch)
    x = np.array(x)
    y = np.array(y)
    
    b = y.shape[0]
    d = x.shape[1]
    y = y.reshape(b,d,-1)
    
    x = torch.tensor(x).to(dtype=torch.float32)
    y = torch.tensor(y).to(dtype=torch.float32)
    
    return x, y

def get_set_and_loader(data, l, p, batch_size = 64, shuffle = True):
    dataset = StockDataset(data, l, p)

    if batch_size == 0:
        batch_size = len(dataset)
        
    loader = DataLoader(dataset = dataset, 
                        batch_size = batch_size, 
                        shuffle = shuffle, 
                        collate_fn=collate_fn)

    return dataset, loader