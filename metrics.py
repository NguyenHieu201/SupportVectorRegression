import torch

def MAPE(x, y):
    return torch.mean(torch.abs((y - x) / y)) * 100.0 

def SMAPE(x, y):
    return torch.mean(torch.abs(y - x)/((x + y)/2)) * 100.0 

def RMSE(x, y):
    criterion = torch.nn.MSELoss()
    return torch.sqrt(criterion(x, y))

def R2(x, y):
    target_mean = torch.mean(y)
    ss_tot = torch.sum((y - target_mean) ** 2)
    ss_res = torch.sum((y - x) ** 2)
    
    r2 = 1 - ss_res / ss_tot
    
    return r2