from Models import MLP, LSTM
from dataloader import get_set_and_loader

import pandas as pd

def get_data(path):
    df = pd.read_csv(path)
    return df.close.to_numpy().reshape(-1,1)

def prepare_dataset(data, l, p, val, test, batch_size, shuffle):
    # 60-20-20 split
    training_ind = int(len(data) * (1 - val - test))
    validating_ind = int(len(data) * (1 - test))
    
    training_set_and_loader = get_set_and_loader(data[:training_ind], l, p, batch_size, shuffle)
    validation_set_and_loader = get_set_and_loader(data[training_ind:validating_ind], l, p, batch_size, False)
    test_set_and_loader = get_set_and_loader(data[validating_ind:], l, p, batch_size=0, shuffle=False)

    return training_set_and_loader, validation_set_and_loader, test_set_and_loader

def get_model(model):
    if model == 'MLP':
        return MLP.MLP
    if model == 'LSTM':
        return LSTM.LSTM