from metrics import *
import torch

params = {
    'source path': './Data/stock_data/data_for_teacher',
    'target path': './Data/stock_data/data_for_student',
    'teachers': ['ACB', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'HDB', 'HPG', 'KDH', 
                 'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'PNJ', 'SAB', 'SSI', 
                 'STB', 'VCB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'],
    'students': ['GVR', 'POW', 'TCB', 'TPB', 'VHM'],
    'input size': 22,
    'output size': 1,
    
    'hyperparameters': {
        'l': 22,
        'p': 1,
        'BETA': 0.1,
        'SIGMA': 0.001, 
        'p_level': 1, 
        'conf_base': 0.06, 
        'confidence': 0.5

    },
    'optimizer': torch.optim.SGD,
    'lr': 0.001,
    'criterion': torch.nn.MSELoss(),
    'metrics': {
        'MAPE': MAPE,
        'SMAPE': SMAPE,
        'RMSE': RMSE,
        'R2': R2
    },
    
}