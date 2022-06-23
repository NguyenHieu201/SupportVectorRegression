from random import shuffle
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

from utils import *
from params import * 
from Models.MultiSrcTL.MultiSrcTL import CustomMultiSrcTL, weight_data_mmd

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--val_pct', type=float, default=0.2)
parser.add_argument('--test_pct', type=float, default=0.2)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--model', type=str)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
val_pct = args.val_pct
test_pct = args.test_pct
shuffle = args.shuffle
model = args.model
mode = args.mode

def indicator(metrics_dict, x, y):
    result = {}
    for metric, metric_fn in metrics_dict.items():
        result[metric] = metric_fn(x, y).item()

    return result


def build_path(path):
    try:
        if not os.path.exists(f'./Results/{path}'):
            os.makedirs(f'./Results/{path}')
    except FileExistsError:
        # directory already exists
        pass
    try:
        if not os.path.exists(f'./Saved models/{path}'):
            os.makedirs(f'./Saved models/{path}')
    except FileExistsError:
        # directory already exists
        pass


filename = f'model_{model}_epochs_{epochs}_batch_{batch_size if batch_size > 0 else "FULL"}_val_{val_pct}_test_{test_pct}_shuffle_{shuffle}'
build_path(filename)

def train_model(net, src_train_set, tar_train_set):
    # Get dataset and dataloader for source training and target training
    src_train_dataset, src_train_dataloader = src_train_set
    tar_train_dataset, tar_train_dataloader = tar_train_set

    for src_x, src_y in src_train_dataloader:
        for tar_x, tar_y in tar_train_dataloader:
            src_x = src_x.squeeze()
            src_y = src_y.squeeze()
            tar_x = tar_x.squeeze()
            tar_y = tar_y.squeeze()

            # Training phase
            result, mmd = weight_data_mmd(source_data=src_x, target_data=tar_x, gamma=0.1)
            sample_weight = result.x
            net.fit(src_x, src_y, sample_weight)
            return net, mmd, {
                'input-data': src_x,
                'output-data': src_y
            }

def mlti_tl_transfer_phase(tar_path, src_data, src_mmd, train_data, train_label, test_data, test_label, y_scaler_target, fig_path):
    # Transfer phase
    custom_mlti = CustomMultiSrcTL(src_path=tar_path,
                                    src_data=src_data,
                                    src_mmd=src_mmd)
    custom_mlti.fit(train_data, train_label)

    pred = y_scaler_target.inverse_transform(custom_mlti.predict(test_data))
    true = y_scaler_target.inverse_transform(test_label.reshape(-1, 1))
    plt.plot(true, label='Ground truth')
    plt.plot(pred, label='Prediction')
    plt.legend()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()

    return pred, true, custom_mlti


def run():
    result = {}
    for student in params['students']:
        # read data feature
        print(student)
        tar_data = get_data(f"{params['target path']}/{student}.csv")
        tar_train, tar_val, tar_test = prepare_dataset(tar_data, 
                                                       params['input size'], params['output size'], 
                                                       val_pct, test_pct, batch_size, shuffle)
        src_mmd = {}
        src_data = {}
        build_path(f"{filename}/{student}")
        tar_path = f"./Saved models/{filename}/{student}"
        fig_path = f"./Results/{filename}/{student}.png"
        train_data, train_label, test_data, test_label = None, None, None, None
        for x, y in tar_train[1]:
            train_data = x.squeeze()
            train_label = y.squeeze()

        for x, y in tar_test[1]:
            test_data = x.squeeze()
            test_label = y.squeeze()
        for teacher in params['teachers']:
            # read data feature
            print(f"\t{teacher}")
            src_data_set = get_data(f"{params['source path']}/{teacher}.csv")
            src_train, src_val, src_test = prepare_dataset(src_data_set, 
                                                           params['input size'], params['output size'], 
                                                           val_pct, test_pct, batch_size, shuffle)
            net = get_model(model)()
            net, mmd, data = train_model(net, src_train, tar_train)
            src_mmd[teacher] = mmd
            src_data[teacher] = data
            save_path = f"{tar_path}/{teacher}"
            pickle.dump(net, open(save_path, 'wb'))

        # Transfer phase
        y_scaler_target = tar_test[0].scaler
        pred, true, custom_mlti = mlti_tl_transfer_phase(tar_path, 
                                            src_data, src_mmd, 
                                            train_data, train_label, 
                                            test_data, test_label, 
                                            y_scaler_target, fig_path)
        
        # Saved model
        with open(f'./Saved models/{filename}/{student}', 'wb') as model_file:
            pickle.dump(custom_mlti, model_file)

        # Save metric result
        metrics = params['metrics']
        results = indicator(metrics, torch.Tensor(pred), torch.Tensor(true))

        for metric, val in results.items():
            print(f'{metric}: {val:10.8f}')
        
        result[student] = results
    result_table = pd.DataFrame.from_dict(result, orient='index')
    result_table.to_csv(f'./Results/{filename}/result.csv')
        
        
run()          