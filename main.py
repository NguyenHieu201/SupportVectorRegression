from random import shuffle
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import *
from params import params

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--val_pct', type=float, default=0.2)
parser.add_argument('--test_pct', type=float, default=0.2)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--model', type=str)
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()

device = 'cpu'
if torch.cuda.is_available() and args.cuda:
    device = 'cuda'

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

epochs = args.epochs
batch_size = args.batch_size
val_pct = args.val_pct
test_pct = args.test_pct
shuffle = args.shuffle
model = args.model

assert batch_size >= 0
assert val_pct > 0 and val_pct < 1
assert test_pct > 0 and test_pct < 1

def indicator(metrics_dict, x, y):
    result = {}
    for metric, metric_fn in metrics_dict.items():
        result[metric] = metric_fn(x, y).item()

    return result

def build_model(model, model_params):
    net = get_model(model)(model_params)

    if torch.cuda.is_available():
        net.cuda()

    optimizer = params['optimizer'](net.parameters(), lr=params['lr'])

    #print(net)
    
    return net, optimizer

def train_model(net, optimizer, criterion, metrics, epochs, training_set, validation_set, device, savepath):
    results = {}

    for metric in metrics.keys():
        results[metric] = {
            'train': np.zeros(epochs),
            'val': np.zeros(epochs)
        }

    results['criterion'] = {
        'train': np.zeros(epochs),
        'val': np.zeros(epochs)
    }
    
    train_dataset, train_dataloader = training_set
    val_dataset, val_dataloader = validation_set

    for epoch in tqdm(range(epochs)):    
        optimizer.zero_grad()

        result = {}

        for metric in metrics.keys():
            result[metric] = []
            
        result['criterion'] = []
        
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            pred = net(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            
            result['criterion'].append(loss.item())
            
            pred = train_dataset.reverse_normalize(pred.detach())
            y = train_dataset.reverse_normalize(y)
            
            metric_result = indicator(metrics, pred, y)
            for metric in metrics.keys():
                result[metric].append(metric_result[metric])

        for metric in results.keys():
            res = sum(result[metric])/len(result[metric])
            results[metric]['train'][epoch] = res

        if epoch == 0 or (epoch + 1) % (epochs // 10) == 0:
            print(f'Epoch: {epoch+1:3}')
            print(f'Training loss: {results["criterion"]["train"][epoch]:10.8f}')

        with torch.no_grad():           
            val_result = {}

            for metric in metrics.keys():
                val_result[metric] = []
            
            val_result['criterion'] = []

            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_pred = net(val_x)
                val_loss = criterion(val_pred, val_y)
                
                val_result['criterion'].append(val_loss.item())

                val_pred = val_dataset.reverse_normalize(val_pred.detach())
                val_y = val_dataset.reverse_normalize(val_y)

                val_metric_result = indicator(metrics, val_pred, val_y)
                for metric in metrics.keys():
                    val_result[metric].append(val_metric_result[metric])

            for metric in results.keys():
                res = sum(val_result[metric])/len(val_result[metric])
                results[metric]['val'][epoch] = res

            if epoch == 0 or (epoch + 1) % (epochs // 10) == 0:
                print(f'Validation loss: {results["criterion"]["val"][epoch]:10.8f}')
        
        torch.save(net.state_dict(), savepath)
            
    return results

def visualize_result(metric, name, path):
    epoch = range(1, len(metric['train'])+1)

    plt.plot(epoch, metric['train'], label='Training')
    plt.plot(epoch, metric['val'], label='Validating')

    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def test_model(net, criterion, metrics, test_set, path, device):
    net.eval()
    
    test_dataset, test_dataloader = test_set
    
    x, y = next(iter(test_dataloader))
    x = x.to(device)
    y = y.to(device)

    pred = net(x)
    
    loss = criterion(pred, y).item()
    print(f'Loss: {loss:10.8f}')

    pred = test_dataset.reverse_normalize(pred.detach())
    y = test_dataset.reverse_normalize(y)   
    
    results = indicator(metrics, pred, y)

    for metric, val in results.items():
        print(f'{metric}: {val:10.8f}')

    label = y.numpy().flatten()
    pred = pred.numpy().flatten()

    plt.plot(range(len(label)), label, label='Ground truth')
    plt.plot(range(len(label)), pred, label='Prediction')

    plt.title('Test')
    plt.xlabel('Session')
    plt.ylabel('Close')
    plt.legend()

    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    
    return results

def build_and_train(model, params, training_set, validation_set, test_set, epochs, device, name):
    net, optimizer = build_model(model, params['hyperparameters'])
    print(net)

    results = train_model(net, optimizer, params['criterion'], params['metrics'], epochs, training_set, validation_set, device, f'Saved models/{name}.pth')
    test_result = test_model(net, params['criterion'], params['metrics'], test_set, f'Results/{name}_test.png', device)

    for metric, pack in results.items():
        visualize_result(pack, metric, f'Results/{name}_{metric}.png')

    return test_result

filename = f'model_{model}_epochs_{epochs}_batch_{batch_size}_val_{val_pct}_test_{test_pct}_shuffle_{shuffle}'
try:
    if not os.path.exists(f'./Results/{filename}'):
        os.makedirs(f'./Results/{filename}')
except FileExistsError:
    # directory already exists
    pass
try:
    if not os.path.exists(f'./Saved models/{filename}'):
        os.makedirs(f'./Saved models/{filename}')
except FileExistsError:
    # directory already exists
    pass

result = {}

for teacher in params['students']:
    data = get_data(f"{params['target path']}/{teacher}.csv")
    training_set, validation_set, test_set = prepare_dataset(data, params['input size'], params['output size'], val_pct, test_pct, batch_size, shuffle)

    teacher_filename = f'{filename}/{teacher}'
    result[teacher] = build_and_train(model, params, training_set, validation_set, test_set, epochs, device, teacher_filename)

result_table = pd.DataFrame.from_dict(result, orient='index')
result_table.to_csv(f'./Results/{filename}/result.csv')