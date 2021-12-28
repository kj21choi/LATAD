"""
Adversarial Contrastive Learning (ACL) for anomaly detection in time series via self-supervised learning representation
"""
import argparse

import pytorch_model_summary
import torch
import sys
import pickle
import os
import random
import pandas as pd
from pandas import DataFrame

from model.model import MyModel
from utils.preprocess import MyDataset, get_sliding_data, down_sample, out_iqr
from gat_transformer_tcn_train_smd import train
from gat_transformer_tcn_evaluate_smd import evaluate_coreset, evaluate_euclidean, evaluate_reconstruction, evaluate_recon_coreset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def run(args=None, machine='machine-1-1'):
    is_train, data_type, w, augmentation, window, feature, device, batch, step, lr, n_epochs \
        = args.train, args.data, args.weight, args.augmentation, args.window, args.feature, args.device, args.batch, args.step, args.lr, args.n_epochs

    print('Data load')
    path = ''
    if data_type == 'simulated':
        path = f'./data/{data_type}/'
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(path, 'label.pkl'), 'rb') as f:
            y_test = pickle.load(f)
    elif data_type == 'swat':
        path = f'/home/kj21.choi/hdd/04_AAAI/THOC/{data_type}/'
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.asarray(x_train).T
            # x_train = x_train[:, :-1]  # wadi
            x_train = down_sample(x_train[100000:], 10)  # swat: initial 100000 points are unstable
            # x_train = down_sample(x_train, 10)  # wadi
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.asarray(x_test).T
            x_test = down_sample(x_test, 10)
            y_test = np.asarray(x_test[:, -1])
            x_test = x_test[:, :-1]
            x_test[:, 5] = x_test[:, 4]  # swat special case
            # x_test[:, 96] = x_test[:, 95]  # wadi special case
    elif data_type == 'wadi':
        path = f'/home/kj21.choi/hdd/04_AAAI/THOC/{data_type}/'
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.asarray(x_train).T
            x_train = x_train[:, :-1]  # wadi
            x_train = down_sample(x_train, 10)  # wadi
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.asarray(x_test).T
            x_test = down_sample(x_test, 10)
            y_test = np.asarray(x_test[:, -1])
            x_test = x_test[:, :-1]
            x_test[:, 96] = x_test[:, 95]  # wadi special case
    elif data_type == 'msl':
        path = f'/home/kj21.choi/hdd/04_AAAI/THOC/{data_type}/'
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.asarray(x_train)
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.asarray(x_test)
            x_test = down_sample(x_test, 10)
        with open(os.path.join(path, 'label.pkl'), 'rb') as f:
            y_test = np.asarray(pickle.load(f))
            y_test = down_sample(y_test, 10)
    elif data_type == 'smd':
        path = f'./data/{data_type}/'
        with open(os.path.join(path, f'{machine}_train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
            x_train = np.asarray(x_train)
            x_train = down_sample(x_train, 5)
        with open(os.path.join(path, f'{machine}_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
            x_test = np.asarray(x_test)
            x_test = down_sample(x_test, 5)
        with open(os.path.join(path, f'{machine}_label.pkl'), 'rb') as f:
            y_test = np.asarray(pickle.load(f))
            y_test = down_sample(y_test, 5)

    print('Data pre-process start')
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    df = DataFrame(x_train)
    iqr = df.apply(out_iqr, k=1.5)
    for column in df:
        df[column] = np.where(iqr[column] is True, 'NaN', df[column])
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.interpolate(method='linear', axis=0).bfill().ffill()
    x_train = df.to_numpy(dtype=float)

    # x_train[:, 0] = x_train[:, 1]  # msl special case

    trn_x, trn_y, trn_ind = get_sliding_data(window, x_train, data_y=None, step=step)
    tst_x, tst_y, tst_ind = get_sliding_data(window, x_test, data_y=y_test, step=step)

    train_data = MyDataset(trn_x, trn_y, trn_ind, is_train)   # add argument: is_train
    test_data = MyDataset(tst_x, tst_y, tst_ind, is_train)    # add argument: is_train
    print('Data pre-process end')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    model = MyModel(feature, window, kernel_size=5, dropout=0.2, d_model=128)
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    # src = torch.zeros(batch, window, feature)
    # print(pytorch_model_summary.summary(model, src, show_input=True))

    if is_train:
        train(train_data,
              model,
              lr=lr,
              decay=1e-3,
              n_epochs=n_epochs,
              augmentation=augmentation,
              path=data_type,
              feature=feature,
              device=device,
              batch_size=batch,
              step=step,
              machine=machine
              )
    else:
        # Plot the distribution of the encodings and use the learnt encoders to train a downstream classifier
        model.load_state_dict(torch.load('./checkpoint/%s/model_%s_b%d_n%d.pt' % (data_type, machine, batch, augmentation)))
        model.eval()
        evaluate_coreset(train_data, test_data, model, y_test, data_type, device, machine)
        # evaluate_reconstruction(test_data, encoder, decoder, y_test, data_type, device)


if __name__ == '__main__':
    random.seed(5)  # 0 ~ 5
    parser = argparse.ArgumentParser(description='Run LATAD (Learnable Augmentation-based Time-series Anomaly Detection')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--data', type=str, default='swat')
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--augmentation', type=int, default=3)  # the number of positive/negative augmentation
    parser.add_argument('--window', type=int, default=100)  # the number of negative samples
    parser.add_argument('--step', type=int, default=10)  # the number of negative samples
    parser.add_argument('--feature', type=int, default=51)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    args_summary = str(args.__dict__)
    print('LATAD model with w=%s' % args_summary)
    #
    server_list = ['machine-1-5']
    # server_list = [ 'machine-1-1','machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5', 'machine-1-6','machine-1-7', 'machine-1-8',
    #                 'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8', 'machine-2-9',
    #                 'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11']
    #
    # server_list = ['machine-1-4', 'machine-1-5', 'machine-1-7', 'machine-2-2', 'machine-3-3', 'machine-3-9', 'machine-3-11']
    for machine in server_list:
        run(args, machine)
