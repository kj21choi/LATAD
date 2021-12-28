import pickle

import torch
import torch.nn as nn
import numpy as np
import os

import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from model.model import EncoderRNN, Transformation
from utils.preprocess import MyDataset, get_sliding_data, get_loaders, down_sample
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader
from scipy import stats
from scipy.spatial import distance
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from datetime import datetime


def similarity(x1, x2):
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    return torch.exp(cosine_similarity(x1, x2) * 1 / 1)  # 1 ~ e


def mahalanobis(x=None, mean=None, iv=None):
    delta = x - mean
    left = np.dot(delta, iv)
    mahal_dist = np.dot(left, delta.T)
    return np.sqrt(mahal_dist)


def adjust_predicts(score, label):
    """
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    predict = score > 0.1
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict.astype(np.float32)


def save_result(path, acc, gt_labels, pred_labels, scores, threshold, d_threshold=None, machine_name=''):
    # Save performance plots
    if not os.path.exists('./plots/%s' % path):
        os.mkdir('./plots/%s' % path)
    T = len(pred_labels)
    labels = [gt for gt in gt_labels]
    preds = [pred for pred in pred_labels]
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.set_xticks(np.arange(0, T, 5000))
    ax2.set_xticks(np.arange(0, T, 5000))
    ax3.set_xticks(np.arange(0, T, 5000))

    ax1.plot(np.arange(T), labels, label="ground truth")
    ax2.plot(np.arange(T), preds, label="prediction")

    threshold = np.repeat(threshold, T)
    ax3.plot(np.arange(T), scores, label="anomaly score")
    ax3.plot(np.arange(T), threshold, label="threshold")

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    plt.savefig(os.path.join("./plots/%s" % path, f'{machine_name}_result.pdf'))
    np.savetxt(os.path.join("./plots/%s" % path, f'{machine_name}_accuracy.txt'), acc, fmt='%.4f')


def best_threshold_search(scores, gt_labels, start, end=None, step_num=1, machine_name='machine-1-1'):
    """
    Find best F1 score
    """
    print(machine_name)
    if step_num is None or end is None:
        end = start
        step_num = 1.
    search_step, search_range, search_lower_bound = step_num, end - start, start
    percentile = search_lower_bound

    pred_labels = np.zeros(len(scores))
    best = -1.
    best_threshold = 0.0
    best_percentile = 0.0
    for i in range(search_step):
        percentile += search_range / search_step
        percentile = np.round(percentile, 1)
        threshold = np.percentile(scores, percentile)
        pred_labels = np.zeros(len(scores))
        pred_labels[scores > threshold] = 1
        # adj_pred_labels = adjust_predicts(pred_labels, gt_labels)
        target = f1_score(gt_labels, pred_labels)
        if target > best:
            best_threshold = threshold
            best_percentile = percentile
            best = target

    # best_threshold = np.percentile(scores, best_threshold)
    pred_labels[scores > best_threshold] = 1
    print(f'best percentile={best_percentile}, threshold={best_threshold}')

    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
    acc = [precision, recall, f1]

    adj_pred_labels = adjust_predicts(pred_labels, gt_labels)
    adj_precision = precision_score(gt_labels, adj_pred_labels)
    adj_recall = recall_score(gt_labels, adj_pred_labels)
    adj_f1 = f1_score(gt_labels, adj_pred_labels)
    print(f'adj_precision={adj_precision:.4f}, adj_recall={adj_recall:.4f}, adj_f1-score={adj_f1:.4f}')
    acc.append(adj_precision)
    acc.append(adj_recall)
    acc.append(adj_f1)

    # precision_curve, recall_curve, threshold_curve = precision_recall_curve(gt_labels, scores)
    # plt.plot(recall_curve, precision_curve, label='LATAD')
    # plt.axhline(precision, color='r', linestyle='--')
    # plt.axvline(recall, color='orange', linestyle='--')
    # plt.legend()
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.show()

    save_result('smd', acc, gt_labels, pred_labels, scores, best_threshold, machine_name=machine_name)

    return best, best_threshold, acc


def out_iqr(s, k=1.5, return_threshold=False):
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_threshold:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]


# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/wadi/'
#
# with open(os.path.join(path, 'train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
#     x_train = np.asarray(x_train).T
#     x_train = down_sample(x_train, 10)
#     y_train = np.asarray(x_train[:, -1])
#     x_train = x_train[:, :-1]
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test).T
#     x_test = down_sample(x_test, 10)
#     y_test = np.asarray(x_test[:, -1])
#     x_test = x_test[:, :-1]
#     x_test[:, 96] = x_test[:, 95]  # wadi special case
# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/smd/'
path = './data/smd/'
server_list = [ 'machine-1-1','machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5', 'machine-1-6','machine-1-7', 'machine-1-8',
                    'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8', 'machine-2-9',
                    'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11']
# server_list = ['machine-1-5']
for machine in server_list:
    with open(os.path.join(path, f'{machine}_train.pkl'), 'rb') as f:
        x_train = np.asarray(pickle.load(f))
        x_train = down_sample(x_train, 5)
    with open(os.path.join(path, f'{machine}_test.pkl'), 'rb') as f:
        x_test = np.asarray(pickle.load(f))
        x_test = down_sample(x_test, 5)
    with open(os.path.join(path, f'{machine}_label.pkl'), 'rb') as f:
        y_test = np.asarray(pickle.load(f))
        y_test = down_sample(y_test, 5)
    # preprocess
    # scaler = MinMaxScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    # df = DataFrame(x_train)
    # iqr = df.apply(out_iqr, k=1.5)
    # for column in df:
    #     df[column] = np.where(iqr[column] is True, 'NaN', df[column])
    # df[column] = df[column].apply(pd.to_numeric, errors='coerce')
    # df = df.interpolate(method='linear', axis=0).bfill().ffill()
    # x_train = df.to_numpy(dtype=float)
    #
    # plt.figure()
    # plt.plot(x_train)
    # plt.show()
    # plt.close()
    #
    # plt.figure()
    # plt.plot(x_test)
    # plt.show()

    scores = np.load(f'/home/kj21.choi/PycharmProjects/ANTLAD/result/smd/{machine}_score.npy')
    pred_labels = np.zeros(len(scores))
    T = len(pred_labels)
    gt_labels = np.array(y_test[:T])
    for i in range(T):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    scores[100:] = scores[:-100]
    # best threshold search
    best_threshold_search(scores, gt_labels, 97, 100, step_num=30, machine_name=machine)

    # peaks over threshold
    base_threshold = 88
    window_size = 5000
    # dynamic_threshold(scores, gt_labels, base_threshold, window_size)

    end = datetime.now()
    print('end:', end)


