import os
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

from utils.preprocess import down_sample


def point_adjust_k(scores, targets, thres, k=20):
    """
    :param scores: anomaly score
    :param targets: target label
    :param thres: threshold
    :param k: ratio to apply point adjust(%), 0 equals to conventional point adjust
    :return: point adjusted scores
    """
    # print("Point adjust renew with K: {}".format(k))
    try:
        scores = np.asarray(scores)
        targets = np.asarray(targets)
    except TypeError:
        scores = np.asarray(scores.cpu())
        targets = np.asarray(targets.cpu())

    T = scores.shape
    predict = scores > thres
    actual = targets > 0.1

    one_start_idx = np.where(np.diff(actual, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actual, prepend=0) == -1)[0]
    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        predict = np.append(predict, 1)
        zero_start_idx = np.append(zero_start_idx, -1)

    for i in range(len(one_start_idx)):
        if predict[one_start_idx[i]: zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predict[one_start_idx[i]: zero_start_idx[i]] = 1

    return predict[:T[0]]


def best_threshold_search(scores, gt_labels, start, end=None, step_num=1):
    """
    Find best F1 score
    """
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
        percentile = np.round(percentile, 2)
        threshold = np.percentile(scores, percentile)
        pred_labels = np.zeros(len(scores))
        pred_labels[scores > threshold] = 1
        target = f1_score(gt_labels, pred_labels)
        if target > best:
            best_threshold = threshold
            best_percentile = percentile
            best = target

    # best_threshold = np.percentile(scores, best_threshold)
    pred_labels[scores > best_threshold] = 1
    # print(f'best percentile={best_percentile}, threshold={best_threshold}')
    print(f'{best_percentile},{best_threshold}')

    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    # print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
    acc = [precision, recall, f1]

    return best, best_threshold, acc


############################# start ####################################

# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/wadi/'
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test).T
#     x_test = down_sample(x_test, 5)
#     y_test = np.asarray(x_test[:, -1])
#     x_test = x_test[:, :-1]
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test).T
#     y_test = np.asarray(x_test[:, -1])
#     x_test = x_test[:, :-1]
# with open(os.path.join(path, 'label.pkl'), 'rb') as f:
#     y_test = np.asarray(pickle.load(f))
#     y_test = down_sample(y_test, 10)

# scores = np.load(f'/home/kj21.choi/PycharmProjects/ANTLAD/result/swat.npy')
# path_scores = '/home/kj21.choi/PycharmProjects/ANTLAD/result/siwon_OmniAnomaly/'
# with open(os.path.join(path_scores, 'wadi.pkl'), 'rb') as f:
#     scores = np.asarray(pickle.load(f))
#
# pred_labels = np.zeros(len(scores))
# T = len(pred_labels)
# gt_labels = np.array(y_test[:T])
# for i in range(T):
#     pred_labels[i] = int(pred_labels[i])
#     gt_labels[i] = int(gt_labels[i])
#
# # best threshold search
# best_f1, thres, acc = best_threshold_search(scores, gt_labels, 85, 100, step_num=300)
# # print(f'best F1-score:{best_f1}, threshold:{thres}, accuracy={acc}')
#
# list_k = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# for item in list_k:
#     # print('K:{}'.format(item))
#
#     point_adjusted_scores = point_adjust_k(scores, gt_labels, thres, k=item)
#     precision = precision_score(gt_labels, point_adjusted_scores)
#     recall = recall_score(gt_labels, point_adjusted_scores)
#     f1 = f1_score(gt_labels, point_adjusted_scores)
#     # print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
#     print(f'{precision:.4f}, {recall:.4f}, {f1:.4f}')


path = './data/smd/'
server_list = ['machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5', 'machine-1-6','machine-1-7', 'machine-1-8',
               'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5', 'machine-2-6', 'machine-2-7', 'machine-2-8', 'machine-2-9',
               'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5', 'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11']
for machine in server_list:
    print(f'{machine} PAK results:')
    with open(os.path.join(path, f'{machine}_label.pkl'), 'rb') as f:
        y_test = np.asarray(pickle.load(f))
        y_test = down_sample(y_test, 5)

    # scores = np.load(f'/home/kj21.choi/PycharmProjects/ANTLAD/result/siwon_MSCRED/{machine}_scores.npy')
    path_scores = '/home/kj21.choi/PycharmProjects/ANTLAD/result/siwon_OmniAnomaly/'
    with open(os.path.join(path_scores, f'd_{machine}.pkl'), 'rb') as f:
        scores = np.asarray(pickle.load(f))

    pred_labels = np.zeros(len(scores))
    T = len(pred_labels)
    gt_labels = np.array(y_test[:T])
    for i in range(T):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    # best threshold search
    best_f1, thres, acc = best_threshold_search(scores, gt_labels, 85, 100, step_num=150)
    # print(f'best F1-score:{best_f1}, threshold:{thres}, accuracy={acc}')

    list_k = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for item in list_k:
        # print('K:{}'.format(item))

        point_adjusted_scores = point_adjust_k(scores, gt_labels, thres, k=item)
        precision = precision_score(gt_labels, point_adjusted_scores)
        recall = recall_score(gt_labels, point_adjusted_scores)
        f1 = f1_score(gt_labels, point_adjusted_scores)
        # print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
        print(f'{precision:.4f}, {recall:.4f}, {f1:.4f}')