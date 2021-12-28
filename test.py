import pickle

import torch
import torch.nn as nn
import numpy as np
import os

import pandas as pd
from matplotlib.dates import DateFormatter
from pandas import DataFrame
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from model.model import EncoderRNN, Transformation, MyModel
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


def save_result(path, acc, gt_labels, pred_labels, scores, threshold, d_threshold=None):
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
    plt.savefig(os.path.join("./plots/%s" % path, "result.pdf"))
    np.savetxt(os.path.join("./plots/%s" % path, "accuracy.txt"), acc, fmt='%.4f')
    np.save(os.path.join("./result/%s" % path, "score.npy"), scores)
    np.save(os.path.join("./result/%s" % path, "preds.npy"), preds)



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
        adj_pred_labels = adjust_predicts(pred_labels, gt_labels)
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

    precision_curve, recall_curve, threshold_curve = precision_recall_curve(gt_labels, scores)
    plt.plot(recall_curve, precision_curve, label='LATAD')
    plt.axhline(precision, color='r', linestyle='--')
    plt.axvline(recall, color='orange', linestyle='--')
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

    save_result('swat', acc, gt_labels, pred_labels, scores, best_threshold)

    pd.DataFrame(adj_pred_labels).to_csv("adj_preds.csv")

    return best, best_threshold, acc


# def dynamic_threshold(scores, gt_labels, base_threshold, window_size):
#     pred_labels = np.zeros(len(scores))
#     base_percentile = np.percentile(scores, base_threshold)
#     d_threshold = np.repeat(base_percentile, len(scores))
#     for i in range(int(len(scores) / window_size) + 1):
#         base_percentile = np.percentile(scores, base_threshold)
#         start = i * window_size
#         end = start + window_size if start + window_size < len(scores) else len(scores) - 1
#         sub_scores = scores[start: end]
#         sub_scores = sub_scores[sub_scores > base_percentile]
#         sub_pred_labels = np.zeros(end - start)
#         if len(sub_scores) > 0:
#             local_percentile = np.percentile(sub_scores, 5)
#             if i == 4:
#                 local_percentile = np.percentile(sub_scores, 0)
#             sub_pred_labels[scores[start: end] > local_percentile] = 1
#             pred_labels[start: end] = sub_pred_labels
#             d_threshold[start: end] = local_percentile
#         else:
#             d_threshold[start: end] = local_percentile
#         if i < 2:
#             base_threshold += 0.5
#         elif i < 4:
#             base_threshold -= 1
#             base_threshold *= 1.03
#         elif i < 5:
#             base_threshold *= 0.996
#         elif i < 6:
#             base_threshold *= 0.99
#
#     precision = precision_score(gt_labels, pred_labels)
#     recall = recall_score(gt_labels, pred_labels)
#     f1 = f1_score(gt_labels, pred_labels)
#     print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
#     acc = [precision, recall, f1]
#
#     adj_pred_labels = adjust_predicts(pred_labels, gt_labels)
#     adj_precision = precision_score(gt_labels, adj_pred_labels)
#     adj_recall = recall_score(gt_labels, adj_pred_labels)
#     adj_f1 = f1_score(gt_labels, adj_pred_labels)
#     print(f'adj_precision={adj_precision:.4f}, adj_recall={adj_recall:.4f}, adj_f1-score={adj_f1:.4f}')
#     acc.append(adj_precision)
#     acc.append(adj_recall)
#     acc.append(adj_f1)
#
#     precision_curve, recall_curve, threshold_curve = precision_recall_curve(gt_labels, scores)
#     plt.plot(recall_curve, precision_curve, label='LATAD')
#     plt.axhline(precision, color='r', linestyle='--')
#     plt.axvline(recall, color='orange', linestyle='--')
#     plt.legend()
#     plt.xlabel('recall')
#     plt.ylabel('precision')
#     plt.show()
#
#     save_result('smap', acc, gt_labels, pred_labels, scores, base_percentile, d_threshold)
#
#     return None


def out_iqr(s, k=1.5, return_threshold=False):
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_threshold:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]


path = f'/home/kj21.choi/hdd/04_AAAI/THOC/swat/'
with open(os.path.join(path, 'train.pkl'), 'rb') as f:
    x_train = pickle.load(f)
    x_train = np.asarray(x_train).T
    x_train = x_train[100000:]
    # x_train = down_sample(x_train[100000:], 10)  # swat: initial 100000 points are unstable
with open(os.path.join(path, 'test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
    x_test = np.asarray(x_test).T
    # x_test = down_sample(x_test, 10)
    y_test = np.asarray(x_test[:, -1])
    x_test = x_test[:, :-1]
    x_test[:, 5] = x_test[:, 4]
# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/wadi/'
# with open(os.path.join(path, 'train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
#     x_train = np.asarray(x_train).T
#     x_train = x_train[:, :-1]  # wadi
#     x_train = down_sample(x_train, 10)  # wadi
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test).T
#     x_test = down_sample(x_test, 10)
#     y_test = np.asarray(x_test[:, -1])
#     x_test = x_test[:, :-1]
#     x_test[:, 96] = x_test[:, 95]


#     x_test[:, 96] = x_test[:, 95]  # wadi special case
# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/msl/'
# path = f'./data/kdd/'
# with open(os.path.join(path, 'train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
#     x_train = np.asarray(x_train)
#     x_train[:, 0] = x_train[:, 1]  # msl special case
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test)
#     x_test = down_sample(x_test, 10)
# with open(os.path.join(path, 'label.pkl'), 'rb') as f:
#     y_test = np.asarray(pickle.load(f))
#     y_test = down_sample(y_test, 10)
# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/kdd/'
# with open(os.path.join(path, 'train.npy'), 'rb') as f:
#     x_train = np.load(f)
#     y_train = x_train[:, -1]
#     x_train = x_train[:, :-1]
#     x_train = down_sample(x_train, 10)
# with open(os.path.join(path, 'test.npy'), 'rb') as f:
#     x_test = np.load(f)
#     y_test = x_test[:, -1]
#     x_test = x_test[:, :-1]
#     x_test = down_sample(x_test, 10)
# path = f'/home/kj21.choi/hdd/04_AAAI/THOC/smap/'
# with open(os.path.join(path, 'train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
#     x_train = np.asarray(x_train)
    # x_train = down_sample(x_train, 10)
# with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#     x_test = pickle.load(f)
#     x_test = np.asarray(x_test)
#     x_test = down_sample(x_test, 10)
# with open(os.path.join(path, 'label.pkl'), 'rb') as f:
#     y_test = np.asarray(pickle.load(f))
#     y_test = down_sample(y_test, 10)
# preprocess

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# for i in range(x_train.shape[0]):
#     for j in range(x_train.shape[1]):
#         x_train[i,j] = x_train[i,j] + np.random.random()*1e-4
x_test = scaler.transform(x_test)
#
# df = DataFrame(x_train)
# iqr = df.apply(out_iqr, k=1.5)
# for column in df:
#     df[column] = np.where(iqr[column] is True, 'NaN', df[column])
# cols = df.columns
# df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
# df = df.interpolate(method='linear', axis=0).bfill().ffill()

# x_train = df
# x_train = x_train.apply(pd.to_numeric, errors='coerce')
# x_train.fillna(method='ffill')

# plt.figure()
# plt.plot(x_train)
# plt.show()
# plt.close()


#############################Correlation test #############################
# import seaborn as sns
# from matplotlib import cm
# colormap = plt.cm.coolwarm
# colormap.set_bad("lightgrey")
# plt.figure(figsize=(10, 8))
# plt.title("Pearson Correlation of Features", y=1.05, size=12)
# sns.heatmap(x_train.astype(float).corr(), linewidths=0.01, vmin=-1.0, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=False, annot_kws={"size":16})
#
# plt.show()







# window, step, feature = 100, 10, x_train[1].size
# trn_x, trn_y, trn_ind = get_sliding_data(window, x_train, data_y=None, step=step)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'device:{device}')
# model = MyModel(feature, window, kernel_size=5, dropout=0.2, d_model=128)
# if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
#     model = nn.DataParallel(model, device_ids=[0, 1, 2])
# data_type, batch, augmentation = 'swat', 72, 10
# model.load_state_dict(torch.load('./checkpoint/%s/model_b%d_n%d.pt' % (data_type, batch, augmentation)))
# model.to(device)
# model.eval()
# transforms = []
# batch = 20
# for i in range(augmentation):
#     transforms.append(Transformation(input_size=feature))
#     transforms[i].load_state_dict(torch.load('./checkpoint/%s/transform_%d_b%d_n%d.pt' % (data_type, i, batch, augmentation)))
#     transforms[i].to(device)
#     transforms[i].eval()
#
# index = np.random.randint(0, trn_x.shape[0] - 1, 1)
# sample = torch.from_numpy(trn_x[index]).cuda()
# sample = sample[:, :, :10]
# index = np.random.randint(0, trn_x.shape[0] - 1, 1)
# mixup_sample = torch.from_numpy(trn_x[index]).cuda()
# mixup_sample = mixup_sample[:, :, :10]
# transformed_sample = []
# for i in range(augmentation):
#     transformed_sample.append(transforms[i](sample)[:, :, 30:40])
# sample = sample[:, :, 30:40]
# w_t_copy = sample.detach().to(device)
# for i in range(augmentation):
#     w_t_copy = sample.clone().detach()
#     # cut out
#     if i % 5 == 0:
#         transformed_sample.append(torch.cat((w_t_copy[:, int(window/2), :].unsqueeze(1).repeat(1, int(window/2), 1), w_t_copy[:, -int(window/2):, :]), dim=1))
#     # gaussian noise
#     elif i % 5 == 1:
#         transformed_sample.append(w_t_copy + torch.randn_like(w_t_copy))
#     # mix up
#     elif i % 5 == 2:
#         transformed_sample.append(0.6 * w_t_copy + 0.4 * mixup_sample.to(device))
#     # trend
#     elif i % 5 == 3:
#         for i in range(window):
#             w_t_copy[:, i, :] += 0.02 * i
#         transformed_sample.append(w_t_copy)
#     # shift cut?
#     else:
#         half_len = int(window / 2)
#         for i in range(half_len):
#             w_t_copy[:, i + half_len, :] = w_t_copy[:, i, :]
#         w_t_copy[:, :half_len, :] = w_t_copy[:, 0, :].unsqueeze(1).repeat(1, half_len, 1)
#         transformed_sample.append(w_t_copy)


# x = np.linspace(0, window)
# fig, ax = plt.subplots(6, 1, constrained_layout=True, figsize=(5, 15))
# ax[0].plot(sample.squeeze(0).detach().cpu().numpy())
# plt.xticks(visible=False)
# ax[0].set_title('Original input', fontsize=10)
# plt.ylabel('amplitude')
# for i in range(5):
#     # ax[i + 1].subplot(6, i + 2, sharex=ax[0], sharey=ax[0])
#     ax[i + 1].plot(transformed_sample[i].squeeze(0).detach().cpu().numpy())
#     if i == 0:
#         ax[i + 1].set_title(f'cut out', fontsize=10)
#     elif i == 1:
#         ax[i + 1].set_title(f'gaussian noise', fontsize=10)
#     elif i == 2:
#         ax[i + 1].set_title(f'mix up', fontsize=10)
#     elif i == 3:
#         ax[i + 1].set_title(f'trend', fontsize=10)
#     elif i == 4:
#         ax[i + 1].set_title(f'shift + cut out', fontsize=10)

    # if i < 5 - 1:
        # ax[i + 1].set_xticks(visible=False)
# for i in range(5):
#     # ax[i + 1].subplot(6, i + 2, sharex=ax[0], sharey=ax[0])
#     ax[i + 1].plot(transformed_sample[i].squeeze(0).detach().cpu().numpy())
#     ax[i + 1].set_title(f'Transformation #{i+1}', fontsize=10)
# plt.xlabel('time')
# # plt.show()
# fig.tight_layout()
# plt.savefig('transformations_trainable.pdf')
# df = DataFrame(x_train)
# iqr = df.apply(out_iqr, k=1.5)
# for column in df:
#     df[column] = np.where(iqr[column] is True, 'NaN', df[column])
# cols = df.columns
# df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
# df = df.interpolate(method='linear', axis=0).bfill().ffill()
# x_train = df.to_numpy(dtype=float)

# plt.figure()
# plt.plot(x_train[:,:13])
# plt.show()
# plt.close()
#
# plt.figure()
# # x = np.linspace(0, x_test.shape[0], 1)
x = pd.date_range(start='2015-12-31 1:17:08', end='2015-12-31 9:37:07', periods=30000)

# plt.plot(x, x_test[20000:23000, 0:8])
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(x, x_test[227828:257828, :], color="lightgrey")
ax.plot(x, x_test[227828:257828, 24], label='P-302')
ax.plot(x, x_test[227828:257828, 17], label='FIT-301')
ax.plot(x, x_test[227828:257828, 16], label='DPIT-301')
ax.plot(x, x_test[227828:257828, 7], label='AIT-203')

# plt.plot(x_train[:, 1], label='LIT101')
# plt.plot(x_train[:, 0], label='FIT101')
# plt.plot(x_train[:, 8], label='FIT201')
plt.legend(loc='best')
plt.xlabel('date time')
plt.ylabel('scaled value')
# plt.xlim(227828, 257828)
date_form = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim(['2015-12-31 01:40:00', '2015-12-31 01:55:00'])
ax.set_ylim(-0.3, 1.2)
# fig.autofmt_xdate()
plt.show()
plt.close()

# plt.plot(x_test[:, 24], label='P302')
# # plt.plot(x_test[:, 7], label='AIT203')
# # # plt.plot(x_test[:, 19], label='MV301')
# plt.plot(x_test[:, 39], label='FIT502')
# plt.plot(x_test[:, 35], label='AIT502')
# plt.plot(x_test[:, 18], label='LIT301')
# # plt.plot(x_test[:, 47], label='FIT601')
# # plt.plot(x_test[:, 20], label='MV302')
# # plt.plot(x_test[:, 1], label='LIT-101')
# # plt.plot(x_test[:, 3], label='P-101')
# # plt.plot(x_test[:, 6], label='AIT-202')
# # plt.plot(x_test[:, 12], label='P-203')
#
# # plt.plot(x_test)
# plt.legend(loc='best')
# plt.xlim(44622, 44800)
# plt.show()

# plt.close()
#
# for i in range(x_test.shape[0]):
#     if np.max(x_test[i, :]) > 50:
#         print(i)
# for i in range(x_train.shape[0]):
#     if np.min(x_train[i, :]) < -1:
#         print(i)

scores = np.load('/home/kj21.choi/PycharmProjects/ANTLAD/result/msl/triplet/score.npy')

# pred = np.load('/home/kj21.choi/PycharmProjects/ANTLAD/result/swat/preds.npy')
# pd.DataFrame(pred).to_csv('pred.csv')
# scores = down_sample(scores, 10)
# gdn_score_swat = np.load('result/swat_scores.npy')
# gdn_score_wadi = np.load('result/wadi_scores.npy')
# gdn_score_msl = np.load('result/msl_scores.npy')
# gdn_score_smap = np.load('result/smap_scores.npy')
# gdn_score_wadi = down_sample(scores, 10)
pred_labels = np.zeros(len(scores))
T = len(pred_labels)
gt_labels = np.array(y_test[:T])
for i in range(T):
    pred_labels[i] = int(pred_labels[i])
    gt_labels[i] = int(gt_labels[i])


# best threshold search
best_threshold_search(scores, gt_labels, 85, 95, step_num=200)

# peaks over threshold
# base_threshold = 88
# window_size = 5000
# dynamic_threshold(scores, gt_labels, base_threshold, window_size)

end = datetime.now()
print('end:', end)


