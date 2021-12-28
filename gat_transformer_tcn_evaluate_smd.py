import torch
import torch.nn as nn
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from model.model import EncoderRNN, Transformation
from utils.preprocess import MyDataset, get_sliding_data, get_loaders
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader
from scipy import stats
from scipy.spatial import distance
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
from tsmoothie.smoother import *

mse = nn.MSELoss(reduction='mean')


def similarity(x1, x2):
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    return torch.exp(cosine_similarity(x1, x2) * 1 / 1)  # 1 ~ e


def cosine_distance(x1, x2):
    cosine_similarity = nn.CosineSimilarity(dim=-1)  # -1 ~ 1
    sim = (cosine_similarity(x1, x2) + 1) / 2  # 0 ~ 1
    return 1.0 - sim * 1 / 1  # 1 ~ 0


def mahalanobis(x=None, mean=None, iv=None):
    delta = x - mean
    left = np.dot(delta, iv)
    mahal_dist = np.dot(left, delta.T)
    return np.sqrt(mahal_dist)[0]


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


def evaluate_reconstruction(test_data, encoder, decoder, gt_labels, path, device):
    start = datetime.now()
    print('start:', start)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
    pbar = tqdm(test_loader)
    scores = []
    criterion = nn.MSELoss()
    for w_t, y_t, t in pbar:
        w_t = w_t.float().to(device)
        z_t = encoder(w_t)
        w_t_hat = decoder(z_t)
        dist = criterion(w_t_hat, w_t).detach().cpu().numpy()
        scores.append(dist)
        pbar.set_description(f'score: {dist.item():.8f}')

    # scaler = MinMaxScaler()
    # scaler.fit(scores)
    # scores = scaler.transform(scores)

    threshold = np.percentile(scores, 87.66)
    print(f'threshold={threshold}')
    pred_labels = np.zeros(len(scores))
    pred_labels[scores > threshold] = 1

    T = len(pred_labels)
    gt_labels = np.array(gt_labels[:T])
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    adj_pred_labels = adjust_predicts(pred_labels, gt_labels)

    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
    acc = [precision, recall, f1]

    adj_precision = precision_score(gt_labels, adj_pred_labels)
    adj_recall = recall_score(gt_labels, adj_pred_labels)
    adj_f1 = f1_score(gt_labels, adj_pred_labels)
    print(f'adj_precision={adj_precision:.4f}, adj_recall={adj_recall:.4f}, adj_f1-score={adj_f1:.4f}')
    acc.append(adj_precision)
    acc.append(adj_recall)
    acc.append(adj_f1)

    save_result(path, acc, gt_labels, pred_labels, scores, threshold)

    end = datetime.now()
    print('end:', end)


def evaluate_coreset(train_data, test_data, model, gt_labels, path, device='cuda:0', machine='machine-1-1'):
    start = datetime.now()
    print(f'{machine} start:{start}')

    train_loader = DataLoader(train_data, batch_size=120, shuffle=True, drop_last=True)
    train_z_t = []
    pbar = tqdm(train_loader)
    i = 0
    T = len(train_loader)
    with torch.no_grad():
        for w_t, _, _, _ in pbar:
            i += 1
            z_t, _ = model(w_t.float().to(device))
            train_z_t.append(z_t.detach().cpu().numpy())
            pbar.set_description(f'progress: {i}/{T}')
    train_z_t = np.array(train_z_t)
    feature = train_z_t.shape[2]
    train_z_t = train_z_t.reshape(-1, feature)

    cluster_centers = torch.tensor(KMeans(n_clusters=10, init='k-means++', max_iter=300).fit(train_z_t).cluster_centers_).to(device)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
    pbar = tqdm(test_loader)
    scores = []
    dist = []
    forecast = []
    i = 0
    with torch.no_grad():
        for w_t, y_t, t, nxt in pbar:
            z_t, pred = model(w_t.float().to(device)) #[:, :, -1]
            # z_t = z_t.to(device)
            # pred = pred.to(device)
            # nxt = nxt.to(device)
            # forecast.append(mse(pred, nxt).detach().cpu().numpy())
            max_dist = np.min([(cosine_distance(z_t, center) * torch.norm(z_t)).detach().cpu().numpy() for center in cluster_centers])
            # max_dist = np.max([cosine_distance(z_t, center) for center in cluster_centers]).detach().cpu().numpy()
            dist.append(max_dist)
            # if max_dist < 0.5:
            # pbar.set_description(f'{i}-th max distance:{dist[i]:.8f}, forecast error:{forecast[i]:.4f}')
            pbar.set_description(f'{i}-th max distance:{dist[i]:.8f}')
            # pbar.set_description(f'forecast error:{forecast[i]:.4f}')
            i += 1

    scaler = MinMaxScaler()
    dist = scaler.fit_transform(np.asarray(dist).reshape(-1, 1)).squeeze(1)
    # smoother = ExponentialSmoother(window_len=100, alpha=0.3)
    # smoother.smooth(dist)
    # dist[:-100] = smoother.data[0]
    # np.save(os.path.join("./result/%s" % path, "tri_score.npy"), dist)
    # forecast = scaler.fit_transform(np.asarray(forecast).reshape(-1, 1)).squeeze(1)
    # forecast[-1] = 0
    # np.save(os.path.join("./result/%s" % path, f'{machine}_score.npy'), forecast)

    # gamma = 0.2
    # scores = (gamma * dist + forecast) / (1 + gamma)
    scores = dist
    # scores = forecast

    threshold = np.percentile(scores, 95)
    print(f'threshold={threshold}')
    pred_labels = np.zeros(len(scores))
    pred_labels[scores > threshold] = 1

    T = len(pred_labels)
    gt_labels = np.array(gt_labels[:T])
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    adj_pred_labels = adjust_predicts(pred_labels, gt_labels)

    np.set_printoptions(suppress=True)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
    acc = [precision, recall, f1]

    adj_precision = precision_score(gt_labels, adj_pred_labels)
    adj_recall = recall_score(gt_labels, adj_pred_labels)
    adj_f1 = f1_score(gt_labels, adj_pred_labels)
    print(f'adj_precision={adj_precision:.4f}, adj_recall={adj_recall:.4f}, adj_f1-score={adj_f1:.4f}')
    acc.append(adj_precision)
    acc.append(adj_recall)
    acc.append(adj_f1)

    save_result(path, acc, gt_labels, pred_labels, scores, threshold, machine)

    end = datetime.now()
    print('end:', end)


def evaluate_recon_coreset(train_data, test_data, encoder, decoder, tri_encoder, gt_labels, path, device='cuda:0'):
    start = datetime.now()
    print('start:', start)

    train_loader = DataLoader(train_data, batch_size=200, shuffle=True, drop_last=True)
    train_z_t = []
    pbar = tqdm(train_loader)
    i = 0
    T = len(train_loader)
    for w_t, _, _ in pbar:
        i += 1
        z_t = encoder(w_t).detach().cpu().numpy()
        train_z_t.append(z_t)
        pbar.set_description(f'progress: {i}/{T}')
    train_z_t = np.array(train_z_t)
    feature = train_z_t.shape[2]
    train_z_t = train_z_t.reshape(-1, feature)

    cluster_centers = torch.tensor(KMeans(n_clusters=10, init='k-means++', max_iter=300).fit(train_z_t).cluster_centers_).to(device)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
    pbar = tqdm(test_loader)
    scores, recon_score, final_scores = [], [], []
    i = 0
    with torch.no_grad():
        for w_t, y_t, t in pbar:
            z_t = encoder(w_t).to(device)
            min_dist = np.min([cosine_distance(z_t, center) * torch.norm(z_t) for center in cluster_centers]).detach().cpu().numpy()
            scores.append(min_dist[0])

            w_t = w_t.to(device)
            w_t_hat = decoder(z_t).to(device)
            recon = mse(w_t, w_t_hat).float().detach().cpu().numpy().max()
            recon_score.append(recon)

            # final_scores.append(max_dist[0] * recon)

            pbar.set_description(f'{i}-th max distance:{scores[i]:.8f}, recon score:{recon_score[i]:.5f}')#, final score:{final_scores[i]:.4f}')
            i += 1
    scaler = MinMaxScaler()
    recon_scores = scaler.fit_transform(np.asarray(recon_score).reshape(-1, 1))
    tri_scores = scaler.fit_transform(np.asarray(scores).reshape(-1, 1))
    final_scores = np.add(recon_scores, tri_scores).reshape(-1)

    threshold = np.percentile(final_scores, 87.66)
    print(f'threshold={threshold}')
    pred_labels = np.zeros(len(final_scores))
    pred_labels[final_scores > threshold] = 1

    T = len(pred_labels)
    gt_labels = np.array(gt_labels[:T])
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    adj_pred_labels = adjust_predicts(pred_labels, gt_labels)

    np.set_printoptions(suppress=True)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    print(f'precision={precision:.4f}, recall={recall:.4f}, f1-score={f1:.4f}')
    acc = [precision, recall, f1]

    adj_precision = precision_score(gt_labels, adj_pred_labels)
    adj_recall = recall_score(gt_labels, adj_pred_labels)
    adj_f1 = f1_score(gt_labels, adj_pred_labels)
    print(f'adj_precision={adj_precision:.4f}, adj_recall={adj_recall:.4f}, adj_f1-score={adj_f1:.4f}')
    acc.append(adj_precision)
    acc.append(adj_recall)
    acc.append(adj_f1)

    save_result(path, acc, gt_labels, pred_labels, final_scores, threshold)

    end = datetime.now()
    print('end:', end)


def save_result(path, acc, gt_labels, pred_labels, scores, threshold, machine):
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
    plt.savefig(os.path.join("./plots/%s" % path, f'{machine}_result.pdf'))
    np.savetxt(os.path.join("./plots/%s" % path, f'{machine}_accuracy.txt'), acc, fmt='%.4f')
    np.save(os.path.join("./result/%s" % path, f'{machine}_score.npy'), scores)
    np.save(os.path.join("./result/%s" % path, f'{machine}_preds.npy'), preds)


def evaluate_euclidean(train_data, test_data, encoder, path='simulated', feature=25, device='cuda:0', positive_sample=10, step=10):
    train_loader = DataLoader(train_data, batch_size=40, shuffle=True, drop_last=True)
    train_z_t = []
    for w_t, _, _ in train_loader:
        z_t = encoder(w_t).detach().cpu().numpy()
        train_z_t.append(z_t)
    train_z_t = np.array(train_z_t)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
    result = []
    i = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for w_t, y_t, t in pbar:
            z_t = encoder(w_t).detach().cpu().numpy()
            dist = distance.cdist(z_t.reshape(1, -1), train_z_t.reshape(-1, 128), 'euclidean').mean()
            # sim = 0
            # for z in train_z_t:
            #     sim += torch.log(similarity(z_t.unsqueeze(), z.unsqueeze()))
            result.append(dist)
            pbar.set_description(f'score: {dist.item():.8f}')
            i += 1
            # if dist > 1.3:
            #     print(f'{i}-th Euclidean distance:{dist}')
    plt.plot(result)
    plt.show()

    print('end')