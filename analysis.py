import argparse
import seaborn as sns
import torch
import sys
import pickle
import os
import random

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import EncoderRNN, Transformation, TCN, MyModel
from utils.preprocess import MyDataset, get_sliding_data, down_sample, get_loaders
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid", {'axes.grid': False})

sns.set()

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def plot_distribution(data, y_test, encoder, transforms=None, device='cuda'):
    # idx = 1000
    # idx = np.random.randint(0, len(train_data) - 1)
    # idx_list = [1000,1010,1020,1030,1040,1050,1060,1070]
    # z_t, z_p, z_n = [], [], []
    test_loader, valid_loader = get_loaders(dataset=data, batch_size=1, val_ratio=0)
    z_p, z_n = [], []
    pbar = tqdm(test_loader)
    i = 0
    with torch.no_grad():
        for w_t, y_t, t, _ in pbar:
            w_t = w_t.to(device).float()
            z_t = encoder(w_t)
            if y_test[i] == 1:  # anomaly
                z_n.append(z_t)
            else:
                z_p.append(z_t)
            i += 1

    n_z_p = len(z_p)
    n_z_n = len(z_n)
    embedding = z_p + z_n
    embedding = [z[0].squeeze(0).detach().cpu().numpy() for z in embedding]

    # pca = PCA(n_components=3)
    # pca.fit(embedding)
    # embedding_pca = pca.transform(embedding)
    tsne = TSNE(n_components=2)
    embedding_tsne = tsne.fit_transform(embedding)

    df_positive = pd.DataFrame.from_dict({"f1": embedding_tsne[0:n_z_p, 0], "f2": embedding_tsne[0:n_z_p, 1]})#, "f3": embedding_tsne[8:88, 2]})
    df_negative = pd.DataFrame.from_dict({"f1": embedding_tsne[n_z_p:n_z_p+n_z_n, 0], "f2": embedding_tsne[n_z_p:n_z_p+n_z_n, 1]})#, "f3": embedding_tsne[88:168, 2]})

    return df_positive, df_negative


# def plot_latent(args=None):
#     data_type, w, augmentation, window, feature, device, batch, step, lr, n_epochs = args.data, args.weight, args.augmentation, args.window, args.feature, args.device, args.batch, args.step, args.lr, args.n_epochs
#
#     path = f'/home/kj21.choi/hdd/04_AAAI/THOC/{data_type}/'
#     with open(os.path.join(path, 'train.pkl'), 'rb') as f:
#         x_train = pickle.load(f)
#         x_train = np.asarray(x_train).T
#         x_train = down_sample(x_train, 10)
#     with open(os.path.join(path, 'test.pkl'), 'rb') as f:
#         x_test = pickle.load(f)
#         x_test = np.asarray(x_test).T
#         x_test = down_sample(x_test, 10)
#         y_test = np.asarray(x_test[:, -1])
#         x_test = x_test[:, :-1]
#
#     print('Data pre-process')
#     trn_x, trn_y, trn_ind = get_sliding_data(window, x_train, data_y=None, step=step)
#     tst_x, tst_y, tst_ind = get_sliding_data(window, x_test, data_y=y_test, step=step)
#
#     train_data = MyDataset(trn_x, trn_y, trn_ind)
#     test_data = MyDataset(tst_x, tst_y, tst_ind)
#
#     encoder = EncoderRNN(hidden_size=128, in_channel=feature, encoding_size=128, device=device)
#
#     # Plot the distribution of the encodings and use the learnt encoders to train a downstream classifier
#     encoder.load_state_dict(torch.load('./checkpoint/%s/09_triplet_loss_real_kl_no_limit_window_100_smooth_threshold_90/encoder_b%d_n%d.pt' % (data_type, batch, augmentation)))
#     encoder.to(device).eval()
#
#     # Save plots
#     if not os.path.exists(os.path.join("./plots/%s" % data_type)):
#         os.mkdir(os.path.join("./plots/%s" % data_type))
#     fig = plt.figure(figsize=(6, 6))
#     # ax = Axes3D(fig)
#     train_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=False)
#     test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)
#
#     embedding =[]
#     with torch.no_grad():
#         pbar = tqdm(train_loader)
#         for w_t, y_t, t in pbar:
#             w_t = torch.tensor(w_t, device=device)
#             # original z
#             z_t = encoder(w_t).squeeze(0)
#             embedding.append(z_t)
#         tsne = TSNE(n_components=2)
#         embedding = [z.detach().cpu().numpy() for z in embedding]
#         embedding_tsne = tsne.fit_transform(embedding)
#
#         df = pd.DataFrame.from_dict({"f1": embedding_tsne[:, 0], "f2": embedding_tsne[:, 1]})
#
#     plt.scatter(df.get("f1"), df.get("f2"), c='blue', marker='o')
#
#     # embedding2 = []
#     # with torch.no_grad():
#     #     pbar = tqdm(test_loader)
#     #     for w_t, y_t, t in pbar:
#     #         w_t = torch.tensor(w_t, device=device)
#     #         # original z
#     #         z_t = encoder(w_t).squeeze(0)
#     #         embedding2.append(z_t)
#     #     tsne = TSNE(n_components=2)
#     #     embedding2 = [z.detach().cpu().numpy() for z in embedding2]
#     #     embedding_tsne2 = tsne.fit_transform(embedding2)
#     #
#     #     df = pd.DataFrame.from_dict({"f1": embedding_tsne2[:, 0], "f2": embedding_tsne2[:, 1]})
#
#     # plt.scatter(df.get("f1"), df.get("f2"), c='red', marker='o')
#
#     plt.xlabel('feature 1')
#     plt.ylabel('feature 2')
#     plt.title("TSNE in the feature space", fontweight="bold")
#     plt.legend(['train', 'test'], loc='best')
#     plt.show()
#     # plt.savefig(os.path.join("./plots/%s" % data_type, "feature_space.pdf"))


def run(args=None):
    data_type, w, augmentation, window, feature, device, batch, step, lr, n_epochs = args.data, args.weight, args.augmentation, args.window, args.feature, args.device, args.batch, args.step, args.lr, args.n_epochs

    path = f'/home/kj21.choi/hdd/04_AAAI/THOC/swat/'
    with open(os.path.join(path, 'train.pkl'), 'rb') as f:
        x_train = pickle.load(f)
        x_train = np.asarray(x_train).T
        x_train = down_sample(x_train[100000:], 10)  # swat: initial 100000 points are unstable
    with open(os.path.join(path, 'test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
        x_test = np.asarray(x_test).T
        # x_test = down_sample(x_test, 10)
        y_test = np.asarray(x_test[:, -1])
        x_test = x_test[:, :-1]
        x_test[:, 5] = x_test[:, 4]

    print('Data pre-process')
    trn_x, trn_y, trn_ind = get_sliding_data(window, x_test, data_y=y_test, step=step)
    # trn_x, trn_y, trn_ind = get_sliding_data(window, x_train, data_y=None, step=step)

    test_data = MyDataset(trn_x, trn_y, trn_ind, is_train=0)
    # train_data = MyDataset(trn_x, trn_y, trn_ind)

    encoder = EncoderRNN(hidden_size=128, in_channel=feature, encoding_size=128, device=device)
    # encoder = TCN(input_size=feature, output_size=128, num_channels=[128] * 4, kernel_size=5, dropout=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'device:{device}')
    # encoder = MyModel(feature, window, kernel_size=5, dropout=0.2, d_model=128)
    # if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
    #     encoder = nn.DataParallel(encoder, device_ids=[0, 1, 2])
    # encoder.to(device)
    # Plot the distribution of the encodings and use the learnt encoders to train a downstream classifier
    encoder.load_state_dict(torch.load('./checkpoint/swat/01_baseline_encoder_decoder_mse_loss/encoder_b30_p0_n0.pt'))
    encoder.to(device).eval()
    # transforms = []
    # for i in range(augmentation):
    #     transforms.append(Transformation(input_size=feature).to(device))
    #     transforms[i].load_state_dict(torch.load('./checkpoint/%s/transform_%d_b%d_n%d.pt' % (data_type, i, batch, augmentation)))
    #     transforms[i].to(device).eval()

    # Save plots
    if not os.path.exists(os.path.join("./plots/%s" % data_type)):
        os.mkdir(os.path.join("./plots/%s" % data_type))
    fig = plt.figure(figsize=(6, 6))
    # ax = Axes3D(fig)
    df_positive, df_negative = plot_distribution(test_data, y_test, encoder, device)
    # df_positive, df_negative = plot_distribution(train_data, None, encoder, transforms)
    # ax.scatter(df_original.get("f1"), df_original.get("f2"), df_original.get("f3"), c='blue', marker='o')
    # ax.scatter(df_positive.get("f1"), df_positive.get("f2"), df_positive.get("f3"), c='green', marker='o')
    # ax.scatter(df_negative.get("f1"), df_negative.get("f2"), df_negative.get("f3"), c='red', marker='o')
    plt.scatter(df_positive.get("f1"), df_positive.get("f2"), c='green', marker='o')
    plt.scatter(df_negative.get("f1"), df_negative.get("f2"), c='red', marker='o')

    # plt.xlabel('feature 1')
    # plt.ylabel('feature 2')
    # plt.title("TSNE in the feature space", fontweight="bold")
    plt.legend(['normal', 'anomaly'], loc='best')
    plt.show()
    # plt.savefig(os.path.join("./plots/%s" % data_type, "feature_space.pdf"))


if __name__ == '__main__':
    random.seed(5)  # 0 ~ 5
    parser = argparse.ArgumentParser(description='Analyze ANTLAD result')
    parser.add_argument('--data', type=str, default='simulated')
    parser.add_argument('--weight', type=float, default=0.05)
    parser.add_argument('--augmentation', type=int, default=10)  # the number of augmentation
    parser.add_argument('--window', type=int, default=100)  # the number of negative samples
    parser.add_argument('--step', type=int, default=10)  # the number of negative samples
    parser.add_argument('--feature', type=int, default=51)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print('LATAD model with w=%s' % args)
    # plot_latent(args)
    # plot_distribution(args)
    run(args)
