import gc

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from model.model import EncoderRNN, Transformation
from utils.preprocess import MyDataset, get_sliding_data, get_loaders
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def save_plot(n_epochs, path, performance, early_stop_win):
    # Save performance plots
    n_epochs -= early_stop_win  # n_epochs - early stop window
    if not os.path.exists('./plots/%s' % path):
        os.mkdir('./plots/%s' % path)
    train_loss = [t[0] for t in performance]
    valid_loss = [t[1] for t in performance]
    plt.figure()
    plt.xticks(np.arange(0, n_epochs, 5))
    plt.plot(np.arange(0, n_epochs, 1), train_loss[:n_epochs], label="Train")
    plt.plot(np.arange(0, n_epochs, 1), valid_loss[:n_epochs], label="Valid")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join("./plots/%s" % path, "loss.pdf"))


# TODO: find_neighborhood
def find_neighborhood(x, t, window_size, adf=True):
    if adf:
        gap = window_size
        corr = []
        for w_t in range(window_size, 4 * window_size, gap):  # 4ea
            try:
                p_val = 0
                for f in range(x.shape[-2]):  # window length
                    p = adfuller(np.array(x[f, max(0, t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                    p_val += 0.01 if np.isnan(p) else p
                corr.append(p_val / x.shape[-2])
            except:
                corr.append(0.6)
        epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0]) == 0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
        delta = int((5 * epsilon * window_size) / window_size)

    # # Random from a Gaussian
    # t_p = [int(t + np.random.randn() * epsilon * window_size) for _ in range(10)]
    # t_p = [max(window_size // 2 + 1, min(t_pp, T - window_size // 2)) for t_pp in t_p]
    # x_p = torch.stack([x[:, t_ind - window_size // 2:t_ind + window_size // 2] for t_ind in t_p])
    return delta


def similarity(x1, x2):
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    return torch.exp(cosine_similarity(x1, x2) * 1 / 1)  # 1 ~ e


def cosine_distance(x1, x2):
    cosine_similarity = nn.CosineSimilarity(dim=-1)  # -1 ~ 1
    sim = (cosine_similarity(x1, x2) + 1) / 2  # 0 ~ 1
    return 1.0 - sim * 1 / 1  # 1 ~ 0


def fgsm_attack(input, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_input = input + epsilon * sign_data_grad
    perturbed_input = torch.clamp(perturbed_input, 0, 1)
    return perturbed_input


mse = nn.MSELoss(reduction='mean')
kl_loss = nn.KLDivLoss(reduction='batchmean')
log_softmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)
np.set_printoptions(suppress=True)


def run(data, loader, transforms, model, margins, optimizer=None, optimizer_t=None, augmentation=3, is_train=True, step=10, device='cuda:0'):
    total_loss, iter_count = 0, 0
    pbar = tqdm(loader)
    # add_margins = torch.rand(augmentation) / 10.0
    # margins += add_margins
    # margins = torch.clamp(margins, max=0.999)
    # print(f'margins:{margins}')
    for w_t, y_t, t, w_nxt in pbar:
        batch_size, len_size, f_size = w_t.shape

        w_t = w_t.float()
        w_t.requires_grad = True
        anchor, pred = model(w_t)  # anchor

        # forecast = mse(w_nxt.float().cuda(), pred)
        smoothness = torch.zeros(1, device=device)
        for i in range(batch_size - 1):
            smoothness += kl_loss(log_softmax(anchor[i].unsqueeze(0)), softmax(anchor[i + 1].unsqueeze(0)))
        # TODO: find_neighborhood
        # get_positive_sample
        w_p_list = []
        T = data.index.shape[0]
        # delta = find_neighborhood(w_t, t, len_size, adf=True)
        for _ in range(augmentation):
            idx = torch.divide(t[:, 0] - np.random.randint(-100, 100), step).int()  # start indices (per sliding window size)
            for i, value in enumerate(idx):
                if value <= 0:
                    idx[i] = 0
                elif value >= T:
                    idx[i] = T - 1
            w_p_list.append(torch.tensor(data[idx][0]))

        # triplet loss among anchor, positive, and negative
        triplet_loss = torch.zeros(batch_size, device=device)
        regularizer = torch.zeros(batch_size, device=device)
        compactness = torch.zeros(batch_size, device=device)
        for i in range(augmentation):
            positive, _ = model(w_p_list[i].float())
            w_n = transforms[i](w_t)
            negative, _ = model(w_n)
            margin = margins[i]

            # 20211014 add compactness
            # compactness += mse(anchor, positive)
            compactness += cosine_distance(anchor, positive)

            criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=margin, reduction='mean', swap=True)
            triplet_loss += criterion(anchor, positive, negative)

            # regularizer += kl_loss(log_softmax(negative), softmax(positive))
            # regularizer += kl_loss(log_softmax(w_n), softmax(w_p_list[i].float()))

        # compactness = 10 * torch.mean(compactness / augmentation)
        # triplet_loss = 100 * torch.mean(triplet_loss / augmentation)
        # regularizer = 10 * torch.mean(regularizer / augmentation)
        # smoothness = 10 * torch.mean(smoothness)
        compactness = 1000 * torch.mean(compactness / augmentation)
        triplet_loss = 1000 * torch.mean(triplet_loss / augmentation)
        regularizer = 1 * torch.mean(regularizer / augmentation)
        smoothness = 1000 * torch.mean(smoothness)
        # forecast = 1000 * torch.mean(forecast)
        iter_loss = compactness + triplet_loss + smoothness + regularizer  # + forecast
        # iter_loss = forecast

        if is_train:
            optimizer.zero_grad()
            optimizer_t.zero_grad()
            iter_loss.backward()
            data_grad = w_t.grad.data
            for i in range(augmentation):
                transforms[i].sign_data_grad = data_grad.sign()
            optimizer.step()
            optimizer_t.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2.0)
        total_loss += iter_loss.item()
        iter_count += 1
        # pbar.set_description(f'Loss: {iter_loss.item():.4f}')

        if is_train and iter_count % 8 == 0:
            transformed_sample = []
            for i in range(augmentation):
                transformed_sample.append(transforms[i](w_t))

            fig, ax = plt.subplots(6, 1, constrained_layout=True, figsize=(5, 15))
            ax[0].plot(w_t[0][:, :10].squeeze(0).detach().cpu().numpy())
            plt.xticks(visible=False)
            ax[0].set_title('Original input', fontsize=10)
            for i in range(5):
                # ax[i + 1].subplot(6, i + 2, sharex=ax[0], sharey=ax[0])
                ax[i + 1].plot(transformed_sample[i][0][:, :10].squeeze(0).detach().cpu().numpy())
                ax[i + 1].set_title(f'Transformation #{i + 1}', fontsize=10)
            plt.xlabel('time')
            fig.tight_layout()
            plt.savefig('transformations_trainable.pdf')

        pbar.set_description(
            f'Loss: {iter_loss.item():.4f}, compactness:{compactness.item():.4f}, triplet: {triplet_loss.item():.12f}, smoothness: {smoothness.item():.4f}, KL: {regularizer.item():.12f}')  # , forecast:{forecast.item():.4f}')#, # )
        del iter_loss

    return total_loss / iter_count


def train(train_data, model, lr=0.001, decay=0.005, augmentation=10, n_epochs=100, path='swat',
          feature=51, window=100, device='cuda:0', batch_size=10, step=10):
    transforms = []
    # margins = torch.rand(augmentation)
    # margins = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    margins = [0.7, 0.8, 0.9, 0.99, 0.9, 0.9, 0.99, 0.9, 0.9, 0.99]
    print(f'margins:{margins}')
    for i in range(augmentation):
        transforms.append(Transformation(input_size=feature, window_size=window))
        transforms[i].train()
    model.train()

    params = list(model.parameters())
    prams_t = []
    for t in transforms:
        prams_t += list(t.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
    optimizer_t = torch.optim.Adam(prams_t, lr=lr, weight_decay=decay)

    train_loader, valid_loader = get_loaders(dataset=train_data, batch_size=batch_size, val_ratio=0.2)

    performance, losses = [], []
    best_loss = np.inf
    early_stop_win = 10
    early_stop_epoch = 0
    stop_improve_count = 0

    for epoch in range(n_epochs):
        early_stop_epoch = epoch + 1
        train_loss = run(train_data,
                         train_loader,
                         transforms,
                         model,
                         margins,
                         optimizer=optimizer,
                         optimizer_t=optimizer_t,
                         augmentation=augmentation,
                         is_train=True,
                         step=step,
                         device=device)
        gc.collect()
        with torch.no_grad():
            valid_loss = run(train_data,
                             valid_loader,
                             transforms,
                             model,
                             margins,
                             augmentation=augmentation,
                             is_train=False,
                             step=step,
                             device=device)
        if early_stop_epoch >= 2:
            performance.append((train_loss, valid_loss))
        print('Epoch %d Loss ============================================================> Training Loss: %.8f \t Valid Loss: %.8f' % (epoch + 1, train_loss, valid_loss))
        if best_loss > valid_loss:
            # if early_stop_epoch >= 10:
            if not os.path.exists('./checkpoint/%s' % path):
                os.mkdir('./checkpoint/%s' % path)
            best_loss = valid_loss
            torch.save(model.state_dict(), './checkpoint/%s/model_b%d_n%d.pt' % (path, batch_size, augmentation))
            # torch.save(model, './checkpoint/%s/model_b%d_n%d.pt' % (path, batch_size, augmentation))
            stop_improve_count = 0
            for i, n in enumerate(transforms):
                torch.save(n.state_dict(), './checkpoint/%s/transform_%d_b%d_n%d.pt' % (path, i, batch_size, augmentation))
        else:
            # if early_stop_epoch >= 10:
            stop_improve_count += 1
        if stop_improve_count >= early_stop_win:
            print(f'{early_stop_epoch}: early stop!!')
            break

    save_plot(early_stop_epoch, path, performance, early_stop_win)

    print('=======> Performance Summary:')
    # print(f'Loss: {best_loss.detach().cpu().numpy(): .4f}')
    print(f'Loss: {best_loss.__float__(): .4f}')

    return model
