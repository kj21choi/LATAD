import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, Subset, DataLoader


def get_sliding_data(window, data_x, data_y=None, step=1):
    samples = data_x
    labels = data_y

    seq_length = window
    seq_step = step

    arr_samples = []
    if labels is not None:
        arr_labels = []

    arr_indexes = []
    idx = np.asarray(list(range(0, np.shape(data_x)[0])))

    s_index = 0
    e_index = s_index + seq_length
    if e_index < samples.shape[0]:
        while e_index < samples.shape[0]:
            arr_samples.append(samples[s_index:e_index])
            if labels is not None:
                arr_labels.append(labels[s_index:e_index])
            arr_indexes.append(idx[s_index:e_index])
            s_index = s_index + seq_step
            e_index = e_index + seq_step
        if s_index < (samples.shape[0] - 1):
            arr_samples.append(samples[-seq_length:])
            if labels is not None:
                arr_labels.append(labels[-seq_length:])
            arr_indexes.append(idx[-seq_length:])
    else:
        arr_samples.append(samples)
        if labels is not None:
            arr_labels.append(labels)
        arr_indexes.append(idx)

    arr_samples = np.stack(arr_samples, axis=0)
    if labels is not None:
        arr_labels = np.stack(arr_labels, axis=0)
    arr_indexes = np.stack(arr_indexes, axis=0)

    samples = arr_samples
    if labels is not None:
        labels = arr_labels
    index = arr_indexes

    return samples, labels, index


def get_loaders(dataset, batch_size, val_ratio=0.2):
    dataset_len = int(len(dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
    train_subset = Subset(dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
    val_subset = Subset(dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader


class MyDataset(Dataset):

    def __init__(self, data_x, data_y, index, is_train):
        self.data_x = data_x
        self.data_y = data_y
        self.index = index
        self.is_train = is_train

    def __getitem__(self, _index):
        return_x = self.data_x[_index]

        return_y = -1
        if self.data_y is not None:
            return_y = self.data_y[_index]

        return_index = self.index[_index]

        return_next = self.index[_index]  # triplet only

        # if _index < self.data_x.shape[0] - 1:
        #     return_next = self.data_x[_index + 1][0]  # evaluate
        # if self.is_train:
        #     return_next = self.data_x[_index + 1][0]  # train
        # else:
        #     if _index < self.data_x.shape[0] - 1:
        #         return_next = self.data_x[_index + 1][0]  # evaluate
        #     else:
        #         return_next = self.data_x[_index][0]

        return return_x, return_y, return_index, return_next

    def __len__(self):
        return len(self.index)


def down_sample(data, sr, dim=0):
    """
    :param data: data to downsample
    :param sr: sampling rate, expecting integer
    :param dim: dimension to downsample
    :return: down_sampled data
    """

    print(data.shape)
    data = np.asarray([data[sr*i:sr*(i+1)].mean(axis=0) for i in range(int(len(data)/sr))])
    # data = np.asarray([data[sr*i] for i in range(int(len(data)/sr))])
    print(data.shape)

    return data


def out_iqr(s, k=1.5, return_threshold=False):
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_threshold:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]