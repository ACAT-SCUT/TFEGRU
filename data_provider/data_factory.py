from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def data_provider(args, count, flag):
    data = load_data(args.dataset, count)
    (x, y), (dataX, dataY), sc = genData(data, args.result_folder, args.seq_len,
                                         args.pred_len, args.dataset, flag, args.train_size, args.val_size)
    dataset = TensorDataset(dataX, dataY)
    if flag == 'train':
        shuffle_flag = True
    else:
        shuffle_flag = False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_flag)
    return dataset, dataloader

# loading datasets
def load_data(dataset, count):
    if dataset == 'google':
        file_path = f'./dataset/google_{count}.csv'
        data_array = pd.read_csv(file_path).to_numpy()[:, 1:]
    else:
        file_path = f'./dataset/alibaba_{count}.csv'
        data_array = pd.read_csv(file_path).to_numpy()[:, 2:]

    # adding new dimension
    stacked_data = data_array[np.newaxis, :]
    print("Array shape:", stacked_data.shape)
    return stacked_data

def genData(training_set, result_folder, seq_length=5, pred_length=1, dataset='google', flag='train', train_size=0.6, val_size=0.2):
    sc_features = MinMaxScaler()
    if dataset == 'google':
        current_data = training_set[0, :, :]

        num_rows, num_cols = current_data.shape
        processed_arr = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            if i + 1 <= num_rows:
                processed_arr[i, :] = np.mean(current_data[i:i+1, :], axis=0)

        training_data = processed_arr[np.newaxis, :, :]
    else:
        num_series, time_steps, num_features = training_set.shape
        current_data = training_set[0, :, :]
        normalized_data = np.zeros_like(current_data)
        current_data_normalized = sc_features.fit_transform(current_data)
        normalized_data = current_data_normalized
        training_data = normalized_data[np.newaxis, :, :]

    x, y, dataX, dataY = large_sliding_windows(training_data, seq_length, pred_length, train_size, train_size + val_size, flag)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    dataX = torch.Tensor(dataX)
    dataY = torch.Tensor(dataY)

    with open(f"{result_folder}/results.txt", "a") as f:
        f.write(f"{flag}X shape: {dataX.shape}, {flag}Y shape: {dataY.shape}\n")
    print(f"{flag}X shape: {dataX.shape}, {flag}Y shape: {dataY.shape}")
    return (x, y), (dataX, dataY), sc_features

def large_sliding_windows(data, seq_length, pred_length=1, train_size=0.7, val_size=0.9, flag='train'):
    x = []
    y = []
    dataX = []
    dataY = []
    for series in data:
        for i in range(len(series) - (seq_length + pred_length -1)):
            _x = series[i:i + seq_length * 1:1]
            _y = series[i + seq_length * 1:i + (seq_length + pred_length) * 1:1]
            x.append(_x)
            y.append(_y)
            if i < (len(series) - (seq_length + pred_length -1))*train_size and flag == 'train':
                dataX.append(_x)
                dataY.append(_y)
            elif (len(series) - (seq_length + pred_length -1))*train_size <= i < (len(series) - (seq_length + pred_length -1))*val_size\
                    and flag == 'val':
                dataX.append(_x)
                dataY.append(_y)
            elif i >= (len(series) - (seq_length + pred_length -1))*val_size and flag == 'test':
                dataX.append(_x)
                dataY.append(_y)

    return np.array(x), np.array(y), np.array(dataX), np.array(dataY)
