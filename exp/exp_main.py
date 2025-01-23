from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TFEGRU

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

import os
import time

import warnings
import matplotlib.pyplot as plt

from ptflops import get_model_complexity_info

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'TFEGRU': TFEGRU
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, count, flag):
        data_set, data_loader = data_provider(self.args, count, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # training and evaluating the model
    def train_and_evaluate(self):
        type = []
        type.append(self.args.model)
        result = pd.DataFrame()

        for count in self.args.counts:
            df = pd.DataFrame()

            MSE, MAE, RMSE, MAPE = [], [], [], []

            plt.grid(True)

            for rnn_type in type:
                model = self.train(count, rnn_type)
                _preds, _labels, mae, mse, mape, rmse, _, _, _ = self.test(count, rnn_type)

                MSE.append(mse)
                MAE.append(mae)
                RMSE.append(rmse)
                MAPE.append(mape)

            # summarizing results
            period_df = pd.DataFrame({
                'Dataset': self.args.dataset,
                'Machine Number': [count] * len(type),
                'RNN Type': type,
                'MSE': MSE,
                'MAE': MAE,
                'RMSE': RMSE,
                'MAPE': MAPE
            })

            df = pd.concat([df, period_df], ignore_index=True)
            self.save_results(type, count, MSE, MAE, RMSE, MAPE)
            plt.close("all")

        # saving results to CSV
        df.to_csv(f"{self.args.result_folder}/metrics_{count}.csv", index=False)
        df.T.to_csv(f"{self.args.result_folder}/metrics_transposed_{count}.csv", index=False)

        result = pd.concat([result, df], ignore_index=True)

        # saving overall result
        result.to_csv(f"{self.args.result_folder}/metrics_result.csv", index=False)
        result.T.to_csv(f"{self.args.result_folder}/metrics_result_transposed.csv", index=False)

    def train(self, count, type='lstm'):
        input_size = count
        best_val_loss = float('inf')
        epochs_no_improve = 0
        rnn_type = type

        _, train_loader = self._get_data(count, 'train')
        _, val_loader = self._get_data(count, 'val')

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        scheduler = ExponentialLR(optimizer, gamma=self.args.decay_rate)
        with open(f"{self.args.result_folder}/results.txt", "a") as f:
            f.write(
                f"Training Parameters: num_epochs={self.args.num_epochs}, batch_size={self.args.batch_size}, learning_rate={self.args.learning_rate},"
                f" loss=MSELoss, hidden_size={self.args.hidden_size}, input_size={input_size}, seq_len={self.args.seq_len}, pred_len={self.args.pred_len},"
                f" rnn_type={type}, dropout={self.args.dropout}\n")
        print(
            f"Training Parameters: num_epochs={self.args.num_epochs}, batch_size={self.args.batch_size}, learning_rate={self.args.learning_rate}, "
            f"loss=MSELoss, hidden_size={self.args.hidden_size}, input_size={input_size}, seq_len={self.args.seq_len}, pred_len={self.args.pred_len},"
            f" rnn_type={type}, dropout={self.args.dropout}")

        for epoch in range(self.args.num_epochs):
            start_time = time.time()
            for batch_X, batch_Y in train_loader:

                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)

                self.model.train()
                # with torch.cuda.amp.autocast():
                outputs = self.model(batch_X)
                optimizer.zero_grad()
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
            end_time = time.time()
            training_time = end_time - start_time
            start_time = time.time()
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                # with torch.cuda.amp.autocast():
                for batch_X_val, batch_Y_val in val_loader:
                    batch_X_val = batch_X_val.to(self.device)
                    batch_Y_val = batch_Y_val.to(self.device)
                    val_outputs = self.model(batch_X_val)
                    val_loss += criterion(val_outputs, batch_Y_val)
                val_loss /= len(val_loader)
            end_time = time.time()
            validation_time = end_time - start_time
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(self.model.cuda(), (self.args.batch_size, self.args.seq_len, input_size),
                                                         as_strings=True,
                                                         print_per_layer_stat=False)
                print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                print('{:<30}  {:<8}'.format('Number of parameters: ', params))

            with open(f"{self.args.result_folder}/results.txt", "a") as f:
                f.write(
                    "Epoch: %d, Train Loss: %1.5f, Validation Loss: %1.5f, Training Time: %1.5f, Validation Time: %1.5f, Computational complexity: %s, Number of parameters: %s\n" % (
                        epoch, loss.item(), val_loss.item(), training_time, validation_time, macs, params))

            print(
                "Epoch: %d, Train Loss: %1.5f, Validation Loss: %1.5f, Training Time: %1.5f, Validation Time: %1.5f" % (
                    epoch, loss.item(), val_loss.item(), training_time, validation_time))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), f'best_model_{self.args.dataset}_{count}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.args.patience:
                    with open(f"{self.args.result_folder}/results.txt", "a") as f:
                        f.write("Early stopping!\n")
                    print("Early stopping!")
                    break

        self.model.load_state_dict(torch.load(f'best_model_{self.args.dataset}_{count}.pth', map_location=self.device), strict=False)
        return self.model

    def test(self, count, type='lstm'):
        # test_dataset = TensorDataset(dataX, dataY)
        test_dataset, test_loader = self._get_data(count, 'test')
        dataY = torch.stack([target for _, target in test_dataset])

        self.model.eval()

        data_predict = []
        dataY_plot = []

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch_dataX, batch_dataY in test_loader:
                    batch_dataX = batch_dataX.to(self.device)
                    batch_dataY = batch_dataY.to(self.device)
                    batch_predict = self.model(batch_dataX)
                    batch_data_predict = batch_predict.data.cpu().numpy()
                    batch_dataY_plot = batch_dataY.data.cpu().numpy()

                    data_predict.extend(batch_data_predict.tolist())
                    dataY_plot.extend(batch_dataY_plot.tolist())

        data_predict = np.array(data_predict)
        dataY_plot = np.array(dataY_plot)
        if self.args.save_data == True:
            os.makedirs(f"{self.args.result_folder}/{type}", exist_ok=True)
            np.save(f"{self.args.result_folder}/{type}/predicted_{type}_{1}.npy", data_predict)
            np.save(f"{self.args.result_folder}/{type}/actual_{type}_{1}.npy", dataY_plot)
        mae_list = np.zeros_like(dataY_plot[:, 0, :])  # saving the MAE for each data point
        mse_list = np.zeros_like(dataY_plot[:, 0, :])  # saving the MSE for each data point
        mape_list = np.zeros_like(dataY_plot[:, 0, :])  # saving the MAPE for each data point

        for i in range(len(dataY[0])):
            predict = data_predict[:, i, :]
            actual = dataY_plot[:, i, :]
            mae_list += np.abs(predict - actual)
            mse_list += (predict - actual) ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_list += np.abs((predict - actual) / actual)
                mape_list = np.where(np.isfinite(mape_list), mape_list, 0)
        mae_list /= len(dataY[0])
        mse_list /= len(dataY[0])
        mape_list /= len(dataY[0])
        mae_avg = np.mean(mae_list)
        mse_avg = np.mean(mse_list)
        mape_avg = np.mean(mape_list)
        rmse_avg = np.sqrt(np.mean((dataY_plot - data_predict) ** 2))
        # Error at each point
        mae_list = np.mean(mae_list, axis=1)
        mse_list = np.mean(mse_list, axis=1)
        mape_list = np.mean(mape_list, axis=1)
        # Averaging, because the actual formula is defined to calculate the error from the sum of the current point to the previous point
        mae_list = np.array([np.mean(mae_list[:i + 1]) for i in range(len(mae_list))])
        mse_list = np.array([np.mean(mse_list[:i + 1]) for i in range(len(mse_list))])
        mape_list = np.array([np.mean(mape_list[:i + 1]) for i in range(len(mape_list))])

        return (data_predict, dataY_plot, mae_avg, mse_avg, mape_avg, rmse_avg, mae_list, mse_list, mape_list)

    # saving results to the file
    def save_results(self, type, count, MSE, MAE, RMSE, MAPE):
        with open(f"{self.args.result_folder}/results.txt", "a") as f:
            f.write(f"rnn_type: {type}, machine: {count}\n")
            f.write(f"MSE: {MSE}, MEAN MSE: {np.mean(MSE)}\n")
            f.write(f"MAE: {MAE}, MEAN MAE: {np.mean(MAE)}\n")
            f.write(f"RMSE: {RMSE}, MEAN RMSE: {np.mean(RMSE)}\n")
            f.write(f"MAPE: {MAPE}, MEAN MAPE: {np.mean(MAPE)}\n")

        print(f"rnn_type: {type}, machine: {count}")
        print(f"MSE: {MSE}, MEAN MSE: {np.mean(MSE)}")
        print(f"MAE: {MAE}, MEAN MAE: {np.mean(MAE)}")
        print(f"RMSE: {RMSE}, MEAN RMSE: {np.mean(RMSE)}")
        print(f"MAPE: {MAPE}, MEAN MAPE: {np.mean(MAPE)}")
