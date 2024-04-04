import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, recall_score


class CNNLSTMModel(nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # Change to (batch, sequence_length, features) for LSTM
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Use the output of the last sequence step
        x = self.fc(x)
        return x


class CSVDataset(Dataset):
    def __init__(self, filepath, mean, std, lookback=20, forecast=3):
        self.filepath = filepath
        self.mean = mean
        self.std = std
        self.lookback = lookback
        self.forecast = forecast
        self.data, self.labels = self.load_and_process_data()

    def load_and_process_data(self):
        data = pd.read_csv(self.filepath)
        data = data[['quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']]
        # mean-standard error 標準化
        data = (data - self.mean) / self.std
        return self.create_features_and_labels(data)

    def create_features_and_labels(self, data):
        X, y = [], []
        for i in range(self.lookback, len(data) - self.forecast):
            window = data.iloc[i - self.lookback:i].values
            future_prices = data['quote.USD.price'][i:i + self.forecast]
            label = int(np.any(future_prices > data['quote.USD.price'].iloc[i - 1]))
            X.append(window)
            y.append(label)
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).transpose(0, 1)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean, self.std = self.compute_mean_std(self.config['train_data_path'])
        self.net = CNNLSTMModel().to(self.device)
        self.train_iter = DataLoader(CSVDataset(filepath=self.config['train_data_path'], mean=self.mean, std=self.std),
                                     batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        self.val_iter = DataLoader(CSVDataset(filepath=self.config['val_data_path'], mean=self.mean, std=self.std),
                                   batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        self.criterion = self.weight_criterion()
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.75)
        self.record_train = []
        self.record_test = []

    @staticmethod
    def compute_mean_std(filepath):
        data = pd.read_csv(filepath)
        data = data[['quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']]
        mean = data.mean().values
        std = data.std().values
        return mean, std

    def weight_criterion(self):
        class_weights = torch.tensor([3, 1], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    def configure_optimizer(self):
        optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.config['learning_rate'],  # 學習率
            weight_decay=self.config['weight_decay']  # 權重衰減
        )
        return optimizer

    def train(self):
        self.net.train()
        num_print = len(self.train_iter) // 4

        for epoch in range(self.config['num_epochs']):
            print(f"========== number {epoch + 1} epoch training ==========")
            total, correct, train_loss = 0, 0, 0
            start = time.time()

            for i, (X, y) in tqdm(enumerate(self.train_iter)):
                X, y = X.to(self.device), y.to(self.device)
                output = self.net(X)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total += y.size(0)
                correct += (output.argmax(dim=1) == y).sum().item()
                train_acc = 100.0 * correct / total

                if (i + 1) % num_print == 0:
                    print(
                        f"進度: [{i + 1}/{len(self.train_iter)}], 訓練損失: {train_loss / (i + 1):.3f} | 訓練準確度: {train_acc:6.3f}% | 學習率: {self.get_cur_lr()}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"花費時間: {time.time() - start:.4f}秒")

            if self.val_iter is not None:
                self.record_test.append(self.validate())
            self.record_train.append(train_acc)
            torch.save(self.net.state_dict(
            ), os.path.join(self.config['weight_path'],
                            f"1DCNN_AND_LSTM_modelTrainer_{epoch + 1}_acc={self.record_test[epoch]:.3f}.pt"))
            print("儲存權重完成")
        torch.save(self.net.state_dict(),
                   os.path.join(self.config['weight_path'], f"1DCNN_AND_LSTM__modelTrainer_full.pt"))

    def validate(self):
        total, correct = 0, 0
        self.net.eval()
        labels = [0, 1]
        class_names = ['no_up_trend', 'up_trend']
        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            print("*************** validation ***************")
            for X, y in tqdm(self.val_iter):
                X, y = X.to(self.device), y.to(self.device)
                output = self.net(X)
                _, preds = torch.max(output, 1)
                loss = self.criterion(output, y)

                probs = F.softmax(output, dim=1)[:, 1]  # 取得屬於類別1的機率
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())  # 累積預測結果

                total += y.size(0)
                correct += (preds == y).sum().item()

            val_acc = 100.0 * correct / total
            print(f"validation loss: {loss.item():.3f} | validation accuracy: {val_acc:6.3f}%")

            # 二分類的AUC計算
            auc_score = roc_auc_score(all_labels, all_probs)
            print(f"Validation AUC: {auc_score:.3f}")

            # 混淆矩陣
            cm = confusion_matrix(all_labels, all_preds)
            print("Confusion Matrix:\n", cm)

            # 計算每個類別的召回率
            recall = recall_score(all_labels, all_preds, average=None, labels=labels)
            for i, class_name in enumerate(class_names):
                print(f"{class_name} Recall: {recall[i]:.3f}")

            # 計算每個類別的精確度
            precision = precision_score(all_labels, all_preds, average=None, labels=labels)
            for i, class_name in enumerate(class_names):
                print(f"{class_name} Precision: {precision[i]:.3f}")

            print("************************************\n")
        self.net.train()
        return val_acc

    def get_cur_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def learning_curve(self):
        plt.style.use("ggplot")
        plt.plot(range(1, len(self.record_train) + 1),
                 self.record_train, label="Training ACC")
        if self.record_test:
            plt.plot(range(1, len(self.record_test) + 1),
                     self.record_test, label="Validation ACC")
        plt.legend(loc=4)
        plt.title("Learning Curve")
        plt.xticks(range(0, len(self.record_train) + 1, 5))
        plt.yticks(range(0, 101, 5))
        plt.xlabel("Nums of Epoch")
        plt.ylabel("ACC")

        # 保存學習曲線圖像
        learning_curve_plt_path = os.path.join(self.config['weight_path'], 'learning_curve.png')
        plt.savefig(learning_curve_plt_path)
        plt.close()


def main():
    config = {
        'train_data_path': "你的訓練集路徑",
        'val_data_path': "你的驗證集路徑",
        'weight_path': "訓練玩模型的儲存路徑",
        'batch_size': 4,
        'num_epochs': 300,
        'learning_rate': 0.000125,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_classes': 2
    }
    os.makedirs(config['weight_path'], exist_ok=True)
    trainer = Trainer(config)
    start_time = time.time()
    trainer.train()
    trainer.learning_curve()
    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 打印結果
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
    trainer.learning_curve()


if __name__ == '__main__':
    main()
