import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score


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
        self.data, self.labels, self.timestamps = self.load_and_process_data()

    def load_and_process_data(self):
        data = pd.read_csv(self.filepath)
        data = data[['timestamp', 'quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']]
        # 標準化
        data[['quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']] = (
                                                                                            data[['quote.USD.price',
                                                                                                  'quote.USD.volume_24h',
                                                                                                  'quote.USD.market_cap']] - self.mean) / self.std
        return self.create_features_and_labels(data)

    def create_features_and_labels(self, data):
        X, y, timestamps = [], [], []
        for i in range(self.lookback, len(data) - self.forecast):
            window = data.iloc[i - self.lookback:i]
            X.append(window[['quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']].values)
            future_prices = data['quote.USD.price'][i:i + self.forecast]
            label = int(np.any(future_prices > data['quote.USD.price'].iloc[i - 1]))
            y.append(label)
            timestamps.append(window['timestamp'].iloc[-1])
        return np.array(X), np.array(y), np.array(timestamps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).transpose(0, 1)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        timestamp = self.timestamps[idx]
        return data_tensor, label_tensor, timestamp


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mean, self.std = self.compute_mean_std(self.config['train_data_path'])
        self.test_iter = DataLoader(CSVDataset(filepath=self.config['test_data_path'], mean=self.mean, std=self.std),
                                    batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        self.net = CNNLSTMModel().to(self.device)
        self.load_model()
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def compute_mean_std(filepath):
        data = pd.read_csv(filepath)
        data = data[['quote.USD.price', 'quote.USD.volume_24h', 'quote.USD.market_cap']]
        mean = data.mean().values
        std = data.std().values
        return mean, std

    def load_model(self):
        self.net.load_state_dict(torch.load(self.config['model_path']))
        self.net.to(self.device)

    def analyze(self):
        self.net.eval()
        true_labels = []
        predictions = []

        timestamps = self.dataset.timestamps

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_iter, desc="Processing", leave=True):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)

                # 收集真實標籤和預測標籤
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())

        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:")
        print(cm)

        # 計算精確率
        precision = precision_score(true_labels, predictions, average='binary')  # 'binary'適用於二分類問題
        print(f"Precision: {precision:.2f}")

        # 計算每個類別的精確率
        precision_each_class = precision_score(true_labels, predictions, average=None)  # 傳回每個類別的精確率
        print(f"Precision for each class: {precision_each_class}")

        class_mapping = {0: 'no_up_trend', 1: 'up_trend'}
        predictions = [class_mapping[pred] if pred in class_mapping else pred for pred in predictions]

        # 建立DataFrame並儲存
        prediction_df = pd.DataFrame({"Timestamp": timestamps, "Prediction": predictions})
        prediction_df.sort_values(by="Timestamp", inplace=True)  # 根據指定列排序
        os.makedirs(self.config['Prediction_csv_save_path'], exist_ok=True)
        prediction_df.to_csv(
            os.path.join(self.config['Prediction_csv_save_path'], 'prediction.csv'),
            index=False)


def main():
    config = {
        'batch_size': "能放幾個就幾個",
        'train_data_path': "你的訓練集資料",
        'test_data_path': "你的測試集資料",
        'model_path': '你的模型處存的絕對路徑，結尾是.pt的檔案',
        'Prediction_csv_save_path': '你的預測csv結果想存去哪個資料夾'
    }

    # 加載模型並進行錯誤分析
    start_time = time.time()
    analyzer = Tester(config)
    analyzer.analyze()
    end_time = time.time()
    total_time = end_time - start_time

    # 計算小時、分鐘和秒
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # 打印結果
    print(f"Total testing time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")


if __name__ == '__main__':
    main()
