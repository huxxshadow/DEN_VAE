import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


import pandas as pd
from datetime import timedelta


# 2. 时间序列预测分析 - 逐步预测实现
class TimeSeriesPredictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.scaler = MinMaxScaler()

    def prepare_prediction_data(self, district_name, feature_index=0):
        """准备指定地区的预测数据"""
        district_indices = np.where(self.model_loader.district_labels == district_name)[0]

        if len(district_indices) == 0:
            raise ValueError(f"No data found for district: {district_name}")

        # 收集时间序列数据
        times = []
        values = []
        for idx in district_indices:
            data = self.model_loader.dataset[idx].numpy()
            times.append(pd.to_datetime(self.model_loader.time_labels[idx]))
            values.append(data[:, feature_index].mean())

        # 确保时间顺序
        time_series = pd.DataFrame({
            'time': times,
            'value': values
        }).sort_values('time')

        return time_series

    def predict_next_step(self, current_sequence, feature_index):
        """
        使用模型预测下一个时间步的值
        current_sequence: [1, time_window, input_dim]
        feature_index: 目标特征的索引
        """
        with torch.no_grad():
            recon, mu, log_var = self.model_loader.model(current_sequence)
            # 只获取最后一个时间步的预测
            next_pred = recon[:, -1:, :]  # [1, 1, input_dim]
            # 选择目标特征
            next_value = next_pred[:, :, feature_index]  # [1, 1]
        return next_value.squeeze().cpu().numpy()  # 返回标量

    def predict_next_steps_iterative(self, district_name, steps=6, feature_name='AGI'):
        """逐步预测未来几个时间步的值"""
        # 获取特征索引
        feature_index = self.model_loader.feature_names.index(feature_name)

        # 准备数据
        time_series = self.prepare_prediction_data(district_name, feature_index)

        # 获取最后一个完整序列
        district_indices = np.where(self.model_loader.district_labels == district_name)[0]
        last_idx = district_indices[-1]
        last_sequence = self.model_loader.dataset[last_idx].unsqueeze(0).to(
            self.model_loader.device)  # [1, time_window, input_dim]

        # 初始化预测列表
        predictions = []
        current_sequence = last_sequence.clone()

        for _ in range(steps):
            # 预测下一个时间步
            next_val = self.predict_next_step(current_sequence, feature_index)  # 标量

            # 创建新的预测点，保持其他特征不变或使用历史平均值
            input_dim = current_sequence.shape[2]
            new_pred = np.zeros(input_dim)
            new_pred[feature_index] = next_val
            new_pred = torch.FloatTensor(new_pred).unsqueeze(0).unsqueeze(0).to(
                self.model_loader.device)  # [1,1,input_dim]

            # 更新当前序列
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],  # 移除最早的时间步
                new_pred  # 添加新的预测时间步
            ], dim=1)  # [1, time_window, input_dim]

            # 保存预测值
            predictions.append(next_val)

        # 生成未来时间点
        last_time = time_series['time'].iloc[-1]
        future_times = [last_time + timedelta(days=30 * (i + 1)) for i in range(steps)]

        return future_times, predictions

    def plot_predictions_iterative(self, district_name, feature_name='AGI', steps=6, window_size=3):
        """可视化历史数据和逐步预测结果，并进行平滑处理"""
        feature_index = self.model_loader.feature_names.index(feature_name)

        # 获取历史数据
        historical_data = self.prepare_prediction_data(district_name, feature_index)

        # 获取预测
        future_times, predictions = self.predict_next_steps_iterative(
            district_name, steps, feature_name
        )

        # 应用移动平均平滑
        if len(predictions) >= window_size:
            smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')
            # 为保持与时间点对齐，前面填充
            smoothed_predictions = np.pad(smoothed_predictions, (window_size - 1, 0), 'edge')
        else:
            smoothed_predictions = predictions

        # 创建图形
        plt.figure(figsize=(15, 6))

        # 绘制历史数据
        plt.plot(historical_data['time'], historical_data['value'],
                 'b.-', label='Historical Data')

        # 绘制预测（原始）
        plt.plot(future_times, predictions,
                 'r.--', label='Predictions (Original)')

        # 绘制预测（平滑）
        plt.plot(future_times, smoothed_predictions,
                 'g.-', label=f'Predictions (Smoothed, window={window_size})')

        # 添加置信区间（使用平滑后的标准差）
        pred_std = historical_data['value'].std()
        plt.fill_between(future_times,
                         smoothed_predictions - 1.96 * pred_std,
                         smoothed_predictions + 1.96 * pred_std,
                         color='g', alpha=0.2,
                         label='95% Confidence Interval (Smoothed)')

        plt.title(f'{feature_name} Prediction for {district_name} (Iterative)')
        plt.xlabel('Time')
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True)

        # 添加垂直线分隔历史数据和预测
        plt.axvline(x=historical_data['time'].iloc[-1],
                    color='gray', linestyle='--', alpha=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return historical_data, future_times, predictions, smoothed_predictions

    def analyze_prediction_uncertainty_iterative(self, district_name, feature_name='AGI',
                                                 steps=6, n_samples=100):
        """通过多次采样分析预测的不确定性（逐步预测）"""
        feature_index = self.model_loader.feature_names.index(feature_name)
        all_predictions = []

        for _ in range(n_samples):
            _, predictions = self.predict_next_steps_iterative(
                district_name, steps, feature_name
            )
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)

        # 计算统计量
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions
