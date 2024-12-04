# 11. 总体趋势分析和预测 - 修改版

from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




class GlobalTrendAnalyzer:
    def __init__(self, model_loader, predictor):
        self.model_loader = model_loader
        self.time_series_predictor = predictor

    def collect_global_data(self, feature_name='AGI'):
        """收集所有地区的数据并计算总体趋势"""
        feature_index = self.model_loader.feature_names.index(feature_name)

        # 创建时间序列数据框
        all_data = []
        for idx in range(len(self.model_loader.dataset)):
            data = self.model_loader.dataset[idx].numpy()
            time = pd.to_datetime(self.model_loader.time_labels[idx])
            district = self.model_loader.district_labels[idx]
            value = data[:, feature_index].mean()

            all_data.append({
                'time': time,
                'district': district,
                'value': value
            })

        df = pd.DataFrame(all_data)
        return df

    def analyze_global_trends(self, feature_name='AGI'):
        """分析和可视化总体趋势"""
        df = self.collect_global_data(feature_name)

        # 计算全局平均趋势
        global_trend = df.groupby('time', observed=False)['value'].agg(['mean', 'std']).reset_index()

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 1. 所有地区的时间序列
        for district in df['district'].unique():
            district_data = df[df['district'] == district]
            ax1.plot(district_data['time'], district_data['value'],
                     alpha=0.3, label=district)

        # 添加全局平均线
        ax1.plot(global_trend['time'], global_trend['mean'],
                 'r-', linewidth=2, label='Global Mean')

        ax1.set_title(f'Individual District {feature_name} Trends')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(feature_name)
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. 箱型图显示分布变化
        ax2.boxplot([df[df['time'] == time]['value']
                     for time in df['time'].unique()],
                    tick_labels=[t.strftime('%Y-%m') for t in df['time'].unique()])

        ax2.set_title(f'Distribution of {feature_name} Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(feature_name)
        ax2.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return global_trend

    def predict_global_trend(self, feature_name='AGI', steps=6, n_samples=100):
        """预测总体趋势"""
        predictor = self.time_series_predictor

        # 收集所有地区的预测
        all_predictions = []
        districts = np.unique(self.model_loader.district_labels)

        # 禁用matplotlib的交互式显示
        plt.ioff()

        for district in districts:
            try:
                mean_pred, std_pred = predictor.analyze_prediction_uncertainty_iterative(
                    district, feature_name, steps, n_samples=20  # 使用逐步预测方法
                )
                all_predictions.append(mean_pred)
            except ValueError as e:
                print(e)
                continue

        # 重新启用matplotlib的交互式显示
        plt.ion()

        all_predictions = np.array(all_predictions)

        # 计算全局预测统计量
        global_mean = np.mean(all_predictions, axis=0)
        global_std = np.std(all_predictions, axis=0)

        # 获取历史数据
        historical_data = self.collect_global_data(feature_name)
        global_historical = historical_data.groupby('time', observed=False)['value'].mean()

        # 生成未来时间点
        last_time = historical_data['time'].max()
        future_times = [last_time + timedelta(days=30 * (i + 1)) for i in range(steps)]

        # 可视化
        plt.figure(figsize=(15, 8))

        # 绘制历史趋势
        plt.plot(global_historical.index, global_historical.values,
                 'b.-', label='Historical Global Mean')

        # 绘制预测趋势
        plt.plot(future_times, global_mean,
                 'r.--', label='Predicted Global Mean')

        # 添加预测区间
        plt.fill_between(future_times,
                         global_mean - 2 * global_std,
                         global_mean + 2 * global_std,
                         color='r', alpha=0.2,
                         label='95% Prediction Interval')

        plt.title(f'Global {feature_name} Trend and Prediction')
        plt.xlabel('Time')
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # 添加垂直线分隔历史数据和预测
        plt.axvline(x=last_time, color='gray',
                    linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return global_historical, future_times, global_mean, global_std

    def analyze_seasonal_patterns(self, feature_name='AGI'):
        """分析季节性模式"""
        df = self.collect_global_data(feature_name)
        df['month'] = df['time'].dt.month
        df['season'] = pd.cut(df['month'],
                              bins=[0, 3, 6, 9, 12],
                              labels=['Spring', 'Summer', 'Fall', 'Winter'])

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. 月度箱型图
        sns.boxplot(data=df, x='month', y='value', ax=ax1)
        ax1.set_title(f'Monthly {feature_name} Distribution')
        ax1.set_xlabel('Month')
        ax1.set_ylabel(feature_name)

        # 2. 季节性箱型图
        sns.boxplot(data=df, x='season', y='value', ax=ax2)
        ax2.set_title(f'Seasonal {feature_name} Distribution')
        ax2.set_xlabel('Season')
        ax2.set_ylabel(feature_name)

        plt.tight_layout()
        plt.show()

        # 计算季节性统计量
        seasonal_stats = df.groupby('season', observed=False)['value'].agg(['mean', 'std'])
        return seasonal_stats

    def generate_trend_report(self, feature_name='AGI'):
        """生成趋势分析报告"""
        # 收集数据
        df = self.collect_global_data(feature_name)
        global_trend = df.groupby('time')['value'].agg(['mean', 'std'])

        # 计算关键指标
        overall_mean = df['value'].mean()
        overall_std = df['value'].std()
        trend_slope = np.polyfit(range(len(global_trend)), global_trend['mean'], 1)[0]

        # 打印报告
        print(f"\n=== {feature_name} 趋势分析报告 ===")
        print(f"\n总体统计:")
        print(f"平均值: {overall_mean:.2f}")
        print(f"标准差: {overall_std:.2f}")
        print(f"趋势斜率: {trend_slope:.2f}")

        # 季节性分析
        seasonal_stats = self.analyze_seasonal_patterns(feature_name)
        print("\n季节性统计:")
        print(seasonal_stats)

        return {
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'trend_slope': trend_slope,
            'seasonal_stats': seasonal_stats
        }
