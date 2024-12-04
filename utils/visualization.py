# 1. 导入必要的库和设置
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.vae import TemporalVAE
from utils.data_loader import DengueDataset
from config import Config
import pandas as pd

# 2. 加载数据和模型
class ModelLoader:
    def __init__(self, model_path="checkpoints/best_models/best_model.pth"):
        self.config = Config()
        self.device = self.config.DEVICE

        # 加载数据
        self.dataset = DengueDataset(self.config)
        self.district_labels = np.load('Data/vae_district_labels.npy', allow_pickle=True)
        self.time_labels = np.load('Data/vae_time_labels.npy', allow_pickle=True)

        # 加载特征名称
        with open('Data/vae_feature_names.txt', 'r') as f:
            self.feature_names = f.read().splitlines()

        # 初始化模型
        input_dim = self.dataset.data.shape[-1]
        self.model = TemporalVAE(
            input_dim=input_dim,
            hidden_dim=self.config.HIDDEN_DIM,
            latent_dim=self.config.LATENT_DIM,
            time_window=self.config.TIME_WINDOW
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

# 4. 时间序列可视化
def plot_sequence(idx, model_loader):
    """可视化单个时间序列"""
    with torch.no_grad():
        data = model_loader.dataset[idx].unsqueeze(0).to(model_loader.device)
        recon, mu, log_var = model_loader.model(data)

    original = data.cpu().numpy()[0]
    reconstruction = recon.cpu().numpy()[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # 原始序列
    sns.heatmap(original.T, ax=ax1,
                xticklabels=range(model_loader.config.TIME_WINDOW),
                yticklabels=model_loader.feature_names,
                cmap='YlOrRd')
    ax1.set_title(f'Original Sequence - District: {model_loader.district_labels[idx]}')

    # 重建序列
    sns.heatmap(reconstruction.T, ax=ax2,
                xticklabels=range(model_loader.config.TIME_WINDOW),
                yticklabels=model_loader.feature_names,
                cmap='YlOrRd')
    ax2.set_title('Reconstructed Sequence')

    plt.tight_layout()
    plt.show()

    return original, reconstruction


# 5. 重建误差分析
def analyze_reconstruction_error(original, reconstruction, model_loader):
    error = np.abs(original - reconstruction)

    plt.figure(figsize=(12, 4))
    sns.heatmap(error.T,
                xticklabels=range(model_loader.config.TIME_WINDOW),
                yticklabels=model_loader.feature_names,
                cmap='YlOrRd')
    plt.title('Reconstruction Error')
    plt.show()

    # 打印每个特征的平均误差
    mean_error = error.mean(axis=0)
    error_df = pd.DataFrame({
        'Feature': model_loader.feature_names,
        'Mean Error': mean_error
    })
    return error_df.sort_values('Mean Error', ascending=False)


# 6. 潜在空间可视化
def visualize_latent_space(model_loader, n_samples=1000):
    encoded_data = []
    districts = []

    with torch.no_grad():
        for i in range(min(n_samples, len(model_loader.dataset))):
            data = model_loader.dataset[i].unsqueeze(0).to(model_loader.device)
            mu, _ = model_loader.model.encoder(data)
            encoded_data.append(mu.cpu().numpy())
            districts.append(model_loader.district_labels[i])

    encoded_data = np.vstack(encoded_data)

    # t-SNE降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    encoded_2d = tsne.fit_transform(encoded_data)

    # 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1],
                          c=pd.Categorical(districts).codes,
                          cmap='tab20', alpha=0.6)
    plt.title('Latent Space Visualization (t-SNE)')
    plt.colorbar(scatter, label='District')
    plt.show()


# 7. 时间趋势分析
def analyze_temporal_trends(district_name, model_loader):
    district_indices = np.where(model_loader.district_labels == district_name)[0]

    if len(district_indices) == 0:
        print(f"No data found for district: {district_name}")
        return

    # 收集该地区的AGI和ADI值
    times = []
    agi_values = []
    adi_values = []

    for idx in district_indices:
        data = model_loader.dataset[idx].numpy()
        times.append(model_loader.time_labels[idx])
        # 假设AGI和ADI是前两个特征
        agi_values.append(data[:, 0].mean())  # 取平均值
        adi_values.append(data[:, 1].mean())

    # 创建时间序列图
    plt.figure(figsize=(15, 6))
    plt.plot(times, agi_values, label='AGI', marker='o')
    plt.plot(times, adi_values, label='ADI', marker='o')
    plt.title(f'Temporal Trends for {district_name}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 8. 生成新样本
def generate_and_visualize_samples(model_loader, n_samples=3):
    with torch.no_grad():
        # 从正态分布采样潜在向量
        z = torch.randn(n_samples, model_loader.config.LATENT_DIM).to(model_loader.device)
        # 生成新样本
        generated = model_loader.model.decoder(z)
        generated = generated.cpu().numpy()

    # 可视化生成的样本
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 5 * n_samples))
    for i in range(n_samples):
        sns.heatmap(generated[i].T, ax=axes[i],
                    xticklabels=range(model_loader.config.TIME_WINDOW),
                    yticklabels=model_loader.feature_names,
                    cmap='YlOrRd')
        axes[i].set_title(f'Generated Sample {i + 1}')

    plt.tight_layout()
    plt.show()


# 9. 特征相关性分析
def analyze_feature_correlations(model_loader):
    # 获取所有数据
    all_data = model_loader.dataset.data.numpy()
    # 重塑数据为2D形式
    reshaped_data = all_data.reshape(-1, all_data.shape[-1])

    # 计算相关性矩阵
    corr_matrix = np.corrcoef(reshaped_data.T)

    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                xticklabels=model_loader.feature_names,
                yticklabels=model_loader.feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.show()