# config.py
import torch

class Config:
    # 数据相关
    DATA_PATH = "Data/"
    BATCH_SIZE = 64
    TIME_WINDOW = 6  # 时间窗口大小

    # 模型相关
    HIDDEN_DIM = 256  # 增加隐藏层维度
    LATENT_DIM = 128   # 增加潜在空间维度
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 300
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.2

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练相关
    BETA = 0.5  # KL损失权重
    GAMMA = 0.1  # DTW损失权重
    TREND_WEIGHT = 0.3  # 趋势损失权重

    # 优化器相关
    WEIGHT_DECAY = 1e-5