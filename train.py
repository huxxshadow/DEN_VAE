# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.vae import TemporalVAE
from utils.data_loader import get_data_loaders
from config import Config
import numpy as np
from dtaidistance import dtw  # 引入DTW库
import matplotlib.pyplot as plt

def dtw_loss(recon_x, x):
    """
    计算DTW损失，优化为批量计算
    recon_x: [batch_size, time_window, input_dim]
    x: [batch_size, time_window, input_dim]
    """
    batch_size, time_window, input_dim = recon_x.size()
    recon_x = recon_x.view(batch_size, time_window, input_dim).detach().cpu().numpy()
    x = x.view(batch_size, time_window, input_dim).detach().cpu().numpy()
    loss = 0
    for i in range(batch_size):
        for j in range(input_dim):
            loss += dtw.distance(recon_x[i,:,j], x[i,:,j])
    return torch.FloatTensor([loss]).to(config.DEVICE)

def loss_function(recon_x, x, mu, log_var, beta, gamma):
    """
    计算总损失，包括重建损失、KL散度和DTW损失
    """
    reconstruction_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    temporal_loss = dtw_loss(recon_x, x)
    return reconstruction_loss + beta * kl_loss + gamma * temporal_loss

def evaluate(model, data_loader, config):
    """评估函数，返回平均损失和平均MAE"""
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(config.DEVICE)
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var, config.BETA, config.GAMMA)
            total_loss += loss.item()
            total_mae += nn.L1Loss(reduction='sum')(recon_batch, data).item()
    avg_loss = total_loss / len(data_loader.dataset)
    avg_mae = total_mae / len(data_loader.dataset)
    return avg_loss, avg_mae

def train_model(config):
    # 创建检查点目录
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建最佳模型目录
    best_model_dir = os.path.join(checkpoint_dir, 'best_models')
    os.makedirs(best_model_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(config)

    input_dim = next(iter(train_loader)).shape[-1]
    model = TemporalVAE(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        time_window=config.TIME_WINDOW
    ).to(config.DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    # 创建日志目录并打开日志文件
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'training_log.txt')

    # 初始化最佳损失和早停计数器
    best_loss = float('inf')
    best_epoch = 0
    patience = 30  # 早停耐心值
    patience_counter = 0

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'best_epochs': []
    }

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mae = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(config.DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                recon_batch, mu, log_var = model(data)
                loss = loss_function(recon_batch, data, mu, log_var, config.BETA, config.GAMMA)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mae += nn.L1Loss(reduction='sum')(recon_batch, data).item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mae = train_mae / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_mae = evaluate(model, val_loader, config)

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # 打印当前轮次的训练情况
        print(f'Epoch: {epoch+1}/{config.NUM_EPOCHS}')
        print(f'Training Loss: {avg_train_loss:.6f}, Training MAE: {avg_train_mae:.6f}')
        print(f'Validation Loss: {val_loss:.6f}, Validation MAE: {val_mae:.6f}')

        # 保存日志
        with open(log_file, 'a') as f:
            f.write(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss:.6f}, Training MAE: {avg_train_mae:.6f}, '
                    f'Validation Loss: {val_loss:.6f}, Validation MAE: {val_mae:.6f}\n')

        # 调整学习率
        scheduler.step()

        # 检查是否是最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            history['best_epochs'].append(epoch + 1)

            # 保存最佳模型
            best_model_path = os.path.join(best_model_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config.__dict__
            }, best_model_path)

            print(f'新的最佳模型已在第 {epoch+1} 轮次保存')
        else:
            patience_counter += 1

        # 检查早停
        if patience_counter >= patience:
            print(f'在第 {epoch+1} 轮次触发早停')
            break

    # 保存训练历史
    np.save(os.path.join(log_dir, 'training_history.npy'), history)

    # 打印最终训练结果
    print("\n训练完成!")
    print(f"最佳模型在第 {best_epoch} 轮次保存，损失为 {best_loss:.6f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.scatter(history['best_epochs'],
                [history['val_loss'][i-1] for i in history['best_epochs']],
                c='r', marker='*', label='最佳模型')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True)

    # 设置中文字体路径，例如 SimHei
    plt.rcParams['font.family'] = ['SimHei']  # 或者 'Microsoft YaHei', 'Noto Sans CJK'
    plt.savefig(os.path.join(log_dir, 'training_curve.png'))
    plt.close()

    return best_model_path

if __name__ == "__main__":
    config = Config()
    best_model_path = train_model(config)
    print(f"最佳模型保存在 {best_model_path}")