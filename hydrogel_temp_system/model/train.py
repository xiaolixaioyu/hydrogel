"""
水凝胶温度智能识别系统 - 模型训练脚本
包含EarlyStopping、ReduceLROnPlateau、ModelCheckpoint回调
"""

import os
import argparse
import json
from datetime import datetime

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model

from model import build_cnn_model
from data_loader import HydrogelDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='训练水凝胶温度预测模型')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='数据集路径（默认: dataset）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='最大训练轮数（默认: 100）')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小（默认: 16）')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像尺寸（默认: 128）')
    parser.add_argument('--output', type=str, default='model_output',
                        help='输出目录（默认: model_output）')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率（默认: 0.001）')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值（默认: 15）')
    return parser.parse_args()


def plot_training_history(history, output_dir):
    """
    绘制训练曲线

    Args:
        history: 训练历史记录
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='训练损失')
    axes[0].plot(history.history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失 (MSE)')
    axes[0].set_title('训练损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='训练MAE')
    axes[1].plot(history.history['val_mae'], label='验证MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('平均绝对误差 (°C)')
    axes[1].set_title('平均绝对误差曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    loss_curve_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()

    return loss_curve_path


def save_training_history(history, output_dir):
    """
    保存训练历史为JSON

    Args:
        history: 训练历史记录
        output_dir: 输出目录
    """
    history_dict = {
        'epoch': history.epoch,
        'history': {k: [float(v) for v in values] for k, values in history.history.items()}
    }

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)

    return history_path


def train_model(dataset_dir, epochs=100, batch_size=16, img_size=128,
                output_dir='model_output', lr=0.001, patience=15):
    """
    训练模型

    Args:
        dataset_dir: 数据集路径
        epochs: 最大训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        output_dir: 输出目录
        lr: 初始学习率
        patience: 早停耐心值

    Returns:
        训练好的模型和历史记录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("水凝胶温度预测模型训练")
    print("=" * 60)
    print(f"数据集: {dataset_dir}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"学习率: {lr}")
    print(f"早停耐心值: {patience}")
    print("=" * 60)

    print("\n[1/4] 初始化数据加载器...")
    data_loader = HydrogelDataLoader(
        dataset_dir,
        img_size=(img_size, img_size),
        batch_size=batch_size
    )

    dataflow = data_loader.validate_dataflow()
    print(f"训练集: {dataflow['train']['samples']} 样本")
    print(f"验证集: {dataflow['val']['samples']} 样本")

    if dataflow['train']['samples'] == 0:
        raise ValueError("训练集为空，请检查数据集路径")

    train_generator = data_loader.get_train_generator()
    val_generator = data_loader.get_val_generator()

    print("\n[2/4] 构建模型...")
    model = build_cnn_model(input_shape=(img_size, img_size, 3))
    print("模型构建完成")

    print("\n[3/4] 配置回调函数...")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n[4/4] 开始训练...")
    print("-" * 60)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    print("-" * 60)
    print("\n训练完成!")

    print("\n保存训练结果...")
    model_path = os.path.join(output_dir, 'final_model.h5')
    model.save(model_path)
    print(f"最终模型: {model_path}")

    plot_path = plot_training_history(history, output_dir)
    print(f"训练曲线: {plot_path}")

    history_path = save_training_history(history, output_dir)
    print(f"训练历史: {history_path}")

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]

    print("\n" + "=" * 60)
    print("训练结果摘要")
    print("=" * 60)
    print(f"最终训练损失: {final_train_loss:.4f}")
    print(f"最终验证损失: {final_val_loss:.4f}")
    print(f"最终训练MAE: {final_train_mae:.4f}°C")
    print(f"最终验证MAE: {final_val_mae:.4f}°C")
    print(f"训练轮数: {len(history.history['loss'])}")
    print("=" * 60)

    return model, history


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"错误: 数据集目录不存在: {args.dataset}")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'run_{timestamp}')

    train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=output_dir,
        lr=args.lr,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
