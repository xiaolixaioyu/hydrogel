"""
水凝胶温度智能识别系统 - 模型评估脚本
计算MAE、RMSE、R²指标并生成可视化图表
"""

import os
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from data_loader import HydrogelDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='评估水凝胶温度预测模型')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径 (.h5)')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='数据集路径（默认: dataset）')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小（默认: 16）')
    parser.add_argument('--img_size', type=int, default=128,
                        help='图像尺寸（默认: 128）')
    parser.add_argument('--output', type=str, default='eval_output',
                        help='输出目录（默认: eval_output）')
    return parser.parse_args()


def predict_with_model(model, generator):
    """
    使用模型进行预测

    Args:
        model: 加载好的模型
        generator: 数据生成器

    Returns:
        tuple: (真实值数组, 预测值数组)
    """
    predictions = []
    true_values = []

    num_batches = np.ceil(generator.samples / generator.batch_size).astype(int)

    for i in range(num_batches):
        batch_x, batch_y = next(generator)
        batch_pred = model.predict(batch_x, verbose=0)

        predictions.extend(batch_pred.flatten())
        true_values.extend(batch_y.flatten())

        if i >= num_batches - 1:
            break

    predictions = np.array(predictions[:generator.samples])
    true_values = np.array(true_values[:generator.samples])

    return true_values, predictions


def calculate_metrics(true_values, predictions):
    """
    计算评估指标

    Args:
        true_values: 真实值
        predictions: 预测值

    Returns:
        dict: 评估指标
    """
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)

    mse = mean_squared_error(true_values, predictions)

    residuals = predictions - true_values

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'residuals': residuals,
        'mean_error': np.mean(residuals),
        'std_error': np.std(residuals),
        'max_error': np.max(np.abs(residuals))
    }


def plot_scatter(true_values, predictions, output_dir):
    """
    绘制预测散点图

    Args:
        true_values: 真实值
        predictions: 预测值
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(true_values, predictions, alpha=0.5, s=30)

    min_val = min(true_values.min(), predictions.min()) - 0.5
    max_val = max(true_values.max(), predictions.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')

    ax.set_xlabel('真实温度 (°C)', fontsize=12)
    ax.set_ylabel('预测温度 (°C)', fontsize=12)
    ax.set_title('温度预测散点图', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    ax.text(0.05, 0.95, f'MAE: {mae:.3f}°C\nR²: {r2:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    scatter_path = os.path.join(output_dir, 'prediction_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()

    return scatter_path


def plot_residuals(true_values, residuals, output_dir):
    """
    绘制残差分布图

    Args:
        true_values: 真实值
        residuals: 残差
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(true_values, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('真实温度 (°C)', fontsize=11)
    axes[0].set_ylabel('残差 (°C)', fontsize=11)
    axes[0].set_title('残差分布图', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('残差 (°C)', fontsize=11)
    axes[1].set_ylabel('频数', fontsize=11)
    axes[1].set_title('残差直方图', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    residual_path = os.path.join(output_dir, 'residual_analysis.png')
    plt.savefig(residual_path, dpi=150, bbox_inches='tight')
    plt.close()

    return residual_path


def print_evaluation_report(metrics, dataset_name='验证集'):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print(f"模型评估报告 - {dataset_name}")
    print("=" * 60)
    print(f"平均绝对误差 (MAE): {metrics['MAE']:.4f}°C")
    print(f"均方根误差 (RMSE): {metrics['RMSE']:.4f}°C")
    print(f"均方误差 (MSE): {metrics['MSE']:.6f}")
    print(f"决定系数 (R²): {metrics['R2']:.4f}")
    print("-" * 60)
    print(f"平均误差: {metrics['mean_error']:.4f}°C")
    print(f"误差标准差: {metrics['std_error']:.4f}°C")
    print(f"最大误差: {metrics['max_error']:.4f}°C")
    print("=" * 60)

    print("\n性能指标评估:")
    if metrics['MAE'] < 0.5:
        print("  ✓ MAE < 0.5°C (目标达成)")
    else:
        print("  ✗ MAE >= 0.5°C (未达标)")

    if metrics['RMSE'] < 0.7:
        print("  ✓ RMSE < 0.7°C (目标达成)")
    else:
        print("  ✗ RMSE >= 0.7°C (未达标)")

    if metrics['R2'] > 0.95:
        print("  ✓ R² > 0.95 (目标达成)")
    else:
        print("  ✗ R² <= 0.95 (未达标)")


def evaluate_model(model_path, dataset_dir, batch_size=16, img_size=128,
                  output_dir='eval_output'):
    """
    评估模型

    Args:
        model_path: 模型文件路径
        dataset_dir: 数据集路径
        batch_size: 批次大小
        img_size: 图像尺寸
        output_dir: 输出目录

    Returns:
        dict: 评估指标
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("水凝胶温度预测模型评估")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"数据集: {dataset_dir}")
    print("=" * 60)

    print("\n[1/3] 加载模型...")
    model = load_model(model_path)
    print("模型加载完成")

    print("\n[2/3] 准备数据...")
    data_loader = HydrogelDataLoader(
        dataset_dir,
        img_size=(img_size, img_size),
        batch_size=batch_size
    )
    val_generator = data_loader.get_val_generator()
    print(f"验证集样本数: {val_generator.samples}")

    print("\n[3/3] 执行评估...")
    true_values, predictions = predict_with_model(model, val_generator)

    metrics = calculate_metrics(true_values, predictions)

    print_evaluation_report(metrics)

    print("\n生成可视化图表...")
    scatter_path = plot_scatter(true_values, predictions, output_dir)
    print(f"散点图: {scatter_path}")

    residual_path = plot_residuals(true_values, metrics['residuals'], output_dir)
    print(f"残差图: {residual_path}")

    np.savez(
        os.path.join(output_dir, 'evaluation_results.npz'),
        true_values=true_values,
        predictions=predictions,
        residuals=metrics['residuals']
    )

    return metrics


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    if not os.path.exists(args.dataset):
        print(f"错误: 数据集目录不存在: {args.dataset}")
        return

    evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
