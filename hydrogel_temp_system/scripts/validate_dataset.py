"""
水凝胶温度智能识别系统 - 数据集验证脚本
验证数据集完整性并生成统计报告
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser(description='验证数据集')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='数据集路径（默认: dataset）')
    parser.add_argument('--output', type=str, default='.',
                        help='输出报告路径（默认: 当前目录）')
    return parser.parse_args()


def get_image_count(directory):
    """统计目录下图像数量"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                count += 1

    return count


def validate_dataset(dataset_dir):
    """验证数据集完整性"""
    expected_temps = [f"{t:.1f}" for t in [35.0, 35.5, 36.0, 36.5, 37.0, 37.5,
                                            38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0]]

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    if not os.path.exists(train_dir):
        results['valid'] = False
        results['errors'].append(f"训练集目录不存在: {train_dir}")
    else:
        for temp in expected_temps:
            temp_dir = os.path.join(train_dir, temp)
            if not os.path.exists(temp_dir):
                results['warnings'].append(f"训练集缺少温度目录: {temp}°C")

    if not os.path.exists(val_dir):
        results['valid'] = False
        results['errors'].append(f"验证集目录不存在: {val_dir}")
    else:
        for temp in expected_temps:
            temp_dir = os.path.join(val_dir, temp)
            if not os.path.exists(temp_dir):
                results['warnings'].append(f"验证集缺少温度目录: {temp}°C")

    train_counts = {}
    val_counts = {}
    total_train = 0
    total_val = 0

    for temp in expected_temps:
        train_count = get_image_count(os.path.join(train_dir, temp)) if os.path.exists(os.path.join(train_dir, temp)) else 0
        val_count = get_image_count(os.path.join(val_dir, temp)) if os.path.exists(os.path.join(val_dir, temp)) else 0

        train_counts[temp] = train_count
        val_counts[temp] = val_count
        total_train += train_count
        total_val += val_count

    results['statistics'] = {
        'train_counts': train_counts,
        'val_counts': val_counts,
        'total_train': total_train,
        'total_val': total_val,
        'total': total_train + total_val
    }

    if total_train == 0:
        results['valid'] = False
        results['errors'].append("训练集为空")

    if total_val == 0:
        results['valid'] = False
        results['errors'].append("验证集为空")

    return results


def generate_visualization(results, output_dir):
    """生成数据分布可视化图表"""
    train_counts = results['statistics']['train_counts']
    val_counts = results['statistics']['val_counts']

    temps = sorted(train_counts.keys(), key=float)
    train_values = [train_counts[t] for t in temps]
    val_values = [val_counts[t] for t in temps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(temps))
    width = 0.35

    ax1.bar([i - width/2 for i in x], train_values, width, label='训练集', color='#3498db')
    ax1.bar([i + width/2 for i in x], val_values, width, label='验证集', color='#e74c3c')
    ax1.set_xlabel('温度 (°C)')
    ax1.set_ylabel('图像数量')
    ax1.set_title('各温度点样本分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels(temps, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(['训练集', '验证集'], [results['statistics']['total_train'],
                                  results['statistics']['total_val']],
            color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('图像数量')
    ax2.set_title('训练集/验证集划分')
    for i, v in enumerate([results['statistics']['total_train'],
                           results['statistics']['total_val']]):
        ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    chart_path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    return chart_path


def print_report(results):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("数据集验证报告")
    print("=" * 60)

    if results['valid']:
        print("✓ 数据集验证通过")
    else:
        print("✗ 数据集验证失败")

    print("\n错误:")
    if results['errors']:
        for error in results['errors']:
            print(f"  ✗ {error}")
    else:
        print("  无")

    print("\n警告:")
    if results['warnings']:
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    else:
        print("  无")

    stats = results['statistics']
    print("\n统计信息:")
    print("-" * 60)
    print(f"{'温度(°C)':<12} {'训练集':<12} {'验证集':<12} {'总计':<12}")
    print("-" * 60)

    for temp in sorted(stats['train_counts'].keys(), key=float):
        train = stats['train_counts'][temp]
        val = stats['val_counts'][temp]
        print(f"{temp:<12} {train:<12} {val:<12} {train + val:<12}")

    print("-" * 60)
    print(f"{'总计':<12} {stats['total_train']:<12} {stats['total_val']:<12} {stats['total']:<12}")
    print("=" * 60)


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"错误: 数据集目录不存在: {args.dataset}")
        return

    print(f"正在验证数据集: {args.dataset}")

    results = validate_dataset(args.dataset)

    print_report(results)

    chart_path = generate_visualization(results, args.output)
    print(f"\n数据分布图表已保存至: {chart_path}")

    report_path = os.path.join(args.output, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"验证报告已保存至: {report_path}")


if __name__ == '__main__':
    main()
