"""
水凝胶温度智能识别系统 - 数据集整理脚本
用于从原始采集图片按温度分类，并划分训练集/验证集
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import json


def parse_args():
    parser = argparse.ArgumentParser(description='整理水凝胶图像数据集')
    parser.add_argument('--source', type=str, required=True,
                        help='原始数据集路径')
    parser.add_argument('--output', type=str, default='dataset',
                        help='输出路径（默认: dataset）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    return parser.parse_args()


def scan_images(source_dir):
    """扫描源目录，返回所有图像文件路径"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def extract_temperature_from_filename(filename):
    """
    从文件名提取温度值
    文件命名规则: [温度]_[四位序号].jpg (如 37.5_0001.jpg)
    """
    basename = os.path.basename(filename)
    name_without_ext = Path(basename).stem

    parts = name_without_ext.split('_')
    if len(parts) >= 1:
        try:
            temp = float(parts[0])
            if 35.0 <= temp <= 41.0:
                return temp
        except ValueError:
            pass

    return None


def organize_by_temperature(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    按温度分类并划分数据集

    Args:
        source_dir: 原始数据集路径
        output_dir: 输出路径
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)

    image_files = scan_images(source_dir)
    print(f"发现 {len(image_files)} 张图像文件")

    temp_groups = defaultdict(list)

    for img_path in image_files:
        temp = extract_temperature_from_filename(img_path)
        if temp is not None:
            temp_groups[temp].append(img_path)
        else:
            print(f"警告: 无法从文件提取温度: {img_path}")

    temp_distribution = {}
    for temp, files in sorted(temp_groups.items()):
        temp_str = f"{temp:.1f}"
        random.shuffle(files)

        train_count = int(len(files) * train_ratio)
        train_files = files[:train_count]
        val_files = files[train_count:]

        temp_distribution[temp_str] = {
            'total': len(files),
            'train': len(train_files),
            'val': len(val_files)
        }

        for f in train_files:
            dest_dir = os.path.join(output_dir, 'train', temp_str)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))

        for f in val_files:
            dest_dir = os.path.join(output_dir, 'val', temp_str)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))

    return temp_distribution


def generate_report(temp_distribution, output_path='dataset_report.json'):
    """生成数据集统计报告"""
    total_train = sum(d['train'] for d in temp_distribution.values())
    total_val = sum(d['val'] for d in temp_distribution.values())
    total = total_train + total_val

    report = {
        'summary': {
            'total_images': total,
            'train_images': total_train,
            'val_images': total_val,
            'train_ratio': total_train / total if total > 0 else 0,
            'val_ratio': total_val / total if total > 0 else 0
        },
        'temperature_distribution': temp_distribution
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("数据集统计报告")
    print("=" * 50)
    print(f"总图像数: {total}")
    print(f"训练集: {total_train} ({total_train/total*100:.1f}%)")
    print(f"验证集: {total_val} ({total_val/total*100:.1f}%)")
    print(f"\n报告已保存至: {output_path}")
    print("\n各温度点分布:")
    print("-" * 40)
    print(f"{'温度':<10} {'总数':<8} {'训练集':<8} {'验证集':<8}")
    print("-" * 40)
    for temp in sorted(temp_distribution.keys(), key=float):
        d = temp_distribution[temp]
        print(f"{temp:<10} {d['total']:<8} {d['train']:<8} {d['val']:<8}")

    return report


def main():
    args = parse_args()

    if not os.path.exists(args.source):
        print(f"错误: 源目录不存在: {args.source}")
        return

    print(f"开始整理数据集...")
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"随机种子: {args.seed}")

    temp_distribution = organize_by_temperature(
        args.source,
        args.output,
        args.train_ratio,
        args.seed
    )

    generate_report(temp_distribution)

    print("\n数据集整理完成!")


if __name__ == '__main__':
    main()
