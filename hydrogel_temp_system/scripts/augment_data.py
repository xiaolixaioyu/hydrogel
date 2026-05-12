"""
水凝胶温度智能识别系统 - 数据增强脚本
对现有数据集进行离线增强，扩充样本多样性
"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据集路径')
    parser.add_argument('--output', type=str, default='dataset_augmented',
                        help='输出路径（默认: dataset_augmented）')
    parser.add_argument('--augment_factor', type=int, default=2,
                        help='增强倍数（每张图片生成N张增强图，默认: 2）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    return parser.parse_args()


def apply_brightness(image, factor_range=(0.7, 1.3)):
    """调整亮度"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast(image, factor_range=(0.8, 1.2)):
    """调整对比度"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_saturation(image, factor_range=(0.8, 1.3)):
    """调整饱和度"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def apply_rotation(image, angle_range=(-15, 15)):
    """随机旋转"""
    angle = random.uniform(*angle_range)
    return image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def apply_horizontal_flip(image):
    """水平翻转"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def apply_zoom(image, zoom_range=(0.8, 1.2)):
    """随机缩放（裁剪并调整大小）"""
    zoom = random.uniform(*zoom_range)
    w, h = image.size

    new_w = int(w * zoom)
    new_h = int(h * zoom)

    left = random.randint(0, max(0, w - new_w))
    top = random.randint(0, max(0, h - new_h))

    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def apply_shift(image, shift_range=0.1):
    """随机平移"""
    w, h = image.size
    shift_x = int(w * random.uniform(-shift_range, shift_range))
    shift_y = int(h * random.uniform(-shift_range, shift_range))

    from PIL import ImageOps
    shifted = ImageOps.expand(image, border=(abs(shift_x), abs(shift_y)), fillcolor=(128, 128, 128))
    return shifted.crop((
        abs(shift_x) - shift_x if shift_x > 0 else 0,
        abs(shift_y) - shift_y if shift_y > 0 else 0,
        w + abs(shift_x) - shift_x if shift_x > 0 else w,
        h + abs(shift_y) - shift_y if shift_y > 0 else h
    ))


def generate_augmented_image(image):
    """对一张图片应用随机增强"""
    augmentation_functions = [
        ('brightness', apply_brightness),
        ('contrast', apply_contrast),
        ('saturation', apply_saturation),
        ('rotation', apply_rotation),
        ('zoom', apply_zoom),
        ('shift', apply_shift),
    ]

    num_augmentations = random.randint(2, 4)

    selected_augmentations = random.sample(augmentation_functions, num_augmentations)

    result = image.copy()
    for name, func in selected_augmentations:
        try:
            result = func(result)
        except Exception as e:
            print(f"警告: 应用 {name} 增强失败: {e}")

    if random.random() > 0.5:
        result = apply_horizontal_flip(result)

    return result


def augment_dataset(input_dir, output_dir, augment_factor=2, seed=42):
    """
    增强数据集

    Args:
        input_dir: 输入数据集路径
        output_dir: 输出数据集路径
        augment_factor: 增强倍数
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    print(f"发现 {len(image_files)} 张图像文件")

    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc="增强图像"):
        try:
            img = Image.open(img_path)

            rel_path = os.path.relpath(img_path, input_dir)
            base_name = Path(rel_path).stem
            ext = Path(rel_path).suffix

            output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(output_subdir, exist_ok=True)

            original_out = os.path.join(output_subdir, f"{base_name}_orig{ext}")
            img.save(original_out)

            for i in range(augment_factor):
                augmented = generate_augmented_image(img)
                aug_name = f"{base_name}_aug{i+1}{ext}"
                aug_out = os.path.join(output_subdir, aug_name)
                augmented.save(aug_out)

        except Exception as e:
            print(f"\n错误: 处理 {img_path} 失败: {e}")

    print(f"\n增强完成！")
    print(f"原始图像: {len(image_files)} 张")
    print(f"增强后图像: {len(image_files) * (augment_factor + 1)} 张")


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return

    print(f"开始数据增强...")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"增强倍数: {args.augment_factor}")
    print(f"随机种子: {args.seed}")

    augment_dataset(
        args.input,
        args.output,
        args.augment_factor,
        args.seed
    )

    print("\n数据增强完成!")


if __name__ == '__main__':
    main()
