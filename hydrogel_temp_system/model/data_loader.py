"""
水凝胶温度智能识别系统 - 数据加载器
使用ImageDataGenerator实现数据加载和预处理
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class HydrogelDataLoader:
    """
    水凝胶温度数据加载器
    负责加载训练集和验证集图像
    """

    def __init__(self, dataset_dir, img_size=(128, 128), batch_size=16):
        """
        初始化数据加载器

        Args:
            dataset_dir: 数据集根目录路径
            img_size: 图像目标尺寸，默认 (128, 128)
            batch_size: 批次大小，默认 16
        """
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=[0.8, 1.2],
            brightness_range=[0.7, 1.3],
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.val_datagen = ImageDataGenerator(rescale=1./255)

    def get_train_generator(self):
        """
        获取训练集数据生成器

        Returns:
            DirectoryIterator: 训练数据生成器
        """
        train_dir = os.path.join(self.dataset_dir, 'train')

        if not os.path.exists(train_dir):
            raise ValueError(f"训练集目录不存在: {train_dir}")

        return self.train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='raw',
            shuffle=True
        )

    def get_val_generator(self):
        """
        获取验证集数据生成器

        Returns:
            DirectoryIterator: 验证数据生成器
        """
        val_dir = os.path.join(self.dataset_dir, 'val')

        if not os.path.exists(val_dir):
            raise ValueError(f"验证集目录不存在: {val_dir}")

        return self.val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='raw',
            shuffle=False
        )

    def get_steps_per_epoch(self, dataset_type='train'):
        """
        计算每个epoch的步数

        Args:
            dataset_type: 'train' 或 'val'

        Returns:
            int: 每个epoch的步数
        """
        if dataset_type == 'train':
            generator = self.get_train_generator()
        else:
            generator = self.get_val_generator()

        return generator.samples // self.batch_size

    def validate_dataflow(self):
        """
        验证数据流是否正确

        Returns:
            dict: 验证结果
        """
        results = {
            'train': {'exists': False, 'samples': 0, 'classes': 0},
            'val': {'exists': False, 'samples': 0, 'classes': 0}
        }

        train_dir = os.path.join(self.dataset_dir, 'train')
        val_dir = os.path.join(self.dataset_dir, 'val')

        if os.path.exists(train_dir):
            results['train']['exists'] = True
            train_gen = self.get_train_generator()
            results['train']['samples'] = train_gen.samples
            results['train']['classes'] = len(train_gen.class_indices)

        if os.path.exists(val_dir):
            results['val']['exists'] = True
            val_gen = self.get_val_generator()
            results['val']['samples'] = val_gen.samples
            results['val']['classes'] = len(val_gen.class_indices)

        return results


def get_data_generators(dataset_dir, img_size=(128, 128), batch_size=16):
    """
    便捷函数：获取训练和验证数据生成器

    Args:
        dataset_dir: 数据集根目录
        img_size: 图像尺寸
        batch_size: 批次大小

    Returns:
        tuple: (train_generator, val_generator)
    """
    loader = HydrogelDataLoader(dataset_dir, img_size, batch_size)
    return loader.get_train_generator(), loader.get_val_generator()


if __name__ == '__main__':
    print("测试数据加载器...")
    loader = HydrogelDataLoader('dataset')
    results = loader.validate_dataflow()
    print(f"验证结果: {results}")
