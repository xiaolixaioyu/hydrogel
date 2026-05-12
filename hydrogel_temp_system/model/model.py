"""
水凝胶温度智能识别系统 - CNN模型定义
四层卷积块 + 全局平均池化 + 全连接回归头
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D,
    Input
)
from tensorflow.keras.optimizers import Adam


def build_cnn_model(input_shape=(128, 128, 3)):
    """
    构建CNN回归模型

    Args:
        input_shape: 输入图像尺寸，默认 (128, 128, 3)

    Returns:
        编译好的Keras模型
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def get_model_summary(model):
    """获取模型摘要信息"""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


if __name__ == '__main__':
    print("构建CNN模型...")
    model = build_cnn_model()
    print("\n模型架构:")
    print(get_model_summary(model))

    total_params = model.count_params()
    print(f"\n总参数量: {total_params:,}")
    print(f"模型已准备就绪")
