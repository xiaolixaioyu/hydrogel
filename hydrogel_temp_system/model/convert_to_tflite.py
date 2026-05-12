"""
水凝胶温度智能识别系统 - TensorFlow Lite 模型转换脚本
将训练好的H5模型转换为TFLite格式，支持float16量化
"""

import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='转换模型为TensorFlow Lite格式')
    parser.add_argument('--model', type=str, required=True,
                        help='输入模型路径 (.h5)')
    parser.add_argument('--output', type=str, default='model_output/hydrogel_temp.tflite',
                        help='输出TFLite模型路径（默认: model_output/hydrogel_temp.tflite）')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='启用float16量化（默认: True）')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='验证转换后的模型（默认: True）')
    return parser.parse_args()


def convert_to_tflite(model_path, output_path, quantize=True, verify=True):
    """
    将Keras模型转换为TensorFlow Lite格式

    Args:
        model_path: 输入模型路径
        output_path: 输出TFLite模型路径
        quantize: 是否应用量化
        verify: 是否验证转换结果

    Returns:
        str: 输出文件路径
    """
    print("=" * 60)
    print("TensorFlow Lite 模型转换")
    print("=" * 60)
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"量化模式: {'float16' if quantize else 'None'}")
    print("=" * 60)

    print("\n[1/3] 加载模型...")
    model = load_model(model_path)
    print("模型加载完成")

    print("\n[2/3] 创建TFLite转换器...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("已启用float16量化优化")

    print("\n[3/3] 执行转换...")
    tflite_model = converter.convert()
    print("转换完成")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size = os.path.getsize(output_path)
    print(f"\n模型已保存至: {output_path}")
    print(f"模型大小: {file_size / 1024 / 1024:.2f} MB")

    if quantize and file_size > 5 * 1024 * 1024:
        print("警告: 模型大小超过5MB，建议检查量化设置")

    if verify:
        print("\n开始验证...")
        verify_conversion(output_path, model)
        print("验证完成")

    return output_path


def verify_conversion(tflite_path, keras_model):
    """
    验证TFLite模型转换的正确性

    Args:
        tflite_path: TFLite模型路径
        keras_model: 原始Keras模型
    """
    print("\n验证转换结果...")
    print("-" * 40)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    input_shape = interpreter.get_input_details()[0]['shape']
    output_shape = interpreter.get_output_details()[0]['shape']

    print(f"TFLite输入形状: {input_shape}")
    print(f"TFLite输出形状: {output_shape}")
    print(f"输入dtype: {interpreter.get_input_details()[0]['dtype']}")
    print(f"输出dtype: {interpreter.get_output_details()[0]['dtype']}")

    dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)

    interpreter.set_tensor(input_index, dummy_input)
    interpreter.invoke()

    tflite_output = interpreter.get_tensor(output_index)

    keras_output = keras_model.predict(dummy_input, verbose=0)

    diff = np.abs(tflite_output - keras_output).max()
    mean_diff = np.abs(tflite_output - keras_output).mean()

    print(f"\nTFLite输出: {tflite_output[0][0]:.4f}")
    print(f"Keras输出: {keras_output[0][0]:.4f}")
    print(f"最大差异: {diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")

    if diff < 0.1:
        print("✓ 验证通过: TFLite与Keras输出差异 < 0.1")
    else:
        print("✗ 警告: TFLite与Keras输出差异较大")

    print("-" * 40)


def print_model_info(tflite_path):
    """
    打印TFLite模型详细信息

    Args:
        tflite_path: TFLite模型路径
    """
    print("\n模型信息:")
    print("-" * 40)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    print(f"输入层数量: {len(interpreter.get_input_details())}")
    print(f"输出层数量: {len(interpreter.get_output_details())}")

    input_details = interpreter.get_input_details()[0]
    print(f"\n输入层:")
    print(f"  索引: {input_details['index']}")
    print(f"  形状: {input_details['shape']}")
    print(f"  数据类型: {input_details['dtype']}")
    print(f"  量化参数: {input_details.get('quantization', 'N/A')}")

    output_details = interpreter.get_output_details()[0]
    print(f"\n输出层:")
    print(f"  索引: {output_details['index']}")
    print(f"  形状: {output_details['shape']}")
    print(f"  数据类型: {output_details['dtype']}")
    print(f"  量化参数: {output_details.get('quantization', 'N/A')}")

    print("-" * 40)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        quantize=args.quantize,
        verify=args.verify
    )

    print_model_info(args.output)


if __name__ == '__main__':
    main()
