"""
水凝胶温度智能识别系统 - TFLite模型推理测试脚本
测试TFLite模型加载、推理和结果一致性
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='测试TFLite模型推理')
    parser.add_argument('--model', type=str, required=True,
                        help='TFLite模型路径 (.tflite)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='测试图像路径（可选）')
    return parser.parse_args()


def load_tflite_model(model_path):
    """
    加载TFLite模型

    Args:
        model_path: 模型文件路径

    Returns:
        tuple: (interpreter, input_details, output_details)
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    return interpreter, input_details, output_details


def preprocess_image(image_path, target_size=(128, 128)):
    """
    预处理图像

    Args:
        image_path: 图像路径
        target_size: 目标尺寸

    Returns:
        np.ndarray: 预处理后的图像数组
    """
    img = Image.open(image_path).convert('RGB')

    img = img.resize(target_size, Image.LANCZOS)

    img_array = np.array(img, dtype=np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_inference(interpreter, input_data, input_details, output_details):
    """
    运行推理

    Args:
        interpreter: TFLite解释器
        input_data: 输入数据
        input_details: 输入层信息
        output_details: 输出层信息

    Returns:
        float: 预测温度值
    """
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])

    return float(output_data[0][0])


def test_with_random_input(interpreter, input_details, output_details, num_tests=5):
    """
    使用随机输入测试模型

    Args:
        interpreter: TFLite解释器
        input_details: 输入层信息
        output_details: 输出层信息
        num_tests: 测试次数

    Returns:
        list: 测试结果列表
    """
    print(f"\n使用随机输入测试模型 ({num_tests} 次)...")
    print("-" * 40)

    results = []

    for i in range(num_tests):
        input_shape = input_details['shape']
        random_input = np.random.rand(*input_shape).astype(np.float32)

        prediction = run_inference(
            interpreter, random_input, input_details, output_details
        )

        results.append(prediction)
        print(f"测试 {i+1}: 输入形状 {input_shape}, 预测温度: {prediction:.2f}°C")

    print(f"\n预测温度范围: {min(results):.2f}°C - {max(results):.2f}°C")
    print(f"预测温度均值: {np.mean(results):.2f}°C")

    print("-" * 40)

    return results


def test_with_image(interpreter, image_path, input_details, output_details):
    """
    使用真实图像测试模型

    Args:
        interpreter: TFLite解释器
        image_path: 图像路径
        input_details: 输入层信息
        output_details: 输出层信息

    Returns:
        float: 预测温度值
    """
    print(f"\n使用图像测试模型: {image_path}")
    print("-" * 40)

    try:
        input_data = preprocess_image(image_path)
        print(f"输入形状: {input_data.shape}")
        print(f"输入范围: [{input_data.min():.3f}, {input_data.max():.3f}]")

        prediction = run_inference(
            interpreter, input_data, input_details, output_details
        )

        print(f"\n预测温度: {prediction:.2f}°C")

        status = get_temperature_status(prediction)
        print(f"状态: {status['label']} - {status['message']}")

        print("-" * 40)

        return prediction

    except Exception as e:
        print(f"错误: 无法处理图像: {e}")
        return None


def get_temperature_status(temp):
    """
    获取温度预警状态

    Args:
        temp: 温度值

    Returns:
        dict: 状态信息
    """
    if temp < 37.5:
        return {
            'level': 'normal',
            'label': '正常',
            'color': '#2ecc71',
            'message': '体温正常，请继续观察'
        }
    elif temp <= 38.5:
        return {
            'level': 'mild',
            'label': '轻度发热',
            'color': '#f1c40f',
            'message': '轻度发热，建议物理降温并密切观察'
        }
    else:
        return {
            'level': 'high',
            'label': '高烧预警',
            'color': '#e74c3c',
            'message': '高烧预警，请立即就医！'
        }


def test_model_inference(model_path, test_image=None):
    """
    测试TFLite模型推理

    Args:
        model_path: 模型路径
        test_image: 测试图像路径（可选）
    """
    print("=" * 60)
    print("TFLite 模型推理测试")
    print("=" * 60)
    print(f"模型路径: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    file_size = os.path.getsize(model_path)
    print(f"模型大小: {file_size / 1024 / 1024:.2f} MB")

    print("\n[1/3] 加载模型...")
    interpreter, input_details, output_details = load_tflite_model(model_path)
    print("模型加载成功")

    print("\n[2/3] 检查模型配置...")
    print(f"输入形状: {input_details['shape']}")
    print(f"输入类型: {input_details['dtype']}")
    print(f"输出形状: {output_details['shape']}")
    print(f"输出类型: {output_details['dtype']}")

    print("\n[3/3] 运行测试...")
    test_with_random_input(interpreter, input_details, output_details, num_tests=3)

    if test_image and os.path.exists(test_image):
        test_with_image(interpreter, test_image, input_details, output_details)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def main():
    args = parse_args()

    test_model_inference(args.model, args.test_image)


if __name__ == '__main__':
    main()
