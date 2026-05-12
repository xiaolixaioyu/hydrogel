# 任务清单 - 基于CNN的水凝胶温度智能识别系统

## 1. 项目环境与依赖配置

- [x] **1.1 创建项目目录结构**
  - [x] 创建 `hydrogel_temp_system/` 根目录
  - [x] 创建 `dataset/train/` 和 `dataset/val/` 目录结构（13个温度子文件夹）
  - [x] 创建 `scripts/`、`model/`、`wechat_miniprogram/` 目录

- [x] **1.2 配置Python训练环境**
  - [x] 创建 `requirements.txt` 包含：tensorflow==2.13.0, numpy, opencv-python, scikit-learn, matplotlib
  - [x] 验证Python 3.8+/3.9+/3.10环境

## 2. 数据处理脚本

- [x] **2.1 数据集整理脚本** (`scripts/organize_dataset.py`)
  - [x] 实现从原始采集图片按温度分类到指定目录的功能
  - [x] 实现训练集/验证集自动划分（8:2比例）
  - [x] 支持随机种子保证可复现性
  - [x] 生成数据集统计报告

- [x] **2.2 数据增强脚本** (`scripts/augment_data.py`)
  - [x] 实现额外的离线数据增强（补充在线增强）
  - [x] 支持亮度调整、旋转、翻转等操作
  - [x] 输出增强后的数据集到指定目录

- [x] **2.3 数据集验证脚本** (`scripts/validate_dataset.py`)
  - [x] 验证数据集完整性（各温度点样本数量）
  - [x] 检查图像格式和尺寸
  - [x] 生成数据分布可视化图表

## 3. CNN模型训练

- [x] **3.1 模型定义** (`model/model.py`)
  - [x] 实现四层卷积块架构
  - [x] 包含BatchNormalization和GlobalAveragePooling
  - [x] Dropout正则化层
  - [x] 全连接回归头

- [x] **3.2 数据加载器** (`model/data_loader.py`)
  - [x] 使用ImageDataGenerator实现数据加载
  - [x] 配置训练集增强参数
  - [x] 配置验证集（无增强）
  - [x] 验证数据流正确性

- [x] **3.3 训练脚本** (`model/train.py`)
  - [x] 配置EarlyStopping回调
  - [x] 配置ReduceLROnPlateau回调
  - [x] 配置ModelCheckpoint保存最佳模型
  - [x] 实现训练历史记录保存
  - [x] 生成训练曲线可视化

- [x] **3.4 模型评估脚本** (`model/evaluate.py`)
  - [x] 计算MAE、RMSE、R²指标
  - [x] 生成预测散点图
  - [x] 生成残差分布图
  - [x] 输出评估报告

## 4. 模型转换与部署

- [x] **4.1 TensorFlow Lite转换脚本** (`model/convert_to_tflite.py`)
  - [x] 加载训练好的.h5模型
  - [x] 配置TFLite转换器
  - [x] 应用float16量化
  - [x] 保存.tflite文件
  - [x] 验证转换后模型推理结果一致性

- [x] **4.2 模型推理测试脚本** (`model/test_inference.py`)
  - [x] 测试TFLite模型加载
  - [x] 验证输入输出形状
  - [x] 对比H5和TFLite模型预测结果

## 5. 微信小程序开发

- [x] **5.1 项目基础配置**
  - [x] 创建 `app.json` 全局配置
  - [x] 创建 `app.js` 全局逻辑
  - [x] 创建 `app.wxss` 全局样式

- [x] **5.2 主页面前端** (`pages/index/`)
  - [x] 创建 `index.wxml` - 界面布局（拍照按钮、结果显示区域）
  - [x] 创建 `index.wxss` - 页面样式
  - [x] 创建 `index.json` - 页面配置

- [x] **5.3 核心推理逻辑** (`pages/index/index.js`)
  - [x] 实现图片选择/拍照功能
  - [x] 实现Canvas图像预处理
  - [x] 实现TFLite模型加载和推理
  - [x] 实现温度预警逻辑
  - [x] 实现震动反馈（高烧预警时）

- [x] **5.4 推理工具模块** (`utils/predictor.js`)
  - [x] 封装TFLite推理逻辑
  - [x] 封装图像预处理函数
  - [x] 封装温度预警判断逻辑

## 6. 操作文档

- [x] **6.1 README.md**
  - [x] 项目介绍
  - [x] 环境配置说明
  - [x] 数据集准备说明
  - [x] 模型训练步骤
  - [x] 模型转换步骤
  - [x] 小程序部署说明
  - [x] 常见问题解答

## 任务依赖关系

```
[1.1] → [2.1] → [2.3]
   ↓          ↓
[1.2]    [2.2]
   ↓
[3.1] → [3.2] → [3.3] → [3.4]
                          ↓
[4.1] ←─────────────[3.3] (best_model.h5)
   ↓
[4.2]
   ↓
[5.1] → [5.2] → [5.3] → [5.4]
                          ↓
                      [6.1]
```

## 并行执行说明

以下任务可以并行执行：
- 任务 1.1（创建目录结构）和 1.2（配置环境）
- 任务 2.1、2.2、2.3（数据处理脚本）
- 任务 5.1、5.2（小程序前端）

以下任务有依赖关系，必须串行执行：
- 任务 3.2 依赖 3.1
- 任务 3.3 依赖 3.2
- 任务 4.1 依赖 3.3（需要训练好的模型）
- 任务 5.3 依赖 5.4（TFLite推理模块）
