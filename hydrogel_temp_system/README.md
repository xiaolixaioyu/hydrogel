# 水凝胶温度智能识别系统

基于卷积神经网络（CNN）的水凝胶温度智能识别系统，通过智能手机拍照实现非接触式体温检测。

## 项目结构

```
hydrogel_temp_system/
├── dataset/                     # 数据集目录
│   ├── train/                  # 训练集（80%）
│   │   ├── 35.0/
│   │   ├── 35.5/
│   │   ├── ...
│   │   └── 41.0/
│   └── val/                    # 验证集（20%）
├── scripts/                     # 数据处理脚本
│   ├── organize_dataset.py      # 数据集整理脚本
│   ├── augment_data.py           # 数据增强脚本
│   └── validate_dataset.py       # 数据集验证脚本
├── model/                       # 模型相关
│   ├── model.py                  # CNN模型定义
│   ├── data_loader.py           # 数据加载器
│   ├── train.py                 # 模型训练脚本
│   ├── evaluate.py              # 模型评估脚本
│   ├── convert_to_tflite.py     # 模型转换脚本
│   └── test_inference.py        # TFLite推理测试
├── wechat_miniprogram/          # 微信小程序
│   ├── pages/index/             # 主页面
│   ├── utils/                   # 工具模块
│   ├── model/                   # 模型文件夹
│   └── app.js                   # 全局配置
├── requirements.txt             # Python依赖
└── README.md                    # 项目说明
```

## 环境配置

### Python环境要求

- Python 3.8 / 3.9 / 3.10
- TensorFlow 2.13.0

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- tensorflow==2.13.0
- numpy>=1.21
- opencv-python>=4.5
- scikit-learn>=1.0
- matplotlib>=3.5
- Pillow>=9.0.0

## 数据集准备

### 采集要求

- **温度范围**：35.0°C 至 41.0°C，间隔0.5°C，共13个温度点
- **每温度照片数**：40-50张
- **总数据量**：520-650张

### 条件覆盖

- **肤色背景**：4种（Fitzpatrick I-VI型）
- **光照条件**：4种（自然日光、室内白光、室内暖光、混合光源）
- **拍摄角度**：正面垂直、左倾15度、右倾15度
- **拍摄距离**：10cm、20cm、30cm

### 数据整理

将采集的图片按以下格式命名：`[温度]_[序号].jpg`（如 `37.5_0001.jpg`）

然后运行整理脚本：

```bash
python scripts/organize_dataset.py --source /path/to/raw/images --output dataset
```

### 数据验证

```bash
python scripts/validate_dataset.py --dataset dataset
```

### 数据增强（可选）

```bash
python scripts/augment_data.py --input dataset/train --output dataset_augmented/train --augment_factor 2
```

## 模型训练

### 训练命令

```bash
python model/train.py \
    --dataset dataset \
    --epochs 100 \
    --batch_size 16 \
    --img_size 128 \
    --output model_output
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --epochs | 100 | 最大训练轮数 |
| --batch_size | 16 | 批次大小 |
| --img_size | 128 | 图像尺寸 |
| --lr | 0.001 | 初始学习率 |
| --patience | 15 | 早停耐心值 |

### 模型评估

```bash
python model/evaluate.py \
    --model model_output/best_model.h5 \
    --dataset dataset \
    --output eval_output
```

### 评估指标

- **MAE**（平均绝对误差）：目标值 < 0.5°C
- **RMSE**（均方根误差）：目标值 < 0.7°C
- **R²**（决定系数）：目标值 > 0.95

## 模型转换

将训练好的H5模型转换为TFLite格式：

```bash
python model/convert_to_tflite.py \
    --model model_output/best_model.h5 \
    --output wechat_miniprogram/model/hydrogel_temp.tflite \
    --quantize
```

转换后，将 `hydrogel_temp.tflite` 文件复制到微信小程序的 `model/` 目录下。

### TFLite推理测试

```bash
python model/test_inference.py --model wechat_miniprogram/model/hydrogel_temp.tflite
```

## 微信小程序

### 项目配置

1. 引入TFLite插件（需要微信公众平台开通插件权限）
2. 将转换好的模型文件放入 `model/` 目录

### 运行项目

1. 打开微信开发者工具
2. 导入项目目录 `wechat_miniprogram`
3. 选择项目类型为"小程序"
4. 点击"编译"运行

### 功能说明

1. **拍照/选择照片**：点击按钮拍摄或从相册选择水凝胶图像
2. **图像预处理**：Canvas缩放至128x128，归一化处理
3. **温度预测**：TFLite模型推理
4. **预警提示**：
   - < 37.5°C：正常（绿色）
   - 37.5-38.5°C：轻度发热（黄色）
   - > 38.5°C：高烧预警（红色）+ 震动提示

## 常见问题

### Q: 模型推理速度慢？
A: 确保使用float16量化后的TFLite模型，可减少约50%推理时间。

### Q: 预测温度偏差大？
A: 检查数据采集质量，确保光照条件覆盖全面，训练数据充足。

### Q: 小程序模型加载失败？
A: 确认TFLite插件已正确配置，模型文件路径正确。

## 技术支持

如有问题，请提交Issue。
