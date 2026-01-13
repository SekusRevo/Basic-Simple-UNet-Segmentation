# UNet Ultrasound Segmentation

基于 UNet 的简单超声图像分割项目，支持评估生成数据增广对模型性能的影响。

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据集结构

```
data/
├── train/
│   ├── image/
│   └── mask/
├── test/
│   ├── image/
│   └── mask/
└── xxgenmodel_train/      # 生成的增广数据
    ├── image/
    └── mask/
```

## 训练

### 基础训练

```bash
# 仅使用原始数据
python train_ultrasound.py \
    --train-dirs ./data/train \
    --test-dir ./data/test \
    --epochs 100 \
    --batch-size 8 \
    --seed 42

# 使用原始数据 + 生成增广数据
python train_ultrasound.py \
    --train-dirs "./data/train,./data/xxgenmodel_train" \
    --test-dir ./data/test \
    --epochs 100 \
    --batch-size 8
```

### 五折交叉验证

```bash
python train_ultrasound.py \
    --train-dirs ./data/train \
    --test-dir ./data/test \
    --kfold \
    --n-folds 5 \
    --epochs 100
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-dirs` | - | 训练数据目录（多个用逗号分隔） |
| `--test-dir` | - | 测试数据目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 8 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--img-size` | 256 | 输入图像尺寸 |
| `--seed` | 42 | 随机种子 |
| `--kfold` | False | 启用 K-Fold 交叉验证 |
| `--n-folds` | 5 | 交叉验证折数 |
| `--patience` | 15 | Early stopping 耐心值 |

## 评估

对比基线模型和增广模型的性能：

```bash
python evaluate_ultrasound.py \
    --baseline-model ./outputs/baseline/checkpoints/best_model.pth \
    --augmented-model ./outputs/augmented/checkpoints/best_model.pth \
    --test-dir ./data/test \
    --output-dir ./evaluation_results \
    --save-predictions
```

输出指标：
- Dice coefficient
- IoU (Intersection over Union)
- Precision / Recall / F1

## 查看训练日志

```bash
tensorboard --logdir ./outputs
```

## 项目结构

```
├── unet/                     # UNet 模型
├── utils/                    # 工具函数
├── ultrasound_dataset.py     # 数据集类
├── train_ultrasound.py       # 训练脚本
├── evaluate_ultrasound.py    # 评估脚本
└── requirements.txt          # 依赖
```
