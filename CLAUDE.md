# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 训练（--encoding 可选 onehot / ctd / esm2，默认 esm2）
python scripts/train/train_mlp_multitask.py --encoding esm2
python scripts/train/train_bnn_multitask.py --encoding esm2
python scripts/train/train_rf_multitask.py --encoding esm2
python scripts/train/train_xgb_multitask.py --encoding esm2

# 推理（单序列 / FASTA 批量）
python scripts/inference/inference_mlp_multitask.py --sequence "MVLSPADKTNV..."
python scripts/inference/inference_mlp_multitask.py --fasta proteins.fasta --output results.json

# 评估 + 可视化
python scripts/evaluate/evaluate_all.py
python scripts/visualize/plot_results.py

# 运行测试
pytest
```

## 架构要点

### 插件注册模式（核心设计）

编码器和分类器均使用装饰器注册，无需修改 base.py 即可扩展：

- **编码器**: 继承 `src/encodings/base.py` 的 `ProteinEncoder`，用 `@register_encoder("name")` 装饰，通过 `EncoderRegistry.get("name")` 获取实例
- **分类器**: 继承 `src/algorithms/base.py` 的 `ProteinClassifier`，用 `@register_classifier("name")` 装饰，通过 `ClassifierRegistry.get("name")` 获取实例

添加新编码器/算法只需在对应目录新建文件，继承基类并加装饰器即可。

### 多任务分类架构

所有模型同时预测三个任务：EC 编号（6类）、细胞定位（11类）、分子功能（17类）。类别定义集中在 `configs/config.py` 的 `MULTITASK_CONFIG`。

- **sklearn 类算法**（RF/XGBoost/SVM/LR）：每个任务训练一个独立模型实例
- **PyTorch 类算法**（MLP/BNN）：共享编码层 + 三个独立任务头（`ec_head` / `loc_head` / `func_head`），如 `scripts/inference/inference_mlp_multitask.py:18-38` 所示
- **BNN 特有**: 实现 `UncertainClassifier` 接口，通过 30 次 MC Dropout 采样输出预测熵作为不确定性

### 数据流

```
原始序列字符串
  → ProteinEncoder.encode() → numpy (dim,)
  → 多任务模型前向传播 → 3 组 logits
  → softmax → Top-1/Top-3 类别 + 置信度
  → JSON 输出
```

编码器 `encode()` 内部会调用 `validate_sequence()` 自动去除非标准氨基酸（仅保留 20 种标准 AA）。

### 配置

`configs/config.py` 是所有路径、超参数、类别定义的中心。`TrainConfig` dataclass 控制训练超参数，`MULTITASK_CONFIG` 定义三个任务的类别映射。`ensure_dirs()` 确保目录结构存在。

### 训练脚本模式

每个 `scripts/train/train_*.py` 脚本结构一致：
1. 加载 train/val/test parquet 文件
2. 通过 `EncoderRegistry.get(encoding)` 获取编码器并生成特征（缓存为 `.npy` 文件）
3. 创建模型并训练（PyTorch 类使用 `ReduceLROnPlateau` + early stopping patience=15）
4. 在 test 集评估并保存 `model.pt`/`model.pkl` + `results.json` 到 `models/{alg}_{enc}_multitask/`

### ESM2 注意

ESM2 编码器（`src/encodings/esm2.py`）需要 `pip install fair-esm`，首次运行会下载 `facebook/esm2_t6_8M_UR50D` 模型。特征维度 480（config 中定义），使用均值池化。ESM2 特征显著优于 OneHot/CTD（准确率高出 20-30 个百分点）。

### 环境

```bash
conda env create -f env.yaml
conda activate protein-classifier
pip install fair-esm
```
