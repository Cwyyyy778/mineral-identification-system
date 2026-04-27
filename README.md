# 矿物标本智能识别系统

基于深度学习与传统特征工程的矿物标本智能识别系统，支持模型对比、知识蒸馏、增量学习等高级功能。由于文件训练好的模型过大无法上传，需要自己重新训练

## 功能特点

### 🎯 核心功能
- **多模型识别**：支持ResNet18、ResNet50等多种深度学习模型
- **模型对比**：多模型识别结果实时对比分析
- **知识蒸馏**：ResNet101→ResNet50模型压缩
- **增量学习**：样本增量与类别增量学习
- **传统特征工程**：SIFT、HOG、LBP、颜色直方图

### 🖥️ 交互界面
- **Streamlit Web应用**：友好的图形化界面
- **FastAPI 服务**：RESTful API接口
- **实时进度显示**：训练过程可视化

### 📊 支持的图片格式
JPG, JPEG, PNG, WEBP, BMP, TIFF

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision streamlit pillow numpy matplotlib pandas scikit-learn opencv-python
```

### 2. 运行 Streamlit 应用

```bash
streamlit run app_streamlit.py
```

### 3. 运行 FastAPI 服务

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看API文档

## 项目结构

```
identification/
├── app.py                          # FastAPI 后端服务
├── app_streamlit.py                # Streamlit 前端应用
├── incremental_sample_manager.py   # 增量学习管理模块
├── baseline_resnet18.pth           # ResNet18 基线模型
├── baseline_resnet50_improved.pth # ResNet50 优化模型
├── baseline_resnet50_distilled.pth # 知识蒸馏模型
├── incremental_resnet50_after.pth  # 增量学习模型
├── minet/                          # 矿物数据集
│   ├── biotite/                    # 黑云母
│   ├── bornite/                    # 斑铜矿
│   ├── chrysocolla/                # 硅孔雀石
│   ├── malachite/                  # 孔雀石
│   ├── muscovite/                  # 白云母
│   ├── pyrite/                    # 黄铁矿
│   └── quartz/                    # 石英
├── increase_learn_data/            # 增量学习数据目录
│   ├── samples/                    # 样本图片存储
│   ├── buffer/                    # GSS缓冲区
│   ├── checkpoints/                # 训练模型保存
│   └── logs/                      # 训练日志
├── FlyGCL-main/                   # 类别增量学习框架
├── Gradient-based-Sample-Selection-master/  # 样本增量学习框架
└── README.md
```

## API 接口文档

### FastAPI 服务接口

启动服务后访问 `http://localhost:8000/docs` 查看完整API文档

#### 预测接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/predict/{model_name}` | POST | 使用指定模型预测 |
| `/predict/all` | POST | 所有模型预测并对比 |
| `/predict/incremental` | POST | 增量学习模型预测 |
| `/predict/distillation` | POST | 知识蒸馏模型预测 |

#### 模型信息接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/models` | GET | 获取所有可用模型列表 |
| `/models/summary` | GET | 模型性能汇总表 |
| `/model/{name}/info` | GET | 获取指定模型详细信息 |

#### 增量学习接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/incremental/info` | GET | 增量学习详细介绍 |
| `/incremental/add_samples` | POST | 添加新样本（预留） |
| `/incremental/add_classes` | POST | 添加新类别（预留） |

#### 项目信息接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/project/tech-stack` | GET | 项目技术栈概览 |
| `/project/comparison` | GET | 模型对比分析 |

#### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | API首页与接口列表 |
| `/health` | GET | 健康检查与模型状态 |

## 模型性能对比

| 模型 | 类型 | 准确率 | 训练时间 | 模型大小 |
|------|------|--------|----------|----------|
| ResNet18基线 | 深度学习 | 80% | 约2小时 | 44MB |
| ResNet18数据清洗版 | 深度学习 | 81% | 约2.5小时 | 44MB |
| ResNet50优化版 | 深度学习 | 87.50% | 约4小时 | 98MB |
| 知识蒸馏模型 | 蒸馏 | 89.00% | 约3小时 | 98MB |
| FlyGCL类别增量 | 增量学习 | 类别增量(9类) | 多阶段 | 100MB |
| GSS样本增量 | 增量学习 | 79.17% | 渐进式 | 98MB |

### 传统特征工程

| 方法 | 准确率 | 特点 |
|------|--------|------|
| SIFT | 约65% | 关键点匹配，对旋转尺度鲁棒 |
| HOG | 约72% | 形状特征，对几何形状敏感 |
| LBP | 约68% | 纹理特征，对光照鲁棒 |
| 颜色直方图 | 约55% | 颜色分布，计算简单 |

## 增量学习使用

### 样本增量学习

1. 进入"📈 增量学习" → "📝 样本增量学习" Tab
2. 选择目标矿物类别
3. 上传该类别的新样本图片
4. 点击"📤 保存样本到数据集"
5. 选择基础模型和训练轮数
6. 点击"🚀 开始样本增量训练"

**数据目录**: `increase_learn_data/samples/`

### 类别增量学习

1. 进入"📈 增量学习" → "🎓 类别增量学习" Tab
2. 输入新矿物类别名称
3. 上传至少50张新类别样本图片
4. 点击"🎓 开始类别增量训练"

### 增量学习类型说明

| 类型 | 说明 | 使用场景 |
|------|------|----------|
| **样本增量** | 增加已有类别的样本 | 提高识别精度、防止遗忘 |
| **类别增量** | 学习新的矿物类别 | 扩展识别范围 |

## 技术栈

### 深度学习
- PyTorch
- TorchVision (ResNet18/50/101)

### 传统特征工程
- OpenCV (SIFT, HOG)
- scikit-learn (LBP, 颜色直方图)

### 增量学习框架
- FlyGCL (类别增量)
- GSS (样本增量)

### Web框架
- Streamlit (前端)
- FastAPI (后端)

## 版本历史

- **v2.0**: 集成增量学习功能，支持样本增量和类别增量
- **v1.0**: 基础深度学习模型和特征工程

## 矿物类别

系统支持识别以下7种矿物：
1. **Biotite** (黑云母)
2. **Bornite** (斑铜矿)
3. **Chrysocolla** (硅孔雀石)
4. **Malachite** (孔雀石)
5. **Muscovite** (白云母)
6. **Pyrite** (黄铁矿)
7. **Quartz** (石英)

## 许可证

MIT License
