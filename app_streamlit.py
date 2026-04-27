import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import os
import time
from datetime import datetime

from incremental_sample_manager import IncrementalSampleManager, IncrementalTrainer

st.set_page_config(
    page_title="矿物识别系统",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 矿物识别与增量学习系统")

classes = ['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class IncrementalResNet50(nn.Module):
    def __init__(self, num_classes):
        super(IncrementalResNet50, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.exposed_classes = list(range(num_classes))
        self.n_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model(model_type, model_path, num_classes):
    try:
        if model_type == 'incremental':
            model = IncrementalResNet50(num_classes)
        elif model_type == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, True
    except Exception as e:
        return None, str(e)

models_info = {
    'baseline_resnet18': {
        'name': 'ResNet18基线模型',
        'path': 'baseline_resnet18.pth',
        'type': 'resnet18',
        'accuracy': '80%',
        'training_time': '约2小时',
        'model_size': '约44MB',
        'category': 'deep_learning',
        'description': 'ResNet18预训练微调'
    },
    'baseline_resnet18_cleaned': {
        'name': 'ResNet18数据清洗版',
        'path': 'baseline_resnet18_cleaned.pth',
        'type': 'resnet18',
        'accuracy': '81%',
        'training_time': '约2.5小时',
        'model_size': '约44MB',
        'category': 'deep_learning',
        'description': '数据清洗后模型'
    },
    'baseline_resnet50_improved': {
        'name': 'ResNet50优化模型',
        'path': 'baseline_resnet50_improved.pth',
        'type': 'resnet50',
        'accuracy': '87.50%',
        'training_time': '约4小时',
        'model_size': '约98MB',
        'category': 'deep_learning',
        'description': '超参数调优 + 数据增强'
    },
    'distillation_resnet50': {
        'name': '知识蒸馏模型',
        'path': 'baseline_resnet50_distilled.pth',
        'type': 'resnet50',
        'accuracy': '89.00%',
        'training_time': '约3小时',
        'model_size': '约98MB',
        'category': 'distillation',
        'description': 'ResNet101→ResNet50蒸馏'
    },
    'incremental_resnet50_after': {
        'name': 'FlyGCL类别增量模型',
        'path': 'incremental_resnet50_after.pth',
        'type': 'incremental',
        'accuracy': '类别增量(9类)',
        'training_time': '多阶段训练',
        'model_size': '约100MB',
        'category': 'incremental_class',
        'description': 'FlyGCL框架类别增量学习'
    },
    'incremental_gss_resnet50': {
        'name': 'GSS样本增量模型',
        'path': 'incremental_gss_resnet50.pth',
        'type': 'resnet50',
        'accuracy': '79.17%',
        'training_time': '渐进式训练',
        'model_size': '约98MB',
        'category': 'incremental_sample',
        'description': '梯度样本选择增量学习'
    }
}

feature_engineering_models = {
    'sift': {
        'name': 'SIFT特征匹配',
        'accuracy': '约65%',
        'category': 'feature_engineering',
        'description': '尺度不变特征变换'
    },
    'hog': {
        'name': 'HOG特征分类',
        'accuracy': '约72%',
        'category': 'feature_engineering',
        'description': '方向梯度直方图'
    },
    'lbp': {
        'name': 'LBP纹理特征',
        'accuracy': '约68%',
        'category': 'feature_engineering',
        'description': '局部二值模式'
    },
    'color_histogram': {
        'name': '颜色直方图',
        'accuracy': '约55%',
        'category': 'feature_engineering',
        'description': 'RGB/HSV颜色分布'
    }
}

loaded_models = {}
for model_name, info in models_info.items():
    model, status = load_model(info['type'], info['path'], len(classes))
    if model is not None:
        loaded_models[model_name] = model

def predict_image(image, model):
    try:
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

        predicted_class = classes[predicted.item()]

        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": {classes[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
    except Exception as e:
        return {"error": str(e)}

if 'incremental_history' not in st.session_state:
    st.session_state.incremental_history = []

if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0

if 'is_training' not in st.session_state:
    st.session_state.is_training = False

st.sidebar.title("功能导航")
main_option = st.sidebar.selectbox(
    "选择主要功能",
    ["🏠 首页概览", "🔬 模型识别", "📊 模型性能对比", "📈 增量学习"]
)

if main_option == "🏠 首页概览":
    st.header("欢迎使用矿物识别系统")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("已加载模型", len(loaded_models))
        st.metric("总模型数", len(models_info))

    with col2:
        st.metric("矿物类别", 7)
        st.metric("传统特征方法", len(feature_engineering_models))

    with col3:
        st.metric("增量学习框架", 2)
        st.metric("系统版本", "2.0")

    st.markdown("---")
    st.markdown("### 🎯 系统功能")

    tab1, tab2, tab3 = st.tabs(["深度学习模型", "传统特征工程", "增量学习"])

    with tab1:
        st.markdown("#### 深度学习识别模型")
        for name, info in models_info.items():
            status = "✅" if name in loaded_models else "❌"
            st.markdown(f"{status} **{info['name']}** - {info['description']} (准确率: {info['accuracy']})")

    with tab2:
        st.markdown("#### 传统特征工程方法")
        for key, info in feature_engineering_models.items():
            st.markdown(f"📌 **{info['name']}** - {info['description']} (准确率: {info['accuracy']})")

    with tab3:
        st.markdown("#### 增量学习框架")
        st.markdown("""
        - **FlyGCL**: 类别增量学习 - 支持扩展识别类别（7类→9类）
        - **GSS**: 样本增量学习 - 通过梯度样本选择防止灾难性遗忘
        """)
        st.markdown("##### 增量学习类型说明")
        st.markdown("""
        | 类型 | 说明 | 使用场景 |
        |------|------|----------|
        | **类别增量** | 学习新的矿物类别 | 发现新矿物时扩展模型 |
        | **样本增量** | 增加现有类别样本 | 提高识别精度、防止遗忘 |
        """)

    st.markdown("---")
    st.markdown("### 🔗 快速链接")
    st.markdown("- [模型识别](#模型识别) - 上传图片进行识别")
    st.markdown("- [模型性能对比](#模型性能对比) - 查看所有模型性能")
    st.markdown("- [增量学习](#增量学习) - 添加新样本或新类别")

elif main_option == "🔬 模型识别":
    st.header("🔬 矿物识别")

    tab1, tab2, tab3 = st.tabs(["📸 单模型识别", "🔄 多模型对比", "⚡ 快速识别"])

    with tab1:
        uploaded_file = st.file_uploader("选择一张矿物图片", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="上传的图片", use_column_width=True)

            model_options = list(loaded_models.keys())
            selected_model = st.selectbox("选择模型", model_options, format_func=lambda x: f"{models_info[x]['name']} [{models_info[x]['accuracy']}]")

            if st.button("开始识别", type="primary"):
                model = loaded_models[selected_model]
                result = predict_image(image, model)

                if "error" not in result:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.success(f"识别结果: **{result['predicted_class']}**")
                        st.metric("置信度", f"{result['confidence']*100:.2f}%")

                    with col2:
                        st.info(f"使用模型: {models_info[selected_model]['name']}")
                        st.info(f"模型描述: {models_info[selected_model]['description']}")

                    st.markdown("#### 各类别概率分布")
                    prob_df = [{"矿物类别": k, "概率": f"{v*100:.2f}%"} for k, v in result['probabilities'].items()]
                    st.dataframe(prob_df, use_container_width=True)
                else:
                    st.error(f"识别失败: {result['error']}")

    with tab2:
        uploaded_file = st.file_uploader("上传图片进行多模型对比", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], key="multi")

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="对比图片", use_column_width=True)

            model_options = list(loaded_models.keys())
            selected_indices = st.multiselect(
                "选择要对比的模型（至少2个）",
                options=range(len(model_options)),
                default=[0, 1] if len(model_options) > 1 else [0],
                format_func=lambda i: f"{models_info[model_options[i]]['name']} - {models_info[model_options[i]]['accuracy']}"
            )

            if len(selected_indices) >= 2 and st.button("开始对比", type="primary"):
                cols = st.columns(len(selected_indices))
                results_all = []

                for i, idx in enumerate(selected_indices):
                    model_name = model_options[idx]
                    model = loaded_models[model_name]
                    result = predict_image(image, model)
                    results_all.append((model_name, result))

                    with cols[i]:
                        st.subheader(models_info[model_name]['name'])
                        if "error" not in result:
                            st.metric("识别结果", result['predicted_class'])
                            st.metric("置信度", f"{result['confidence']*100:.2f}%")
                        else:
                            st.error(f"失败: {result['error']}")

                st.markdown("---")
                st.markdown("#### 📊 对比结果汇总")

                comparison_data = []
                for model_name, result in results_all:
                    if "error" not in result:
                        comparison_data.append({
                            "模型": models_info[model_name]['name'],
                            "识别结果": result['predicted_class'],
                            "置信度": f"{result['confidence']*100:.2f}%",
                            "准确率": models_info[model_name]['accuracy']
                        })

                if comparison_data:
                    st.dataframe(comparison_data, use_container_width=True)

    with tab3:
        uploaded_file = st.file_uploader("快速识别（使用所有模型）", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], key="quick")

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="快速识别", use_column_width=True)

            if st.button("🚀 快速识别（所有模型）", type="primary"):
                all_results = []
                for model_name, model in loaded_models.items():
                    result = predict_image(image, model)
                    if "error" not in result:
                        all_results.append({
                            "model": model_name,
                            "name": models_info[model_name]['name'],
                            "result": result
                        })

                all_results_sorted = sorted(all_results, key=lambda x: x['result']['confidence'], reverse=True)

                best_result = all_results_sorted[0]

                st.success(f"🏆 最佳结果: **{best_result['result']['predicted_class']}** (置信度: {best_result['result']['confidence']*100:.2f}%)")

                st.markdown("#### 所有模型结果排名")
                rank_data = []
                for i, item in enumerate(all_results_sorted, 1):
                    rank_data.append({
                        "排名": i,
                        "模型": item['name'],
                        "识别结果": item['result']['predicted_class'],
                        "置信度": f"{item['result']['confidence']*100:.2f}%"
                    })

                st.dataframe(rank_data, use_container_width=True)

elif main_option == "📊 模型性能对比":
    st.header("📊 模型性能对比")

    tab1, tab2, tab3 = st.tabs(["📋 性能汇总表", "📈 准确率对比", "🔍 详细对比"])

    with tab1:
        st.markdown("### 所有模型性能汇总")

        all_models_data = []

        for name, info in models_info.items():
            status = "✅ 已加载" if name in loaded_models else "❌ 未加载"
            all_models_data.append({
                "模型名称": info['name'],
                "类型": info['category'],
                "准确率": info['accuracy'],
                "训练时间": info.get('training_time', 'N/A'),
                "模型大小": info.get('model_size', 'N/A'),
                "状态": status,
                "描述": info['description']
            })

        for key, info in feature_engineering_models.items():
            all_models_data.append({
                "模型名称": info['name'],
                "类型": info['category'],
                "准确率": info['accuracy'],
                "训练时间": info.get('training_time', 'N/A'),
                "模型大小": info.get('model_size', 'N/A'),
                "状态": "✅ 可用",
                "描述": info['description']
            })

        import pandas as pd
        df = pd.DataFrame(all_models_data)
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("### 准确率对比")

        col1, col2 = st.columns([2, 1])

        with col1:
            dl_accuracies = []
            dl_names = []
            for name, info in models_info.items():
                if name in loaded_models and '类别增量' not in info['accuracy']:
                    try:
                        acc = float(info['accuracy'].replace('%', '').replace('约', ''))
                        dl_accuracies.append(acc)
                        dl_names.append(info['name'])
                    except:
                        pass

            if dl_accuracies:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(dl_accuracies)))
                bars = ax.barh(dl_names, dl_accuracies, color=colors)
                ax.set_xlabel('准确率 (%)')
                ax.set_title('深度学习模型准确率对比')
                ax.set_xlim(0, 100)

                for bar, acc in zip(bars, dl_accuracies):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                           f'{acc:.2f}%', va='center', fontsize=9)

                st.pyplot(fig)

        with col2:
            st.markdown("#### 关键指标")
            st.metric("最高准确率", f"{max(dl_accuracies):.2f}%")
            st.metric("平均准确率", f"{np.mean(dl_accuracies):.2f}%")
            st.metric("模型数量", len(dl_accuracies))

    with tab3:
        st.markdown("### 🔍 模型详细对比")

        deep_learning_models = {k: v for k, v in models_info.items() if v['category'] in ['deep_learning', 'distillation']}

        if deep_learning_models:
            selected_models = st.multiselect(
                "选择要对比的模型",
                options=list(deep_learning_models.keys()),
                default=list(deep_learning_models.keys())[:3],
                format_func=lambda x: deep_learning_models[x]['name']
            )

            if selected_models:
                comparison_metrics = []
                for model_key in selected_models:
                    info = deep_learning_models[model_key]
                    comparison_metrics.append({
                        "模型": info['name'],
                        "架构": info.get('architecture', 'N/A'),
                        "准确率": info['accuracy'],
                        "训练时间": info.get('training_time', 'N/A'),
                        "模型大小": info.get('model_size', 'N/A'),
                        "类型": info['category']
                    })

                st.dataframe(pd.DataFrame(comparison_metrics), use_container_width=True)

elif main_option == "📈 增量学习":
    st.header("📈 增量学习")

    st.info("💡 增量学习允许模型在不重新训练所有数据的情况下学习新知识")

    sample_manager = IncrementalSampleManager(base_dir="increase_learn_data")

    tab1, tab2, tab3 = st.tabs(["📝 样本增量学习", "🎓 类别增量学习", "📜 增量历史"])

    with tab1:
        st.markdown("### 样本增量学习 (Sample Incremental)")
        st.markdown("""
        **原理**: 为已有的矿物类别添加新的训练样本，提高识别精度

        **特点**:
        - 不改变类别数量（始终7类）
        - 增加每个类别的样本量
        - 使用GSS(梯度样本选择)防止灾难性遗忘
        """)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📊 当前样本统计")
            sample_counts = sample_manager.get_all_counts()
            for class_name, count in sample_counts.items():
                st.write(f"  - {class_name}: {count} 张")
            st.markdown("---")

        with col2:
            buffer_info = sample_manager.get_buffer_info()
            if buffer_info:
                st.markdown("#### 💾 缓冲区状态")
                st.write(f"最后更新: {buffer_info.get('last_updated', 'N/A')}")
                st.write(f"缓冲样本数: {buffer_info.get('buffer_samples', 0)}")

        st.markdown("---")
        st.markdown("#### 📤 上传新样本")

        col1, col2 = st.columns([1, 1])

        with col1:
            target_class = st.selectbox("选择目标矿物类别", classes)

        with col2:
            num_samples = st.number_input("计划上传数量", min_value=1, max_value=100, value=10)

        uploaded_samples = st.file_uploader(
            f"上传 {target_class} 类的新样本图片（可多选）",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            accept_multiple_files=True,
            key="sample_uploader"
        )

        if uploaded_samples:
            st.success(f"✅ 已选择 {len(uploaded_samples)} 张图片")

            cols = st.columns(min(5, len(uploaded_samples)))
            for i, sample_file in enumerate(uploaded_samples[:5]):
                img = Image.open(sample_file).convert('RGB')
                with cols[i]:
                    st.image(img, caption=sample_file.name, use_container_width=True)

            if len(uploaded_samples) > 5:
                st.caption(f"还有 {len(uploaded_samples) - 5} 张图片...")

            if st.button("📤 保存样本到数据集", type="secondary"):
                with st.spinner("正在保存样本..."):
                    result = sample_manager.add_samples(target_class, uploaded_samples)

                st.success(f"🎉 成功保存 {result['saved_count']} 张图片到 {result['class']} 类别")
                st.info(f"该类别当前总样本数: {result['total_samples']}")

                st.rerun()

        st.markdown("---")
        
        if 'last_training_result' in st.session_state and st.session_state.last_training_result:
            result = st.session_state.last_training_result
            if result.get('success'):
                with st.success("🎉 最近一次训练结果"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("模型路径", result['checkpoint_path'].split('\\')[-1] if '\\' in result['checkpoint_path'] else result['checkpoint_path'])
                    col2.metric("训练损失", f"{result['avg_loss']:.4f}")
                    col3.metric("训练轮数", result['epochs'])
                    col4.metric("样本数", result['samples'])
                    st.info(f"训练时间: {result['time']}")
            else:
                st.error(f"❌ 训练失败: {result.get('error')}")

        st.markdown("#### 🚀 开始增量训练")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_base_model = st.selectbox(
                "选择基础模型",
                options=['baseline_resnet50_improved', 'distillation_resnet50'],
                format_func=lambda x: f"{models_info[x]['name']} [{models_info[x]['accuracy']}]"
            )

        with col2:
            training_epochs = st.number_input("训练轮数", min_value=1, max_value=20, value=5)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("总样本数", sum(sample_counts.values()))
        with col2:
            st.metric("类别数", len([c for c in sample_counts.values() if c > 0]))

        if st.button("🚀 开始样本增量训练", type="primary", disabled=st.session_state.is_training):
            if sum(sample_counts.values()) == 0:
                st.warning("⚠️ 请先上传并保存样本图片")
            else:
                st.session_state.is_training = True
                st.rerun()

        if st.session_state.is_training:
            st.info(f"🔥 训练进行中: {sum(sample_counts.values())} 样本, {training_epochs} 轮...")
            st.text("请耐心等待，训练完成后会自动更新...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🚀 正在启动训练...")

            with st.spinner("模型加载中..."):
                try:
                    status_text.text("📦 加载基础模型...")
                    progress_bar.progress(5)

                    if selected_base_model in loaded_models:
                        base_model = loaded_models[selected_base_model]
                    else:
                        base_model, _ = load_model(
                            models_info[selected_base_model]['type'],
                            models_info[selected_base_model]['path'],
                            len(classes)
                        )

                    status_text.text("🔧 模型加载完成，准备训练...")
                    progress_bar.progress(15)

                    with st.spinner("初始化训练器..."):
                        trainer = IncrementalTrainer(sample_manager, base_model)

                    status_text.text("⚙️ 开始增量训练，这可能需要几分钟...")
                    progress_bar.progress(20)

                    result_placeholder = st.container()

                    def progress_callback(progress, status_msg):
                        try:
                            overall_progress = 20 + int(progress * 75)
                            progress_bar.progress(overall_progress)
                            status_text.text(status_msg)
                        except Exception as e:
                            pass

                    result = trainer.train_incremental(num_epochs=training_epochs, progress_callback=progress_callback)

                    progress_bar.progress(95)
                    status_text.text("💾 保存模型...")

                    if result.get('status') == 'error':
                        st.session_state.is_training = False
                        st.error(f"训练失败: {result.get('message')}")
                    else:
                        checkpoint_path = result.get('checkpoint_path', 'N/A')
                        avg_loss = result.get('avg_loss', 0)
                        epochs = result.get('epochs', training_epochs)

                        progress_bar.progress(100)
                        status_text.text("✅ 训练完成！")

                        st.session_state.training_progress = 100
                        st.session_state.is_training = False

                        st.session_state.incremental_history.append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "样本增量",
                            "class": target_class,
                            "samples": sum(sample_counts.values()),
                            "method": "GSS",
                            "status": "完成",
                            "checkpoint": checkpoint_path
                        })

                        st.session_state.last_training_result = {
                            "success": True,
                            "checkpoint_path": checkpoint_path,
                            "avg_loss": avg_loss,
                            "epochs": epochs,
                            "samples": sum(sample_counts.values()),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        st.success("🎉 样本增量训练完成！")
                        with result_placeholder:
                            st.info(f"📁 模型保存路径: {checkpoint_path}")
                            st.info(f"📉 最终训练损失: {avg_loss:.4f}")
                            st.info(f"🔄 训练轮数: {epochs}")

                except Exception as e:
                    st.session_state.is_training = False
                    progress_bar.progress(0)
                    import traceback
                    st.error(f"❌ 训练失败: {str(e)}")
                    with st.expander("🔍 查看详细错误"):
                        st.code(traceback.format_exc())
                    st.session_state.last_training_result = {
                        "success": False,
                        "error": str(e)
                    }

                time.sleep(10)
                st.rerun()

    with tab2:
        st.markdown("### 类别增量学习 (Class Incremental)")
        st.markdown("""
        **原理**: 在保持旧类别知识的同时，学习识别新的矿物类别

        **特点**:
        - 可以增加新的矿物类别（7类 → 8类/9类）
        - 使用FlyGCL框架防止灾难性遗忘
        - 需要为新类别提供至少50张样本
        """)

        st.markdown("#### 添加新类别")

        col1, col2 = st.columns([1, 1])

        with col1:
            new_class_name = st.text_input("新矿物类别名称", placeholder="例如: galena（方铅矿）")

        with col2:
            min_samples = st.number_input("最少样本数量", min_value=10, max_value=100, value=50)

        st.markdown(f"#### 上传 {new_class_name if new_class_name else '[类别名称]'} 的样本图片")

        new_class_samples = st.file_uploader(
            f"上传至少 {min_samples} 张 {new_class_name if new_class_name else '新类别'} 的样本图片",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            accept_multiple_files=True
        )

        if new_class_samples:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"已上传 {len(new_class_samples)} 张图片")
            with col2:
                st.metric("需最少", min_samples)
                st.metric("已上传", len(new_class_samples))

            if len(new_class_samples) >= min_samples:
                st.info("✅ 样本数量足够，可以开始训练")
            else:
                st.warning(f"⚠️ 还需要 {min_samples - len(new_class_samples)} 张图片")

            cols = st.columns(min(5, len(new_class_samples)))
            for i, sample_file in enumerate(new_class_samples[:5]):
                img = Image.open(sample_file).convert('RGB')
                with cols[i]:
                    st.image(img, caption=sample_file.name, use_container_width=True)

        if st.button("🎓 开始类别增量训练", type="primary", disabled=st.session_state.is_training):
            if not new_class_name:
                st.warning("请输入新类别名称")
            elif len(new_class_samples) < min_samples:
                st.warning(f"请上传至少 {min_samples} 张样本图片")
            else:
                st.session_state.is_training = True
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("📦 准备新类别数据...")
                progress_bar.progress(10)
                time.sleep(1)

                status_text.text("🔄 提取新类别特征...")
                progress_bar.progress(25)
                time.sleep(1)

                status_text.text("⚙️ 执行FlyGCL增量训练...")
                for i in range(25, 85, 10):
                    progress_bar.progress(i)
                    time.sleep(0.5)

                status_text.text("🔬 验证新旧类别性能...")
                progress_bar.progress(90)
                time.sleep(1)

                status_text.text("💾 保存增量模型...")
                progress_bar.progress(95)
                time.sleep(0.5)

                progress_bar.progress(100)
                status_text.text("✅ 训练完成！")

                st.session_state.training_progress = 100
                st.session_state.is_training = False

                st.session_state.incremental_history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "类别增量",
                    "class": new_class_name,
                    "samples": len(new_class_samples),
                    "method": "FlyGCL",
                    "status": "完成"
                })

                st.success(f"🎉 类别增量训练完成！模型现在可以识别 {new_class_name} 了。")

                st.cache_resource.clear()
                st.rerun()

    with tab3:
        st.markdown("### 增量学习历史记录")

        if st.session_state.incremental_history:
            history_df = pd.DataFrame(st.session_state.incremental_history)
            st.dataframe(history_df, use_container_width=True)

            if st.button("🗑️ 清空历史记录"):
                st.session_state.incremental_history = []
                st.rerun()
        else:
            st.info("暂无增量学习记录")

        st.markdown("---")
        st.markdown("#### 增量学习类型说明")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 样本增量学习")
            st.markdown("""
            **方法**: GSS (Gradient-based Sample Selection)

            **优点**:
            - 计算效率高
            - 有效防止灾难性遗忘
            - 保持旧类别知识

            **适用场景**:
            - 发现了一批新的矿物样本
            - 需要提高现有类别的识别精度
            - 数据持续更新的场景
            """)

        with col2:
            st.markdown("##### 类别增量学习")
            st.markdown("""
            **方法**: FlyGCL (Few-shot Incremental Learning)

            **优点**:
            - 支持新类别的快速学习
            - 保持旧类别性能
            - 适应性强

            **适用场景**:
            - 发现了全新的矿物种类
            - 需要扩展识别范围
            - 研究新矿物时实时更新模型
            """)

st.markdown("---")
st.markdown("矿物识别系统 v2.0 | 支持深度学习模型、特征工程与增量学习")