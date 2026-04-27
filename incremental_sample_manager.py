import os
import shutil
import json
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

class IncrementalSampleManager:
    def __init__(self, base_dir="increase_learn_data"):
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "samples"
        self.buffer_dir = self.base_dir / "buffer"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.classes = ['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']
        self.metadata_file = self.base_dir / "metadata.json"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._init_metadata()

    def _init_metadata(self):
        if not self.metadata_file.exists():
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_samples": {c: 0 for c in self.classes},
                "incremental_history": [],
                "buffer_info": {}
            }
            self._save_metadata(metadata)
        else:
            metadata = self._load_metadata()
            if "total_samples" not in metadata:
                metadata["total_samples"] = {c: 0 for c in self.classes}
                self._save_metadata(metadata)
            if "buffer_info" not in metadata:
                metadata["buffer_info"] = {}
                self._save_metadata(metadata)

    def _load_metadata(self):
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_metadata(self, metadata):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def add_samples(self, class_name, image_files):
        if class_name not in self.classes:
            raise ValueError(f"类别 {class_name} 不在支持列表中")

        class_dir = self.samples_dir / class_name
        class_dir.mkdir(exist_ok=True)

        saved_files = []
        metadata = self._load_metadata()

        for idx, img_file in enumerate(image_files):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{class_name}_{timestamp}_{idx}.jpg"
            filepath = class_dir / filename

            if isinstance(img_file, bytes):
                img = Image.open(io.BytesIO(img_file)).convert('RGB')
            elif hasattr(img_file, 'read'):
                img = Image.open(img_file).convert('RGB')
            else:
                img = img_file

            img.save(filepath, 'JPEG')

            saved_files.append(str(filepath))
            metadata["total_samples"][class_name] = metadata["total_samples"].get(class_name, 0) + 1

        history_entry = {
            "time": datetime.now().isoformat(),
            "type": "sample_incremental",
            "class": class_name,
            "count": len(image_files),
            "files": saved_files
        }
        metadata["incremental_history"].append(history_entry)

        self._save_metadata(metadata)

        return {
            "status": "success",
            "class": class_name,
            "saved_count": len(saved_files),
            "total_samples": metadata["total_samples"][class_name],
            "files": saved_files
        }

    def get_class_samples(self, class_name, limit=None):
        class_dir = self.samples_dir / class_name
        if not class_dir.exists():
            return []

        image_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.png"))

        if limit:
            return [str(f) for f in image_files[:limit]]
        return [str(f) for f in image_files]

    def get_sample_count(self, class_name):
        metadata = self._load_metadata()
        return metadata["total_samples"].get(class_name, 0)

    def get_all_counts(self):
        metadata = self._load_metadata()
        return metadata["total_samples"]

    def get_incremental_history(self):
        metadata = self._load_metadata()
        return metadata.get("incremental_history", [])

    def save_buffer(self, buffer_data):
        buffer_file = self.buffer_dir / "gss_buffer.pt"
        torch.save(buffer_data, buffer_file)

        metadata = self._load_metadata()
        metadata["buffer_info"] = {
            "last_updated": datetime.now().isoformat(),
            "buffer_samples": buffer_data[0].shape[0] if buffer_data else 0
        }
        self._save_metadata(metadata)

        return str(buffer_file)

    def load_buffer(self):
        buffer_file = self.buffer_dir / "gss_buffer.pt"
        if buffer_file.exists():
            return torch.load(buffer_file)
        return None

    def get_buffer_info(self):
        metadata = self._load_metadata()
        return metadata.get("buffer_info", {})

    def create_sample_tensor(self, class_name):
        sample_files = self.get_class_samples(class_name)
        if not sample_files:
            return None

        tensors = []
        for img_path in sample_files:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            tensors.append(tensor)

        if tensors:
            return torch.stack(tensors)
        return None

    def get_training_data(self):
        all_tensors = []
        all_labels = []

        for class_name in self.classes:
            class_tensors = self.create_sample_tensor(class_name)
            if class_tensors is not None:
                all_tensors.append(class_tensors)
                class_idx = self.classes.index(class_name)
                labels = torch.full((class_tensors.shape[0],), class_idx, dtype=torch.long)
                all_labels.append(labels)

        if all_tensors:
            return torch.cat(all_tensors, dim=0), torch.cat(all_labels, dim=0)
        return None, None

class IncrementalTrainer:
    def __init__(self, sample_manager, model=None):
        self.sample_manager = sample_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = None

    def set_model(self, model):
        self.model = model.to(self.device)

    def gss_sample_selection(self, gradients_list, memory_per_class=50):
        import numpy as np

        selected_indices = []

        for class_idx in range(len(self.sample_manager.classes)):
            class_name = self.sample_manager.classes[class_idx]
            class_tensors = self.sample_manager.create_sample_tensor(class_name)

            if class_tensors is None or class_tensors.shape[0] == 0:
                continue

            n_samples = min(memory_per_class, class_tensors.shape[0])

            if len(gradients_list) >= class_tensors.shape[0]:
                grads = torch.stack([gradients_list[i] for i in range(class_tensors.shape[0])])
            else:
                grads = torch.randn(class_tensors.shape[0], 512)

            norm_grads = torch.norm(grads, dim=1)
            _, top_indices = torch.topk(norm_grads, min(n_samples, len(norm_grads)))

            selected_indices.extend(top_indices.tolist())

        return selected_indices

    def train_incremental(self, num_epochs=5, progress_callback=None):
        if self.model is None:
            raise ValueError("模型未设置")

        X, y = self.sample_manager.get_training_data()
        if X is None:
            return {"status": "error", "message": "没有可用的训练数据"}

        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        total_loss = 0
        total_batches = num_epochs * len(dataloader)
        current_batch = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                current_batch += 1

                if progress_callback and current_batch % max(1, len(dataloader) // 10) == 0:
                    progress = current_batch / total_batches
                    progress_callback(progress, f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss

        checkpoint_path = self.sample_manager.checkpoint_dir / f"incremental_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_info': {
                'date': datetime.now().isoformat(),
                'total_samples': int(X.shape[0]),
                'num_classes': len(self.sample_manager.classes),
                'avg_loss': total_loss / num_epochs,
                'epochs': num_epochs
            }
        }, checkpoint_path)

        return {
            "status": "success",
            "checkpoint_path": str(checkpoint_path),
            "total_samples": int(X.shape[0]),
            "avg_loss": total_loss / num_epochs,
            "epochs": num_epochs
        }

import io
