import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

def pil_loader(path):
    """
    Loads an image in RGB mode.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class IIMinsects202(object):
    def __init__(self, args, split="train"):
        """
        Args:
            args (dict): parameters containing:
                "dataset_root": path to the dataset root directory.
                "meta_file": path to meta.json
                "samples_per_class": number of samples to generate per class
            split (str): "train" or "test"
        """
        self.args = args
        self.use_path = True
        self.split = split.lower()
        # 添加samples_per_class参数
        self.samples_per_class = args.get("samples_per_class", 70)  # 默认值为100
        
        dataset_root = args["dataset_root"]
        # Expect meta.json at the root of dataset_root
        meta_file = args.get("meta_file", os.path.join(dataset_root, "meta.json"))
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as jf:
                self.meta = json.load(jf)
        else:
            self.meta = {}
        # Use stage_ids from meta if available; otherwise default to two stages.
        self.class_stage_mapping = self.meta.get("class_stage_mapping", {})
        self.max_stages = self.meta.get("global_settings", {}).get("max_stages", 2)
        
        # Define transforms (you can adjust these as needed)
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        self.common_trsf = []  # can add common transforms here if needed
        
        # Initialize lists to store samples.
        # Each sample: ( [path for stage0, ..., path for stageT], class_id, [stage0, stage1, ...] )
        self.samples = []  
        self.targets = []  
        
    def download_data(self):
        """
        Parse the dataset from the directory structure.
        It expects that under dataset_root, there are "train" and "test" folders.
        """
        dataset_root = self.args["dataset_root"]
        split_dir = os.path.join(dataset_root, self.split)  # e.g., dataset_root/train or dataset_root/test
        if not os.path.isdir(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist.")
            
        # 构建文件夹名称到class_id的映射
        folder_to_class = {}
        for class_key, class_info in self.class_stage_mapping.items():
            cid = int(class_key.split('_')[-1])
            folder_name = class_info.get('folder_name', f"class_{cid}")
            folder_to_class[folder_name] = cid
            
        # 修改类别目录遍历逻辑
        class_dirs = sorted(os.listdir(split_dir))
        class_ids = []
        for cdir in class_dirs:
            cpath = os.path.join(split_dir, cdir)
            if not os.path.isdir(cpath):
                continue
                
            # 使用文件夹名称查找对应的class_id
            if cdir in folder_to_class:
                cid = folder_to_class[cdir]
                class_ids.append(cid)
                
        class_ids = sorted(class_ids)
        self.class_order = class_ids

        # 在处理每个类别时使用实际的文件夹名称
        for cid in class_ids:
            # 从class_stage_mapping中获取实际的文件夹名称
            folder_name = None
            for class_key, class_info in self.class_stage_mapping.items():
                if int(class_key.split('_')[-1]) == cid:
                    folder_name = class_info.get('folder_name', f"class_{cid}")
                    break
                    
            # 使用实际的文件夹名称
            class_folder = os.path.join(split_dir, folder_name)
            if not os.path.isdir(class_folder):
                continue
                
            # 获取该类别的特定阶段ID
            class_info = self.class_stage_mapping.get(f"class_{cid}", {})
            class_stage_ids = class_info.get("stage_ids", list(range(self.max_stages)))
            
            # 构建stage_map
            stage_map = {}
            stage_dirs = os.listdir(class_folder)
            for sdir in stage_dirs:
                spath = os.path.join(class_folder, sdir)
                if not os.path.isdir(spath):
                    continue
                try:
                    sid = int(sdir.split('_')[-1])
                    # 只处理该类别允许的阶段
                    if sid not in class_stage_ids:
                        continue
                except:
                    continue
                    
                img_files = [os.path.join(spath, f) for f in os.listdir(spath)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if img_files:
                    stage_map[sid] = img_files
                    
            # 对每个样本进行采样
            for _ in range(self.samples_per_class):  # 使用类的samples_per_class参数
                img_paths = []
                stage_list = []
                
                # 使用该类别的特定阶段ID
                for sid in class_stage_ids:
                    if sid in stage_map:
                        pick = random.choice(stage_map[sid])
                        img_paths.append(pick)
                        stage_list.append(sid)
                    else:
                        # 如果某个期望的阶段缺失
                        if len(stage_map) > 0:
                            available_stage = random.choice(list(stage_map.keys()))
                            pick = random.choice(stage_map[available_stage])
                            img_paths.append(pick)
                            stage_list.append(available_stage)
                        else:
                            img_paths.append(None)
                            stage_list.append(-1)
                            
                self.samples.append((img_paths, cid, stage_list))
                self.targets.append(cid)

        # Convert to numpy arrays for compatibility with DataManager.
        self.samples = np.array(self.samples, dtype=object)
        self.targets = np.array(self.targets, dtype=int)


class IIMinsects202Dataset(Dataset):
    """IIMinsects202 Dataset"""
    def __init__(self, samples, targets=None, transform=None, use_path=True, class_stage_mapping=None):
        self.transform = transform
        self.use_path = use_path
        self.class_stage_mapping = class_stage_mapping or {}
        self.samples = samples
        self.targets = targets
        
        # 验证样本格式
        if len(samples) > 0 and self.use_path:
            # 检查第一个样本以了解格式
            sample = samples[0]
            if isinstance(sample, tuple) and len(sample) >= 3:
                # 格式: (img_paths, cid, stage_list)
                for s in samples:
                    for path in s[0]:
                        if path is not None and not os.path.exists(path):
                            print(f"Warning: Image path does not exist: {path}")
            elif isinstance(sample, (list, np.ndarray)) and len(sample) >= 2:
                # 处理其他可能的格式
                pass
            else:
                # 简单格式: 直接是路径或者数组
                pass

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 处理 (img_paths, cid, stage_list) 格式
        if isinstance(sample, tuple) and len(sample) >= 3:
            img_paths, _, stage_list = sample
            
            if self.use_path:
                # 加载所有有效图像
                imgs = []
                valid_stages = []
                
                for i, path in enumerate(img_paths):
                    if path is not None and os.path.exists(path):
                        img = Image.open(path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        imgs.append(img)
                        valid_stages.append(stage_list[i])
                
                if len(imgs) == 0:
                    # 如果没有有效图像，创建一个空白图像
                    img = torch.zeros((3, 224, 224))
                    stage = torch.tensor([0], dtype=torch.long)
                elif len(imgs) == 1:
                    img = imgs[0]
                    stage = torch.tensor(valid_stages, dtype=torch.long)
                else:
                    img = torch.stack(imgs)  # [T, C, H, W]
                    stage = torch.tensor(valid_stages, dtype=torch.long)
                    
            else:
                # 处理非路径数据
                if isinstance(img_paths, (list, np.ndarray)):
                    img = torch.tensor(img_paths)
                    # 修改: 只对非tensor数据应用转换
                    if self.transform and not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                else:
                    img = img_paths
                    # 修改: 只对非tensor数据应用转换
                    if self.transform and not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                stage = torch.tensor(stage_list, dtype=torch.long)
        
        # 处理其他格式
        else:
            if self.use_path:
                if isinstance(sample, (str, bytes)):
                    # 纯路径
                    img = Image.open(sample).convert('RGB')
                else:
                    # 非路径且非元组，可能是数组等其他格式
                    img = torch.zeros((3, 224, 224))
            else:
                img = sample
                
            # 修改: 只对非tensor数据应用转换
            if self.transform and not isinstance(img, torch.Tensor):
                img = self.transform(img)
                
            # 默认阶段
            stage = torch.tensor([0], dtype=torch.long)
        
        # 返回结果
        if self.targets is not None:
            target = self.targets[index]
            return index, img, target, stage
        
        return index, img

    def __len__(self):
        return len(self.samples)
