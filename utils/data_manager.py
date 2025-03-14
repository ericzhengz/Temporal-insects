import os
import json
import random
import numpy as np
import logging
from torch.utils.data import Subset
import torchvision.transforms as transforms
from utils.data import IIMinsects202, IIMinsects202Dataset 

class DataManager(object):
    """
    DataManager for IIMinsects202 dataset.
    
    This DataManager reads the dataset from the specified train and test folders.
    It assumes that both "train" and "test" directories share the same folder structure.
    
    It also partitions the training classes into incremental tasks.
    """
    def __init__(self, shuffle, seed, init_cls, increment, args):
        """
        Args:
            shuffle (bool): Whether to shuffle the class order.
            seed (int): Random seed.
            init_cls (int): Number of classes in the initial task.
            increment (int): Number of classes in subsequent tasks.
            args (dict): Must contain:
                  "dataset_root": Root directory of the dataset.
                  "meta_file": Path to meta.json.
                  "samples_per_class": (optional) number of samples per class.
        """
        self.args = args
        self.shuffle = shuffle
        self.seed = seed
        self.init_cls = init_cls
        self.increment = increment

        self._setup_data()
        # Build incremental tasks: _increments is a list of number of classes per task.
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def _setup_data(self):
        """
        Loads training and testing data using the IIMinsects202 class.
        Expects a folder structure:
            dataset_root/
                train/
                test/
                meta.json
        Performs an 80:20 class split (by class labels) for train/test.
        """
        dataset_root = self.args["dataset_root"]

        # Create dataset instances for train and test.
        train_data_obj = IIMinsects202(self.args, split="train")
        train_data_obj.download_data()
        test_data_obj = IIMinsects202(self.args, split="test")
        test_data_obj.download_data()

        # Full arrays from train and test
        full_train_samples = train_data_obj.samples
        full_train_targets = train_data_obj.targets
        full_test_samples = test_data_obj.samples
        full_test_targets = test_data_obj.targets

        self.use_path = train_data_obj.use_path

        # Use transformations from iIIMinsects202
        self._train_trsf = train_data_obj.train_trsf
        self._test_trsf = test_data_obj.test_trsf
        self._common_trsf = train_data_obj.common_trsf

        # Build class order from meta (if available) or simply sorted unique classes from training targets.
        unique_classes = sorted(list(np.unique(full_train_targets)))
        if self.shuffle:
            random.seed(self.seed)
            unique_classes = random.sample(unique_classes, len(unique_classes))
        self._class_order = unique_classes
        logging.info(f"Class order: {self._class_order}")

        # Map targets to new indices based on _class_order.
        mapped_train_targets = np.array([self._class_order.index(t) for t in full_train_targets], dtype=int)
        mapped_test_targets = np.array([self._class_order.index(t) for t in full_test_targets], dtype=int)

        # Here, the meta.json might specify a desired 80:20 split by class.
        # We'll assume the test set is already provided.
        self._train_data = full_train_samples
        self._train_targets = mapped_train_targets
        self._test_data = full_test_samples
        self._test_targets = mapped_test_targets

        # 添加对每个类别stage信息的处理
        self.class_stage_mapping = train_data_obj.class_stage_mapping
        self.max_stages = train_data_obj.max_stages

        # 获取每个类别的stage信息
        def get_class_stages(class_id):
            class_info = self.class_stage_mapping.get(f"class_{class_id}", {})
            return class_info.get("stage_ids", list(range(self.max_stages)))

        # 增加类别名称的日志输出
        class_names = []
        for cid in self._class_order:
            class_key = f"class_{cid}"
            class_info = self.class_stage_mapping.get(class_key, {})
            folder_name = class_info.get('folder_name', f"class_{cid}")
            class_names.append(f"{cid}:{folder_name}")
            
        logging.info(f"Class order (id:folder_name): {class_names}")

    def get_dataset(self, indices, source="train", mode="train", ret_data=False):
        """
        Returns a PyTorch Dataset (IIMinsects202Dataset) filtered by given class indices.
        Args:
            indices (list): List of class indices (in the new mapping) to include.
            source (str): "train" or "test"
            mode (str): "train" or "test" for transformation choices.
            ret_data (bool): If True, also return (data_array, targets, dataset_obj).
        """
        if source == "train":
            data_array, targets = self._train_data, self._train_targets
        elif source == "test":
            data_array, targets = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown source: {source}")

        if mode == "train":
            trsf = transforms.Compose(self._train_trsf + self._common_trsf)
        else:
            trsf = transforms.Compose(self._test_trsf + self._common_trsf)

        selected_indices = [i for i in range(len(data_array)) if targets[i] in indices]
        filtered_data = data_array[selected_indices]
        filtered_targets = targets[selected_indices]
        dataset_obj = IIMinsects202Dataset(
            filtered_data, 
            targets=filtered_targets,  # 添加targets参数
            transform=trsf, 
            use_path=self.use_path,
            class_stage_mapping=self.class_stage_mapping
        )
        if ret_data:
            return filtered_data, filtered_targets, dataset_obj
        return dataset_obj

    def get_task_dataset(self, task, source="train", mode="train"):
        """
        Returns a dataset for a given incremental task.
        Args:
            task (int):  Task index (starting from 0).
            source (str): "train" or "test"
            mode (str): "train" or "test"
        """
        start = sum(self._increments[:task])
        end = sum(self._increments[:task + 1])
        class_indices = self._class_order[start:end]
        return self.get_dataset(class_indices, source=source, mode=mode)

    def getlen(self, class_idx):
        """
        Returns the number of samples for a given class index in the training set.
        """
        return np.sum(self._train_targets == class_idx)
