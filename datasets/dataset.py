"""
数据加载模块实现
支持Crack500、CFD、Sun520三种裂缝数据集
"""
import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Callable, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BaseCrackDataset(Dataset):
    """
    裂缝分割数据集基类
    
    参数：
        root_dir: 数据集根目录
        split: 数据集划分 ('train', 'val', 'test')
        transform: 数据增强变换
        target_size: 目标尺寸
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transform: Optional[A.Compose] = None,
            target_size: int = 512
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size

        self.images = []
        self.masks = []

        self._load_data()

    def _load_data(self):
        """加载数据路径，子类需要实现"""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        获取数据样本
        
        返回：
            image: 图像张量 (3, H, W)
            mask: 掩码张量 (1, H, W)
            meta: 元数据字典
        """
        # 读取图像
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask = self._read_mask(idx)

        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 默认变换
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # 元数据
        meta = {
            'image_path': self.images[idx],
            'mask_path': self.masks[idx] if idx < len(self.masks) else None,
            'original_size': image.shape[-2:]
        }

        return image, mask, meta

    def _read_mask(self, idx: int) -> np.ndarray:
        """读取掩码，子类可重写"""
        mask = cv2.imread(self.masks[idx], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask


class Crack500Dataset(BaseCrackDataset):
    """
    Crack500数据集
    
    结构：
    crack500/
    ├── train/
    │   ├── image/
    │   └── mask/
    ├── test/
    │   ├── image/
    │   └── mask/
    └── validation/
        ├── image/
        └── mask/
    """

    def _load_data(self):
        # 根据split确定目录名
        split_dir = 'train' if self.split == 'train' else ('test' if self.split == 'test' else 'validation')

        image_dir = os.path.join(self.root_dir, split_dir, 'image')
        mask_dir = os.path.join(self.root_dir, split_dir, 'mask')

        if not os.path.exists(image_dir):
            print(f"警告: 目录 {image_dir} 不存在")
            return

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)

            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

    def _read_mask(self, idx: int) -> np.ndarray:
        """Crack500的mask可能是3通道，需要转换"""
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)

        # 如果mask有3通道，转换为灰度
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask


class CFDDataset(BaseCrackDataset):
    """
    CFD (Crack Fragmentation Dataset) 数据集
    
    结构：
    cfd/
    ├── train/
    │   ├── image/
    │   │   └── img_001.jpg
    │   └── groundtruth/
    │       └── img_001_label.png
    ├── validation/
    │   ├── image/
    │   └── groundtruth/
    └── test/
        ├── image/
        └── groundtruth/
    """

    def _load_data(self):
        if self.split in ['val', 'validation']:
            split_dir = 'validation'
        else:
            split_dir = self.split

        image_dir = os.path.join(self.root_dir, split_dir, 'image')
        mask_dir = os.path.join(self.root_dir, split_dir, 'groundtruth')

        if not os.path.exists(image_dir):
            print(f"警告: 目录 {image_dir} 不存在")
            return

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)

            # CFD数据集的mask命名规则
            # image: img_001.jpg -> mask: img_001_label.png
            # 或者 validation中特殊命名
            base_name = os.path.splitext(img_file)[0]
            if '_label' in base_name:
                # validation数据集的特殊命名
                mask_name = base_name.replace('_label', '') + '.PNG'
            else:
                mask_name = base_name + '_label.PNG'

            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)


class Sun520Dataset(BaseCrackDataset):
    """
    Sun520数据集 (高分辨率)
    
    结构：
    sun520/
    ├── train/
    │   ├── image/
    │   │   └── xxx_0001.png (3968×2240)
    │   └── gt/
    │       └── xxx_0001.png
    └── test/
        ├── image/
        └── gt/
    
    特点：高分辨率，使用滑动窗口裁剪
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transform: Optional[A.Compose] = None,
            target_size: int = 512,
            crop_size: int = 512,
            stride: int = 256,
            use_sliding_window: bool = True
    ):
        self.crop_size = crop_size
        self.stride = stride
        self.use_sliding_window = use_sliding_window
        self.crops_info = []

        super().__init__(root_dir, split, transform, target_size)

    def _load_data(self):
        split_dir = 'train' if self.split == 'train' else 'test'

        image_dir = os.path.join(self.root_dir, split_dir, 'image')
        mask_dir = os.path.join(self.root_dir, split_dir, 'gt')

        if not os.path.exists(image_dir):
            print(f"警告: 目录 {image_dir} 不存在")
            return

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)

            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

                # 如果使用滑动窗口，计算裁剪位置
                if self.use_sliding_window:
                    self._calculate_crops(img_path)

    def _calculate_crops(self, img_path: str):
        """计算图像的滑动窗口裁剪位置"""
        # 读取图像尺寸
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 计算裁剪位置
        for y in range(0, h - self.crop_size + 1, self.stride):
            for x in range(0, w - self.crop_size + 1, self.stride):
                self.crops_info.append({
                    'image_idx': len(self.images) - 1,
                    'crop_y': y,
                    'crop_x': x
                })

        # 处理边界情况
        if h > self.crop_size and (h - self.crop_size) % self.stride != 0:
            for x in range(0, w - self.crop_size + 1, self.stride):
                self.crops_info.append({
                    'image_idx': len(self.images) - 1,
                    'crop_y': h - self.crop_size,
                    'crop_x': x
                })

        if w > self.crop_size and (w - self.crop_size) % self.stride != 0:
            for y in range(0, h - self.crop_size + 1, self.stride):
                self.crops_info.append({
                    'image_idx': len(self.images) - 1,
                    'crop_y': y,
                    'crop_x': w - self.crop_size
                })

    def __len__(self) -> int:
        if self.use_sliding_window:
            return len(self.crops_info)
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if self.use_sliding_window:
            return self._get_crop_item(idx)
        return super().__getitem__(idx)

    def _get_crop_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """获取滑动窗口裁剪样本"""
        crop_info = self.crops_info[idx]
        img_idx = crop_info['image_idx']
        y, x = crop_info['crop_y'], crop_info['crop_x']

        # 读取图像和掩码
        image = cv2.imread(self.images[img_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[img_idx], cv2.IMREAD_GRAYSCALE)

        # 裁剪
        image = image[y:y + self.crop_size, x:x + self.crop_size]
        mask = mask[y:y + self.crop_size, x:x + self.crop_size]

        # 二值化
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        meta = {
            'image_path': self.images[img_idx],
            'mask_path': self.masks[img_idx],
            'crop_position': (y, x),
            'crop_size': self.crop_size,
            'original_size': (self.crop_size, self.crop_size)
        }

        return image, mask, meta


def get_transforms(
        mode: str = 'train',
        target_size: int = 512
) -> A.Compose:
    """
    获取数据增强变换
    
    参数：
        mode: 'train', 'val', 'test'
        target_size: 目标尺寸
        
    返回：
        Albumentations变换组合
    """
    if mode == 'train':
        transform = A.Compose([
            A.Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC),
            # 增强对比度，凸显细微裂缝
            A.CLAHE(clip_limit=4., tile_grid_size=(8, 8), p=0.8),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Affine(translate_percent=0.1, scale=0.1, rotate=15, p=0.5),
            # A.OneOf([
            #     A.GaussNoise(std_range=(math.sqrt(0.1), math.sqrt(0.2)), p=1),
            #     A.GaussianBlur(blur_limit=3, p=1),
            # ], p=0.3),
            # A.OneOf([
            #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            # ], p=0.3),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC),
            # 测试和验证集必须以 p=1.0 概率做同样的对比度增强
            A.CLAHE(clip_limit=4., tile_grid_size=(8, 8), p=1.),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    return transform


def get_dataloader(
        dataset_name: str,
        root_dir: str,
        split: str = 'train',
        batch_size: int = 8,
        target_size: int = 512,
        num_workers: int = 4,
        **kwargs
) -> DataLoader:
    """
    获取数据加载器
    
    参数：
        dataset_name: 数据集名称 ('crack500', 'cfd', 'sun520')
        root_dir: 数据集根目录
        split: 数据集划分
        batch_size: 批次大小
        target_size: 目标尺寸
        num_workers: 工作进程数
        
    返回：
        DataLoader
    """
    # 获取变换
    transform = get_transforms(mode=split, target_size=target_size)

    # 创建数据集
    dataset_map = {
        'crack500': Crack500Dataset,
        'cfd': CFDDataset,
        'sun520': Sun520Dataset
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"未知数据集: {dataset_name}，支持的数据集: {list(dataset_map.keys())}")

    dataset_class = dataset_map[dataset_name.lower()]

    # Sun520特殊处理
    if dataset_name.lower() == 'sun520':
        dataset = dataset_class(
            root_dir=root_dir,
            split=split,
            transform=transform,
            target_size=target_size,
            use_sliding_window=kwargs.get('use_sliding_window', True)
        )
    else:
        dataset = dataset_class(
            root_dir=root_dir,
            split=split,
            transform=transform,
            target_size=target_size
        )

    # 创建数据加载器
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


class CombinedDataset(Dataset):
    """
    组合多个数据集用于联合训练
    
    参数：
        datasets: 数据集列表
        weights: 各数据集采样权重
    """

    def __init__(
            self,
            datasets: List[Dataset],
            weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)

        # 计算累积长度
        self.cumulative_lengths = [0]
        for dataset in datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # 确定使用哪个数据集
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]

        raise IndexError(f"Index {idx} out of range")


def ai_test():
    print("=" * 60)
    print("测试数据加载模块")
    print("=" * 60)

    # 测试变换
    print("\n[1] 测试数据变换...")
    train_transform = get_transforms('train', 512)
    val_transform = get_transforms('val', 512)
    print(f"  训练变换: {len(train_transform.transforms)} 个")
    print(f"  验证变换: {len(val_transform.transforms)} 个")

    # 创建模拟数据进行测试
    print("\n[2] 创建模拟数据进行测试...")

    # 模拟图像和掩码
    mock_image = np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)
    mock_mask = np.random.randint(0, 2, (320, 640), dtype=np.uint8) * 255

    # 应用变换
    augmented = train_transform(image=mock_image, mask=mock_mask)
    image_tensor = augmented['image']
    mask_tensor = augmented['mask']

    print(f"  原始图像: {mock_image.shape}")
    print(f"  原始掩码: {mock_mask.shape}")
    print(f"  变换后图像: {image_tensor.shape}")
    print(f"  变换后掩码: {mask_tensor.shape}")

    # 测试Crack500Dataset（需要实际数据）
    print("\n[3] 测试Crack500Dataset...")
    print("  (跳过，需要实际数据)")

    # 测试CFDDataset
    print("\n[4] 测试CFDDataset...")
    print("  (跳过，需要实际数据)")

    # 测试Sun520Dataset
    print("\n[5] 测试Sun520Dataset...")
    print("  (跳过，需要实际数据)")

    # 测试组合数据集
    print("\n[6] 测试CombinedDataset...")

    # 创建模拟数据集
    class MockDataset(Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            image = torch.randn(3, 512, 512)
            mask = torch.randint(0, 2, (1, 512, 512)).float()
            meta = {'idx': idx}
            return image, mask, meta

    dataset1 = MockDataset(100)
    dataset2 = MockDataset(50)
    combined = CombinedDataset([dataset1, dataset2])

    print(f"  数据集1大小: {len(dataset1)}")
    print(f"  数据集2大小: {len(dataset2)}")
    print(f"  组合数据集大小: {len(combined)}")

    # 获取样本
    image, mask, meta = combined[0]
    print(f"  样本图像形状: {image.shape}")
    print(f"  样本掩码形状: {mask.shape}")

    print("\n" + "=" * 60)
    print("数据加载模块测试完成！")
    print("注意: 完整测试需要实际数据集")
    print("=" * 60)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    data_loader = get_dataloader('crack500', root_dir='D:/dev/data/crack500', split='train', batch_size=1)
    # 获取第一个 batch
    inputs, targets, _ = next(iter(data_loader))
    img = inputs.detach().cpu().numpy()
    mask = targets.detach().cpu().numpy()
    print("Inputs:", img.shape)
    print("Targets:", mask.shape)
    from matplotlib import pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(0).transpose((1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(0))
    plt.show()
