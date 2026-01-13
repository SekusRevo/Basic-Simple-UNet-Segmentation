"""
Ultrasound segmentation dataset with support for multiple data directories.
Designed for training with original and augmented data.
"""

import logging
import random
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(filename):
    """Load image from various formats."""
    ext = Path(filename).suffix.lower()
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    else:
        return np.array(Image.open(filename).convert('RGB'))


def load_mask(filename):
    """Load mask from various formats."""
    ext = Path(filename).suffix.lower()
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    else:
        img = Image.open(filename)
        mask = np.array(img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask


class UltrasoundDataset(Dataset):
    """
    Dataset for ultrasound image segmentation supporting multiple data directories.

    Args:
        data_dirs: List of data directory paths. Each directory should contain
                   'image' and 'mask' subdirectories.
        img_size: Target image size (height, width) for resizing.
        transform: Optional albumentations transform for data augmentation.
        is_train: Whether this is training data (enables augmentation).
    """

    def __init__(
        self,
        data_dirs: list,
        img_size: tuple = (256, 256),
        transform: A.Compose = None,
        is_train: bool = True
    ):
        self.img_size = img_size
        self.is_train = is_train
        self.samples = []

        # Collect samples from all directories
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            img_dir = data_path / 'image'
            mask_dir = data_path / 'mask'

            if not img_dir.exists():
                logging.warning(f"Image directory not found: {img_dir}")
                continue
            if not mask_dir.exists():
                logging.warning(f"Mask directory not found: {mask_dir}")
                continue

            # Find all images
            img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.npy']
            img_files = []
            for ext in img_extensions:
                img_files.extend(img_dir.glob(ext))
                img_files.extend(img_dir.glob(ext.upper()))

            for img_path in img_files:
                # Find corresponding mask
                stem = img_path.stem
                mask_path = None
                for ext in img_extensions:
                    candidates = list(mask_dir.glob(f"{stem}{ext[1:]}"))
                    candidates.extend(list(mask_dir.glob(f"{stem}{ext[1:].upper()}")))
                    if candidates:
                        mask_path = candidates[0]
                        break

                if mask_path and mask_path.exists():
                    self.samples.append({
                        'image': str(img_path),
                        'mask': str(mask_path),
                        'source': data_path.name
                    })
                else:
                    logging.warning(f"Mask not found for image: {img_path}")

        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs found in {data_dirs}")

        logging.info(f"Created dataset with {len(self.samples)} samples from {len(data_dirs)} directories")

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()

    def _get_train_transform(self):
        """Default training augmentation."""
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _get_val_transform(self):
        """Default validation transform (no augmentation)."""
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = load_image(sample['image'])
        mask = load_mask(sample['mask'])

        # Ensure image is RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Binarize mask (0 or 1)
        mask = (mask > 0).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure mask has correct shape [H, W] -> [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            'image': image.float(),
            'mask': mask.float(),
            'source': sample['source']
        }


def get_dataloaders(
    train_dirs: list,
    test_dir: str,
    img_size: tuple = (256, 256),
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Create train, validation, and test dataloaders.

    Args:
        train_dirs: List of training data directories.
        test_dir: Test data directory.
        img_size: Target image size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        val_split: Fraction of training data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, random_split

    # Set random seed
    set_seed(seed)

    # Create training dataset
    train_dataset = UltrasoundDataset(
        data_dirs=train_dirs,
        img_size=img_size,
        is_train=True
    )

    # Split into train and validation
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create test dataset
    test_dataset = UltrasoundDataset(
        data_dirs=[test_dir],
        img_size=img_size,
        is_train=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logging.info(f"Train: {n_train}, Val: {n_val}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def get_kfold_dataloaders(
    train_dirs: list,
    test_dir: str,
    n_folds: int = 5,
    img_size: tuple = (256, 256),
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create K-Fold cross-validation dataloaders.

    Args:
        train_dirs: List of training data directories.
        test_dir: Test data directory.
        n_folds: Number of folds for cross-validation.
        img_size: Target image size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.

    Yields:
        fold_idx, train_loader, val_loader, test_loader for each fold
    """
    # Set random seed
    set_seed(seed)

    # Create base dataset (without augmentation for proper splitting)
    base_dataset = UltrasoundDataset(
        data_dirs=train_dirs,
        img_size=img_size,
        is_train=False  # No augmentation for base
    )

    # Create augmented dataset for training
    train_dataset = UltrasoundDataset(
        data_dirs=train_dirs,
        img_size=img_size,
        is_train=True  # With augmentation
    )

    # Create test dataset
    test_dataset = UltrasoundDataset(
        data_dirs=[test_dir],
        img_size=img_size,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # K-Fold split
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(base_dataset))

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        logging.info(f"\nFold {fold_idx + 1}/{n_folds}")
        logging.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(base_dataset, val_indices)  # No augmentation for validation

        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        yield fold_idx, train_loader, val_loader, test_loader
