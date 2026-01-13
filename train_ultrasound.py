"""
Modern training script for ultrasound segmentation using UNet.
Supports training with multiple data directories (original + augmented data).

Features:
- AdamW optimizer with cosine annealing scheduler
- Mixed precision training (AMP)
- Dice + BCE combined loss
- Early stopping
- Comprehensive logging with TensorBoard
- Model checkpointing
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import torchvision.utils as vutils

from Models import U_Net
from ultrasound_dataset import set_seed, get_dataloaders, get_kfold_dataloaders


def dice_coeff(pred, target, smooth=1e-6):
    """Calculate Dice coefficient."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation."""
    return 1 - dice_coeff(pred, target, smooth)


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


@torch.no_grad()
def visualize_predictions(model, loader, device, save_dir, epoch, num_samples=4):
    """Visualize segmentation predictions on validation samples."""
    model.eval()

    # Create visualization directory
    vis_dir = Path(save_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    # Get a batch of samples
    batch = next(iter(loader))
    images = batch['image'][:num_samples].to(device)
    masks = batch['mask'][:num_samples].to(device)

    # Get predictions
    outputs = model(images)
    preds = torch.sigmoid(outputs)
    preds_binary = (preds > 0.5).float()

    # Denormalize images for visualization (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Create visualization grid: [image, ground truth, prediction, overlay]
    vis_list = []
    for i in range(num_samples):
        img = images_denorm[i]  # [3, H, W]
        gt = masks[i].expand(3, -1, -1)  # [3, H, W]
        pred = preds_binary[i].expand(3, -1, -1)  # [3, H, W]

        # Create overlay: green for correct, red for false positive, blue for false negative
        overlay = img.clone()
        tp = (preds_binary[i] == 1) & (masks[i] == 1)  # True positive - green
        fp = (preds_binary[i] == 1) & (masks[i] == 0)  # False positive - red
        fn = (preds_binary[i] == 0) & (masks[i] == 1)  # False negative - blue

        overlay[1] = torch.where(tp.squeeze(), torch.ones_like(overlay[1]) * 0.8, overlay[1])
        overlay[0] = torch.where(fp.squeeze(), torch.ones_like(overlay[0]) * 0.8, overlay[0])
        overlay[2] = torch.where(fn.squeeze(), torch.ones_like(overlay[2]) * 0.8, overlay[2])

        vis_list.extend([img, gt, pred, overlay])

    # Create grid: 4 columns (image, gt, pred, overlay) x num_samples rows
    grid = vutils.make_grid(vis_list, nrow=4, normalize=False, padding=2)

    # Save to file
    save_path = vis_dir / f'epoch_{epoch:03d}.png'
    vutils.save_image(grid, save_path)
    logging.info(f'Saved visualization to {save_path}')


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            dice = dice_coeff(outputs, masks)

        total_loss += loss.item()
        total_dice += dice.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_dice = 0

    for batch in tqdm(loader, desc='Validating'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        with autocast('cuda', enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)

        dice = dice_coeff(outputs, masks)
        total_loss += loss.item()
        total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


def train(args):
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Setup TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Parse training directories
    train_dirs = args.train_dirs.split(',')
    train_dirs = [d.strip() for d in train_dirs]
    logging.info(f'Training directories: {train_dirs}')

    # Create dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        train_dirs=train_dirs,
        test_dir=args.test_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
    )

    # Create model
    model = U_Net(in_ch=3, out_ch=1)
    model = model.to(device)

    # Load pretrained weights if provided
    if args.pretrained:
        logging.info(f'Loading pretrained weights from {args.pretrained}')
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict)

    logging.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(args.epochs // 4, 1),
        T_mult=2,
        eta_min=args.lr * 0.01
    )

    # Loss function
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # Mixed precision scaler
    scaler = GradScaler('cuda') if args.amp and device.type == 'cuda' else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # Training loop
    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        logging.info(f'\n{"="*50}')
        logging.info(f'Epoch {epoch}/{args.epochs}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        logging.info(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Dice', {'train': train_dice, 'val': val_dice}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Visualize predictions periodically
        if epoch % args.vis_freq == 0 or epoch == 1:
            visualize_predictions(model, val_loader, device, output_dir, epoch)

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            logging.info(f'New best model saved! Dice: {best_dice:.4f}')

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

        # Early stopping check
        if early_stopping(val_dice):
            logging.info(f'Early stopping triggered at epoch {epoch}')
            break

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')

    writer.close()
    logging.info(f'\nTraining completed! Best validation Dice: {best_dice:.4f}')
    logging.info(f'Results saved to: {output_dir}')

    return output_dir, best_dice


def train_kfold(args):
    """K-Fold cross-validation training."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'kfold_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse training directories
    train_dirs = args.train_dirs.split(',')
    train_dirs = [d.strip() for d in train_dirs]
    logging.info(f'Training directories: {train_dirs}')
    logging.info(f'Running {args.n_folds}-fold cross-validation')

    fold_results = []

    # K-Fold training
    for fold_idx, train_loader, val_loader, _ in get_kfold_dataloaders(
        train_dirs=train_dirs,
        test_dir=args.test_dir,
        n_folds=args.n_folds,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    ):
        logging.info(f'\n{"="*60}')
        logging.info(f'FOLD {fold_idx + 1}/{args.n_folds}')
        logging.info(f'{"="*60}')

        # Create fold directory
        fold_dir = output_dir / f'fold_{fold_idx + 1}'
        fold_dir.mkdir(exist_ok=True)
        checkpoint_dir = fold_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        # Setup TensorBoard for this fold
        writer = SummaryWriter(fold_dir / 'logs')

        # Create model (fresh for each fold)
        model = U_Net(in_ch=3, out_ch=1)
        model = model.to(device)

        if args.pretrained:
            logging.info(f'Loading pretrained weights from {args.pretrained}')
            state_dict = torch.load(args.pretrained, map_location=device)
            model.load_state_dict(state_dict)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(args.epochs // 4, 1),
            T_mult=2,
            eta_min=args.lr * 0.01
        )

        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        scaler = GradScaler('cuda') if args.amp and device.type == 'cuda' else None
        early_stopping = EarlyStopping(patience=args.patience, mode='max')

        # Training loop for this fold
        best_dice = 0
        for epoch in range(1, args.epochs + 1):
            logging.info(f'\nFold {fold_idx + 1} - Epoch {epoch}/{args.epochs}')

            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
            val_loss, val_dice = validate(model, val_loader, criterion, device)

            scheduler.step()

            logging.info(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
            logging.info(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Dice', {'train': train_dice, 'val': val_dice}, epoch)

            # Visualize predictions periodically
            if epoch % args.vis_freq == 0 or epoch == 1:
                visualize_predictions(model, val_loader, device, fold_dir, epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
                logging.info(f'New best model for fold {fold_idx + 1}! Dice: {best_dice:.4f}')

            if early_stopping(val_dice):
                logging.info(f'Early stopping triggered at epoch {epoch}')
                break

        torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')
        writer.close()

        fold_results.append({
            'fold': fold_idx + 1,
            'best_dice': best_dice
        })

        logging.info(f'\nFold {fold_idx + 1} completed. Best Dice: {best_dice:.4f}')

    # Summary
    logging.info(f'\n{"="*60}')
    logging.info('K-FOLD CROSS-VALIDATION SUMMARY')
    logging.info(f'{"="*60}')

    dice_scores = [r['best_dice'] for r in fold_results]
    for r in fold_results:
        logging.info(f"Fold {r['fold']}: Dice = {r['best_dice']:.4f}")

    mean_dice = sum(dice_scores) / len(dice_scores)
    std_dice = (sum((d - mean_dice) ** 2 for d in dice_scores) / len(dice_scores)) ** 0.5

    logging.info(f'\nMean Dice: {mean_dice:.4f} +/- {std_dice:.4f}')
    logging.info(f'Results saved to: {output_dir}')

    # Save summary
    import json
    summary = {
        'n_folds': args.n_folds,
        'fold_results': fold_results,
        'mean_dice': mean_dice,
        'std_dice': std_dice
    }
    with open(output_dir / 'kfold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return output_dir, mean_dice, std_dice


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet for ultrasound segmentation')

    # Data arguments
    parser.add_argument('--train-dirs', type=str, required=True,
                        help='Comma-separated list of training data directories')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Test data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')

    # Model arguments
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (used when not using k-fold)')

    # Training options
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis-freq', type=int, default=5,
                        help='Visualize predictions every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # K-Fold cross-validation
    parser.add_argument('--kfold', action='store_true', default=False,
                        help='Use K-Fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.kfold:
        train_kfold(args)
    else:
        train(args)
