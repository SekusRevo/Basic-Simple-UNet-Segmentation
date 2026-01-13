"""
Evaluation script for ultrasound segmentation models.
Compares model performance with and without augmented training data.

Metrics:
- Dice coefficient
- IoU (Intersection over Union)
- Precision, Recall, F1
- Hausdorff Distance (optional)
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from Models import U_Net
from Metrics import dice_coeff as numpy_dice_coeff
from ultrasound_dataset import UltrasoundDataset


class SegmentationMetrics:
    """Calculate segmentation metrics."""

    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.dice_scores = []
        self.iou_scores = []

    def update(self, pred, target):
        """
        Update metrics with predictions and targets.

        Args:
            pred: Model predictions (logits), shape [B, 1, H, W]
            target: Ground truth masks, shape [B, 1, H, W]
        """
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()

        # Flatten for metric calculation
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)

        # Calculate confusion matrix elements
        self.tp += ((pred_flat == 1) & (target_flat == 1)).sum().item()
        self.fp += ((pred_flat == 1) & (target_flat == 0)).sum().item()
        self.tn += ((pred_flat == 0) & (target_flat == 0)).sum().item()
        self.fn += ((pred_flat == 0) & (target_flat == 1)).sum().item()

        # Calculate per-sample Dice and IoU
        for i in range(pred.shape[0]):
            p = pred_binary[i].view(-1)
            t = target[i].view(-1)

            intersection = (p * t).sum()
            union = p.sum() + t.sum()

            # Dice
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            self.dice_scores.append(dice.item())

            # IoU
            iou = (intersection + self.smooth) / (union - intersection + self.smooth)
            self.iou_scores.append(iou.item())

    def compute(self):
        """Compute all metrics."""
        total = self.tp + self.fp + self.tn + self.fn

        # Pixel-level metrics
        accuracy = (self.tp + self.tn) / (total + self.smooth)
        precision = self.tp / (self.tp + self.fp + self.smooth)
        recall = self.tp / (self.tp + self.fn + self.smooth)
        specificity = self.tn / (self.tn + self.fp + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)

        # Sample-level metrics
        dice_mean = np.mean(self.dice_scores)
        dice_std = np.std(self.dice_scores)
        iou_mean = np.mean(self.iou_scores)
        iou_std = np.std(self.iou_scores)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'dice_mean': dice_mean,
            'dice_std': dice_std,
            'iou_mean': iou_mean,
            'iou_std': iou_std,
            'num_samples': len(self.dice_scores)
        }


@torch.no_grad()
def evaluate_model(model, dataloader, device, save_predictions=False, output_dir=None):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        save_predictions: Whether to save prediction images
        output_dir: Directory to save predictions

    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = SegmentationMetrics()

    if save_predictions and output_dir:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    for batch in tqdm(dataloader, desc='Evaluating'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        metrics.update(outputs, masks)

        # Save predictions
        if save_predictions and output_dir:
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()

            for i in range(images.shape[0]):
                # Save prediction mask
                pred_np = (pred_binary[i, 0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_np).save(pred_dir / f'pred_{sample_idx:04d}.png')

                # Save ground truth for comparison
                gt_np = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(gt_np).save(pred_dir / f'gt_{sample_idx:04d}.png')

                sample_idx += 1

    return metrics.compute()


def compare_models(args):
    """Compare models trained with and without augmented data."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Create test dataloader
    test_dataset = UltrasoundDataset(
        data_dirs=[args.test_dir],
        img_size=(args.img_size, args.img_size),
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logging.info(f'Test dataset: {len(test_dataset)} samples')

    results = {}

    # Evaluate baseline model (without augmentation)
    if args.baseline_model:
        logging.info('\n' + '='*50)
        logging.info('Evaluating BASELINE model (without augmented data)')
        logging.info('='*50)

        model_baseline = U_Net(in_ch=3, out_ch=1)
        model_baseline.load_state_dict(torch.load(args.baseline_model, map_location=device))
        model_baseline = model_baseline.to(device)

        baseline_metrics = evaluate_model(
            model_baseline, test_loader, device,
            save_predictions=args.save_predictions,
            output_dir=Path(args.output_dir) / 'baseline' if args.output_dir else None
        )

        results['baseline'] = baseline_metrics
        logging.info(f"Baseline Results:")
        logging.info(f"  Dice: {baseline_metrics['dice_mean']:.4f} +/- {baseline_metrics['dice_std']:.4f}")
        logging.info(f"  IoU:  {baseline_metrics['iou_mean']:.4f} +/- {baseline_metrics['iou_std']:.4f}")
        logging.info(f"  Precision: {baseline_metrics['precision']:.4f}")
        logging.info(f"  Recall: {baseline_metrics['recall']:.4f}")
        logging.info(f"  F1: {baseline_metrics['f1']:.4f}")

    # Evaluate augmented model
    if args.augmented_model:
        logging.info('\n' + '='*50)
        logging.info('Evaluating AUGMENTED model (with generated data)')
        logging.info('='*50)

        model_augmented = U_Net(in_ch=3, out_ch=1)
        model_augmented.load_state_dict(torch.load(args.augmented_model, map_location=device))
        model_augmented = model_augmented.to(device)

        augmented_metrics = evaluate_model(
            model_augmented, test_loader, device,
            save_predictions=args.save_predictions,
            output_dir=Path(args.output_dir) / 'augmented' if args.output_dir else None
        )

        results['augmented'] = augmented_metrics
        logging.info(f"Augmented Results:")
        logging.info(f"  Dice: {augmented_metrics['dice_mean']:.4f} +/- {augmented_metrics['dice_std']:.4f}")
        logging.info(f"  IoU:  {augmented_metrics['iou_mean']:.4f} +/- {augmented_metrics['iou_std']:.4f}")
        logging.info(f"  Precision: {augmented_metrics['precision']:.4f}")
        logging.info(f"  Recall: {augmented_metrics['recall']:.4f}")
        logging.info(f"  F1: {augmented_metrics['f1']:.4f}")

    # Compare results
    if 'baseline' in results and 'augmented' in results:
        logging.info('\n' + '='*50)
        logging.info('COMPARISON: Augmented vs Baseline')
        logging.info('='*50)

        dice_diff = results['augmented']['dice_mean'] - results['baseline']['dice_mean']
        iou_diff = results['augmented']['iou_mean'] - results['baseline']['iou_mean']
        f1_diff = results['augmented']['f1'] - results['baseline']['f1']

        logging.info(f"Dice improvement: {dice_diff:+.4f} ({dice_diff/results['baseline']['dice_mean']*100:+.2f}%)")
        logging.info(f"IoU improvement:  {iou_diff:+.4f} ({iou_diff/results['baseline']['iou_mean']*100:+.2f}%)")
        logging.info(f"F1 improvement:   {f1_diff:+.4f} ({f1_diff/results['baseline']['f1']*100:+.2f}%)")

        results['comparison'] = {
            'dice_improvement': dice_diff,
            'dice_improvement_pct': dice_diff / results['baseline']['dice_mean'] * 100,
            'iou_improvement': iou_diff,
            'iou_improvement_pct': iou_diff / results['baseline']['iou_mean'] * 100,
            'f1_improvement': f1_diff,
            'f1_improvement_pct': f1_diff / results['baseline']['f1'] * 100,
        }

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'evaluation_results_{timestamp}.json'

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        results_serializable = convert_to_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logging.info(f'\nResults saved to: {results_file}')

    return results


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate ultrasound segmentation models')

    # Model paths
    parser.add_argument('--baseline-model', type=str, default=None,
                        help='Path to baseline model (trained without augmented data)')
    parser.add_argument('--augmented-model', type=str, default=None,
                        help='Path to augmented model (trained with generated data)')

    # Data arguments
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Test data directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')

    # Model arguments

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    compare_models(args)
