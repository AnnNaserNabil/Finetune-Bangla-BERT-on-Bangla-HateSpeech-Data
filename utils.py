"""
Utility functions for Bangla BERT Hate Speech Detection
Includes seed setting, model metrics calculation, and other helper functions
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_model_metrics(model):
    """
    Calculate model size and parameter counts.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary containing total parameters, trainable parameters, and model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(size_mb, 2)
    }


def print_experiment_header(config):
    """
    Print formatted experiment header with configuration details.

    Args:
        config: Configuration object with experiment parameters
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT: {config.author_name} (Hate Speech Detection)")
    print("="*60)
    print(f"Model: {config.model_path}")
    print(f"Batch Size: {config.batch}")
    print(f"Learning Rate: {config.lr}")
    print(f"Max Epochs: {config.epochs}")
    print(f"Max Length: {config.max_length}")
    print(f"Freeze Base: {config.freeze_base}")
    print(f"Stratification: {config.stratification_type}")
    print(f"K-Folds: {config.num_folds}")
    print(f"Dropout: {config.dropout}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Warmup Ratio: {config.warmup_ratio}")
    print(f"Gradient Clip Norm: {config.gradient_clip_norm}")
    print(f"MLflow Experiment: {config.mlflow_experiment_name}")
    print("="*60 + "\n")


def print_fold_summary(fold_num, best_metrics, best_epoch):
    """
    Print summary of fold performance.

    Args:
        fold_num (int): Fold number (0-indexed)
        best_metrics (dict): Best metrics achieved in this fold
        best_epoch (int): Epoch number where best performance was achieved
    """
    print("\n" + "-"*60)
    print(f"FOLD {fold_num + 1} SUMMARY")
    print("-"*60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best F1: {best_metrics['f1']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Precision: {best_metrics['precision']:.4f}")
    print(f"Best Recall: {best_metrics['recall']:.4f}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print("-"*60 + "\n")


def print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics):
    """
    Print final experiment summary.

    Args:
        best_fold_idx (int): Index of best performing fold
        best_fold_metrics (dict): Metrics from the best fold
        model_metrics (dict): Model size and parameter information
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Best performing fold: Fold {best_fold_idx + 1}")
    print(f"  Accuracy: {best_fold_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_fold_metrics['precision']:.4f}")
    print(f"  Recall: {best_fold_metrics['recall']:.4f}")
    print(f"  F1: {best_fold_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {best_fold_metrics['roc_auc']:.4f}")
    print("\nModel Information:")
    print(f"  Model size: {model_metrics['model_size_mb']} MB")
    print(f"  Total parameters: {model_metrics['total_parameters']:,}")
    print(f"  Trainable parameters: {model_metrics['trainable_parameters']:,}")
    print("="*60)
