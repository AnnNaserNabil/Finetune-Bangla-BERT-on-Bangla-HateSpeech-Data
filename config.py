"""
Configuration module for Bangla BERT Hate Speech Detection
Includes all hyperparameters and experiment settings
"""

import argparse


def parse_arguments():
    """
    Parse command-line arguments for experiment configuration.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Transformer models for binary Bangla hate speech detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Maximum number of training epochs.')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the CSV dataset file.')
    parser.add_argument('--model_path', type=str, default='sagorsarker/bangla-bert-base',
                       help='Pre-trained model name or path.')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization.')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of folds for K-Fold cross-validation.')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze the base transformer layers during fine-tuning.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility.')
    parser.add_argument('--stratification_type', type=str, default='binary',
                       choices=['binary', 'none'],
                       help='Type of stratification for K-fold splitting. '
                            'binary: stratifies based on single hate speech label, '
                            'none: no stratification (regular K-fold).')
    parser.add_argument('--author_name', type=str, required=True,
                       help='Author name for MLflow run tagging.')
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-HateSpeech-Detection',
                       help='MLflow experiment name for tracking.')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for the classification head.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of total steps for learning rate warmup.')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Maximum norm for gradient clipping.')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Number of epochs without improvement before early stopping.')

    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("Batch size must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.num_folds < 2:
        raise ValueError("Number of folds must be at least 2")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("Dropout must be between 0 and 1")
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        raise ValueError("Warmup ratio must be between 0 and 1")

    return args


def print_config(config):
    """
    Print configuration in a formatted way.

    Args:
        config: Configuration namespace
    """
    print("\n" + "="*60)
    print("CONFIGURATION (Hate Speech Detection)")
    print("="*60)
    print("\nTraining Parameters:")
    print(f"  Batch Size: {config.batch}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Max Epochs: {config.epochs}")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")
    print("\nModel Parameters:")
    print(f"  Model: {config.model_path}")
    print(f"  Max Sequence Length: {config.max_length}")
    print(f"  Freeze Base: {config.freeze_base}")
    print(f"  Dropout: {config.dropout}")
    print("\nOptimizer Parameters:")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Warmup Ratio: {config.warmup_ratio}")
    print(f"  Gradient Clip Norm: {config.gradient_clip_norm}")
    print("\nExperiment Parameters:")
    print(f"  Author: {config.author_name}")
    print(f"  K-Folds: {config.num_folds}")
    print(f"  Stratification: {config.stratification_type}")
    print(f"  Random Seed: {config.seed}")
    print(f"  MLflow Experiment: {config.mlflow_experiment_name}")
    print("\nData Parameters:")
    print(f"  Dataset Path: {config.dataset_path}")
    print("="*60 + "\n")
