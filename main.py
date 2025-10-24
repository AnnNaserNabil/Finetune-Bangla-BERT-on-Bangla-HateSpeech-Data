"""
Main script for Bangla BERT Hate Speech Detection
Runs K-fold cross-validation training for binary classification
"""

from transformers import AutoTokenizer
import torch
from config import parse_arguments, print_config
from data import load_and_preprocess_data, prepare_kfold_splits
from train import run_kfold_training
from utils import set_seed


def main():
    # Parse arguments
    config = parse_arguments()
    print_config(config)
    config.freeze_base = False
    # Set random seed for reproducibility
    set_seed(config.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    # Load and preprocess data
    comments, labels = load_and_preprocess_data(config.dataset_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Run K-fold training
    run_kfold_training(config, comments, labels, tokenizer, device)


if __name__ == "__main__":
    main()
