"""
Data loading and preprocessing module for binary hate speech detection
Supports stratified K-fold for binary classification
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold


# Label column for the hate speech detection task
LABEL_COLUMN = 'HateSpeech'


class HateSpeechDataset(Dataset):
    """
    PyTorch Dataset for hate speech detection.
    Handles tokenization and label conversion for binary classification.
    """

    def __init__(self, comments, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.

        Args:
            comments: Array of text comments
            labels: Array of binary labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.comments = comments
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]  # Single float (0 or 1)

        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)  # Single float
        }


def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the dataset from CSV.

    Args:
        dataset_path (str): Path to the CSV file

    Returns:
        tuple: (comments array, labels array)
    """
    df = pd.read_csv(dataset_path)

    # Ensure required columns exist
    if 'Comments' not in df.columns or 'HateSpeech' not in df.columns:
        raise ValueError("Dataset must contain 'Comments' and 'HateSpeech' columns")

    comments = df['Comments'].values
    labels = df['HateSpeech'].values  # Shape: (n_samples,)

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(comments)}")
    print(f"Label distribution:")
    positive_count = np.sum(labels)
    percentage = (positive_count / len(labels)) * 100
    print(f"  HateSpeech: {positive_count}/{len(labels)} ({percentage:.2f}% positive)")

    return comments, labels


def prepare_kfold_splits(comments, labels, num_folds=5, stratification_type='binary', seed=42):
    """
    Prepare K-fold cross-validation splits with optional stratification.

    Args:
        comments: Array of text comments
        labels: Array of binary labels
        num_folds (int): Number of folds for cross-validation
        stratification_type (str): 'binary' or 'none'
        seed (int): Random seed for reproducibility

    Returns:
        generator: K-fold split indices
    """
    if stratification_type == 'binary':
        print(f"Using StratifiedKFold with {num_folds} folds for binary hate speech classification")
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments, labels)
    else:
        print(f"Using regular KFold with {num_folds} folds (no stratification)")
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments)


def calculate_class_weights(labels):
    """
    Calculate class weights for handling slight imbalance in data.

    Args:
        labels: Array of binary labels

    Returns:
        torch.FloatTensor: Class weight for the positive class
    """
    pos_count = np.sum(labels)
    neg_count = len(labels) - pos_count

    # Avoid division by zero
    weight = neg_count / pos_count if pos_count > 0 else 1.0

    print("\nClass weights for handling slight imbalance:")
    print(f"  HateSpeech: {weight:.3f} (based on {pos_count} positive, {neg_count} negative samples)")

    return torch.FloatTensor([weight])
