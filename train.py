import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import mlflow
import pandas as pd
import time
from data import HateSpeechDataset, calculate_class_weights, prepare_kfold_splits
from model import TransformerBinaryClassifier
from utils import get_model_metrics, print_fold_summary, print_experiment_summary

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics for binary classification with threshold exploration, optimizing for macro F1.
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
    Returns:
        dict: Dictionary containing accuracy, F1 (positive and negative), macro F1, ROC-AUC, and best threshold
    """
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    metrics = {}
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    best_macro_f1 = -1
    best_threshold = None
    best_threshold_metrics = {}

    for thresh in thresholds:
        y_pred_binary = (y_pred > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, zero_division=0)
        macro_f1 = (f1[0] + f1[1]) / 2 if len(f1) == 2 else f1[0]
        metrics[f'macro_f1_th_{thresh}'] = macro_f1

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = thresh
            best_threshold_metrics = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision[1],  # Positive class (hate speech)
                'recall': recall[1],
                'f1': f1[1],
                'macro_f1': macro_f1,
                'precision_negative': precision[0],  # Negative class (non-hate speech)
                'recall_negative': recall[0],
                'f1_negative': f1[0]
            }

    metrics.update({
        'accuracy': best_threshold_metrics['accuracy'],
        'precision': best_threshold_metrics['precision'],
        'recall': best_threshold_metrics['recall'],
        'f1': best_threshold_metrics['f1'],
        'macro_f1': best_threshold_metrics['macro_f1'],
        'precision_negative': best_threshold_metrics['precision_negative'],
        'recall_negative': best_threshold_metrics['recall_negative'],
        'f1_negative': best_threshold_metrics['f1_negative'],
        'roc_auc': roc_auc_score(y_true, y_pred) if y_pred is not None else 0.0,
        'best_threshold': best_threshold
    })

    return metrics

def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None, max_norm=1.0):
    """
    Train the model for one epoch.
    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run training on
        class_weights: Optional class weight for loss
        max_norm: Maximum norm for gradient clipping
    Returns:
        dict: Training metrics including loss and performance metrics
    """
    model.train()
    total_loss = 0
    all_train_predictions = []
    all_train_labels = []

    if class_weights is not None:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        loss_fct = nn.BCEWithLogitsLoss()

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=None)
        loss = loss_fct(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = torch.sigmoid(outputs['logits'])
        all_train_predictions.extend(predictions.detach().cpu().numpy())
        all_train_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    train_metrics = calculate_metrics(np.array(all_train_labels), np.array(all_train_predictions))
    train_metrics['loss'] = avg_loss

    return train_metrics

def evaluate_model(model, dataloader, device, class_weights=None):
    """
    Evaluate the model on validation data.
    Args:
        model: The transformer model
        dataloader: Validation data loader
        device: Device to run evaluation on
        class_weights: Optional class weight for loss calculation
    Returns:
        dict: Validation metrics including loss and performance metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    if class_weights is not None:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        loss_fct = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).view(-1, 1)

            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            loss = loss_fct(outputs['logits'], labels)
            total_loss += loss.item()

            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss

    return metrics

def print_epoch_metrics(epoch, num_epochs, fold, num_folds, train_metrics, val_metrics, best_macro_f1, best_epoch):
    """
    Print epoch metrics, focusing on key metrics for a nearly balanced dataset.
    Args:
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        fold: Current fold (0-indexed)
        num_folds: Total number of folds
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        best_macro_f1: Best validation macro F1 score so far
        best_epoch: Epoch with best macro F1 score (0-indexed)
    """
    print("\n" + "="*60)
    print(f"Epoch {epoch+1}/{num_epochs} | Fold {fold+1}/{num_folds}")
    print("="*60)
    print("TRAINING:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {train_metrics['macro_f1']:.4f} (Threshold: {train_metrics['best_threshold']})")
    print(f"  F1 (Hate): {train_metrics['f1']:.4f}")
    print(f"  F1 (Non-Hate): {train_metrics['f1_negative']:.4f}")
    print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}")
    print("\nVALIDATION:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f} (Threshold: {val_metrics['best_threshold']})")
    print(f"  F1 (Hate): {val_metrics['f1']:.4f}")
    print(f"  F1 (Non-Hate): {val_metrics['f1_negative']:.4f}")
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"\nBest Macro F1 so far: {best_macro_f1:.4f} (Epoch {best_epoch})")
    print("="*60)

def run_kfold_training(config, comments, labels, tokenizer, device):
    """
    Run K-fold cross-validation training for hate speech detection, optimized for a nearly balanced dataset.
    Args:
        config: Configuration object with hyperparameters
        comments: Array of text comments
        labels: Array of binary labels
        tokenizer: Tokenizer for text encoding
        device: Device to run training on
    """
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{config.author_name}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}"):
        mlflow.log_params({
            'batch_size': config.batch,
            'learning_rate': config.lr,
            'num_epochs': config.epochs,
            'num_folds': config.num_folds,
            'max_length': config.max_length,
            'freeze_base': config.freeze_base,
            'dropout': config.dropout,
            'weight_decay': config.weight_decay,
            'warmup_ratio': config.warmup_ratio,
            'gradient_clip_norm': config.gradient_clip_norm,
            'early_stopping_patience': config.early_stopping_patience,
            'author_name': config.author_name,
            'model_path': config.model_path,
            'seed': config.seed,
            'stratification_type': config.stratification_type
        })

        kfold_splits = prepare_kfold_splits(
            comments, labels,
            num_folds=config.num_folds,
            stratification_type=config.stratification_type,
            seed=config.seed
        )

        fold_results = []
        best_fold_model92 = None
        best_fold_idx = -1
        best_overall_macro_f1 = 0
        best_overall_metrics = {}
        best_overall_epoch = 0

        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{config.num_folds}")
            print('='*60)

            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Log class distribution
            train_hate_count = np.sum(train_labels)
            train_non_hate_count = len(train_labels) - train_hate_count
            val_hate_count = np.sum(val_labels)
            val_non_hate_count = len(val_labels) - val_hate_count
            mlflow.log_metrics({
                f'fold_{fold+1}_train_hate_samples': train_hate_count,
                f'fold_{fold+1}_train_non_hate_samples': train_non_hate_count,
                f'fold_{fold+1}_val_hate_samples': val_hate_count,
                f'fold_{fold+1}_val_non_hate_samples': val_non_hate_count
            })

            # No class weights needed for nearly balanced dataset
            class_weights = None

            train_dataset = HateSpeechDataset(train_comments, train_labels, tokenizer, config.max_length)
            val_dataset = HateSpeechDataset(val_comments, val_labels, tokenizer, config.max_length)

            train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False, num_workers=2, pin_memory=True)

            model = TransformerBinaryClassifier(config.model_path, dropout=config.dropout)
            if config.freeze_base:
                model.freeze_base_layers()
            model.to(device)

            if fold == 0:
                model_metrics = get_model_metrics(model)
                mlflow.log_metrics({
                    'total_parameters': model_metrics['total_parameters'],
                    'trainable_parameters': model_metrics['trainable_parameters'],
                    'model_size_mb': model_metrics['model_size_mb']
                })

            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
            total_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(config.warmup_ratio * total_steps),
                num_training_steps=total_steps
            )

            best_macro_f1 = 0
            best_metrics = {}
            best_epoch = 0
            patience = config.early_stopping_patience
            patience_counter = 0

            for epoch in range(config.epochs):
                train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights, max_norm=config.gradient_clip_norm)
                val_metrics = evaluate_model(model, val_loader, device, class_weights)

                if val_metrics['macro_f1'] > best_macro_f1:
                    best_macro_f1 = val_metrics['macro_f1']
                    best_metrics = val_metrics.copy()
                    best_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Log threshold exploration for best epoch
                    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                        mlflow.log_metric(f'fold_{fold+1}_best_epoch_val_macro_f1_th_{thresh}', val_metrics[f'macro_f1_th_{thresh}'])

                    if best_macro_f1 > best_overall_macro_f1:
                        best_overall_macro_f1 = best_macro_f1
                        best_fold_idx = fold
                        best_fold_model = model.state_dict()
                        best_overall_metrics = best_metrics.copy()
                        best_overall_epoch = best_epoch  # Track the best epoch
                else:
                    patience_counter += 1

                print_epoch_metrics(epoch, config.epochs, fold, config.num_folds,
                                   train_metrics, val_metrics, best_macro_f1, best_epoch)

                # Log only validation metrics per epoch
                mlflow.log_metrics({
                    f'fold_{fold+1}_epoch_{epoch+1}_val_loss': val_metrics['loss'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_accuracy': val_metrics['accuracy'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_f1': val_metrics['f1'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_f1_negative': val_metrics['f1_negative'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_macro_f1': val_metrics['macro_f1'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_roc_auc': val_metrics['roc_auc'],
                    f'fold_{fold+1}_epoch_{epoch+1}_val_best_threshold': val_metrics['best_threshold']
                })

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

            best_metrics['best_epoch'] = best_epoch
            fold_results.append(best_metrics)

            # Log all best metrics for the fold, including precision and recall for both classes
            for metric_name, metric_value in best_metrics.items():
                if not metric_name.startswith('train_'):
                    mlflow.log_metric(f"fold_{fold+1}_best_{metric_name}", metric_value)

            print_fold_summary(fold, best_metrics, best_epoch)

        # Aggregate metrics across folds
        aggregate_metrics = {
            'mean_val_accuracy': np.mean([fr['accuracy'] for fr in fold_results]),
            'std_val_accuracy': np.std([fr['accuracy'] for fr in fold_results]),
            'mean_val_precision': np.mean([fr['precision'] for fr in fold_results]),
            'std_val_precision': np.std([fr['precision'] for fr in fold_results]),
            'mean_val_recall': np.mean([fr['recall'] for fr in fold_results]),
            'std_val_recall': np.std([fr['recall'] for fr in fold_results]),
            'mean_val_f1': np.mean([fr['f1'] for fr in fold_results]),
            'std_val_f1': np.std([fr['f1'] for fr in fold_results]),
            'mean_val_precision_negative': np.mean([fr['precision_negative'] for fr in fold_results]),
            'std_val_precision_negative': np.std([fr['precision_negative'] for fr in fold_results]),
            'mean_val_recall_negative': np.mean([fr['recall_negative'] for fr in fold_results]),
            'std_val_recall_negative': np.std([fr['recall_negative'] for fr in fold_results]),
            'mean_val_f1_negative': np.mean([fr['f1_negative'] for fr in fold_results]),
            'std_val_f1_negative': np.std([fr['f1_negative'] for fr in fold_results]),
            'mean_val_macro_f1': np.mean([fr['macro_f1'] for fr in fold_results]),
            'std_val_macro_f1': np.std([fr['macro_f1'] for fr in fold_results]),
            'mean_val_roc_auc': np.mean([fr['roc_auc'] for fr in fold_results]),
            'std_val_roc_auc': np.std([fr['roc_auc'] for fr in fold_results]),
            'mean_val_loss': np.mean([fr['loss'] for fr in fold_results]),
            'std_val_loss': np.std([fr['loss'] for fr in fold_results])
        }
        mlflow.log_metrics(aggregate_metrics)

        # Create and log fold summary table (including training metrics and best epoch)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Create unique filename for fold_summary.csv
        model_name_safe = config.model_path.replace('/', '_')  # Replace slashes for safe filename
        fold_summary_filename = f'fold_summary_{model_name_safe}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv'
        summary_data = {
            'Fold': [f'Fold {i+1}' for i in range(config.num_folds)],
            'Best Epoch': [fr['best_epoch'] for fr in fold_results],
            'Val Accuracy': [fr['accuracy'] for fr in fold_results],
            'Val Precision (Hate)': [fr['precision'] for fr in fold_results],
            'Val Recall (Hate)': [fr['recall'] for fr in fold_results],
            'Val F1 (Hate)': [fr['f1'] for fr in fold_results],
            'Val Precision (Non-Hate)': [fr['precision_negative'] for fr in fold_results],
            'Val Recall (Non-Hate)': [fr['recall_negative'] for fr in fold_results],
            'Val F1 (Non-Hate)': [fr['f1_negative'] for fr in fold_results],
            'Val Macro F1': [fr['macro_f1'] for fr in fold_results],
            'Val ROC-AUC': [fr['roc_auc'] for fr in fold_results],
            'Val Loss': [fr['loss'] for fr in fold_results],
            'Best Threshold': [fr['best_threshold'] for fr in fold_results],
            'Train Accuracy': [fr['train_accuracy'] for fr in fold_results],
            'Train Precision (Hate)': [fr['train_precision'] for fr in fold_results],
            'Train Recall (Hate)': [fr['train_recall'] for fr in fold_results],
            'Train F1 (Hate)': [fr['train_f1'] for fr in fold_results],
            'Train Precision (Non-Hate)': [fr['train_precision_negative'] for fr in fold_results],
            'Train Recall (Non-Hate)': [fr['train_recall_negative'] for fr in fold_results],
            'Train F1 (Non-Hate)': [fr['train_f1_negative'] for fr in fold_results],
            'Train Macro F1': [fr['train_macro_f1'] for fr in fold_results],
            'Train ROC-AUC': [fr['train_roc_auc'] for fr in fold_results],
            'Train Loss': [fr['train_loss'] for fr in fold_results]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.loc['Mean'] = summary_df.select_dtypes(include=[np.number]).mean()
        summary_df.loc['Std'] = summary_df.select_dtypes(include=[np.number]).std()
        try:
            summary_df.to_csv(fold_summary_filename)
            mlflow.log_artifact(fold_summary_filename)
        except Exception as e:
            print(f"Error saving or logging {fold_summary_filename}: {e}")

        # Validate required keys in best_overall_metrics
        required_keys = [
            'accuracy', 'precision', 'recall', 'f1', 'precision_negative', 'recall_negative',
            'f1_negative', 'macro_f1', 'roc_auc', 'loss', 'best_threshold',
            'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
            'train_precision_negative', 'train_recall_negative', 'train_f1_negative',
            'train_macro_f1', 'train_roc_auc', 'train_loss'
        ]
        for key in required_keys:
            if key not in best_overall_metrics:
                raise KeyError(f"Missing key '{key}' in best_overall_metrics")

        # Create and log best metrics CSV (global best based on macro F1)
        best_metrics_filename = f'best_metrics_{model_name_safe}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}_{timestamp}.csv'
        best_metrics_data = {
            'Best Fold': [f'Fold {best_fold_idx+1}'],
            'Best Epoch': [best_overall_epoch],
            'Val Accuracy': [best_overall_metrics['accuracy']],
            'Val Precision (Hate)': [best_overall_metrics['precision']],
            'Val Recall (Hate)': [best_overall_metrics['recall']],
            'Val F1 (Hate)': [best_overall_metrics['f1']],
            'Val Precision (Non-Hate)': [best_overall_metrics['precision_negative']],
            'Val Recall (Non-Hate)': [best_overall_metrics['recall_negative']],
            'Val F1 (Non-Hate)': [best_overall_metrics['f1_negative']],
            'Val Macro F1': [best_overall_metrics['macro_f1']],
            'Val ROC-AUC': [best_overall_metrics['roc_auc']],
            'Val Loss': [best_overall_metrics['loss']],
            'Best Threshold': [best_overall_metrics['best_threshold']],
            'Train Accuracy': [best_overall_metrics['train_accuracy']],
            'Train Precision (Hate)': [best_overall_metrics['train_precision']],
            'Train Recall (Hate)': [best_overall_metrics['train_recall']],
            'Train F1 (Hate)': [best_overall_metrics['train_f1']],
            'Train Precision (Non-Hate)': [best_overall_metrics['train_precision_negative']],
            'Train Recall (Non-Hate)': [best_overall_metrics['train_recall_negative']],
            'Train F1 (Non-Hate)': [best_overall_metrics['train_f1_negative']],
            'Train Macro F1': [best_overall_metrics['train_macro_f1']],
            'Train ROC-AUC': [best_overall_metrics['train_roc_auc']],
            'Train Loss': [best_overall_metrics['train_loss']]
        }
        best_metrics_df = pd.DataFrame(best_metrics_data)
        try:
            best_metrics_df.to_csv(best_metrics_filename, index=False)
            mlflow.log_artifact(best_metrics_filename)
        except Exception as e:
            print(f"Error saving or logging {best_metrics_filename}: {e}")

        # Log global best metrics to MLflow
        mlflow.log_metric('best_fold_index', best_fold_idx + 1)
        mlflow.log_metric('best_epoch', best_overall_epoch)
        for metric_name, metric_value in best_overall_metrics.items():
            mlflow.log_metric(f"best_{metric_name}", metric_value)

        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'f1_negative', 'precision_negative', 'recall_negative', 'macro_f1', 'roc_auc']:
            best_value = max([fold_result[metric_name] for fold_result in fold_results])
            mlflow.log_metric(f'best_{metric_name}', best_value)

        best_loss = min([fold_result['loss'] for fold_result in fold_results])
        mlflow.log_metric('best_loss', best_loss)

        print_experiment_summary(best_fold_idx, best_overall_metrics, model_metrics)
