import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import mlflow
import os
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
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]  # Explore multiple thresholds
    metrics = {}
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Initialize variables to track the best threshold based on macro F1
    best_macro_f1 = -1
    best_threshold = None
    best_threshold_metrics = {}

    # Calculate macro F1 for each threshold
    for thresh in thresholds:
        y_pred_binary = (y_pred > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, zero_division=0)
        macro_f1 = (f1[0] + f1[1]) / 2 if len(f1) == 2 else f1[0]
        metrics[f'macro_f1_th_{thresh}'] = macro_f1

        # Track the best threshold based on macro F1
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

    # Update metrics with best threshold values and ROC-AUC
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
        best_fold_model = None
        best_fold_idx = -1
        best_overall_macro_f1 = 0

        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{config.num_folds}")
            print('='*60)

            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

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

                # Use macro F1 for early stopping and model saving
                if val_metrics['macro_f1'] > best_macro_f1:
                    best_macro_f1 = val_metrics['macro_f1']
                    best_metrics = val_metrics.copy()
                    best_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
                    best_epoch = epoch + 1
                    patience_counter = 0

                    if best_macro_f1 > best_overall_macro_f1:
                        best_overall_macro_f1 = best_macro_f1
                        best_fold_idx = fold
                        best_fold_model = model.state_dict()
                else:
                    patience_counter += 1

                print_epoch_metrics(epoch, config.epochs, fold, config.num_folds,
                                   train_metrics, val_metrics, best_macro_f1, best_epoch)

                # Log key metrics to MLflow
                mlflow.log_metrics({
                    f'fold_{fold+1}_epoch_{epoch+1}_train_loss': train_metrics['loss'],
                    f'fold_{fold+1}_epoch_{epoch+1}_train_accuracy': train_metrics['accuracy'],
                    f'fold_{fold+1}_epoch_{epoch+1}_train_f1': train_metrics['f1'],
                    f'fold_{fold+1}_epoch_{epoch+1}_train_f1_negative': train_metrics['f1_negative'],
                    f'fold_{fold+1}_epoch_{epoch+1}_train_macro_f1': train_metrics['macro_f1'],
                    f'fold_{fold+1}_epoch_{epoch+1}_train_roc_auc': train_metrics['roc_auc'],
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

            for metric_name, metric_value in best_metrics.items():
                if not metric_name.startswith('train_'):
                    mlflow.log_metric(f"fold_{fold+1}_best_{metric_name}", metric_value)

            print_fold_summary(fold, best_metrics, best_epoch)

        best_fold_metrics = fold_results[best_fold_idx]
        mlflow.log_metric('best_fold_index', best_fold_idx + 1)

        for metric_name in ['accuracy', 'f1', 'f1_negative', 'macro_f1', 'roc_auc']:
            best_value = max([fold_result[metric_name] for fold_result in fold_results])
            mlflow.log_metric(f'best_{metric_name}', best_value)

        best_loss = min([fold_result['loss'] for fold_result in fold_results])
        mlflow.log_metric('best_loss', best_loss)

        if best_fold_model is not None:
            final_model = TransformerBinaryClassifier(config.model_path, dropout=config.dropout)
            final_model.load_state_dict(best_fold_model)
            mlflow.pytorch.log_model(
                final_model,
                name="model",
                registered_model_name=f"bangla_hatespeech_model_fold{best_fold_idx+1}_macro_f1_{best_overall_macro_f1:.4f}"
            )

        print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics)
