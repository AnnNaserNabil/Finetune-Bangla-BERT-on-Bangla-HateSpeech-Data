# BanglaBERT Hate Speech Detection Fine-Tuning

![BanglaBERT Logo](https://img.shields.io/badge/Model-BanglaBERT-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Project Overview

This project provides a modular Python framework for fine-tuning the BanglaBERT model (from `sagorsarker/bangla-bert-base`) on a Bangla hate speech dataset for **binary classification**. The dataset consists of comments labeled with a single binary category: `HateSpeech` (1 for hate speech, 0 for non-hate speech), with a balanced distribution of 8,326 positive and 8,081 negative samples.

The framework uses PyTorch for training, Hugging Face Transformers for model handling, and MLflow for experiment tracking. It supports K-Fold cross-validation (default: 5 folds), class weighting to handle slight imbalance, early stopping, and optional freezing of base layers during fine-tuning. Key metrics include accuracy, precision, recall, F1-score, and ROC-AUC, with threshold exploration at 0.4, 0.5, and 0.6 to optimize the decision boundary.

Key features:
- **Modular Structure**: Separate files for configuration (`config.py`), data handling (`data.py`), model definition (`model.py`), training logic (`train.py`), utilities (`utils.py`), and entry point (`main.py`).
- **Experiment Tracking**: All runs are logged to MLflow, including parameters, per-fold metrics, per-epoch metrics, and threshold exploration results.
- **Customization**: Easily experiment with hyperparameters via command-line arguments.
- **Binary Classification**: Uses `BCEWithLogitsLoss` with a single output and sigmoid activation for binary predictions.

This setup is designed for reproducibility and collaboration, ideal for researchers or contributors experimenting with different hyperparameters on Bangla NLP tasks for hate speech detection.

## Usage

Run fine-tuning via `main.py` with command-line arguments. All experiments log to MLflow under `./mlruns`.

### Running in Google Colab (Recommended for Free GPU)
1. Open Colab: [colab.research.google.com](https://colab.research.google.com).
2. Enable GPU: Runtime > Change runtime type > T4 GPU.
3. Clone repo: `!git clone https://github.com/AnnNaserNabil/Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data`
4. `%cd Finetune-Bangla-BERT-on-Bangla-HateSpeech-Data`
5. Install dependencies:
   ```
   !pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow
   ```
6. Run command (replace with your values):
   ```
   !python main.py \
    --author_name "your_name" \
    --dataset_path "path/to/your/HateSpeech.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 15 \
    --model_path "sagorsarker/bangla-bert-base" \
    --stratification_type binary \
    --seed 42 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --gradient_clip_norm 1.0 \
    --early_stopping_patience 5 \
    --num_folds 5
   ```
   - Full arguments:
     - `--batch`: Batch size (e.g., 16, 32, 64).
     - `--lr`: Learning rate (e.g., 1e-5, 2e-5, 3e-5).
     - `--epochs`: Number of epochs (e.g., 10-20).
     - `--author_name`: Your name (tags the MLflow run).
     - `--dataset_path`: Path to CSV (required).
     - `--model_path`: Pre-trained model (default: `sagorsarker/bangla-bert-base`).
     - `--num_folds`: Folds for cross-validation (default: 5).
     - `--max_length`: Token max length (default: 128).
     - `--stratification_type`: Stratification type (default: `binary`).
     - `--seed`: Random seed (default: 42).
     - `--dropout`: Dropout rate for classification head (default: 0.1).
     - `--weight_decay`: Weight decay for AdamW (default: 0.01).
     - `--warmup_ratio`: Ratio of total steps for learning rate warmup (default: 0.1).
     - `--gradient_clip_norm`: Maximum norm for gradient clipping (default: 1.0).
     - `--early_stopping_patience`: Epochs without improvement before early stopping (default: 5).
     - `--freeze_base`: Freeze BERT base layers.
     - `--mlflow_experiment_name`: Experiment name (default: `Bangla-HateSpeech-Detection`).

7. After run: Zip and download MLflow logs:
   ```
   !zip -r mlruns_yourname_batch_32_lr_2e-5_epochs_15_dropout_0.1.zip ./mlruns
   ```
   - Download `{mlruns_yourname}_batch_32_lr_2e-5_epochs_15_dropout_0.1.zip` from Colab's files sidebar.

### Viewing Results Locally
1. Unzip `mlruns_yourname_batch_32_lr_2e-5_epochs_15_dropout_0.1.zip` to a local directory (e.g., `experiments/mlruns_yourname_batch_32_lr_2e-5_epochs_15_dropout_0.1`).
2. In VS Code or terminal: Navigate to the directory, activate your virtual environment, and run:
   ```
   mlflow ui
   ```
3. Open `http://localhost:5000` in your browser to view experiments, metrics (accuracy, precision, recall, F1, ROC-AUC), parameters, and saved models.

### Running Locally (No Colab)
Same as above, but use `python main.py ...` instead of `!python`.

## Collaboration Guide

To collaborate with minimal effort:
1. Fork the repo and clone to Colab or your local machine.
2. Run with your name and dataset:
   ```
   python main.py --batch 32 --lr 2e-5 --epochs 15 --author_name 'collaborator' --dataset_path 'your/path/HateSpeech.csv'
   ```
3. Zip/download `mlruns.zip` and view results locally as described above.
4. Add new experiments? Create a new configuration in your script or modify hyperparameters, commit, and submit a pull request (PR).
5. Issues/PRs: Welcome! Describe your changes (e.g., "Adjusted learning rate to 1e-5 for better F1-score").

## Notes
- **Dataset**: The dataset is balanced (8,326 positive, 8,081 negative samples). Class weights are computed to handle the slight imbalance (~0.97 for positive class).
- **Threshold Exploration**: The framework evaluates F1-scores at thresholds 0.4, 0.5, and 0.6 to optimize the decision boundary. Check MLflow logs to select the best threshold.
- **Preprocessing**: If your comments contain noise (e.g., URLs, emojis), add preprocessing in `data.py` (e.g., remove URLs with regex).
- **Hyperparameter Tuning**: Experiment with `--lr`, `--batch`, `--max_length`, or `--dropout` to improve performance. For example, try `--lr 1e-5` or `--max_length 256` for longer comments.
- **Model Choice**: If `sagorsarker/bangla-bert-base` underperforms, explore other Bangla-specific models on Hugging Face.

For questions, open an issue or contact the repository owner.

