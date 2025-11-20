"""
Model Evaluation
Comprehensive evaluation with metrics and visualizations using matplotlib
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend

from model_classifier import create_classifier
from train_classifier import TicketDataset
from transformers import AutoTokenizer
from utils import load_config, setup_logging, get_device, ensure_dir


def load_trained_model(config: dict, device: torch.device):
    """
    Load trained classifier model
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Model, tokenizer, and label encoder
    """
    logger = setup_logging(config)
    
    # Load model
    model = create_classifier(config)
    model_path = f"{config['classifier']['save_dir']}/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['classifier']['save_dir'])
    
    # Load label encoder
    with open(f"{config['classifier']['save_dir']}/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder


def get_predictions(model, data_loader, device):
    """
    Get predictions and true labels
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
    
    Returns:
        Predictions, probabilities, and true labels
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Get predictions
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def calculate_metrics(y_true, y_pred, y_probs, label_encoder):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        label_encoder: Label encoder
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Per-class metrics
    class_report = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    return metrics, class_report


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Label encoder
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_roc_curves(y_true, y_probs, label_encoder, save_path):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_probs: Prediction probabilities
        label_encoder: Label encoder
        save_path: Path to save plot
    """
    n_classes = len(label_encoder.classes_)
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curves saved to {save_path}")


def plot_metrics_comparison(metrics, save_path):
    """
    Plot comparison of different metrics
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save plot
    """
    metric_names = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_values = [metrics[name] for name in metric_names]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_labels, metric_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics comparison saved to {save_path}")


def evaluate_model(config: dict):
    """
    Main evaluation function
    
    Args:
        config: Configuration dictionary
    """
    logger = setup_logging(config)
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Load test data
    data_path = config['data']['processed_path']
    df = pd.read_csv(data_path)
    
    # Load label encoder
    with open(f"{config['classifier']['save_dir']}/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    
    df['label_encoded'] = label_encoder.transform(df['category'])
    
    # Use last 10% as test set (simple approach)
    test_size = int(len(df) * config['data']['test_split'])
    test_df = df.tail(test_size)
    
    logger.info(f"Test set size: {len(test_df)}")
    
    # Load model and tokenizer
    model, tokenizer, label_encoder = load_trained_model(config, device)
    
    # Create test dataset and loader
    test_dataset = TicketDataset(
        test_df['cleaned_text'].tolist(),
        test_df['label_encoded'].tolist(),
        tokenizer,
        config['classifier']['max_length']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False
    )
    
    # Get predictions
    logger.info("\nGenerating predictions...")
    y_pred, y_probs, y_true = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    metrics, class_report = calculate_metrics(y_true, y_pred, y_probs, label_encoder)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL METRICS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    logger.info(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    logger.info(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    logger.info(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PER-CLASS METRICS")
    logger.info("=" * 80)
    for class_name in label_encoder.classes_:
        if class_name in class_report:
            metrics_class = class_report[class_name]
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"  Precision: {metrics_class['precision']:.4f}")
            logger.info(f"  Recall:    {metrics_class['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics_class['f1-score']:.4f}")
            logger.info(f"  Support:   {metrics_class['support']}")
    
    # Create visualizations
    output_dir = config['evaluation']['output_dir']
    ensure_dir(output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    if config['evaluation'].get('save_confusion_matrix', True):
        plot_confusion_matrix(
            y_true, y_pred, label_encoder,
            f"{output_dir}/confusion_matrix.png"
        )
    
    if config['evaluation'].get('save_roc_curve', True):
        plot_roc_curves(
            y_true, y_probs, label_encoder,
            f"{output_dir}/roc_curves.png"
        )
    
    plot_metrics_comparison(metrics, f"{output_dir}/metrics_comparison.png")
    
    # Sample predictions
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 80)
    
    sample_indices = np.random.choice(len(test_df), min(5, len(test_df)), replace=False)
    
    for idx in sample_indices:
        text = test_df.iloc[idx]['text'][:100]
        true_label = label_encoder.classes_[y_true[idx]]
        pred_label = label_encoder.classes_[y_pred[idx]]
        confidence = y_probs[idx][y_pred[idx]]
        
        logger.info(f"\nText: {text}...")
        logger.info(f"True: {true_label}")
        logger.info(f"Pred: {pred_label} (confidence: {confidence:.3f})")
        logger.info(f"Correct: {'✓' if true_label == pred_label else '✗'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80)


def main():
    """Main execution"""
    config = load_config()
    ensure_dir(config['evaluation']['output_dir'])
    evaluate_model(config)
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()
