"""
Train Ticket Classifier
Handles training pipeline with optional PySpark support for distributed processing
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm

from model_classifier import create_classifier, ClassifierTrainer
from utils import load_config, setup_logging, get_device, ensure_dir, format_time

# Optional PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import StringType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


class TicketDataset(Dataset):
    """Dataset for ticket classification"""
    
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 128):
        """
        Initialize dataset
        
        Args:
            texts: List of text strings
            labels: List of labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def preprocess_with_pyspark(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocess data using PySpark for distributed processing
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
    
    Returns:
        Processed DataFrame
    """
    if not PYSPARK_AVAILABLE:
        print("PySpark not available, skipping distributed processing")
        return df
    
    logger = setup_logging(config)
    logger.info("Initializing PySpark session...")
    
    pyspark_config = config.get('pyspark', {})
    
    spark = SparkSession.builder \
        .appName(pyspark_config.get('app_name', 'ServicingIntelligence')) \
        .master(pyspark_config.get('master', 'local[*]')) \
        .config("spark.executor.memory", pyspark_config.get('executor_memory', '4g')) \
        .config("spark.driver.memory", pyspark_config.get('driver_memory', '2g')) \
        .getOrCreate()
    
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Define UDF for text length
    def text_length(text):
        return len(text) if text else 0
    
    length_udf = udf(text_length, StringType())
    
    # Apply transformations
    spark_df = spark_df.withColumn('text_length', length_udf(col('processed_text')))
    
    # Filter by length
    min_length = 10
    spark_df = spark_df.filter(col('text_length') > min_length)
    
    # Convert back to Pandas
    result_df = spark_df.toPandas()
    
    spark.stop()
    logger.info(f"PySpark processing complete: {len(result_df)} records")
    
    return result_df


def prepare_data(config: dict, logger):
    """
    Prepare data for training
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Train, validation, and test datasets
    """
    # Load processed data
    data_path = config['data']['processed_path']
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Optional PySpark preprocessing
    if config['classifier'].get('use_pyspark', False):
        df = preprocess_with_pyspark(df, config)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Save label encoder
    import pickle
    ensure_dir(config['classifier']['save_dir'])
    with open(f"{config['classifier']['save_dir']}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved with {len(label_encoder.classes_)} classes")
    
    # Split data
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['validation_split']
    test_ratio = config['data']['test_split']
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=42, stratify=df['label_encoded']
    )
    
    # Second split: train and val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=42, stratify=train_val_df['label_encoded']
    )
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df, label_encoder


def create_data_loaders(train_df, val_df, test_df, tokenizer, config):
    """
    Create PyTorch data loaders
    
    Args:
        train_df, val_df, test_df: DataFrames
        tokenizer: Tokenizer instance
        config: Configuration dictionary
    
    Returns:
        Train, validation, and test data loaders
    """
    max_length = config['classifier']['max_length']
    batch_size = config['classifier']['batch_size']
    
    # Create datasets
    train_dataset = TicketDataset(
        train_df['cleaned_text'].tolist(),
        train_df['label_encoded'].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = TicketDataset(
        val_df['cleaned_text'].tolist(),
        val_df['label_encoded'].tolist(),
        tokenizer,
        max_length
    )
    
    test_dataset = TicketDataset(
        test_df['cleaned_text'].tolist(),
        test_df['label_encoded'].tolist(),
        tokenizer,
        max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(config: dict):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    logger = setup_logging(config)
    device = get_device()
    
    logger.info("=" * 80)
    logger.info("TICKET CLASSIFIER TRAINING")
    logger.info("=" * 80)
    
    # Prepare data
    train_df, val_df, test_df, label_encoder = prepare_data(config, logger)
    
    # Load tokenizer
    model_name = config['classifier']['model_name']
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save tokenizer
    tokenizer.save_pretrained(config['classifier']['save_dir'])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, config
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_classifier(config)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    trainer = ClassifierTrainer(model, config, device)
    
    # Setup scheduler
    epochs = config['classifier']['epochs']
    num_training_steps = len(train_loader) * epochs
    trainer.setup_scheduler(num_training_steps)
    
    # Training loop
    logger.info("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = f"{config['classifier']['save_dir']}/best_model.pt"
            trainer.save_model(save_path)
            logger.info(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']:.4f}"
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)


def main():
    """Main execution"""
    config = load_config()
    ensure_dir(config['classifier']['save_dir'])
    train_model(config)
    print("\n✓ Classifier training completed!")


if __name__ == "__main__":
    main()
