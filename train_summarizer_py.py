"""
Train Ticket Summarizer
Fine-tune T5 model for abstractive summarization
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

from model_summarizer import create_summarizer, SummarizerTrainer
from utils import load_config, setup_logging, get_device, ensure_dir, format_time


class SummarizationDataset(Dataset):
    """Dataset for summarization task"""
    
    def __init__(self, texts: list, summaries: list, tokenizer, 
                 max_source_length: int = 512, max_target_length: int = 128):
        """
        Initialize dataset
        
        Args:
            texts: List of source texts
            summaries: List of target summaries
            tokenizer: T5 tokenizer
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        source_text = str(self.texts[idx])
        target_text = str(self.summaries[idx])
        
        # Add T5 prefix
        source_text = f"summarize: {source_text}"
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids']
        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }


def prepare_data(config: dict, logger):
    """
    Prepare summarization data
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Train, validation, test DataFrames
    """
    # Load data
    data_path = config['data']['processed_path']
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use cleaned_text (without lemmatization) for summarization
    # as T5 works better with natural text
    df = df[['cleaned_text', 'summary']].copy()
    df = df.dropna()
    
    logger.info(f"Loaded {len(df)} samples with summaries")
    
    # Split data
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['validation_split']
    test_ratio = config['data']['test_split']
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
    
    # Second split: train and val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42)
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def create_data_loaders(train_df, val_df, test_df, tokenizer, config):
    """
    Create PyTorch data loaders
    
    Args:
        train_df, val_df, test_df: DataFrames
        tokenizer: T5 tokenizer
        config: Configuration dictionary
    
    Returns:
        Train, validation, and test data loaders
    """
    max_source_length = config['summarizer']['max_source_length']
    max_target_length = config['summarizer']['max_target_length']
    batch_size = config['summarizer']['batch_size']
    
    # Create datasets
    train_dataset = SummarizationDataset(
        train_df['cleaned_text'].tolist(),
        train_df['summary'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
    )
    
    val_dataset = SummarizationDataset(
        val_df['cleaned_text'].tolist(),
        val_df['summary'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
    )
    
    test_dataset = SummarizationDataset(
        test_df['cleaned_text'].tolist(),
        test_df['summary'].tolist(),
        tokenizer,
        max_source_length,
        max_target_length
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
    logger.info("TICKET SUMMARIZER TRAINING")
    logger.info("=" * 80)
    
    # Prepare data
    train_df, val_df, test_df = prepare_data(config, logger)
    
    # Create model and tokenizer
    logger.info(f"Loading model: {config['summarizer']['model_name']}")
    model, tokenizer = create_summarizer(config)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, config
    )
    
    # Create trainer
    trainer = SummarizerTrainer(model, tokenizer, config, device)
    
    # Setup scheduler
    epochs = config['summarizer']['epochs']
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
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_dir = config['summarizer']['save_dir']
            trainer.save_model(save_dir)
            logger.info(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Generate sample summaries
    logger.info("\nGenerating sample summaries on test set...")
    sample_texts = test_df['cleaned_text'].head(3).tolist()
    sample_refs = test_df['summary'].head(3).tolist()
    
    generated_summaries = trainer.generate_summaries(sample_texts, batch_size=3)
    
    logger.info("\nSample Predictions:")
    for i, (text, ref, gen) in enumerate(zip(sample_texts, sample_refs, generated_summaries)):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"Reference: {ref}")
        logger.info(f"Generated: {gen}")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)


def main():
    """Main execution"""
    config = load_config()
    ensure_dir(config['summarizer']['save_dir'])
    train_model(config)
    print("\n✓ Summarizer training completed!")


if __name__ == "__main__":
    main()
