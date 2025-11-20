"""
T5-based Ticket Summarization Model
Fine-tuned T5 for abstractive summarization of customer service tickets
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import Dict, List, Optional
from utils import load_config, setup_logging


class TicketSummarizer(nn.Module):
    """
    T5-based summarizer for ticket content
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize summarizer
        
        Args:
            model_name: Pretrained T5 model name
        """
        super(TicketSummarizer, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained T5
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.model.config
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs (for training)
        
        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate_summary(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        max_length: int = 128, min_length: int = 30,
                        num_beams: int = 4, length_penalty: float = 2.0) -> torch.Tensor:
        """
        Generate summary using beam search
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
        
        Returns:
            Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
        
        return outputs
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SummarizerTrainer:
    """
    Trainer for ticket summarizer
    """
    
    def __init__(self, model: TicketSummarizer, tokenizer: T5Tokenizer,
                 config: dict, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: TicketSummarizer model
            tokenizer: T5 tokenizer
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = setup_logging(config)
        
        summarizer_config = config['summarizer']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=summarizer_config.get('learning_rate', 3e-5),
            weight_decay=summarizer_config.get('weight_decay', 0.01)
        )
        
        self.scheduler = None
    
    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        from transformers import get_linear_schedule_with_warmup
        
        warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def evaluate(self, eval_loader) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            eval_loader: Evaluation data loader
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def generate_summaries(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Generate summaries for a list of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for generation
        
        Returns:
            List of generated summaries
        """
        self.model.eval()
        summaries = []
        
        summarizer_config = self.config['summarizer']
        max_length = summarizer_config.get('max_target_length', 128)
        min_length = summarizer_config.get('min_target_length', 30)
        num_beams = summarizer_config.get('beam_size', 4)
        length_penalty = summarizer_config.get('length_penalty', 2.0)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Prepend "summarize: " prefix for T5
            batch_texts = [f"summarize: {text}" for text in batch_texts]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=summarizer_config.get('max_source_length', 512),
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Generate
            outputs = self.model.generate_summary(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
            
            # Decode
            batch_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def save_model(self, save_dir: str):
        """
        Save model and tokenizer
        
        Args:
            save_dir: Directory to save model
        """
        self.model.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, load_dir: str):
        """
        Load model and tokenizer
        
        Args:
            load_dir: Directory to load model from
        """
        self.model.model = T5ForConditionalGeneration.from_pretrained(load_dir)
        self.model = self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(load_dir)
        self.logger.info(f"Model loaded from {load_dir}")


def create_summarizer(config: dict) -> tuple:
    """
    Create summarizer model and tokenizer from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (TicketSummarizer, T5Tokenizer)
    """
    summarizer_config = config['summarizer']
    model_name = summarizer_config['model_name']
    
    model = TicketSummarizer(model_name=model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    return model, tokenizer


if __name__ == "__main__":
    # Test summarizer
    config = load_config()
    
    print("Creating summarizer model...")
    model, tokenizer = create_summarizer(config)
    
    print(f"Model: {model.model_name}")
    print(f"Trainable parameters: {model.get_num_parameters():,}")
    
    # Test generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    test_text = "summarize: My internet connection has been dropping for the past three days."
    
    encoded = tokenizer(test_text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate_summary(input_ids, attention_mask, max_length=50)
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nTest generation:")
    print(f"Input: {test_text}")
    print(f"Output: {summary}")
    print(f"✓ Summarizer working correctly!")
