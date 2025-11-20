"""
Transformer-based Ticket Classifier Model
Uses BERT/DistilBERT for multi-class ticket classification
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Tuple, Optional
from utils import load_config, setup_logging


class TicketClassifier(nn.Module):
    """
    Transformer-based classifier for ticket categorization
    """
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        """
        Initialize classifier
        
        Args:
            model_name: Pretrained model name (e.g., 'distilbert-base-uncased')
            num_labels: Number of output categories
            dropout: Dropout probability
        """
        super(TicketClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            Logits [batch_size, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        return logits
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Predicted labels and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        return predictions, probs
    
    def get_num_parameters(self) -> int:
        """
        Count trainable parameters
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_transformer(self):
        """Freeze transformer weights for transfer learning"""
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_transformer(self):
        """Unfreeze transformer weights"""
        for param in self.transformer.parameters():
            param.requires_grad = True


class ClassifierTrainer:
    """
    Trainer for ticket classifier
    """
    
    def __init__(self, model: TicketClassifier, config: dict, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: TicketClassifier model
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = setup_logging(config)
        
        classifier_config = config['classifier']
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=classifier_config.get('learning_rate', 2e-5),
            weight_decay=classifier_config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = None
    
    def setup_scheduler(self, num_training_steps: int):
        """
        Setup learning rate scheduler
        
        Args:
            num_training_steps: Total number of training steps
        """
        from transformers import get_linear_schedule_with_warmup
        
        warmup_steps = self.config['classifier'].get('warmup_steps', 500)
        
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
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, eval_loader) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            eval_loader: Evaluation data loader
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, save_path: str):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load model checkpoint
        
        Args:
            load_path: Path to load model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {load_path}")


def create_classifier(config: dict) -> TicketClassifier:
    """
    Create classifier model from config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TicketClassifier instance
    """
    classifier_config = config['classifier']
    
    model = TicketClassifier(
        model_name=classifier_config['model_name'],
        num_labels=classifier_config['num_labels'],
        dropout=0.1
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = load_config()
    
    print("Creating classifier model...")
    model = create_classifier(config)
    
    print(f"Model: {model.model_name}")
    print(f"Number of labels: {model.num_labels}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Trainable parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 4
    seq_length = 32
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✓ Model working correctly!")
