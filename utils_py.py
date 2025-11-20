"""
Utility functions for Servicing Intelligence NLP Pipeline
Includes logging setup, configuration loading, and helper functions
"""

import logging
import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'servicing_intelligence.log')
    
    # Create logger
    logger = logging.getLogger('servicing_intelligence')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def get_model_size(model_path: str) -> str:
    """
    Get size of saved model in MB
    
    Args:
        model_path: Path to model file
    
    Returns:
        Formatted size string
    """
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ['data', 'preprocessing', 'classifier', 'summarizer']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data paths
    if 'raw_path' not in config['data']:
        raise ValueError("Missing 'raw_path' in data configuration")
    
    # Validate splits sum to 1.0
    splits = [
        config['data'].get('train_split', 0),
        config['data'].get('validation_split', 0),
        config['data'].get('test_split', 0)
    ]
    if not 0.99 <= sum(splits) <= 1.01:
        raise ValueError(f"Data splits must sum to 1.0, got {sum(splits)}")
    
    return True


class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_name: str, value: float):
        """Add a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average of a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
    
    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        return {name: self.get_average(name) for name in self.metrics}


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to max length with ellipsis
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def print_section(title: str, width: int = 80):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        width: Total width of header
    """
    padding = (width - len(title) - 2) // 2
    print("\n" + "=" * width)
    print(" " * padding + title + " " * padding)
    print("=" * width + "\n")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Load config
    config = load_config()
    print(f"✓ Configuration loaded: {len(config)} sections")
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Logging setup successful")
    
    # Check device
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Validate config
    validate_config(config)
    logger.info("✓ Configuration validation passed")
    
    print("\n✓ All utility functions working correctly!")
