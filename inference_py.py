"""
Inference Pipeline
Provides functions for ticket classification and summarization
"""

import torch
import pickle
import numpy as np
from typing import Dict, Tuple
from transformers import AutoTokenizer, T5Tokenizer

from model_classifier import create_classifier
from model_summarizer import create_summarizer
from preprocess import TextPreprocessor
from utils import load_config, setup_logging, get_device


class InferencePipeline:
    """Complete inference pipeline for classification and summarization"""
    
    def __init__(self, config: dict):
        """
        Initialize inference pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logging(config)
        self.device = get_device()
        
        # Load preprocessor
        self.preprocessor = TextPreprocessor(config)
        
        # Load classifier components
        self._load_classifier()
        
        # Load summarizer components
        self._load_summarizer()
        
        self.logger.info("✓ Inference pipeline initialized")
    
    def _load_classifier(self):
        """Load classifier model and components"""
        try:
            # Load model
            self.classifier_model = create_classifier(self.config)
            model_path = f"{self.config['classifier']['save_dir']}/best_model.pt"
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier_model = self.classifier_model.to(self.device)
            self.classifier_model.eval()
            
            # Load tokenizer
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                self.config['classifier']['save_dir']
            )
            
            # Load label encoder
            with open(f"{self.config['classifier']['save_dir']}/label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.logger.info("✓ Classifier loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load classifier: {e}")
            self.classifier_model = None
    
    def _load_summarizer(self):
        """Load summarizer model and components"""
        try:
            # Load model and tokenizer
            self.summarizer_model, self.summarizer_tokenizer = create_summarizer(self.config)
            
            # Load fine-tuned model
            save_dir = self.config['summarizer']['save_dir']
            self.summarizer_model.model = torch.nn.Module()  # Placeholder
            from transformers import T5ForConditionalGeneration
            self.summarizer_model.model = T5ForConditionalGeneration.from_pretrained(save_dir)
            self.summarizer_model = self.summarizer_model.to(self.device)
            self.summarizer_model.eval()
            
            self.summarizer_tokenizer = T5Tokenizer.from_pretrained(save_dir)
            
            self.logger.info("✓ Summarizer loaded successfully")
        except Exception as e:
            self.logger.warning(f"Summarizer not available: {e}")
            self.summarizer_model = None
    
    def classify_text(self, text: str) -> Tuple[str, float]:
        """
        Classify a single text
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (predicted_category, confidence)
        """
        if self.classifier_model is None:
            raise ValueError("Classifier model not loaded")
        
        # Preprocess text
        processed_text = self.preprocessor.clean_text(text)
        
        # Tokenize
        encoding = self.classifier_tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.config['classifier']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.classifier_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
        
        # Decode label
        category = self.label_encoder.classes_[pred_idx]
        
        return category, confidence
    
    def summarize_text(self, text: str) -> str:
        """
        Generate summary for a text
        
        Args:
            text: Input text
        
        Returns:
            Generated summary
        """
        if self.summarizer_model is None:
            raise ValueError("Summarizer model not loaded")
        
        # Preprocess text (light preprocessing for summarization)
        processed_text = self.preprocessor.clean_text(text)
        
        # Add T5 prefix
        input_text = f"summarize: {processed_text}"
        
        # Tokenize
        encoding = self.summarizer_tokenizer(
            input_text,
            max_length=self.config['summarizer']['max_source_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.summarizer_model.generate_summary(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config['summarizer']['max_target_length'],
                min_length=self.config['summarizer']['min_target_length'],
                num_beams=self.config['summarizer']['beam_size'],
                length_penalty=self.config['summarizer']['length_penalty']
            )
        
        # Decode
        summary = self.summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def pipeline(self, text: str) -> Dict[str, any]:
        """
        Complete pipeline: classify and summarize
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with classification and summary results
        """
        result = {}
        
        # Classification
        try:
            category, confidence = self.classify_text(text)
            result['category'] = category
            result['confidence'] = float(confidence)
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            result['category'] = None
            result['confidence'] = None
        
        # Summarization
        try:
            summary = self.summarize_text(text)
            result['summary'] = summary
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            result['summary'] = None
        
        result['original_text'] = text
        
        return result


# Global pipeline instance
_pipeline_instance = None


def get_pipeline(config: dict = None) -> InferencePipeline:
    """
    Get or create pipeline instance (singleton pattern)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        InferencePipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        if config is None:
            config = load_config()
        _pipeline_instance = InferencePipeline(config)
    
    return _pipeline_instance


def classify_text(text: str) -> Tuple[str, float]:
    """
    Classify text using global pipeline
    
    Args:
        text: Input text
    
    Returns:
        Tuple of (category, confidence)
    """
    pipeline = get_pipeline()
    return pipeline.classify_text(text)


def summarize_text(text: str) -> str:
    """
    Summarize text using global pipeline
    
    Args:
        text: Input text
    
    Returns:
        Generated summary
    """
    pipeline = get_pipeline()
    return pipeline.summarize_text(text)


def pipeline(text: str) -> Dict[str, any]:
    """
    Complete classification and summarization pipeline
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with results
    """
    pipe = get_pipeline()
    return pipe.pipeline(text)


def main():
    """Test inference pipeline"""
    config = load_config()
    
    print("=" * 80)
    print("INFERENCE PIPELINE TEST")
    print("=" * 80)
    
    # Test texts
    test_texts = [
        "My internet has been down for 3 hours. I need urgent help fixing this issue.",
        "I was charged twice on my last bill. Can someone refund the extra charge?",
        "Your customer service was excellent! Thank you for the quick resolution."
    ]
    
    pipe = InferencePipeline(config)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}")
        print(f"{'='*80}")
        print(f"Input: {text}")
        
        result = pipe.pipeline(text)
        
        print(f"\nResults:")
        print(f"  Category:   {result.get('category', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Summary:    {result.get('summary', 'N/A')}")
    
    print(f"\n{'='*80}")
    print("✓ Inference pipeline test completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
