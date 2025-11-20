"""
Text Preprocessing Pipeline
Handles text cleaning, tokenization, lemmatization, and PII masking
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from utils import load_config, setup_logging

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocess_config = config.get('preprocessing', {})
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # PII patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        self.ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        self.logger = setup_logging(config)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Convert to lowercase
        if self.preprocess_config.get('lowercase', True):
            text = text.lower()
        
        # Remove punctuation
        if self.preprocess_config.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def mask_pii(self, text: str) -> str:
        """
        Mask personally identifiable information
        
        Args:
            text: Input text
        
        Returns:
            Text with masked PII
        """
        if not self.preprocess_config.get('mask_pii', True):
            return text
        
        # Mask emails
        text = re.sub(self.email_pattern, '[EMAIL]', text)
        
        # Mask phone numbers
        text = re.sub(self.phone_pattern, '[PHONE]', text)
        
        # Mask SSN
        text = re.sub(self.ssn_pattern, '[SSN]', text)
        
        return text
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        # Filter by minimum length
        min_length = self.preprocess_config.get('min_token_length', 2)
        tokens = [t for t in tokens if len(t) >= min_length]
        
        return tokens
    
    def lemmatize_spacy(self, text: str) -> str:
        """
        Lemmatize text using spaCy
        
        Args:
            text: Input text
        
        Returns:
            Lemmatized text
        """
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(lemmas)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered tokens
        """
        return [t for t in tokens if t.lower() not in self.stop_words]
    
    def preprocess(self, text: str, use_lemmatization: bool = True) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            use_lemmatization: Whether to apply lemmatization
        
        Returns:
            Preprocessed text
        """
        # Mask PII first (before lowercasing)
        text = self.mask_pii(text)
        
        # Clean text
        text = self.clean_text(text)
        
        # Apply lemmatization if requested
        if use_lemmatization:
            text = self.lemmatize_spacy(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str], use_lemmatization: bool = True) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of input texts
            use_lemmatization: Whether to apply lemmatization
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, use_lemmatization) for text in texts]


def process_dataset(input_path: str, output_path: str, config: dict) -> pd.DataFrame:
    """
    Process entire dataset
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        config: Configuration dictionary
    
    Returns:
        Processed DataFrame
    """
    logger = setup_logging(config)
    logger.info(f"Loading data from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Process text column
    logger.info("Preprocessing text...")
    df['processed_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x, use_lemmatization=True))
    
    # Also keep a version without lemmatization for certain models
    df['cleaned_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x, use_lemmatization=False))
    
    # Remove empty processed texts
    initial_count = len(df)
    df = df[df['processed_text'].str.len() > 0]
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} empty records after preprocessing")
    
    # Truncate to max length
    max_length = config['preprocessing'].get('max_text_length', 512)
    df['processed_text'] = df['processed_text'].apply(lambda x: x[:max_length])
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x[:max_length])
    
    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Print statistics
    logger.info("\nPreprocessing Statistics:")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Average text length: {df['processed_text'].str.len().mean():.2f} chars")
    logger.info(f"Category distribution:\n{df['category'].value_counts()}")
    
    return df


def main():
    """Main preprocessing execution"""
    # Load configuration
    config = load_config()
    
    # Process dataset
    input_path = config['data']['raw_path']
    output_path = config['data']['processed_path']
    
    df = process_dataset(input_path, output_path, config)
    
    print("\n✓ Preprocessing completed successfully!")
    print(f"✓ Processed data saved to: {output_path}")
    print(f"✓ Total records: {len(df)}")


if __name__ == "__main__":
    main()
