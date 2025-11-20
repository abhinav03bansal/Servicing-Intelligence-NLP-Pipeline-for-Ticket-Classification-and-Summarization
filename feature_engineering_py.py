"""
Feature Engineering Pipeline
Extracts TF-IDF, Word2Vec, and BERT embeddings with statistical feature selection
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import load_config, setup_logging, get_device, ensure_dir


class FeatureEngineer:
    """Extract and engineer features from text data"""
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.logger = setup_logging(config)
        self.device = get_device()
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features
        
        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer
        
        Returns:
            TF-IDF feature matrix
        """
        if not self.feature_config.get('use_tfidf', True):
            return None
        
        self.logger.info("Extracting TF-IDF features...")
        
        if fit or self.tfidf_vectorizer is None:
            max_features = self.feature_config.get('tfidf_max_features', 5000)
            ngram_range = tuple(self.feature_config.get('tfidf_ngram_range', [1, 2]))
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
            
            features = self.tfidf_vectorizer.fit_transform(texts)
            self.logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        self.logger.info(f"TF-IDF feature shape: {features.shape}")
        return features.toarray()
    
    def extract_word2vec_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract Word2Vec embeddings
        
        Args:
            texts: List of text strings
            fit: Whether to train the model
        
        Returns:
            Word2Vec feature matrix (averaged word vectors)
        """
        if not self.feature_config.get('use_word2vec', True):
            return None
        
        self.logger.info("Extracting Word2Vec features...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        if fit or self.word2vec_model is None:
            vector_size = self.feature_config.get('word2vec_size', 100)
            window = self.feature_config.get('word2vec_window', 5)
            min_count = self.feature_config.get('word2vec_min_count', 2)
            
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4,
                epochs=10
            )
            
            self.logger.info(f"Word2Vec vocabulary size: {len(self.word2vec_model.wv)}")
        
        # Get averaged word vectors for each text
        features = []
        for tokens in tokenized_texts:
            vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(self.word2vec_model.vector_size))
        
        features = np.array(features)
        self.logger.info(f"Word2Vec feature shape: {features.shape}")
        return features
    
    def extract_bert_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Extract BERT embeddings using transformers
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
        
        Returns:
            BERT embedding matrix
        """
        if not self.feature_config.get('use_bert_embeddings', True):
            return None
        
        self.logger.info("Extracting BERT embeddings...")
        
        # Load BERT model and tokenizer
        if self.bert_model is None:
            model_name = "bert-base-uncased"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed {i+len(batch_texts)}/{len(texts)} texts")
        
        embeddings = np.array(embeddings)
        self.logger.info(f"BERT embedding shape: {embeddings.shape}")
        return embeddings
    
    def select_features_statistical(self, X: np.ndarray, y: np.ndarray, 
                                   threshold: float = 10.0) -> Tuple[np.ndarray, List[int]]:
        """
        Statistical feature selection using variance inflation factor (VIF)
        
        Args:
            X: Feature matrix
            y: Target labels
            threshold: VIF threshold for multicollinearity
        
        Returns:
            Selected features and their indices
        """
        self.logger.info("Performing statistical feature selection...")
        
        # Calculate VIF for each feature
        n_features = X.shape[1]
        if n_features > 100:
            # Sample features for efficiency
            sample_indices = np.random.choice(n_features, min(100, n_features), replace=False)
            X_sample = X[:, sample_indices]
        else:
            X_sample = X
            sample_indices = np.arange(n_features)
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = range(X_sample.shape[1])
        
        try:
            vif_data["VIF"] = [variance_inflation_factor(X_sample, i) for i in range(X_sample.shape[1])]
            
            # Select features with VIF below threshold
            selected_indices = vif_data[vif_data["VIF"] < threshold]["Feature"].values
            
            if len(selected_indices) > 0:
                # Map back to original indices
                original_indices = sample_indices[selected_indices]
                X_selected = X[:, original_indices]
                self.logger.info(f"Selected {len(original_indices)}/{n_features} features based on VIF")
                return X_selected, original_indices.tolist()
        except:
            self.logger.warning("VIF calculation failed, using all features")
        
        return X, list(range(n_features))
    
    def combine_features(self, tfidf: Optional[np.ndarray], 
                        word2vec: Optional[np.ndarray], 
                        bert: Optional[np.ndarray]) -> np.ndarray:
        """
        Combine different feature types
        
        Args:
            tfidf: TF-IDF features
            word2vec: Word2Vec features
            bert: BERT embeddings
        
        Returns:
            Combined feature matrix
        """
        features = []
        
        if tfidf is not None:
            features.append(tfidf)
            self.logger.info(f"Added TF-IDF features: {tfidf.shape}")
        
        if word2vec is not None:
            features.append(word2vec)
            self.logger.info(f"Added Word2Vec features: {word2vec.shape}")
        
        if bert is not None:
            features.append(bert)
            self.logger.info(f"Added BERT features: {bert.shape}")
        
        if not features:
            raise ValueError("No features to combine")
        
        combined = np.hstack(features)
        self.logger.info(f"Combined feature shape: {combined.shape}")
        return combined
    
    def save_models(self, save_dir: str = "models/features"):
        """
        Save feature extraction models
        
        Args:
            save_dir: Directory to save models
        """
        ensure_dir(save_dir)
        
        if self.tfidf_vectorizer is not None:
            with open(f"{save_dir}/tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            self.logger.info(f"Saved TF-IDF vectorizer to {save_dir}")
        
        if self.word2vec_model is not None:
            self.word2vec_model.save(f"{save_dir}/word2vec_model.bin")
            self.logger.info(f"Saved Word2Vec model to {save_dir}")
    
    def load_models(self, save_dir: str = "models/features"):
        """
        Load feature extraction models
        
        Args:
            save_dir: Directory containing saved models
        """
        try:
            with open(f"{save_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            self.logger.info("Loaded TF-IDF vectorizer")
        except:
            self.logger.warning("Could not load TF-IDF vectorizer")
        
        try:
            self.word2vec_model = Word2Vec.load(f"{save_dir}/word2vec_model.bin")
            self.logger.info("Loaded Word2Vec model")
        except:
            self.logger.warning("Could not load Word2Vec model")


def main():
    """Main feature engineering execution"""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    # Load processed data
    data_path = config['data']['processed_path']
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(config)
    
    # Extract features
    texts = df['processed_text'].tolist()
    
    tfidf_features = engineer.extract_tfidf_features(texts, fit=True)
    word2vec_features = engineer.extract_word2vec_features(texts, fit=True)
    bert_features = engineer.extract_bert_embeddings(texts, batch_size=16)
    
    # Combine features
    combined_features = engineer.combine_features(tfidf_features, word2vec_features, bert_features)
    
    logger.info(f"\nFinal combined feature shape: {combined_features.shape}")
    
    # Save models
    engineer.save_models()
    
    # Save features
    np.save("features_combined.npy", combined_features)
    logger.info("Saved combined features to features_combined.npy")
    
    print("\n✓ Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
