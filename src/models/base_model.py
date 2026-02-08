"""
Base Model Class for StealthMesh
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODELS_DIR, RESULTS_DIR


class BaseModel(ABC):
    """
    Abstract base class for all StealthMesh models
    """
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.is_trained = False
        self.training_time = 0
        self.metrics = {}
        
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model doesn't support probability predictions")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 average: str = 'weighted') -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0),
            'inference_time_ms': (inference_time / len(X_test)) * 1000,
            'total_inference_time': inference_time,
        }
        
        # ROC-AUC for binary classification
        try:
            if len(np.unique(y_test)) == 2:
                y_proba = self.predict_proba(X_test)[:, 1]
                self.metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            pass
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return self.metrics
    
    def print_evaluation(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Print formatted evaluation results
        """
        metrics = self.evaluate(X_test, y_test)
        
        print("\n" + "="*60)
        print(f"MODEL EVALUATION: {self.name}")
        print("="*60)
        print(f"Accuracy:           {metrics['accuracy']*100:.2f}%")
        print(f"Precision:          {metrics['precision']*100:.2f}%")
        print(f"Recall:             {metrics['recall']*100:.2f}%")
        print(f"F1-Score:           {metrics['f1_score']*100:.2f}%")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"Inference Time:     {metrics['inference_time_ms']:.3f} ms/sample")
        print(f"Training Time:      {self.training_time:.2f} seconds")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("="*60)
        
        return metrics
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f'{self.name.lower()}_model.pkl')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None
