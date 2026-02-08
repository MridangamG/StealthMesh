"""
XGBoost Model for StealthMesh
"""

import os
import numpy as np
import time
from typing import Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import XGB_PARAMS
from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for network intrusion detection
    """
    
    def __init__(self, name: str = "XGBoost"):
        super().__init__(name)
        
    def build_model(self, **kwargs):
        """
        Build XGBoost model
        
        Args:
            **kwargs: Override default parameters
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        params = XGB_PARAMS.copy()
        params.update(kwargs)
        
        self.model = XGBClassifier(**params)
        print(f"Built {self.name} model with parameters: {params}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 10,
              **kwargs):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Rounds for early stopping
        """
        if self.model is None:
            self.build_model(**kwargs)
        
        print(f"\nTraining {self.name}...")
        start_time = time.time()
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        self.model.fit(X_train, y_train, **fit_params)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: Optional[list] = None,
                                importance_type: str = 'gain'):
        """
        Get feature importance ranking
        
        Args:
            feature_names: List of feature names
            importance_type: 'weight', 'gain', or 'cover'
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        import pandas as pd
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return df


if __name__ == "__main__":
    # Test the model
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train and evaluate
    model = XGBoostModel()
    model.build_model(n_estimators=50)
    model.train(X_train, y_train)
    model.print_evaluation(X_test, y_test)
