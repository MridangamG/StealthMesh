"""
Feature Engineering Module
Handles feature selection, scaling, and transformation
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TOP_FEATURES, PROCESSED_DIR, RANDOM_STATE


class FeatureEngineer:
    """
    Feature engineering and selection for CICIDS 2017 dataset
    """
    
    def __init__(self):
        self.scaler = None
        self.selected_features = None
        self.feature_importances = None
        self.pca = None
        
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding labels and metadata)
        """
        exclude_cols = [
            'Label', 'Label_Binary', 'Label_Multiclass', 
            'Label_Category', 'Attack_Category', 'Source_File'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only include numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_cols
    
    def select_top_features(self, df: pd.DataFrame, 
                            n_features: int = 40) -> pd.DataFrame:
        """
        Select top features based on predefined list
        """
        available_features = [f for f in TOP_FEATURES if f in df.columns]
        
        if len(available_features) < n_features:
            # Add more features if needed
            all_features = self.get_feature_columns(df)
            additional = [f for f in all_features if f not in available_features]
            available_features.extend(additional[:n_features - len(available_features)])
        
        self.selected_features = available_features[:n_features]
        print(f"Selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def select_features_statistical(self, X: pd.DataFrame, y: pd.Series,
                                     n_features: int = 40,
                                     method: str = 'f_classif') -> List[str]:
        """
        Select features using statistical tests
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            method: 'f_classif' or 'mutual_info'
        """
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=n_features)
        else:
            selector = SelectKBest(mutual_info_classif, k=n_features)
        
        selector.fit(X, y)
        
        # Get selected feature names
        mask = selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        
        # Store feature scores
        self.feature_importances = dict(zip(X.columns, selector.scores_))
        
        print(f"Selected {len(self.selected_features)} features using {method}")
        
        return self.selected_features
    
    def scale_features(self, df: pd.DataFrame, 
                       feature_cols: List[str],
                       method: str = 'standard',
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to scale
            method: 'standard', 'minmax', or 'robust'
            fit: Whether to fit the scaler (False for transform only)
        """
        df = df.copy()
        
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            self.scaler = scaler_class()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            print(f"Fitted and transformed using {method} scaling")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            print(f"Transformed using existing {method} scaler")
        
        return df
    
    def apply_pca(self, X: pd.DataFrame, 
                  n_components: int = 20,
                  fit: bool = True) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            X: Feature DataFrame
            n_components: Number of principal components
            fit: Whether to fit PCA
        """
        if fit:
            self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
            X_pca = self.pca.fit_transform(X)
            
            # Print explained variance
            cum_var = np.cumsum(self.pca.explained_variance_ratio_)
            print(f"PCA: {n_components} components explain {cum_var[-1]*100:.2f}% variance")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important columns
        """
        df = df.copy()
        
        # Packet ratio features
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            df['Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        
        # Byte ratio features
        if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
            df['Byte_Ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
        
        # Flow intensity
        if 'Flow Duration' in df.columns and 'Flow Bytes/s' in df.columns:
            df['Flow_Intensity'] = df['Flow Bytes/s'] * df['Flow Duration'] / 1e6
        
        # Flag combinations
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
                     'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
        existing_flags = [c for c in flag_cols if c in df.columns]
        if existing_flags:
            df['Total_Flags'] = df[existing_flags].sum(axis=1)
        
        print(f"Created interaction features. New shape: {df.shape}")
        
        return df
    
    def save_scaler(self, filepath: Optional[str] = None):
        """
        Save the fitted scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler to save")
        
        if filepath is None:
            filepath = os.path.join(PROCESSED_DIR, 'scaler.pkl')
        
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: Optional[str] = None):
        """
        Load a saved scaler
        """
        if filepath is None:
            filepath = os.path.join(PROCESSED_DIR, 'scaler.pkl')
        
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
    
    def save_selected_features(self, filepath: Optional[str] = None):
        """
        Save selected feature list
        """
        if self.selected_features is None:
            raise ValueError("No features selected")
        
        if filepath is None:
            filepath = os.path.join(PROCESSED_DIR, 'selected_features.pkl')
        
        joblib.dump(self.selected_features, filepath)
        print(f"Features saved to {filepath}")
    
    def load_selected_features(self, filepath: Optional[str] = None) -> List[str]:
        """
        Load saved feature list
        """
        if filepath is None:
            filepath = os.path.join(PROCESSED_DIR, 'selected_features.pkl')
        
        self.selected_features = joblib.load(filepath)
        print(f"Loaded {len(self.selected_features)} features from {filepath}")
        return self.selected_features
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame
        """
        if self.feature_importances is None:
            raise ValueError("No feature importances available")
        
        df = pd.DataFrame([
            {'Feature': k, 'Importance': v} 
            for k, v in self.feature_importances.items()
        ])
        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return df


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    df = loader.load_all_files()
    
    cleaner = DataCleaner()
    df = cleaner.clean_data(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns(df)
    print(f"\nTotal feature columns: {len(feature_cols)}")
    
    # Select top features
    selected = engineer.select_top_features(df)
    print(f"Selected features: {selected[:10]}...")
    
    # Scale features
    df_scaled = engineer.scale_features(df, selected)
    print(f"Scaled data shape: {df_scaled.shape}")
