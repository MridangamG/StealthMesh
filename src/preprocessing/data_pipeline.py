"""
Complete Data Pipeline for StealthMesh
Combines loading, cleaning, and feature engineering
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    SAMPLE_SIZE, USE_SMOTE
)
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer


class DataPipeline:
    """
    End-to-end data pipeline for StealthMesh
    """
    
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.is_fitted = False
        
    def run_pipeline(self, 
                     sample_size: Optional[int] = SAMPLE_SIZE,
                     classification_type: str = 'binary',
                     n_features: int = 40,
                     scale: bool = True,
                     balance: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete data pipeline
        
        Args:
            sample_size: Number of samples to use (None for all)
            classification_type: 'binary', 'multiclass', or 'category'
            n_features: Number of features to select
            scale: Whether to scale features
            balance: Whether to balance classes
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n" + "="*70)
        print("STEALTHMESH DATA PIPELINE")
        print("="*70)
        
        # Step 1: Load data (with early sampling for memory efficiency)
        print("\n[STEP 1] Loading data...")
        df = self.loader.load_all_files()
        self.loader.print_summary(df)
        
        # Step 2: Sample data EARLY if needed (before cleaning to save memory)
        if sample_size and len(df) > sample_size:
            print(f"\n[STEP 2] Early sampling {sample_size:,} records (memory optimization)...")
            # Clean Label column first for stratified sampling
            df.columns = df.columns.str.strip()
            if 'Label' in df.columns:
                df['Label'] = df['Label'].str.strip()
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)
            print(f"  Sampled to {len(df):,} records")
        else:
            print("\n[STEP 2] Using all data (no sampling)...")
        
        # Step 3: Clean data
        print("\n[STEP 3] Cleaning data...")
        df = self.cleaner.clean_data(df)
        
        # Step 4: Feature selection
        print(f"\n[STEP 4] Selecting top {n_features} features...")
        feature_cols = self.engineer.select_top_features(df, n_features)
        
        # Ensure all selected features exist
        feature_cols = [f for f in feature_cols if f in df.columns]
        
        # Step 5: Prepare X and y
        print("\n[STEP 5] Preparing features and labels...")
        X = df[feature_cols].copy()
        
        label_col_map = {
            'binary': 'Label_Binary',
            'multiclass': 'Label_Multiclass',
            'category': 'Label_Category'
        }
        y = df[label_col_map[classification_type]].copy()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Label distribution:\n{y.value_counts()}")
        
        # Step 6: Split data
        print("\n[STEP 6] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"  Train set: {X_train.shape[0]:,} samples")
        print(f"  Test set: {X_test.shape[0]:,} samples")
        
        # Step 7: Scale features
        if scale:
            print("\n[STEP 7] Scaling features...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.engineer.scaler = scaler
            X_train = X_train_scaled
            X_test = X_test_scaled
            print("  Scaling complete")
        
        # Step 8: Balance classes
        if balance and classification_type in ['binary', 'multiclass']:
            print("\n[STEP 8] Balancing classes...")
            X_train, y_train = self._balance_classes(X_train, y_train.values if hasattr(y_train, 'values') else y_train)
            print(f"  Balanced train set: {X_train.shape[0]:,} samples")
        
        self.is_fitted = True
        self.feature_cols = feature_cols
        self.classification_type = classification_type
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print("="*70)
        
        # Convert y_test to numpy array
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
        return X_train, X_test, y_train, y_test
    
    def _stratified_sample(self, df: pd.DataFrame, 
                           sample_size: int,
                           classification_type: str) -> pd.DataFrame:
        """
        Perform stratified sampling
        """
        label_col_map = {
            'binary': 'Label_Binary',
            'multiclass': 'Label_Multiclass',
            'category': 'Label_Category'
        }
        label_col = label_col_map[classification_type]
        
        # Sample while maintaining class distribution
        sampled_dfs = []
        
        for label in df[label_col].unique():
            label_df = df[df[label_col] == label]
            
            # Calculate proportional sample size
            proportion = len(label_df) / len(df)
            label_sample_size = max(int(sample_size * proportion), 10)
            
            if len(label_df) <= label_sample_size:
                sampled_dfs.append(label_df)
            else:
                sampled_dfs.append(label_df.sample(
                    n=label_sample_size, 
                    random_state=RANDOM_STATE
                ))
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using SMOTE and undersampling
        """
        # Get class distribution
        unique, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        min_count = counts.min()
        
        # If already relatively balanced, skip
        if min_count / max_count > 0.5:
            print("  Classes relatively balanced, skipping...")
            return X, y
        
        try:
            # Use a combination of oversampling and undersampling
            over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE)
            
            pipeline = ImbPipeline([
                ('over', over),
                ('under', under)
            ])
            
            X_balanced, y_balanced = pipeline.fit_resample(X, y)
            
            print(f"  Original: {len(y):,} -> Balanced: {len(y_balanced):,}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"  Balancing failed: {e}")
            print("  Returning original data...")
            return X, y
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, 
                            prefix: str = 'processed'):
        """
        Save processed data to disk
        """
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'), X_test)
        np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'), y_test)
        
        # Save feature columns
        joblib.dump(self.feature_cols, os.path.join(PROCESSED_DIR, f'{prefix}_features.pkl'))
        
        # Save scaler
        if self.engineer.scaler:
            self.engineer.save_scaler(os.path.join(PROCESSED_DIR, f'{prefix}_scaler.pkl'))
        
        # Save label mapping
        joblib.dump(
            self.cleaner.get_label_mapping(),
            os.path.join(PROCESSED_DIR, f'{prefix}_label_mapping.pkl')
        )
        
        print(f"\nProcessed data saved to {PROCESSED_DIR}")
    
    def load_processed_data(self, prefix: str = 'processed') -> Tuple:
        """
        Load processed data from disk
        """
        X_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_X_test.npy'))
        y_train = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_train.npy'))
        y_test = np.load(os.path.join(PROCESSED_DIR, f'{prefix}_y_test.npy'))
        
        self.feature_cols = joblib.load(os.path.join(PROCESSED_DIR, f'{prefix}_features.pkl'))
        self.engineer.load_scaler(os.path.join(PROCESSED_DIR, f'{prefix}_scaler.pkl'))
        
        self.is_fitted = True
        
        print(f"Loaded processed data from {PROCESSED_DIR}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run the pipeline
    pipeline = DataPipeline()
    
    # Process for binary classification
    X_train, X_test, y_train, y_test = pipeline.run_pipeline(
        sample_size=100000,  # Use 100k samples for testing
        classification_type='binary',
        n_features=40,
        scale=True,
        balance=True
    )
    
    # Save processed data
    pipeline.save_processed_data(X_train, X_test, y_train, y_test, prefix='binary')
    
    print("\n✅ Pipeline completed successfully!")
