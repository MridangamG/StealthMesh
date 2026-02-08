"""
Data Cleaner Module
Handles data cleaning, missing values, and label encoding
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import COLUMNS_TO_DROP, BINARY_LABELS, MULTICLASS_LABELS, ATTACK_CATEGORIES


class DataCleaner:
    """
    Cleans and preprocesses CICIDS 2017 dataset
    """
    
    def __init__(self):
        self.removed_columns = []
        self.removed_rows = 0
        self.label_mapping = {}
        
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names
        """
        df = df.copy()
        
        # Strip whitespace
        df.columns = df.columns.str.strip()
        
        # Handle duplicate column names (Fwd Header Length appears twice)
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dups = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dups):
                if i > 0:
                    cols[idx] = f"{dup}.{i}"
        df.columns = cols
        
        return df
    
    def clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize label values
        """
        df = df.copy()
        
        if 'Label' not in df.columns:
            raise ValueError("Label column not found in DataFrame")
        
        # Strip whitespace from labels
        df['Label'] = df['Label'].str.strip()
        
        # Fix encoding issues in web attack labels
        label_fixes = {
            'Web Attack � Brute Force': 'Web Attack - Brute Force',
            'Web Attack � XSS': 'Web Attack - XSS', 
            'Web Attack � Sql Injection': 'Web Attack - SQL Injection',
            'Web Attack Brute Force': 'Web Attack - Brute Force',
            'Web Attack XSS': 'Web Attack - XSS',
            'Web Attack Sql Injection': 'Web Attack - SQL Injection',
        }
        
        # Apply fixes using regex for encoding issues
        for old, new in label_fixes.items():
            df['Label'] = df['Label'].str.replace(old, new, regex=False)
        
        # Handle any remaining encoding issues
        df['Label'] = df['Label'].apply(
            lambda x: re.sub(r'[^\x00-\x7F]+', '-', str(x)) if pd.notna(x) else x
        )
        
        # Clean up multiple dashes
        df['Label'] = df['Label'].str.replace('--', '-', regex=False)
        df['Label'] = df['Label'].str.replace(' - ', ' - ', regex=False)
        
        return df
    
    def remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with invalid values (NaN, Infinity)
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Replace infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove rows with NaN in numeric columns
        df = df.dropna(subset=numeric_cols)
        
        self.removed_rows = initial_rows - len(df)
        print(f"Removed {self.removed_rows:,} rows with invalid values")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Drop duplicates, keeping first occurrence
        df = df.drop_duplicates(keep='first')
        
        removed = initial_rows - len(df)
        print(f"Removed {removed:,} duplicate rows")
        
        return df
    
    def drop_unnecessary_columns(self, df: pd.DataFrame, 
                                  additional_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Drop unnecessary columns (IPs, timestamps, identifiers)
        """
        df = df.copy()
        
        cols_to_drop = COLUMNS_TO_DROP.copy()
        if additional_cols:
            cols_to_drop.extend(additional_cols)
        
        # Find columns that exist in the dataframe
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            self.removed_columns = existing_cols
            print(f"Dropped columns: {existing_cols}")
        
        return df
    
    def encode_labels_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels for binary classification (Benign vs Attack)
        """
        df = df.copy()
        
        # Create binary label: 0 = Benign, 1 = Attack
        df['Label_Binary'] = df['Label'].apply(
            lambda x: 0 if x == 'BENIGN' else 1
        )
        
        self.label_mapping['binary'] = {0: 'BENIGN', 1: 'ATTACK'}
        
        return df
    
    def encode_labels_multiclass(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels for multi-class classification
        """
        df = df.copy()
        
        # Get unique labels
        unique_labels = df['Label'].unique()
        
        # Create label mapping
        label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        df['Label_Multiclass'] = df['Label'].map(label_to_int)
        
        self.label_mapping['multiclass'] = {v: k for k, v in label_to_int.items()}
        
        return df
    
    def encode_labels_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels by attack category
        """
        df = df.copy()
        
        # Map to categories
        category_mapping = {
            'BENIGN': 'Benign',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'Heartbleed': 'DoS',
            'DDoS': 'DDoS',
            'PortScan': 'Reconnaissance',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Botnet',
            'Infiltration': 'Infiltration',
            'Web Attack - Brute Force': 'Web Attack',
            'Web Attack - XSS': 'Web Attack',
            'Web Attack - SQL Injection': 'Web Attack',
        }
        
        df['Attack_Category'] = df['Label'].map(category_mapping)
        
        # Fill any unmapped labels
        df['Attack_Category'] = df['Attack_Category'].fillna('Unknown')
        
        # Encode categories
        categories = sorted(df['Attack_Category'].unique())
        cat_to_int = {cat: idx for idx, cat in enumerate(categories)}
        df['Label_Category'] = df['Attack_Category'].map(cat_to_int)
        
        self.label_mapping['category'] = {v: k for k, v in cat_to_int.items()}
        
        return df
    
    def clean_data(self, df: pd.DataFrame, 
                   encode_binary: bool = True,
                   encode_multiclass: bool = True,
                   encode_category: bool = True) -> pd.DataFrame:
        """
        Run full data cleaning pipeline
        """
        print("\n" + "="*60)
        print("DATA CLEANING PIPELINE")
        print("="*60)
        
        # Step 1: Clean column names
        print("\n[1/7] Cleaning column names...")
        df = self.clean_column_names(df)
        
        # Step 2: Clean labels
        print("[2/7] Cleaning labels...")
        df = self.clean_labels(df)
        
        # Step 3: Remove invalid values
        print("[3/7] Removing invalid values...")
        df = self.remove_invalid_values(df)
        
        # Step 4: Remove duplicates
        print("[4/7] Removing duplicates...")
        df = self.remove_duplicates(df)
        
        # Step 5: Drop unnecessary columns
        print("[5/7] Dropping unnecessary columns...")
        df = self.drop_unnecessary_columns(df)
        
        # Step 6: Encode labels
        print("[6/7] Encoding labels...")
        if encode_binary:
            df = self.encode_labels_binary(df)
        if encode_multiclass:
            df = self.encode_labels_multiclass(df)
        if encode_category:
            df = self.encode_labels_category(df)
        
        # Step 7: Final cleanup
        print("[7/7] Final cleanup...")
        
        # Convert numeric columns to appropriate types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == 'float64':
                # Check if can be converted to int
                if (df[col] % 1 == 0).all():
                    df[col] = df[col].astype('int64')
                else:
                    df[col] = df[col].astype('float32')
        
        print("\n" + "="*60)
        print(f"Cleaning complete! Final shape: {df.shape}")
        print("="*60)
        
        return df
    
    def get_label_mapping(self) -> dict:
        """
        Get the label mapping dictionary
        """
        return self.label_mapping


if __name__ == "__main__":
    # Test the data cleaner
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_all_files()
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)
    
    print("\nLabel Mappings:")
    for key, mapping in cleaner.get_label_mapping().items():
        print(f"\n{key}:")
        for k, v in mapping.items():
            print(f"  {k}: {v}")
