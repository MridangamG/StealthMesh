"""
Data Loader Module
Handles loading and merging CICIDS 2017 dataset files
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR, DATASET_FILES


class DataLoader:
    """
    Loads and merges CICIDS 2017 dataset files
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.dataframes = {}
        
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with the loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading {filename}...")
        
        # Read CSV with low_memory=False to avoid dtype warnings
        df = pd.read_csv(filepath, low_memory=False)
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Add source file column for tracking
        df['Source_File'] = filename
        
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def load_all_files(self, files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and merge all dataset files
        
        Args:
            files: List of filenames to load (default: all files)
            
        Returns:
            Merged DataFrame
        """
        if files is None:
            files = DATASET_FILES
            
        dfs = []
        
        for filename in files:
            try:
                df = self.load_single_file(filename)
                self.dataframes[filename] = df
                dfs.append(df)
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No files were loaded successfully")
        
        print(f"\nMerging {len(dfs)} files...")
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Total records: {len(merged_df):,}")
        
        return merged_df
    
    def load_by_day(self, day: str) -> pd.DataFrame:
        """
        Load data for a specific day
        
        Args:
            day: Day name (Monday, Tuesday, Wednesday, Thursday, Friday)
            
        Returns:
            DataFrame with data for that day
        """
        day_files = [f for f in DATASET_FILES if day.lower() in f.lower()]
        
        if not day_files:
            raise ValueError(f"No files found for day: {day}")
        
        return self.load_all_files(day_files)
    
    def load_by_attack(self, attack_type: str) -> pd.DataFrame:
        """
        Load data containing specific attack type
        
        Args:
            attack_type: Type of attack (DDos, PortScan, WebAttacks, etc.)
            
        Returns:
            DataFrame filtered for that attack
        """
        # Mapping of attack types to files
        attack_file_mapping = {
            'ddos': ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'],
            'portscan': ['Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'],
            'bot': ['Friday-WorkingHours-Morning.pcap_ISCX.csv'],
            'infiltration': ['Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'],
            'webattack': ['Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'],
            'dos': ['Wednesday-workingHours.pcap_ISCX.csv'],
            'bruteforce': ['Tuesday-WorkingHours.pcap_ISCX.csv'],
            'benign': ['Monday-WorkingHours.pcap_ISCX.csv'],
        }
        
        attack_key = attack_type.lower().replace(' ', '').replace('-', '')
        
        if attack_key not in attack_file_mapping:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return self.load_all_files(attack_file_mapping[attack_key])
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': 0,  # Skip duplicate check for large datasets
        }
        
        # Label distribution
        if 'Label' in df.columns:
            summary['label_distribution'] = df['Label'].value_counts().to_dict()
        
        return summary
    
    def print_summary(self, df: pd.DataFrame):
        """
        Print formatted summary of the dataset
        """
        summary = self.get_data_summary(df)
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total Records:     {summary['total_records']:,}")
        print(f"Total Features:    {summary['total_features']}")
        print(f"Memory Usage:      {summary['memory_usage_mb']:.2f} MB")
        print(f"Missing Values:    {summary['missing_values']:,}")
        print(f"Duplicate Rows:    {summary['duplicate_rows']:,}")
        
        if 'label_distribution' in summary:
            print("\nLabel Distribution:")
            print("-"*40)
            for label, count in sorted(summary['label_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                pct = count / summary['total_records'] * 100
                print(f"  {label:35} {count:>10,} ({pct:5.2f}%)")
        
        print("="*60)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Load all data
    df = loader.load_all_files()
    loader.print_summary(df)
