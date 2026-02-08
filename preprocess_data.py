"""
Main Preprocessing Script for StealthMesh
Run this script to preprocess the CICIDS 2017 dataset
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.data_pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(description='StealthMesh Data Preprocessing')
    
    parser.add_argument(
        '--classification', 
        type=str, 
        default='binary',
        choices=['binary', 'multiclass', 'category'],
        help='Type of classification (default: binary)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to use (default: all)'
    )
    
    parser.add_argument(
        '--features',
        type=int,
        default=40,
        help='Number of features to select (default: 40)'
    )
    
    parser.add_argument(
        '--no-scale',
        action='store_true',
        help='Skip feature scaling'
    )
    
    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Skip class balancing'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("   STEALTHMESH DATA PREPROCESSING")
    print("   Adaptive Stealth Communication and Decentralized Defense")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Classification type: {args.classification}")
    print(f"  Sample size: {args.samples if args.samples else 'All data'}")
    print(f"  Number of features: {args.features}")
    print(f"  Scaling: {not args.no_scale}")
    print(f"  Balancing: {not args.no_balance}")
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    
    X_train, X_test, y_train, y_test = pipeline.run_pipeline(
        sample_size=args.samples,
        classification_type=args.classification,
        n_features=args.features,
        scale=not args.no_scale,
        balance=not args.no_balance
    )
    
    # Save processed data
    prefix = f'{args.classification}'
    pipeline.save_processed_data(X_train, X_test, y_train, y_test, prefix=prefix)
    
    print("\n" + "="*70)
    print("   PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Run model training: python train_models.py --classification {args.classification}")
    print(f"  2. Or explore data in: notebooks/01_eda.ipynb")
    print("="*70)


if __name__ == "__main__":
    main()
