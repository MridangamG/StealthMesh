"""
Neural Network Model for StealthMesh
"""

import os
import numpy as np
import time
from typing import Optional, List, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NN_PARAMS
from src.models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural Network classifier for network intrusion detection
    Using sklearn MLPClassifier for simplicity
    """
    
    def __init__(self, name: str = "NeuralNetwork"):
        super().__init__(name)
        self.history = None
        
    def build_model(self, **kwargs):
        """
        Build Neural Network model using sklearn
        
        Args:
            **kwargs: Override default parameters
        """
        from sklearn.neural_network import MLPClassifier
        
        params = {
            'hidden_layer_sizes': tuple(NN_PARAMS['hidden_layers']),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': NN_PARAMS['batch_size'],
            'learning_rate_init': NN_PARAMS['learning_rate'],
            'max_iter': NN_PARAMS['epochs'],
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': NN_PARAMS['early_stopping_patience'],
            'random_state': 42,
            'verbose': False
        }
        params.update(kwargs)
        
        self.model = MLPClassifier(**params)
        print(f"Built {self.name} model with layers: {params['hidden_layer_sizes']}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Train the Neural Network model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model is None:
            self.build_model(**kwargs)
        
        print(f"\nTraining {self.name}...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Store training history
        self.history = {
            'loss': self.model.loss_curve_,
            'n_iter': self.model.n_iter_
        }
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Iterations: {self.model.n_iter_}")
        
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
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training loss curve
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{self.name} Training Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class DeepNeuralNetwork(BaseModel):
    """
    Deep Neural Network using TensorFlow/Keras
    """
    
    def __init__(self, name: str = "DeepNN"):
        super().__init__(name)
        self.history = None
        
    def build_model(self, input_dim: int, num_classes: int = 2, **kwargs):
        """
        Build Deep Neural Network model using Keras
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            **kwargs: Override default parameters
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
        
        params = NN_PARAMS.copy()
        params.update(kwargs)
        
        model = Sequential([
            Dense(params['hidden_layers'][0], activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            
            Dense(params['hidden_layers'][1], activation='relu'),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            
            Dense(params['hidden_layers'][2], activation='relu'),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
        ])
        
        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        self.params = params
        print(f"Built {self.name} model")
        model.summary()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs):
        """
        Train the Deep Neural Network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        if self.model is None:
            num_classes = len(np.unique(y_train))
            self.build_model(X_train.shape[1], num_classes, **kwargs)
        
        print(f"\nTraining {self.name}...")
        start_time = time.time()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            validation_split=0.1 if validation_data is None else 0,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = self.model.predict(X, verbose=0)
        
        if predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = self.model.predict(X, verbose=0)
        
        if predictions.shape[1] == 1:
            return np.column_stack([1 - predictions, predictions])
        return predictions


if __name__ == "__main__":
    # Test the model
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Test sklearn NN
    model = NeuralNetworkModel()
    model.build_model()
    model.train(X_train, y_train)
    model.print_evaluation(X_test, y_test)
