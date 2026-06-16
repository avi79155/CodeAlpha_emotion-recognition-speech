"""
Model Training Module

This module handles model training for emotion recognition including:
- Traditional ML models (Logistic Regression, Random Forest, SVM)
- Deep Learning models (CNN, LSTM, CNN-LSTM Hybrid)
- Hyperparameter tuning
- Early stopping and model checkpointing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import joblib
import os
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TraditionalMLModels:
    """
    Wrapper for traditional machine learning models.
    """

    def __init__(self):
        """
        Initialize traditional ML models.
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        logger.info("TraditionalMLModels initialized")

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Test set size
            random_state (int): Random state for reproducibility

        Returns:
            Tuple: X_train, X_test, y_train, y_test, label_encoder
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info(f"Data prepared: Training set {X_train.shape}, Test set {X_test.shape}")
        return X_train, X_test, y_train, y_test, self.label_encoder

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  max_iter: int = 1000) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            max_iter (int): Maximum iterations

        Returns:
            LogisticRegression: Trained model
        """
        model = LogisticRegression(max_iter=max_iter, random_state=42, multi_class='multinomial')
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        logger.info("Logistic Regression model trained")
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: int = 10) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth

        Returns:
            RandomForestClassifier: Trained model
        """
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        logger.info("Random Forest model trained")
        return model

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 kernel: str = 'rbf', C: float = 1.0) -> SVC:
        """
        Train Support Vector Machine model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            kernel (str): Kernel type
            C (float): Regularization parameter

        Returns:
            SVC: Trained model
        """
        model = SVC(kernel=kernel, C=C, random_state=42, probability=True)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        logger.info("SVM model trained")
        return model

    def tune_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Tune Random Forest using GridSearchCV.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels

        Returns:
            RandomForestClassifier: Best model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best RF parameters: {grid_search.best_params_}")
        logger.info(f"Best RF score: {grid_search.best_score_}")

        self.models['random_forest_tuned'] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def tune_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """
        Tune SVM using GridSearchCV.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels

        Returns:
            SVC: Best model
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best SVM parameters: {grid_search.best_params_}")
        logger.info(f"Best SVM score: {grid_search.best_score_}")

        self.models['svm_tuned'] = grid_search.best_estimator_
        return grid_search.best_estimator_

    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            model_name (str): Name of model in self.models
            filepath (str): Path to save model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        else:
            logger.warning(f"Model {model_name} not found")

    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_name (str): Name to assign to loaded model
            filepath (str): Path to load model from
        """
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")


class DeepLearningModels:
    """
    Wrapper for deep learning models using Keras/TensorFlow.
    """

    def __init__(self, num_classes: int, input_shape: Tuple[int, ...]):
        """
        Initialize deep learning models.

        Args:
            num_classes (int): Number of emotion classes
            input_shape (Tuple): Input shape for the models
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.models = {}
        self.history = {}
        logger.info(f"DeepLearningModels initialized with {num_classes} classes")

    def build_cnn(self) -> keras.Model:
        """
        Build CNN model for emotion recognition.

        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Reshape((self.input_shape[0], self.input_shape[1], 1)),

            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten and Dense
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.models['cnn'] = model
        logger.info("CNN model built and compiled")
        return model

    def build_lstm(self) -> keras.Model:
        """
        Build LSTM model for emotion recognition.

        Returns:
            keras.Model: Compiled LSTM model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            # LSTM layers
            layers.LSTM(128, return_sequences=True, activation='relu'),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False, activation='relu'),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.models['lstm'] = model
        logger.info("LSTM model built and compiled")
        return model

    def build_cnn_lstm_hybrid(self) -> keras.Model:
        """
        Build CNN-LSTM Hybrid model for emotion recognition.

        Returns:
            keras.Model: Compiled CNN-LSTM model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Reshape((self.input_shape[0], self.input_shape[1], 1)),

            # CNN layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Reshape for LSTM
            layers.Reshape((layers.Input(shape=self.input_shape[0] // 4).shape[0],
                           64 * (self.input_shape[1] // 4))),

            # LSTM layers
            layers.LSTM(128, return_sequences=True, activation='relu'),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu'),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.models['cnn_lstm'] = model
        logger.info("CNN-LSTM Hybrid model built and compiled")
        return model

    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 32,
                   model_path: str = 'models/best_model.h5') -> Dict[str, Any]:
        """
        Train a deep learning model.

        Args:
            model_name (str): Name of model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels (one-hot encoded)
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels (one-hot encoded)
            epochs (int): Number of epochs
            batch_size (int): Batch size
            model_path (str): Path to save best model

        Returns:
            Dict: Training history
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}

        model = self.models[model_name]

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )

        self.history[model_name] = history.history
        logger.info(f"Model {model_name} trained successfully")
        return history.history

    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            model_name (str): Name of model to save
            filepath (str): Path to save model
        """
        if model_name in self.models:
            self.models[model_name].save(filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        else:
            logger.warning(f"Model {model_name} not found")

    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_name (str): Name to assign to loaded model
            filepath (str): Path to load model from
        """
        self.models[model_name] = keras.models.load_model(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")


if __name__ == "__main__":
    logger.info("Training module loaded successfully")
