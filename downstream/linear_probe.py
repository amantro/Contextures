# # downstream/linear_probe.py

# """
# Linear probe for downstream evaluation of self-supervised learning representations.

# This module provides functionality to train linear classifiers/regressors on top of
# learned representations and evaluate their performance on downstream tasks.
# """

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import numpy as np
# import pandas as pd
# from typing import Dict, Any, Optional, Tuple, Literal, Union
# from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.preprocessing import StandardScaler
# import warnings

# from utils.registry import register_loss
# from utils.types import DataFrame, ArrayLike
# from utils.registry import get_loss


# class LinearProbe(nn.Module):
#     """
#     Linear probe for downstream evaluation.
    
#     Supports both classification and regression tasks with configurable
#     training parameters and evaluation metrics.
#     """
    
#     def __init__(self, 
#                  input_dim: int,
#                  task_type: Literal["classification", "regression"] = "classification",
#                  num_classes: Optional[int] = None,
#                  lr: float = 0.01,
#                  weight_decay: float = 1e-4,
#                  max_epochs: int = 100,
#                  batch_size: int = 256,
#                  early_stopping_patience: int = 10,
#                  device: str = "auto") -> None:
#         """
#         Initialize the linear probe.
        
#         Args:
#             input_dim: Dimension of input features
#             task_type: Type of downstream task ("classification" or "regression")
#             num_classes: Number of classes for classification (required if task_type="classification")
#             lr: Learning rate for training
#             weight_decay: Weight decay for regularization
#             max_epochs: Maximum number of training epochs
#             batch_size: Batch size for training
#             early_stopping_patience: Number of epochs to wait before early stopping
#             device: Device to use ("auto", "cpu", or "cuda")
#         """
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.task_type = task_type
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.max_epochs = max_epochs
#         self.batch_size = batch_size
#         self.early_stopping_patience = early_stopping_patience
        
#         # Set device
#         if device == "auto":
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)
        
#         # Initialize linear layer
#         if task_type == "classification":
#             if num_classes is None:
#                 raise ValueError("num_classes must be specified for classification tasks")
#             self.num_classes = num_classes
#             self.linear = nn.Linear(input_dim, num_classes)
#             self.criterion = nn.CrossEntropyLoss()
#         else:  # regression
#             self.linear = nn.Linear(input_dim, 1)
#             self.criterion = nn.MSELoss()
        
#         self.linear.to(self.device)
#         self._is_fitted = False
    
#     def fit(self, 
#             X_train: Union[torch.Tensor, ArrayLike, DataFrame],
#             y_train: Union[torch.Tensor, ArrayLike, DataFrame],
#             X_val: Optional[Union[torch.Tensor, ArrayLike, DataFrame]] = None,
#             y_val: Optional[Union[torch.Tensor, ArrayLike, DataFrame]] = None,
#             verbose: bool = True) -> Dict[str, Any]:
#         """
#         Fit the linear probe to the training data.
        
#         Args:
#             X_train: Training features
#             y_train: Training targets
#             X_val: Validation features (optional)
#             y_val: Validation targets (optional)
#             verbose: Whether to print training progress
            
#         Returns:
#             Dictionary containing training history and metrics
#         """
#         # Convert inputs to tensors
#         X_train_tensor = self._to_tensor(X_train)
#         y_train_tensor = self._to_tensor(y_train)
        
#         if X_val is not None and y_val is not None:
#             X_val_tensor = self._to_tensor(X_val)
#             y_val_tensor = self._to_tensor(y_val)
#             has_validation = True
#         else:
#             has_validation = False
        
#         # Create data loaders
#         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
#         if has_validation:
#             val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#             val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
#         # Initialize optimizer
#         optimizer = optim.Adam(self.linear.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
#         # Training loop
#         best_val_loss = float('inf')
#         patience_counter = 0
#         train_losses = []
#         val_losses = []
        
#         for epoch in range(self.max_epochs):
#             # Training
#             self.train()
#             train_loss = 0.0
#             for batch_X, batch_y in train_loader:
#                 batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
#                 optimizer.zero_grad()
#                 outputs = self.linear(batch_X)
#                 loss = self.criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
                
#                 train_loss += loss.item()
            
#             train_loss /= len(train_loader)
#             train_losses.append(train_loss)
            
#             # Validation
#             if has_validation:
#                 self.eval()
#                 val_loss = 0.0
#                 with torch.no_grad():
#                     for batch_X, batch_y in val_loader:
#                         batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
#                         outputs = self.linear(batch_X)
#                         loss = self.criterion(outputs, batch_y)
#                         val_loss += loss.item()
                
#                 val_loss /= len(val_loader)
#                 val_losses.append(val_loss)
                
#                 # Early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
                
#                 if patience_counter >= self.early_stopping_patience:
#                     if verbose:
#                         print(f"Early stopping at epoch {epoch + 1}")
#                     break
                
#                 if verbose and (epoch + 1) % 10 == 0:
#                     print(f"Epoch {epoch + 1}/{self.max_epochs}: "
#                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#             else:
#                 if verbose and (epoch + 1) % 10 == 0:
#                     print(f"Epoch {epoch + 1}/{self.max_epochs}: Train Loss: {train_loss:.4f}")
        
#         self._is_fitted = True
        
#         # Return training history
#         history = {
#             "train_losses": train_losses,
#             "final_train_loss": train_losses[-1] if train_losses else None,
#             "epochs_trained": len(train_losses)
#         }
        
#         if has_validation:
#             history.update({
#                 "val_losses": val_losses,
#                 "final_val_loss": val_losses[-1] if val_losses else None,
#                 "best_val_loss": best_val_loss
#             })
        
#         return history
    
#     def predict(self, X: Union[torch.Tensor, ArrayLike, DataFrame]) -> Union[torch.Tensor, ArrayLike]:
#         """
#         Make predictions on new data.
        
#         Args:
#             X: Input features
            
#         Returns:
#             Predictions
#         """
#         if not self._is_fitted:
#             raise RuntimeError("Linear probe must be fitted before making predictions")
        
#         self.eval()
#         X_tensor = self._to_tensor(X)
        
#         with torch.no_grad():
#             X_tensor = X_tensor.to(self.device)
#             outputs = self.linear(X_tensor)
            
#             if self.task_type == "classification":
#                 predictions = torch.argmax(outputs, dim=1)
#             else:  # regression
#                 predictions = outputs.squeeze()
        
#         # Convert back to original format
#         if isinstance(X, np.ndarray):
#             return predictions.cpu().numpy()
#         elif isinstance(X, pd.DataFrame):
#             return predictions.cpu().numpy()
#         else:
#             return predictions
    
#     def predict_proba(self, X: Union[torch.Tensor, ArrayLike, DataFrame]) -> Union[torch.Tensor, ArrayLike]:
#         """
#         Get prediction probabilities (classification only).
        
#         Args:
#             X: Input features
            
#         Returns:
#             Prediction probabilities
#         """
#         if self.task_type != "classification":
#             raise ValueError("predict_proba is only available for classification tasks")
        
#         if not self._is_fitted:
#             raise RuntimeError("Linear probe must be fitted before making predictions")
        
#         self.eval()
#         X_tensor = self._to_tensor(X)
        
#         with torch.no_grad():
#             X_tensor = X_tensor.to(self.device)
#             outputs = self.linear(X_tensor)
#             probabilities = torch.softmax(outputs, dim=1)
        
#         # Convert back to original format
#         if isinstance(X, np.ndarray):
#             return probabilities.cpu().numpy()
#         elif isinstance(X, pd.DataFrame):
#             return probabilities.cpu().numpy()
#         else:
#             return probabilities
    
#     def evaluate(self, 
#                 X: Union[torch.Tensor, ArrayLike, DataFrame],
#                 y: Union[torch.Tensor, ArrayLike, DataFrame]) -> Dict[str, float]:
#         """
#         Evaluate the linear probe on test data.
        
#         Args:
#             X: Test features
#             y: Test targets
            
#         Returns:
#             Dictionary containing evaluation metrics
#         """
#         if not self._is_fitted:
#             raise RuntimeError("Linear probe must be fitted before evaluation")
        
#         y_true = self._to_numpy(y)
#         y_pred = self._to_numpy(self.predict(X))
        
#         metrics = {}
        
#         if self.task_type == "classification":
#             metrics["accuracy"] = accuracy_score(y_true, y_pred)
            
#             # Try to get probabilities for additional metrics
#             try:
#                 y_proba = self._to_numpy(self.predict_proba(X))
#                 # Add more classification metrics here if needed
#             except Exception as e:
#                 warnings.warn(f"Could not compute probability-based metrics: {e}")
        
#         else:  # regression
#             metrics["r2_score"] = r2_score(y_true, y_pred)
#             metrics["mse"] = mean_squared_error(y_true, y_pred)
#             metrics["rmse"] = np.sqrt(metrics["mse"])
        
#         return metrics
    
#     def _to_tensor(self, data: Union[torch.Tensor, ArrayLike, DataFrame]) -> torch.Tensor:
#         """Convert input data to tensor."""
#         if isinstance(data, torch.Tensor):
#             return data
#         elif isinstance(data, np.ndarray):
#             # For classification targets, use long dtype
#             if self.task_type == "classification" and data.dtype in [np.int32, np.int64, int]:
#                 return torch.tensor(data, dtype=torch.long)
#             else:
#                 return torch.tensor(data, dtype=torch.float32)
#         elif isinstance(data, pd.DataFrame):
#             return torch.tensor(data.values, dtype=torch.float32)
#         elif isinstance(data, pd.Series):
#             # For classification targets, use long dtype
#             if self.task_type == "classification" and data.dtype in ['int32', 'int64', 'int']:
#                 return torch.tensor(data.values, dtype=torch.long)
#             else:
#                 return torch.tensor(data.values, dtype=torch.float32)
#         else:
#             raise TypeError(f"Unsupported data type: {type(data)}")
    
#     def _to_numpy(self, data: Union[torch.Tensor, ArrayLike, DataFrame]) -> np.ndarray:
#         """Convert input data to numpy array."""
#         if isinstance(data, torch.Tensor):
#             return data.cpu().numpy()
#         elif isinstance(data, np.ndarray):
#             return data
#         elif isinstance(data, pd.DataFrame):
#             return data.values
#         elif isinstance(data, pd.Series):
#             return data.values
#         else:
#             raise TypeError(f"Unsupported data type: {type(data)}")


# def train_linear_probe(
#     features: Union[torch.Tensor, ArrayLike, DataFrame],
#     targets: Union[torch.Tensor, ArrayLike, DataFrame],
#     task_type: Literal["classification", "regression"] = "classification",
#     test_size: float = 0.2,
#     val_size: float = 0.2,
#     random_state: int = 42,
#     X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
#     **probe_kwargs
# ) -> Tuple[LinearProbe, Dict[str, Any]]:
#     """
#     Train a linear probe on extracted features.
#     If X_train/y_train/X_val/y_val/X_test/y_test are provided, use them directly (no splitting).
#     Otherwise, split features/targets internally.
#     """
#     # Convert to numpy for splitting
#     if isinstance(features, torch.Tensor):
#         features_np = features.cpu().numpy()
#     elif isinstance(features, pd.DataFrame):
#         features_np = features.values
#     else:
#         features_np = features
    
#     if isinstance(targets, torch.Tensor):
#         targets_np = targets.cpu().numpy()
#     elif isinstance(targets, pd.DataFrame):
#         targets_np = targets.values
#     elif isinstance(targets, pd.Series):
#         targets_np = targets.values
#     else:
#         targets_np = targets
    
#     # Determine input dimension
#     if features_np.ndim == 1:
#         input_dim = 1
#         features_np = features_np.reshape(-1, 1)
#     else:
#         input_dim = features_np.shape[1]
    
#     # Determine number of classes for classification
#     if task_type == "classification":
#         num_classes = len(np.unique(targets_np))
#         probe_kwargs["num_classes"] = num_classes
    
#     # Use provided splits if available
#     if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
#         # Optionally use val if provided
#         if X_val is not None and y_val is not None:
#             pass
#         else:
#             X_val, y_val = None, None
#     else:
#         # Split data
#         np.random.seed(random_state)
#         n_samples = len(features_np)
#         indices = np.random.permutation(n_samples)
#         test_split = int(n_samples * (1 - test_size))
#         val_split = int(test_split * (1 - val_size))
#         train_indices = indices[:val_split]
#         val_indices = indices[val_split:test_split]
#         test_indices = indices[test_split:]
#         X_train = features_np[train_indices]
#         y_train = targets_np[train_indices]
#         X_val = features_np[val_indices]
#         y_val = targets_np[val_indices]
#         X_test = features_np[test_indices]
#         y_test = targets_np[test_indices]
    
#     # Initialize and train probe
#     probe = LinearProbe(input_dim=input_dim, task_type=task_type, **probe_kwargs)
    
#     # Train the probe
#     train_history = probe.fit(X_train, y_train, X_val, y_val, verbose=probe_kwargs.get('verbose', True))
    
#     # Evaluate on test set
#     test_metrics = probe.evaluate(X_test, y_test)
    
#     # Compile results
#     results = {
#         "train_history": train_history,
#         "test_metrics": test_metrics,
#         "data_splits": {
#             "train_size": len(X_train),
#             "val_size": len(X_val) if X_val is not None else 0,
#             "test_size": len(X_test)
#         }
#     }
    
#     return probe, results


# def extract_features(encoder: nn.Module,
#                     dataloader: DataLoader,
#                     device: str = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Extract features from a trained encoder.
    
#     Args:
#         encoder: Trained encoder model
#         dataloader: DataLoader containing the data
#         device: Device to use for extraction
        
#     Returns:
#         Tuple of (features, targets)
#     """
#     if device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)
    
#     encoder.to(device)
#     encoder.eval()
    
#     features_list = []
#     targets_list = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             if len(batch) == 2:
#                 x, y = batch
#             else:
#                 x = batch[0]
#                 y = batch[1] if len(batch) > 1 else None
            
#             x = x.to(device)
#             features = encoder(x)
            
#             features_list.append(features.cpu())
#             if y is not None:
#                 targets_list.append(y.cpu())
    
#     features = torch.cat(features_list, dim=0)
#     targets = torch.cat(targets_list, dim=0) if targets_list else None
    
#     return features, targets


# # Registry registration for easy access
# @register_loss('linear_probe')
# class LinearProbeLoss(nn.Module):
#     """
#     Linear probe loss for integration with the registry system.
#     This is a wrapper around LinearProbe for integration with the existing loss registry system.
#     Optionally, you can use a custom loss from the losses folder by passing custom_loss_name and custom_loss_params.
#     """
#     def __init__(self, 
#                  input_dim: int,
#                  task_type: Literal["classification", "regression"] = "classification",
#                  num_classes: Optional[int] = None,
#                  custom_loss_name: str = None,
#                  custom_loss_params: dict = None,
#                  **kwargs):
#         """
#         Initialize linear probe loss.
#         Args:
#             input_dim: Dimension of input features
#             task_type: Type of downstream task
#             num_classes: Number of classes for classification
#             custom_loss_name: Name of custom loss from losses folder (optional)
#             custom_loss_params: Parameters for custom loss (optional)
#             **kwargs: Additional arguments for LinearProbe
#         """
#         super().__init__()
#         self.probe = LinearProbe(
#             input_dim=input_dim,
#             task_type=task_type,
#             num_classes=num_classes,
#             **kwargs
#         )
#         self.custom_loss = None
#         if custom_loss_name is not None:
#             loss_class = get_loss(custom_loss_name)
#             params = custom_loss_params or {}
#             self.custom_loss = loss_class(**params)
    
#     def forward(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         """
#         Forward pass for linear probe training.
#         If a custom loss is provided, use it. Otherwise, use the default probe loss.
#         Args:
#             features: Input features
#             targets: Target values
#         Returns:
#             Tuple of (loss, metrics_dict)
#         """
#         if self.custom_loss is not None:
#             # Assume custom loss takes (features, targets) and returns (loss, metrics)
#             return self.custom_loss(features, targets)
#         # Default: dummy loss (for registry compatibility)
#         dummy_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
#         metrics = {
#             "probe_initialized": True,
#             "input_dim": self.probe.input_dim,
#             "task_type": self.probe.task_type
#         }
#         return dummy_loss, metrics
# downstream/linear_probe.py

# downstream/linear_probe.py

# downstream/linear_probe.py

"""
Linear probe for downstream evaluation of self-supervised learning representations.

This module provides functionality to train linear classifiers/regressors on top of
learned representations and evaluate their performance on downstream tasks.
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Literal, Union
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.types import DataFrame, ArrayLike

class LinearProbe:
    """
    Linear probe for downstream evaluation using scikit-learn.
    Supports both classification (LogisticRegression) and regression (Ridge).
    """
    def __init__(self, 
                 task_type: Literal["binary", "classification", "regression"] = "classification",
                 weight_decay: float = 1.0,
                 max_iter: int = 1000,
                 **kwargs):
        self.task_type = task_type
        self.weight_decay = weight_decay
        if task_type == "classification" or task_type == "binary":
            # C is inverse of regularization strength
            self.model = LogisticRegression(penalty='l2', C=1/weight_decay if weight_decay > 0 else 1e12, max_iter=max_iter, **kwargs)
        else:
            self.model = Ridge(alpha=weight_decay, max_iter=max_iter, **kwargs)
        self._is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        # No explicit validation/early stopping, but keep API for compatibility
        return {"epochs_trained": 1, "final_train_loss": None}

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Linear probe must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if self.task_type != "classification" and self.task_type != "binary":
            raise ValueError("predict_proba is only available for classification tasks")
        if not self._is_fitted:
            raise RuntimeError("Linear probe must be fitted before making predictions")
        return self.model.predict_proba(X)

    def evaluate(self, X, y) -> Dict[str, float]:
        y_true = y
        y_pred = self.predict(X)
        metrics = {}
        if self.task_type == "classification" or self.task_type == "binary":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        else:
            metrics["r2_score"] = r2_score(y_true, y_pred)
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
        return metrics


def train_linear_probe(
    features: Union[torch.Tensor, ArrayLike, DataFrame],
    targets: Union[torch.Tensor, ArrayLike, DataFrame],
    task_type: Literal["binary", "classification", "regression"] = "classification",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
    weight_decay: float = 1.0,
    max_iter: int = 1000,
    **probe_kwargs
) -> Tuple[LinearProbe, Dict[str, Any]]:
    """
    Train a linear probe on extracted features using scikit-learn.
    If X_train/y_train/X_val/y_val/X_test/y_test are provided, use them directly (no splitting).
    Otherwise, split features/targets internally.
    """
    # Convert to numpy for splitting
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    elif isinstance(features, pd.DataFrame):
        features_np = features.values
    else:
        features_np = features
    
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    elif isinstance(targets, pd.DataFrame):
        targets_np = targets.values
    elif isinstance(targets, pd.Series):
        targets_np = targets.values
    else:
        targets_np = targets
    
    # Use provided splits if available
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        if X_val is not None and y_val is not None:
            pass
        else:
            X_val, y_val = None, None
    else:
        np.random.seed(random_state)
        n_samples = len(features_np)
        indices = np.random.permutation(n_samples)
        test_split = int(n_samples * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        train_indices = indices[:val_split]
        val_indices = indices[val_split:test_split]
        test_indices = indices[test_split:]
        X_train = features_np[train_indices]
        y_train = targets_np[train_indices]
        X_val = features_np[val_indices]
        y_val = targets_np[val_indices]
        X_test = features_np[test_indices]
        y_test = targets_np[test_indices]
    
    # Initialize and train probe
    probe = LinearProbe(task_type=task_type, weight_decay=weight_decay, max_iter=max_iter, **probe_kwargs)
    train_history = probe.fit(X_train, y_train, X_val, y_val, verbose=probe_kwargs.get('verbose', True))
    
    # Evaluate on test set
    test_metrics = probe.evaluate(X_test, y_test)
    
    # Compile results
    results = {
        "train_history": train_history,
        "test_metrics": test_metrics,
        "data_splits": {
            "train_size": len(X_train),
            "val_size": len(X_val) if X_val is not None else 0,
            "test_size": len(X_test)
        }
    }
    
    return probe, results


def extract_features(encoder: torch.nn.Module, dataloader, device: str = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
    # Run the trained encoder φ in eval mode to extract representations for downstream evaluation (e.g., linear probe)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    encoder.to(device)
    encoder.eval()
    features_list = []
    targets_list = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            x = x.to(device)
            features = encoder(x)
            features_list.append(features.cpu())
            if y is not None:
                targets_list.append(y.cpu())
    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0) if targets_list else None
    return features, targets
