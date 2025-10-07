import logging
import argparse
import pickle
import os
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, temporal_signal_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving"""
    
    def __init__(self, patience: int = 5, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self._is_better(val_score, self.best_score):
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


class SpatialGCNLayer(nn.Module):
    """Spatial Graph Convolutional Network layer with enhanced regularization"""
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.3, use_layernorm: bool = False):
        super(SpatialGCNLayer, self).__init__()
        self.gcn = GCNConv(in_features, out_features)
        if use_layernorm:
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.gcn(x, edge_index, edge_weight)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AttentionLayer(nn.Module):
    """Graph Attention Network layer with enhanced regularization"""
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout_rate: float = 0.3, attn_dropout: float = 0.2, use_layernorm: bool = False):
        super(AttentionLayer, self).__init__()
        self.gat = GATConv(in_features, out_features, heads=heads, dropout=attn_dropout)
        if use_layernorm:
            self.norm = nn.LayerNorm(out_features * heads)
        else:
            self.norm = nn.BatchNorm1d(out_features * heads)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # Note: attn_dropout is now applied inside GATConv, this is additional feature dropout
        self.feature_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.gat(x, edge_index)  # attn_dropout applied inside GAT
        x = self.norm(x)
        x = self.activation(x)
        x = self.feature_dropout(x)  # Additional feature dropout
        return x


class MemoryUnit(nn.Module):
    """GRU-based memory unit for temporal modeling"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super(MemoryUnit, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            hidden: (num_layers, batch_size, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
            hidden: (num_layers, batch_size, hidden_size)
        """
        output, hidden = self.gru(x, hidden)
        return output, hidden


class TGNNModel(nn.Module):
    """
    Temporal Graph Neural Network for hate speech prediction
    
    Architecture:
    1. Spatial GCN layer for graph structure learning
    2. Memory unit (GRU) for temporal modeling
    3. Attention layer (GAT) for importance weighting
    4. Output layer for hate probability prediction
    """
    
    def __init__(self, 
                 node_features: int,
                 gcn_hidden: int = 64,
                 gru_hidden: int = 32,
                 gat_hidden: int = 16,
                 gat_heads: int = 4,
                 output_dim: int = 1,
                 num_gru_layers: int = 2,
                 time_decay: float = 0.95,
                 adaptive_decay: bool = True,
                 use_gat: bool = False,
                 dropout_rate: float = 0.3,
                 attn_dropout: float = 0.2,
                 use_layernorm: bool = False):
        super(TGNNModel, self).__init__()
        
        self.node_features = node_features
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.gat_hidden = gat_hidden
        self.time_decay = time_decay
        self.adaptive_decay = adaptive_decay
        self.use_gat = use_gat
        
        # Spatial layer - GCN with enhanced regularization
        self.spatial_layer = SpatialGCNLayer(node_features, gcn_hidden, dropout_rate=dropout_rate, use_layernorm=use_layernorm)
        
        # Memory unit - GRU
        self.memory_unit = MemoryUnit(gcn_hidden, gru_hidden, num_gru_layers)
        
        # Attention layer - GAT (conditional) with enhanced regularization
        if use_gat:
            self.attention_layer = AttentionLayer(gru_hidden, gat_hidden, gat_heads, dropout_rate=dropout_rate, attn_dropout=attn_dropout, use_layernorm=use_layernorm)
            attention_output_dim = gat_hidden * gat_heads
        else:
            self.attention_layer = None
            attention_output_dim = gru_hidden
        
        # Output layers (pure linear regression, no activation)
        self.output_layer = nn.Sequential(
            nn.Linear(attention_output_dim, gat_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gat_hidden, output_dim)
            # Pure linear output for regression - no sigmoid, no clamp
        )
        
        # Hidden state for GRU
        self.hidden_state = None
        
        # Adaptive time decay components
        if self.adaptive_decay:
            # Learnable decay parameters
            self.decay_gate = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden // 2),
                nn.ReLU(),
                nn.Linear(gru_hidden // 2, 1),
                nn.Sigmoid()
            )
            # Time step counter for adaptive decay
            self.time_step = 0
        
    def forward(self, x, edge_index, edge_weight=None, apply_time_decay=True):
        """
        Forward pass through the TGNN model
        
        Args:
            x: Node features (num_nodes, node_features)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            apply_time_decay: Whether to apply time decay to hidden states
            
        Returns:
            hate_probability: Continuous hate score [0, 1] (num_nodes, 1)
        """
        batch_size = x.size(0)
        
        # 1. Spatial layer - GCN
        spatial_features = self.spatial_layer(x, edge_index, edge_weight)
        
        # 2. Memory unit - GRU
        # Reshape for GRU: (batch_size, seq_len=1, features)
        gru_input = spatial_features.unsqueeze(1)
        
        # Handle variable batch sizes by resetting hidden state if size mismatch
        if self.hidden_state is not None:
            expected_batch_size = self.hidden_state.size(1)
            current_batch_size = batch_size
            
            if expected_batch_size != current_batch_size:
                # Reset hidden state for different batch sizes
                self.hidden_state = None
            elif apply_time_decay:
                # Apply enhanced time decay
                if self.adaptive_decay:
                    # Adaptive decay based on current hidden state
                    decay_factor = self.decay_gate(self.hidden_state.mean(dim=0, keepdim=True))
                    # Combine fixed and adaptive decay
                    adaptive_decay = self.time_decay * (0.5 + 0.5 * decay_factor.mean().item())
                    self.hidden_state = self.hidden_state * adaptive_decay
                else:
                    # Standard exponential decay
                    self.hidden_state = self.hidden_state * self.time_decay
                
                # Detach hidden state to prevent backprop through time
                self.hidden_state = self.hidden_state.detach()
        
        # Increment time step for adaptive decay
        if self.adaptive_decay:
            self.time_step += 1
            
        temporal_features, self.hidden_state = self.memory_unit(gru_input, self.hidden_state)
        
        # Remove sequence dimension: (batch_size, features)
        temporal_features = temporal_features.squeeze(1)
        
        # 3. Attention layer - GAT (conditional)
        if self.use_gat and self.attention_layer is not None:
            attention_features = self.attention_layer(temporal_features, edge_index, edge_weight)
        else:
            # Skip attention layer, use temporal features directly
            attention_features = temporal_features
        
        # 4. Output layer (pure linear regression)
        hate_score = self.output_layer(attention_features)
        
        # No clamping - let the model learn the full range
        return hate_score
    
    def reset_memory(self):
        """Reset the GRU hidden state and time step"""
        self.hidden_state = None
        if self.adaptive_decay:
            self.time_step = 0
        
    def predict_next_timestep(self, x, edge_index, edge_weight=None):
        """
        Predict hate likelihood for the next timestep
        
        Returns:
            predictions: Hate probability for each node
        """
        with torch.no_grad():
            self.eval()
            hate_prob = self.forward(x, edge_index, edge_weight, apply_time_decay=True)
            return hate_prob


class TGNNTrainer:
    """Trainer class for TGNN model"""
    
    def __init__(self, 
                 model: TGNNModel,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 early_stop_patience: int = 5,
                 task: str = "regression",
                 corr_lambda: float = 0.1):
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()
        
        # Combined loss parameters
        self.correlation_weight = corr_lambda  # Weight for correlation loss component
        
        # Early stopping and learning rate scheduling - use correlation for early stopping
        self.early_stopping = EarlyStopping(patience=early_stop_patience, mode='max')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5
        )
        
        # Target normalization parameters (z-score)
        self.target_mean = None
        self.target_std = None
        self.normalization_fitted = False
    
    def fit_target_normalization(self, train_loader):
        """Fit z-score normalization parameters on training data"""
        all_targets = []
        
        # Collect all targets from training data
        try:
            snapshots = list(train_loader)
        except AttributeError:
            # Direct dataset access
            for i in range(train_loader.snapshot_count):
                all_targets.extend(train_loader.targets[i].numpy().flatten())
        else:
            for snapshot in snapshots:
                all_targets.extend(snapshot.y.numpy().flatten())
        
        # Calculate mean and std
        all_targets = np.array(all_targets)
        self.target_mean = np.mean(all_targets)
        self.target_std = np.std(all_targets)
        self.normalization_fitted = True
        
        print(f"Target normalization fitted: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
    
    def normalize_targets(self, targets):
        """Normalize targets using z-score"""
        if not self.normalization_fitted:
            raise ValueError("Target normalization not fitted yet")
        return (targets - self.target_mean) / self.target_std
    
    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to original scale"""
        if not self.normalization_fitted:
            raise ValueError("Target normalization not fitted yet")
        return predictions * self.target_std + self.target_mean
    
    def compute_combined_loss(self, y_pred, y_true):
        """Compute combined loss: MSE + correlation penalty"""
        # Primary MSE loss
        mse_loss = self.criterion(y_pred, y_true)
        
        # Correlation penalty (1 - pearson_corr)
        try:
            # Compute Pearson correlation
            y_pred_flat = y_pred.flatten()
            y_true_flat = y_true.flatten()
            
            # Calculate correlation coefficient
            mean_pred = torch.mean(y_pred_flat)
            mean_true = torch.mean(y_true_flat)
            
            numerator = torch.sum((y_pred_flat - mean_pred) * (y_true_flat - mean_true))
            denominator = torch.sqrt(torch.sum((y_pred_flat - mean_pred) ** 2) * torch.sum((y_true_flat - mean_true) ** 2))
            
            if denominator > 1e-8:
                correlation = numerator / denominator
                correlation_loss = 1.0 - correlation
            else:
                correlation_loss = 0.0
                
        except:
            correlation_loss = 0.0
        
        # Combined loss
        total_loss = mse_loss + self.correlation_weight * correlation_loss
        
        return total_loss, mse_loss, correlation_loss
        
    def train_epoch(self, train_loader, epoch: int, use_next_timestep=True):
        """Train for one epoch with optional t->t+1 prediction"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Reset memory at the beginning of each epoch
        self.model.reset_memory()
        
        # Use direct dataset access for stability and dtype control
        # Create snapshots manually from dataset components with explicit dtype
        snapshots = []
        for i in range(train_loader.snapshot_count):
            from torch_geometric.data import Data
            snapshot = Data(
                x=(train_loader.features[i].detach().clone().to(torch.float32) if torch.is_tensor(train_loader.features[i]) else torch.tensor(train_loader.features[i], dtype=torch.float32)),
                edge_index=(train_loader.edge_indices[i].detach().clone().to(torch.long) if torch.is_tensor(train_loader.edge_indices[i]) else torch.tensor(train_loader.edge_indices[i], dtype=torch.long)),
                edge_attr=((train_loader.edge_weights[i].detach().clone().to(torch.float32) if torch.is_tensor(train_loader.edge_weights[i]) else torch.tensor(train_loader.edge_weights[i], dtype=torch.float32)) if i < len(train_loader.edge_weights) and train_loader.edge_weights[i] is not None else None),
                y=(train_loader.targets[i].detach().clone().to(torch.float32) if torch.is_tensor(train_loader.targets[i]) else torch.tensor(train_loader.targets[i], dtype=torch.float32))
            )
            snapshots.append(snapshot)
        
        for time_step, snapshot in enumerate(tqdm(snapshots, desc=f"Epoch {epoch}")):
            # Move data to device
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_weight = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            
            # Forward pass
            y_pred = self.model(x, edge_index, edge_weight)
            
            # Determine target: current timestep or next timestep
            if use_next_timestep and time_step < len(snapshots) - 1:
                # Use next timestep's toxicity as target (t->t+1 prediction)
                next_snapshot = snapshots[time_step + 1]
                y_true = next_snapshot.y.to(self.device)
                # Ensure target has same number of nodes as prediction
                if y_true.shape[0] != y_pred.shape[0]:
                    # If sizes don't match, use current timestep instead
                    y_true = snapshot.y.to(self.device)
            else:
                # Use current timestep's toxicity as target (t->t prediction)
                y_true = snapshot.y.to(self.device)
            
            # Create comment node mask using is_user indicator (last feature dimension)
            is_user = x[:, -1]  # Last feature is is_user indicator
            comment_mask = (is_user == 0)  # Only compute loss on comment nodes (is_user=0)
            
            # Normalize both predictions and targets for consistent loss computation
            if self.normalization_fitted:
                y_pred_norm = (y_pred - self.target_mean) / (self.target_std + 1e-8)
            
            # Compute combined loss with consistent normalization
            if self.normalization_fitted and comment_mask.sum() > 0:
                y_true_normalized = self.normalize_targets(y_true[comment_mask])
                loss, mse_loss, corr_loss = self.compute_combined_loss(y_pred_norm[comment_mask], y_true_normalized)
            elif comment_mask.sum() > 0:
                # Compute combined loss only on comment nodes
                loss, mse_loss, corr_loss = self.compute_combined_loss(y_pred[comment_mask], y_true[comment_mask])
            else:
                # Fallback if no comment nodes detected
                if self.normalization_fitted:
                    y_true_normalized = self.normalize_targets(y_true)
                    loss, mse_loss, corr_loss = self.compute_combined_loss(y_pred_norm, y_true_normalized)
                else:
                    loss, mse_loss, corr_loss = self.compute_combined_loss(y_pred, y_true)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Backward pass every few steps or at the end
            if (time_step + 1) % 1 == 0:  # Update every step for now
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, test_loader, threshold: float = 0.5, use_next_timestep=True,
                 pos_label_thresh: float = 0.3, temperature: float = 1.0,
                 prob_mode: str = "minmax",
                 val_scores_raw=None, val_labels_bin=None):
        """Evaluate the model with threshold sweeping and t->t+1 alignment"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_snapshots = 0
        
        # Reset memory for evaluation
        self.model.reset_memory()
        
        # Use direct dataset access for stability and dtype control
        # Create snapshots manually from dataset components with explicit dtype
        snapshots = []
        for i in range(test_loader.snapshot_count):
            from torch_geometric.data import Data
            snapshot = Data(
                x=(test_loader.features[i].detach().clone().to(torch.float32) if torch.is_tensor(test_loader.features[i]) else torch.tensor(test_loader.features[i], dtype=torch.float32)),
                edge_index=(test_loader.edge_indices[i].detach().clone().to(torch.long) if torch.is_tensor(test_loader.edge_indices[i]) else torch.tensor(test_loader.edge_indices[i], dtype=torch.long)),
                edge_attr=((test_loader.edge_weights[i].detach().clone().to(torch.float32) if torch.is_tensor(test_loader.edge_weights[i]) else torch.tensor(test_loader.edge_weights[i], dtype=torch.float32)) if i < len(test_loader.edge_weights) and test_loader.edge_weights[i] is not None else None),
                y=(test_loader.targets[i].detach().clone().to(torch.float32) if torch.is_tensor(test_loader.targets[i]) else torch.tensor(test_loader.targets[i], dtype=torch.float32))
            )
            snapshots.append(snapshot)
        
        with torch.no_grad():
            for time_step, snapshot in enumerate(snapshots):
                x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.to(self.device)
                edge_weight = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
                
                # Predict
                y_pred = self.model(x, edge_index, edge_weight)
                
                # Apply temperature scaling if specified (before normalization)
                if temperature != 1.0:
                    # For temperature scaling, we need to work with logits
                    # Since we're doing regression, we can apply temperature to the raw output
                    y_pred = y_pred / temperature
                
                # Determine target: current timestep or next timestep (same as training)
                if use_next_timestep and time_step < len(snapshots) - 1:
                    # Use next timestep's toxicity as target (t->t+1 prediction)
                    next_snapshot = snapshots[time_step + 1]
                    y_true = next_snapshot.y.to(self.device)
                    # Ensure target has same number of nodes as prediction
                    if y_true.shape[0] != y_pred.shape[0]:
                        # If sizes don't match, use current timestep instead
                        y_true = snapshot.y.to(self.device)
                else:
                    # Use current timestep's toxicity as target (t->t prediction)
                    y_true = snapshot.y.to(self.device)
                
                # Create comment node mask using is_user indicator (last feature dimension)
                is_user = x[:, -1]  # Last feature is is_user indicator
                comment_mask = (is_user == 0)  # Only evaluate on comment nodes (is_user=0)
                
                # Normalize predictions for consistent loss computation (same as training)
                if self.normalization_fitted:
                    y_pred_norm = (y_pred - self.target_mean) / (self.target_std + 1e-8)
                
                # Compute loss only on comment nodes with consistent normalization
                if comment_mask.sum() > 0:
                    if self.normalization_fitted:
                        y_true_normalized = self.normalize_targets(y_true[comment_mask])
                        loss = self.criterion(y_pred_norm[comment_mask], y_true_normalized)
                    else:
                        loss = self.criterion(y_pred[comment_mask], y_true[comment_mask])
                    # Store predictions and targets only for comment nodes (keep original scale for metrics)
                    all_predictions.extend(y_pred[comment_mask].cpu().numpy().flatten())
                    all_targets.extend(y_true[comment_mask].cpu().numpy().flatten())
                else:
                    # Fallback if no comment nodes detected
                    if self.normalization_fitted:
                        y_true_normalized = self.normalize_targets(y_true)
                        loss = self.criterion(y_pred_norm, y_true_normalized)
                    else:
                        loss = self.criterion(y_pred, y_true)
                    all_predictions.extend(y_pred.cpu().numpy().flatten())
                    all_targets.extend(y_true.cpu().numpy().flatten())
                
                total_loss += loss.item()
                num_snapshots += 1
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Strictly separate raw regression scores and probability scores
        raw_scores = predictions  # Model's raw regression output (no sigmoid/minmax mapping)
        if self.normalization_fitted:
            y_pred_reg = raw_scores * self.target_std + self.target_mean  # Denormalized for regression metrics
        else:
            y_pred_reg = raw_scores
        
        # Calibrate predictions for probability-based metrics
        prob_scores = self.calibrate_probabilities(predictions, targets, mode=prob_mode,
                                                   val_scores_raw=val_scores_raw,
                                                   val_labels_bin=val_labels_bin)
        
        # Threshold sweeping for optimal F1 and Youden index (full range)
        thresholds = np.linspace(0.0, 1.0, 101)
        best_f1 = 0.0
        best_youden = 0.0
        best_threshold_f1 = 0.2  # Default to 0.2 instead of 0.5
        best_threshold_youden = 0.2
        
        # Store threshold sweep results
        threshold_metrics = {}
        
        for thresh in thresholds:
            binary_predictions = (prob_scores > thresh).astype(int)
            binary_targets = (targets > pos_label_thresh).astype(int)  # Use pos_label_thresh as target threshold
            
            if len(np.unique(binary_targets)) > 1:
                try:
                    f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
                    precision = precision_score(binary_targets, binary_predictions, zero_division=0)
                    recall = recall_score(binary_targets, binary_predictions, zero_division=0)
                    
                    # Youden index = sensitivity + specificity - 1
                    tn = np.sum((binary_targets == 0) & (binary_predictions == 0))
                    fp = np.sum((binary_targets == 0) & (binary_predictions == 1))
                    fn = np.sum((binary_targets == 1) & (binary_predictions == 0))
                    tp = np.sum((binary_targets == 1) & (binary_predictions == 1))
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    youden = sensitivity + specificity - 1
                    
                    # Store metrics for this threshold
                    threshold_metrics[thresh] = {
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'youden': youden
                    }
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold_f1 = thresh
                    
                    if youden > best_youden:
                        best_youden = youden
                        best_threshold_youden = thresh
                        
                except:
                    continue
        
        # Use best threshold for final metrics
        optimal_threshold = best_threshold_f1 if best_f1 > 0 else best_threshold_youden
        binary_predictions = (prob_scores > optimal_threshold).astype(int)
        binary_targets = (targets > pos_label_thresh).astype(int)  # Use pos_label_thresh as target threshold
        
        # Calculate AUC using probability scores and binary targets
        try:
            if len(np.unique(binary_targets)) > 1:
                auc_score = roc_auc_score(binary_targets, prob_scores)
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0
        
        # Continuous metrics using denormalized regression scores
        mse_score = np.mean((y_pred_reg - targets) ** 2)
        mae_score = np.mean(np.abs(y_pred_reg - targets))
        
        # Correlation between predicted and actual hate scores (using denormalized regression scores)
        try:
            correlation = np.corrcoef(y_pred_reg, targets)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        metrics = {
            'loss': total_loss / num_snapshots if num_snapshots > 0 else 0.0,
            'mse': mse_score,
            'mae': mae_score,
            'correlation': correlation,
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'best_f1': best_f1,
            'best_youden': best_youden,
            # Keep binary metrics for threshold-based evaluation
            'accuracy': accuracy_score(binary_targets, binary_predictions),
            'precision': precision_score(binary_targets, binary_predictions, zero_division=0),
            'recall': recall_score(binary_targets, binary_predictions, zero_division=0),
            'f1': f1_score(binary_targets, binary_predictions, zero_division=0)
        }
        
        # Add probability scores to metrics (already computed above)
        metrics['calibrated_predictions'] = prob_scores
        
        return metrics, predictions, targets, threshold_metrics
    
    def calibrate_probabilities(self, predictions, targets, mode="minmax", val_scores_raw=None, val_labels_bin=None):
        """Calibrate predictions to proper probability range [0,1]"""
        if mode == "minmax":
            # Simple min-max scaling to [0,1] range
            pred_min = np.min(predictions)
            pred_max = np.max(predictions)
            
            if pred_max > pred_min:
                calibrated = (predictions - pred_min) / (pred_max - pred_min)
            else:
                calibrated = np.full_like(predictions, 0.5)
        elif mode == "sigmoid_z":
            # Z-score normalization followed by sigmoid
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            if pred_std > 1e-8:
                z_scores = (predictions - pred_mean) / pred_std
                calibrated = 1 / (1 + np.exp(-z_scores))  # Sigmoid
            else:
                calibrated = np.full_like(predictions, 0.5)
        elif mode == "isotonic":
            # Isotonic regression calibration using validation data
            if val_scores_raw is not None and val_labels_bin is not None:
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(val_scores_raw, val_labels_bin)
                calibrated = ir.transform(predictions)
            else:
                # Fallback to minmax if no validation data provided
                pred_min = np.min(predictions)
                pred_max = np.max(predictions)
                if pred_max > pred_min:
                    calibrated = (predictions - pred_min) / (pred_max - pred_min)
                else:
                    calibrated = np.full_like(predictions, 0.5)
        else:
            calibrated = predictions
        
        # Ensure values are in [0,1] range
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        return calibrated


def load_snapshot_data(file_path: str) -> DynamicGraphTemporalSignal:
    """Load preprocessed snapshot data"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def prune_by_threshold(predictions: np.ndarray, 
                      threshold: float = 0.7,
                      action: str = "flag",
                      inclusive: bool = True) -> Dict:
    """
    Prune/flag/ban/hide nodes based on hate probability threshold
    
    Args:
        predictions: Hate probabilities for each node
        threshold: Threshold for flagging
        action: Action to take ("flag", "ban", "hide", "prune")
        inclusive: Whether to use >= (True) or > (False) for comparison
    
    Returns:
        Dictionary with flagged node indices and statistics
    """
    if inclusive:
        flagged_indices = np.where(predictions >= threshold)[0]
    else:
        flagged_indices = np.where(predictions > threshold)[0]
    
    result = {
        'action': action,
        'threshold': threshold,
        'flagged_nodes': flagged_indices.tolist(),
        'num_flagged': len(flagged_indices),
        'total_nodes': len(predictions),
        'flagged_ratio': len(flagged_indices) / len(predictions) if len(predictions) > 0 else 0.0
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="TGNN for Hate Speech Prediction (Regression)")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"], help="Task type: regression (continuous hate score) or classification")
    parser.add_argument("--data_path", type=str, help="Path to snapshot data file (required if train_path/test_path not provided)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (for temporal data)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for flagging hate")
    parser.add_argument("--gcn_hidden", type=int, default=64, help="GCN hidden dimensions")
    parser.add_argument("--gru_hidden", type=int, default=64, help="GRU hidden dimensions")
    parser.add_argument("--gat_hidden", type=int, default=32, help="GAT hidden dimensions")
    parser.add_argument("--gat_heads", type=int, default=4, help="Number of GAT attention heads")
    parser.add_argument("--use_gat", action="store_true", help="Enable GAT attention layer (default: disabled)")
    parser.add_argument("--time_decay", type=float, default=0.7, help="Time decay factor for temporal weighting (0.7 = 30% decay per timestep)")
    parser.add_argument("--adaptive_decay", action="store_true", help="Use adaptive time decay")
    parser.add_argument("--next_timestep", action="store_true", default=True, help="Use t->t+1 prediction (predict next timestep, default: True)")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--log_calibration", action="store_true", help="Log prediction calibration and threshold sweep")
    parser.add_argument("--train_path", type=str, help="Path to training data pkl file(s), comma-separated for multiple files")
    parser.add_argument("--valid_path", type=str, help="Path to validation data pkl file (for threshold tuning)")
    parser.add_argument("--test_path", type=str, help="Path to test data pkl file")
    parser.add_argument("--auto_threshold", type=str, choices=["f1", "youden", "none"], default="f1", help="Use optimal threshold based on f1, youden, or none (default: f1)")
    parser.add_argument("--pos_label_thresh", type=float, default=0.25, help="Threshold for positive label classification")
    parser.add_argument("--flag_percentile", type=float, default=80, help="Percentile threshold as upper limit (default: 80)")
    parser.add_argument("--flag_min_threshold", type=float, default=0.35, help="Minimum threshold for flagging (default: 0.35)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for predictions (default: 1.0, lower = more spread)")
    parser.add_argument("--save_model", type=str, default="tgnn_model.pth", help="Path to save model")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for saving/loading model weights")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for regularization")
    parser.add_argument("--attn_dropout", type=float, default=0.2, help="Attention dropout rate for GAT")
    parser.add_argument("--use_layernorm", action="store_true", help="Use LayerNorm instead of BatchNorm")
    parser.add_argument("--corr_lambda", type=float, default=0.2, help="Weight for correlation loss component")
    parser.add_argument("--prob_mode", type=str, default="sigmoid_z", choices=["minmax", "sigmoid_z", "isotonic"], help="Probability calibration mode")
    parser.add_argument("--inclusive_threshold", action="store_true", help="Use >= (inclusive) for threshold comparison instead of > (exclusive)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train_path and not args.test_path and not args.data_path:
        parser.error("Either --data_path or both --train_path and --test_path must be provided")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    logger.info("[SEED] Set random seeds: torch=42, numpy=42, random=42")
    
    # Load data based on provided paths
    if args.train_path and args.test_path:
        # Load separate train, validation, and test datasets
        logger.info(f"Loading training data from {args.train_path}")
        train_paths = args.train_path.split(',')
        
        # Load and concatenate training datasets
        train_snapshots = []
        for train_path in train_paths:
            train_path = train_path.strip()
            logger.info(f"Loading {train_path}")
            train_data = load_snapshot_data(train_path)
            train_snapshots.extend(train_data)
        
        # Create combined training dataset
        from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
        train_dataset = DynamicGraphTemporalSignal(
            edge_indices=[s.edge_index for s in train_snapshots],
            edge_weights=[s.edge_attr for s in train_snapshots],
            features=[s.x for s in train_snapshots],
            targets=[s.y for s in train_snapshots]
        )
        
        # Load validation data if provided (for threshold tuning)
        valid_dataset = None
        if args.valid_path:
            logger.info(f"Loading validation data from {args.valid_path}")
            valid_dataset = load_snapshot_data(args.valid_path)
            logger.info(f"Validation snapshots: {valid_dataset.snapshot_count}")
        
        logger.info(f"Loading test data from {args.test_path}")
        test_dataset = load_snapshot_data(args.test_path)
        
        logger.info(f"Train snapshots: {train_dataset.snapshot_count}, Test snapshots: {test_dataset.snapshot_count}")
        
        # Get feature dimensions from first training snapshot
        # Workaround for PyTorch version compatibility issue
        try:
            first_snapshot = train_dataset[0]
            node_features = first_snapshot.x.shape[1]
            logger.info(f"Node features: {node_features}")
            logger.info(f"First snapshot - Nodes: {first_snapshot.x.shape[0]}, Edges: {first_snapshot.edge_index.shape[1]}")
        except (AttributeError, TypeError) as e:
            logger.debug(f"Could not access first snapshot directly: {e}")
            # Fallback: get feature dimension from the dataset's features list
            if len(train_dataset.features) > 0:
                node_features = train_dataset.features[0].shape[1]
                logger.info(f"Node features (fallback): {node_features}")
                logger.info(f"First snapshot - Nodes: {train_dataset.features[0].shape[0]}, Edges: {train_dataset.edge_indices[0].shape[1]}")
            else:
                raise ValueError("No snapshots available in training dataset")
        
    else:
        # Fallback to original data loading
        logger.info(f"Loading data from {args.data_path}")
        dataset = load_snapshot_data(args.data_path)
        logger.info(f"Loaded dataset with {dataset.snapshot_count} snapshots")
        
        # Get feature dimensions
        first_snapshot = dataset[0]
        node_features = first_snapshot.x.shape[1]
        logger.info(f"Node features: {node_features}")
        logger.info(f"First snapshot - Nodes: {first_snapshot.x.shape[0]}, Edges: {first_snapshot.edge_index.shape[1]}")
        
        # Split data
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)
        logger.info(f"Train snapshots: {train_dataset.snapshot_count}, Test snapshots: {test_dataset.snapshot_count}")
    
    # Initialize model
    model = TGNNModel(
        node_features=node_features,
        gcn_hidden=args.gcn_hidden,
        gru_hidden=args.gru_hidden,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        time_decay=args.time_decay,
        adaptive_decay=args.adaptive_decay,
        use_gat=args.use_gat,
        dropout_rate=args.dropout,
        attn_dropout=args.attn_dropout,
        use_layernorm=args.use_layernorm
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"GAT Attention: {'ENABLED' if args.use_gat else 'DISABLED'}")
    
    # Initialize trainer
    trainer = TGNNTrainer(
        model, device, args.lr, 
        early_stop_patience=args.early_stop_patience,
        task=args.task,
        corr_lambda=args.corr_lambda
    )
    
    logger.info(f"Task: {args.task.upper()}")
    if args.task == "regression":
        logger.info("Using MSE/L2 loss for continuous hate score prediction")
    else:
        logger.info("Using classification loss")
    
    # Fit target normalization on training data
    logger.info("Fitting target normalization...")
    trainer.fit_target_normalization(train_dataset)
    
    # Training loop
    logger.info("Starting training...")
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_dataset, epoch + 1, use_next_timestep=args.next_timestep)
        
        # Evaluate on validation set for early stopping and learning rate scheduling
        if valid_dataset is not None:
            val_metrics, val_predictions, val_targets, val_threshold_metrics = trainer.evaluate(
                valid_dataset, args.threshold, use_next_timestep=args.next_timestep, 
                pos_label_thresh=args.pos_label_thresh, temperature=args.temperature, prob_mode=args.prob_mode
            )
            
            # Update learning rate scheduler
            trainer.scheduler.step(val_metrics['loss'])
            
            # Check for early stopping using correlation (better for regression)
            if trainer.early_stopping(val_metrics['correlation']):
                logger.info(f"[EARLYSTOP] Early stopping at epoch {epoch + 1}")
                break
            
            # Save best model based on validation correlation
            if val_metrics['correlation'] > best_f1:  # Reuse best_f1 variable for correlation
                best_f1 = val_metrics['correlation']
                save_path = args.ckpt_path if args.ckpt_path else args.save_model
                torch.save(model.state_dict(), save_path)
                logger.info(f"[EARLYSTOP] Saved best model with validation correlation: {best_f1:.4f} to {save_path}")
        
        # Log training progress every epoch
        log_msg = f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}"
        if valid_dataset is not None:
            log_msg += f" | Val MSE: {val_metrics['mse']:.4f}, Corr: {val_metrics['correlation']:.4f}, F1: {val_metrics['f1']:.4f}"
        logger.info(log_msg)
    
    # Final evaluation
    logger.info("Final evaluation...")
    save_path = args.ckpt_path if args.ckpt_path else args.save_model
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path))
            logger.info(f"Loaded best model from {save_path} for final evaluation")
        except RuntimeError as e:
            logger.warning(f"Could not load saved model: {e}")
            logger.info("Using current model for final evaluation")
    else:
        logger.info(f"No saved model found at {save_path}, using current model for final evaluation")
    # If validation data is available, use it for threshold tuning and isotonic calibration
    valid_opt_thr = args.threshold
    val_scores_raw = None
    val_labels_bin = None
    
    if valid_dataset is not None and args.auto_threshold:
        logger.info("Using validation data for threshold tuning...")
        # 先在验证集上跑一遍，拿到 raw 分数和标签
        valid_metrics, val_predictions, val_targets, valid_threshold_metrics = trainer.evaluate(
            valid_dataset, args.threshold, use_next_timestep=args.next_timestep,
            pos_label_thresh=args.pos_label_thresh, temperature=args.temperature, prob_mode=args.prob_mode
        )
        # Select threshold based on auto_threshold parameter
        if args.auto_threshold == "youden":
            valid_opt_thr = max(valid_threshold_metrics.items(), key=lambda x: x[1]['youden'])[0]
            logger.info(f"Validation optimal threshold (Youden): {valid_opt_thr:.3f}")
        elif args.auto_threshold == "f1":
            valid_opt_thr = max(valid_threshold_metrics.items(), key=lambda x: x[1]['f1'])[0]
            logger.info(f"Validation optimal threshold (F1): {valid_opt_thr:.3f}")
        elif args.auto_threshold == "none":
            # Use percentile + minimum threshold strategy only
            valid_opt_thr = 0.0
            logger.info("Using percentile + minimum threshold strategy only (auto_threshold=none)")
        else:
            # Fallback to default
            valid_opt_thr = valid_metrics.get('optimal_threshold', args.threshold)
            logger.info(f"Validation optimal threshold (default): {valid_opt_thr:.3f}")
        
        # 为 isotonic 准备：验证集 raw 分数 + 二值标签
        val_scores_raw = val_predictions
        val_labels_bin = (val_targets > args.pos_label_thresh).astype(int)
        
        # Log calibration if requested
        if args.log_calibration:
            logger.info("[CAL] Validation threshold sweep (top 10 F1 scores):")
            sorted_thresholds = sorted(valid_threshold_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)[:10]
            for thresh, metrics in sorted_thresholds:
                logger.info(f"[CAL]   Threshold {thresh:.3f}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # 测试集评测时把验证集信息传进去（关键）
    final_metrics, final_predictions, final_targets, final_threshold_metrics = trainer.evaluate(
        test_dataset, valid_opt_thr, use_next_timestep=args.next_timestep,
        pos_label_thresh=args.pos_label_thresh, temperature=args.temperature, prob_mode=args.prob_mode,
        val_scores_raw=val_scores_raw, val_labels_bin=val_labels_bin
    )
    
    logger.info(f"[CAL] Mode={args.prob_mode}, Temperature={args.temperature}")
    
    # Print prediction score distribution
    if args.log_calibration:
        logger.info("[CAL] Prediction Score Distribution:")
        logger.info(f"[CAL]   Min: {final_predictions.min():.4f}")
        logger.info(f"[CAL]   Max: {final_predictions.max():.4f}")
        logger.info(f"[CAL]   Mean: {final_predictions.mean():.4f}")
        percentiles = np.percentile(final_predictions, [50, 75, 90, 95, 99])
        logger.info(f"[CAL]   P50: {percentiles[0]:.4f}")
        logger.info(f"[CAL]   P75: {percentiles[1]:.4f}")
        logger.info(f"[CAL]   P90: {percentiles[2]:.4f}")
        logger.info(f"[CAL]   P95: {percentiles[3]:.4f}")
        logger.info(f"[CAL]   P99: {percentiles[4]:.4f}")
        
        # Log test threshold sweep if requested
        logger.info("[CAL] Test threshold sweep (top 10 F1 scores):")
        sorted_thresholds = sorted(final_threshold_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)[:10]
        for thresh, metrics in sorted_thresholds:
            logger.info(f"[CAL]   Threshold {thresh:.3f}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    else:
        logger.info("Prediction Score Distribution:")
        logger.info(f"  Min: {final_predictions.min():.4f}")
        logger.info(f"  Max: {final_predictions.max():.4f}")
        logger.info(f"  Mean: {final_predictions.mean():.4f}")
        percentiles = np.percentile(final_predictions, [50, 75, 90, 95, 99])
        logger.info(f"  P50: {percentiles[0]:.4f}")
        logger.info(f"  P75: {percentiles[1]:.4f}")
        logger.info(f"  P90: {percentiles[2]:.4f}")
        logger.info(f"  P95: {percentiles[3]:.4f}")
        logger.info(f"  P99: {percentiles[4]:.4f}")
    
    # 统一用校准后的概率做分布统计
    dist_source = final_metrics['calibrated_predictions']
    logger.info("="*60)
    logger.info("PREDICTION DISTRIBUTION:")
    logger.info(f"  Min: {dist_source.min():.4f}, Max: {dist_source.max():.4f}, Mean: {dist_source.mean():.4f}")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        logger.info(f"  P{q}: {np.percentile(dist_source, q):.4f}")
    logger.info("")
    logger.info("FINAL RESULTS - Regression Metrics (Primary):")
    logger.info(f"  MSE: {final_metrics['mse']:.4f}")
    logger.info(f"  MAE: {final_metrics['mae']:.4f}")
    logger.info(f"  Correlation: {final_metrics['correlation']:.4f}")
    logger.info("")
    logger.info(f"Classification Metrics (For Reference, threshold={args.pos_label_thresh}):")
    logger.info(f"  AUC: {final_metrics['auc']:.4f}")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  Recall: {final_metrics['recall']:.4f}")
    logger.info(f"  F1: {final_metrics['f1']:.4f}")
    logger.info("="*60)
    
    # Get calibrated predictions for threshold analysis
    calib = final_metrics['calibrated_predictions']
    
    # Calculate final threshold: clamp between [min_threshold, P80] range
    q = args.flag_percentile / 100.0
    try:
        # 保证阈值落在"右侧"台阶上，从而不会一口气吞下大段等值样本
        percentile_threshold = np.quantile(calib, q, method="higher")
    except TypeError:
        # 兼容旧 numpy（无 method 参数时退化为普通 quantile）
        percentile_threshold = np.quantile(calib, q)
    
    # —— Sanity Check: 百分位阈值必须落在校准分布范围内
    cal_min, cal_max = float(np.min(calib)), float(np.max(calib))
    if not (cal_min - 1e-12 <= percentile_threshold <= cal_max + 1e-12):
        logger.warning(
            f"[SANITY] P{args.flag_percentile}={percentile_threshold:.3f} is out of range "
            f"[{cal_min:.3f}, {cal_max:.3f}] of calibrated probs — check source of percentile!"
        )
    
    # New logic: clamp final threshold in [flag_min_threshold, P80] range
    final_threshold = min(valid_opt_thr, percentile_threshold)
    if args.flag_min_threshold is not None:
        final_threshold = max(final_threshold, args.flag_min_threshold)
        logger.info(f"Applied minimum threshold: {args.flag_min_threshold:.3f}")
    
    min_threshold = args.flag_min_threshold if args.flag_min_threshold is not None else 0.0
    logger.info(f"Final threshold clamped to [min_thresh, P{args.flag_percentile}]: [{min_threshold:.3f}, {percentile_threshold:.3f}]")
    
    logger.info(
        f"[CHECK] thresholds -> F1_opt: {valid_opt_thr:.3f}, "
        f"P{args.flag_percentile}: {percentile_threshold:.3f}, "
        f"Min: {min_threshold:.3f}"
    )
    
    logger.info(f"Threshold Analysis (F1 + Percentile + Min) - Based on Calibrated Probabilities:")
    logger.info(f"  F1 optimal threshold: {valid_opt_thr:.3f}")
    logger.info(f"  P{args.flag_percentile} percentile threshold: {percentile_threshold:.3f}")
    logger.info(f"  Final threshold: {final_threshold:.3f}")
    
    # Calculate hit rates using calibrated probabilities
    f1_hits = int(np.sum(calib >= valid_opt_thr))
    percentile_hits = int(np.sum(calib >= percentile_threshold))
    final_hits = int(np.sum(calib >= final_threshold))
    total_nodes = len(calib)
    
    logger.info(f"Hit Rates (Calibrated Probabilities):")
    logger.info(f"  F1 threshold: {f1_hits}/{total_nodes} ({f1_hits/total_nodes:.1%})")
    logger.info(f"  P{args.flag_percentile} threshold: {percentile_hits}/{total_nodes} ({percentile_hits/total_nodes:.1%})")
    logger.info(f"  Final threshold: {final_hits}/{total_nodes} ({final_hits/total_nodes:.1%})")
    
    # —— Sanity Check: 直接统计与 prune 结果应一致
    recount_hits = int(np.sum(calib >= final_threshold))
    if recount_hits != final_hits:
        logger.warning(
            f"[SANITY] final_hits({final_hits}) != recount_hits({recount_hits}) on calibrated probs. "
            f"Please check variable reuse / view slicing."
        )
    
    final_pruning = prune_by_threshold(calib, final_threshold, "flag", inclusive=args.inclusive_threshold)
    logger.info(f"Final flagging: {final_pruning['num_flagged']} nodes flagged "
               f"({final_pruning['flagged_ratio']:.2%}) with final threshold {final_threshold:.3f}")
    
    # Export probability and class columns (plan requirement)
    out_prob = final_metrics['calibrated_predictions']
    out_cls = (out_prob >= final_threshold).astype(int)
    df_out = pd.DataFrame({
        'toxicity_probability': out_prob,
        'class': np.where(out_cls==1, 'toxic', 'non-toxic')
    })
    df_out.to_csv('predictions_calibrated.csv', index=False, encoding='utf-8')
    logger.info("Saved calibrated probabilities to predictions_calibrated.csv")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
