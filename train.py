import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import torch.nn.functional as F
from collections import defaultdict
import pickle
import glob
import math
import random
import numpy as np

def set_seed(seed):
    """
    Set random seeds for reproducible results across different libraries.
    
    Args:
        seed (int): Random seed value
        
    Note:
        For full CUDA determinism, this function sets CUBLAS_WORKSPACE_CONFIG
        which may impact performance slightly but ensures reproducibility.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_graph(file):
    with open(file, "rb") as f:
        return pickle.load(f)

# ======================== ADVANCED LOSS FUNCTIONS ========================

class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss for regression with extreme class imbalance.
    Designed specifically for POI co-visitation prediction where targets span 5 orders of magnitude.
    
    This loss function addresses the head-tail imbalance by:
    1. Applying adaptive weighting based on target magnitude
    2. Using focal-style modulation to focus on harder examples
    3. Incorporating magnitude-aware scaling for extreme value ranges
    
    Args:
        alpha (float): Scaling factor for magnitude-based weighting (default: 2.0)
        gamma (float): Focusing parameter for hard examples (default: 2.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean'):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Compute base MSE
        mse = (pred - target) ** 2
        
        # Magnitude-aware weighting: higher weight for larger target values
        # This addresses the extreme range (1 to 28,000+ co-visits)
        magnitude_weights = torch.clamp(target.abs(), min=1.0) ** 0.3
        
        # Focal-style modulation: focus on examples with high residuals
        # Normalized residuals to make focal term scale-invariant
        normalized_residuals = mse / (magnitude_weights + 1e-8)
        focal_weight = (1 + normalized_residuals) ** self.gamma
        
        # Combine magnitude and focal weighting
        total_weight = (magnitude_weights ** (1/self.alpha)) * focal_weight
        weighted_loss = total_weight * mse
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

class QuantileLoss(nn.Module):
    """
    Multi-Quantile Loss for regression with heavy-tailed distributions.
    
    This loss is particularly effective for POI co-visitation data because:
    1. It handles extreme outliers better than MSE
    2. Provides robust optimization across the full distribution range
    3. Less sensitive to the 5 orders of magnitude variation in targets
    
    Args:
        quantiles (list): List of quantiles to optimize (default: [0.1, 0.5, 0.9])
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], reduction='mean'):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.reduction = reduction
        
    def forward(self, pred, target):
        losses = []
        total_weight = 0
        
        for i, q in enumerate(self.quantiles):
            errors = target - pred
            # Asymmetric loss: penalize over/under-prediction differently
            loss_q = torch.where(errors >= 0, 
                               q * errors, 
                               (q - 1) * errors)
            
            # Weight central quantiles more heavily
            weight = 2.0 if q == 0.5 else 1.0
            losses.append(weight * loss_q.mean())
            total_weight += weight
            
        combined_loss = sum(losses) / total_weight
        return combined_loss

class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that dynamically adjusts focus based on training progress.
    
    This loss combines ideas from focal loss and curriculum learning:
    1. Early training: Focus on easier examples (moderate co-visit values)
    2. Late training: Gradually include extreme values
    3. Adaptive gamma that changes based on epoch progress
    
    Args:
        initial_gamma (float): Starting gamma value (default: 1.0)
        final_gamma (float): Final gamma value (default: 3.0)
        alpha (float): Magnitude weighting factor (default: 1.5)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, initial_gamma=1.0, final_gamma=3.0, alpha=1.5, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.alpha = alpha
        self.reduction = reduction
        self.current_gamma = initial_gamma
        
    def update_gamma(self, epoch, max_epochs):
        """Update gamma based on training progress"""
        progress = min(epoch / max_epochs, 1.0)
        self.current_gamma = self.initial_gamma + progress * (self.final_gamma - self.initial_gamma)
        
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        
        # Target magnitude weighting
        magnitude_weights = torch.pow(target.abs() + 1, 1/self.alpha)
        
        # Adaptive focal weighting
        relative_error = mse / (magnitude_weights + 1e-8)
        focal_weight = torch.pow(1 + relative_error, self.current_gamma)
        
        weighted_loss = focal_weight * magnitude_weights * mse
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

class HuberQuantileLoss(nn.Module):
    """
    Huber-style Quantile Loss that combines robustness of Huber loss with quantile optimization.
    
    This is particularly suited for POI data because:
    1. Robust to extreme outliers (like the very high co-visit counts)
    2. Maintains sensitivity to small differences in moderate ranges
    3. Quantile-based optimization handles distribution skewness
    
    Args:
        quantiles (list): Quantiles to optimize (default: [0.25, 0.5, 0.75])
        delta (float): Huber loss threshold (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, quantiles=[0.25, 0.5, 0.75], delta=1.0, reduction='mean'):
        super(HuberQuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred, target):
        losses = []
        
        for q in self.quantiles:
            errors = target - pred
            
            # Quantile-specific asymmetric loss
            quantile_loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            
            # Apply Huber-style smoothing
            abs_loss = torch.abs(quantile_loss)
            huber_loss = torch.where(abs_loss <= self.delta,
                                   0.5 * quantile_loss**2,
                                   self.delta * (abs_loss - 0.5 * self.delta))
            
            # Weight by quantile importance (center more important)
            weight = 2.0 if q == 0.5 else 1.0
            losses.append(weight * huber_loss.mean())
            
        combined_loss = sum(losses) / sum([2.0 if q == 0.5 else 1.0 for q in self.quantiles])
        return combined_loss

# ======================== END ADVANCED LOSS FUNCTIONS ========================

class EdgeRegressionGNN(nn.Module):
    def __init__(self, layers, node_in_dim, edge_in_dim, hidden_dim, dropout, num_naics_codes=100000, naics_embed_dim=32):
        super(EdgeRegressionGNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.naics_embedding = nn.Embedding(num_naics_codes, naics_embed_dim)
        self.embed_norm = nn.LayerNorm(naics_embed_dim)

        init_mmbed = "constant"
        if init_mmbed == "xavier_uniform":
            nn.init.xavier_uniform_(self.naics_embedding.weight)
        elif init_mmbed == "constant":
            nn.init.constant_(self.naics_embedding.weight[0], 0)

        # self.input_projection = nn.Linear(naics_embed_dim + node_in_dim, hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(naics_embed_dim + node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Create GraphSAGE layers and batch norms
        self.convs, self.batch_norms = self._build_gnn_layers(layers, hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Edge MLP for predicting edge values
        self.edge_mlp = self._build_edge_mlp(hidden_dim, edge_in_dim, dropout)

    def _build_gnn_layers(self, num_layers, hidden_dim):
        """Build GraphSAGE layers and corresponding batch normalization layers."""
        convs = nn.ModuleList()
        batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim if i > 0 else hidden_dim
            convs.append(SAGEConv(in_channels, hidden_dim))
            batch_norms.append(nn.LazyBatchNorm1d(hidden_dim))

        return convs, batch_norms

    def _build_edge_mlp(self, hidden_dim, edge_in_dim, dropout):
        """Build the MLP used for edge prediction."""
        return nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _apply_gnn_layer(self, x, edge_index, conv, batch_norm, activation):
        """Apply a single GraphSAGE layer with batch normalization and residual connection."""
        x_res = x
        x = conv(x, edge_index)
        x = batch_norm(x)
        
        if activation == "relu":
            x = torch.relu(x)
        elif activation == "leaky_relu":
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        elif activation == "tanh":
            x = torch.tanh(x)
        
        x = self.dropout(x)
        x = x + x_res
        return x

    def forward(self, x, edge_index, edge_attr, naics_indices):
        naics_embeds = self.naics_embedding(naics_indices)
        naics_embeds = self.embed_norm(naics_embeds)

        x = torch.cat([naics_embeds, x], dim=1)

        # Input projection
        x = self.input_projection(x)

        # Retrieve activation function from config
        activation = "tanh"

        initial_features = x

        # Apply GNN layers
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = self._apply_gnn_layer(x, edge_index, conv, batch_norm, activation)

            # Global residual connection every 2 layers
            if i % 2 == 1:
                x = x + initial_features

        # Combine node embeddings and edge features
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)

        # Predict edge value (e.g., total covisits)
        return self.edge_mlp(edge_features)

subgraph_files = sorted(glob.glob("../datasets/state_graphs_small/*.pkl"))  # Sort for reproducibility
train_files, val_files, test_files = subgraph_files[:40], subgraph_files[40:45], subgraph_files[45:]

# Build global NAICS mapping
naics_counter = defaultdict(int)
for file in train_files + val_files + test_files:
    graph = load_graph(file)
    naics_codes = graph.x[:, 0].long().unique().cpu().numpy().tolist()
    for code in naics_codes:
        naics_counter[code] += 1

# Create mapping dictionary
unique_naics = sorted(list(naics_counter.keys()))  # Sort for deterministic mapping
naics_to_idx = {code: idx for idx, code in enumerate(unique_naics)}
num_naics_codes = len(unique_naics)

def transform_target(x, method='log'):
    """
    Transform target values using the specified scaling method.
    
    Parameters:
        x (Tensor): Input tensor.
        method (str): Transformation method. Options are:
            - 'log': Log transformation. Shifts data to be positive by subtracting min(x) and adding 1.
            - 'minmax': Scales data to the range [0, 1].
            - 'standard': Standardizes data (zero mean, unit variance).
            
    Returns:
        transformed (Tensor): Transformed tensor.
        params: Parameters needed for inverse transformation.
            For 'log', this is the minimum value.
            For 'minmax', a tuple (min, max).
            For 'standard', a tuple (mean, std).
    """
    if method == 'log':
        min_val = x.min()
        shifted = x - min_val + 1  # shift so that minimum becomes 1
        return torch.log(shifted), min_val
    elif method == 'minmax':
        min_val = x.min()
        max_val = x.max()
        scaled = (x - min_val) / (max_val - min_val)
        return scaled, (min_val, max_val)
    elif method == 'standard':
        mean_val = x.mean()
        std_val = x.std()
        scaled = (x - mean_val) / std_val
        return scaled, (mean_val, std_val)
    elif method == 'rth':
        return x, None
    else:
        raise ValueError(f"Unknown transformation method: {method}")

def inverse_transform_target(x, params, method='log'):
    """
    Inverse the transformation applied to target values.
    
    Parameters:
        x (Tensor): Transformed tensor.
        params: Parameters returned by transform_target().
        method (str): Transformation method (must be the same as used in transform_target).
        
    Returns:
        Tensor: The inverse-transformed (original scale) tensor.
    """
    if method == 'log':
        # params is min_val
        min_val = params
        return torch.exp(x) + min_val - 1
    elif method == 'minmax':
        # params is (min_val, max_val)
        min_val, max_val = params
        return x * (max_val - min_val) + min_val
    elif method == 'standard':
        mean_val, std_val = params
        return x * std_val + mean_val
    elif method == 'rth':
        return x
    else:
        raise ValueError(f"Unknown transformation method: {method}")

def train():
    layers = 5
    hidden_dim = 512
    dropout = 0.2
    lr = 0.001
    weight_decay = 0.0001
    num_epoch = 100
    criterion_name = "MSELoss"
    naics_embed_dim = 16
    scale_method = "standard"
    seed = 12 

    set_seed(seed)
    
    scaler = GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    first_graph = load_graph(train_files[0])
    node_in_dim = first_graph.x.size(1) - 1
    edge_in_dim = first_graph.edge_attr.size(1)

    model = EdgeRegressionGNN(layers, node_in_dim, edge_in_dim, hidden_dim, dropout, num_naics_codes, naics_embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "L1Loss":
        criterion = nn.L1Loss()
    elif criterion_name == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    elif criterion_name == "HuberLoss":
        criterion = nn.HuberLoss()
    elif criterion_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    # ======================== NEW ADVANCED LOSS FUNCTIONS ========================
    elif criterion_name == "FocalMSELoss":
        criterion = FocalMSELoss(alpha=2.0, gamma=2.0)
        print("Using FocalMSELoss: Designed for extreme class imbalance (alpha=2.0, gamma=2.0)")
    elif criterion_name == "QuantileLoss":
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        print("Using QuantileLoss: Robust to outliers with quantiles [0.1, 0.5, 0.9]")
    elif criterion_name == "AdaptiveFocalLoss":
        criterion = AdaptiveFocalLoss(initial_gamma=1.0, final_gamma=3.0, alpha=1.5)
        print("Using AdaptiveFocalLoss: Curriculum learning approach (gamma: 1.0→3.0)")
    elif criterion_name == "HuberQuantileLoss":
        criterion = HuberQuantileLoss(quantiles=[0.25, 0.5, 0.75], delta=1.0)
        print("Using HuberQuantileLoss: Robust quantile optimization (delta=1.0)")
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
    
    # Track if using adaptive loss for epoch updates
    is_adaptive_loss = criterion_name == "AdaptiveFocalLoss"
    # ======================== END LOSS FUNCTION SELECTION ========================

    best_val_loss = -float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(num_epoch):
        # Update adaptive loss parameters if needed
        if is_adaptive_loss:
            criterion.update_gamma(epoch, num_epoch)
            if epoch % 10 == 0:  # Log gamma updates every 10 epochs
                print(f"Epoch {epoch}: AdaptiveFocalLoss gamma = {criterion.current_gamma:.3f}")
        
        all_train_y_true = []
        all_train_y_pred = []

        for file in train_files:
            graph_data = load_graph(file)
            graph_data = graph_data.to(device)

            y_true = graph_data.y
            y_true_trans, trans_params = transform_target(y_true, method=scale_method)

            naics_indices = torch.tensor([naics_to_idx[code.item()] for code in graph_data.x[:, 0]])
            naics_indices = naics_indices.to(device, dtype=torch.long)
            
            model.train()
            optimizer.zero_grad()
            with autocast("cuda"):
                y_pred_trans = model(
                    graph_data.x[:, 1:], 
                    graph_data.edge_index, 
                    graph_data.edge_attr,
                    naics_indices
                )
                loss = criterion(y_pred_trans, y_true_trans)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                y_pred = inverse_transform_target(y_pred_trans, trans_params, method=scale_method)
                all_train_y_true.append(y_true)
                all_train_y_pred.append(y_pred)

            del graph_data, y_pred, y_true, y_pred_trans, y_true_trans
            torch.cuda.empty_cache()

        # Calculate final weighted metrics
        train_y_true = torch.cat(all_train_y_true, dim=0)
        train_y_pred = torch.cat(all_train_y_pred, dim=0)
        train_mse = F.mse_loss(train_y_pred, train_y_true).item()
        train_mae = F.l1_loss(train_y_pred, train_y_true).item()
        train_r2 = r2_score(train_y_true.cpu().detach().numpy(), train_y_pred.cpu().detach().numpy())
        train_rmse = math.sqrt(train_mse)

        all_val_y_true = []
        all_val_y_pred = []
        for file in val_files:
            graph_data = load_graph(file)
            graph_data = graph_data.to(device)

            y_true = graph_data.y
            y_true_trans, trans_params = transform_target(y_true, method=scale_method)

            naics_indices = torch.tensor([naics_to_idx[code.item()] for code in graph_data.x[:, 0]])
            naics_indices = naics_indices.to(device, dtype=torch.long)
            
            model.eval()
            with torch.no_grad(), autocast("cuda"):
                y_pred_trans  = model(
                    graph_data.x[:, 1:], 
                    graph_data.edge_index, 
                    graph_data.edge_attr,
                    naics_indices
                )
                y_pred = inverse_transform_target(y_pred_trans, trans_params, method=scale_method)
                all_val_y_true.append(y_true)
                all_val_y_pred.append(y_pred)

            del graph_data, y_pred, y_true, y_pred_trans, y_true_trans
            torch.cuda.empty_cache()

        val_y_true = torch.cat(all_val_y_true, dim=0)
        val_y_pred = torch.cat(all_val_y_pred, dim=0)
        val_mse = F.mse_loss(val_y_pred, val_y_true).item()
        val_mae = F.l1_loss(val_y_pred, val_y_true).item()
        val_r2 = r2_score(val_y_true.cpu().detach().numpy(), val_y_pred.cpu().detach().numpy())
        val_rmse = math.sqrt(val_mse)

        print(f"Epoch {epoch + 1}:")
        print(f"  Training    - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
        print(f"  Validation  - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
        
        scheduler.step(val_r2)

        # # Early stopping
        if val_r2 > best_val_loss:
            best_val_loss = val_r2
            patience_counter = 0
            # Save best model
            print(f"Saving best model at epoch {epoch + 1} with R2: {val_r2:.4f}")
            torch.save(model.state_dict(), '../models/best_model.pt')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping triggered after {epoch + 1} epochs")
        #         break
        
    all_test_y_true = []
    all_test_y_pred = []
    for file in test_files:
        graph_data = load_graph(file)
        graph_data = graph_data.to(device)

        y_true = graph_data.y
        y_true_trans, trans_params = transform_target(y_true, method=scale_method)
        
        naics_indices = torch.tensor([naics_to_idx[code.item()] for code in graph_data.x[:, 0]])
        naics_indices = naics_indices.to(device, dtype=torch.long)

        model.eval()
        with torch.no_grad(), autocast("cuda"):
            y_pred_trans = model(
                graph_data.x[:, 1:], 
                graph_data.edge_index, 
                graph_data.edge_attr,
                naics_indices
            )
            y_pred = inverse_transform_target(y_pred_trans, trans_params, method=scale_method)
            all_test_y_true.append(y_true)
            all_test_y_pred.append(y_pred)

        # Clean up memory
        del graph_data, y_pred, y_true, y_pred_trans, y_true_trans
        torch.cuda.empty_cache()
            
    test_y_true = torch.cat(all_test_y_true, dim=0)
    test_y_pred = torch.cat(all_test_y_pred, dim=0)
    mse = F.mse_loss(test_y_pred, test_y_true).item()
    mae = F.l1_loss(test_y_pred, test_y_true).item()
    r2 = r2_score(test_y_true.cpu().detach().numpy(), test_y_pred.cpu().detach().numpy())
    rmse = math.sqrt(mse)


    print(f"  Test        - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train()