import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. PYTORCH DATASET WRAPPER
# =============================================================================
class LTPDataset(Dataset):
    """PyTorch Dataset wrapper for the Low-Temperature Plasma (LTP) data."""
    def __init__(self, X, y):
        # Efficiently convert numpy arrays to PyTorch float32 tensors
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 2. DATA LOADING, SPLITTING, AND SCALING
# =============================================================================
def prepare_data_model(data_path, ratio_test_val_train, data_split_seed=42):
    """
    Prepares the dataset with a STRICTLY FIXED random seed (data_split_seed).
    This ensures all models are evaluated on the exact same train/val/test splits.
    
    Returns:
        A dictionary containing the scikit-learn scalers and the scaled train, val, and test sets.
    """
    # Load raw text data
    with open(data_path, 'r') as f:
        data = f.readlines()

    for i in range(len(data)):
        data[i] = data[i].split()
        data[i] = [float(x) for x in data[i]]
        
    data = np.array(data)

    # Split data using the fixed data_split_seed (NOT the model's random seed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data[:, 0:3], data[:, 3:], 
        test_size=ratio_test_val_train, 
        random_state=data_split_seed
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.7, 
        random_state=data_split_seed
    )

    # Initialize and fit standard scalers
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_Y.fit_transform(y_train)

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_Y.transform(y_val)

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_Y.transform(y_test)

    return {
        'scalers': (scaler_X, scaler_Y), 
        'train': (X_scaled, y_scaled),
        'val': (X_val_scaled, y_val_scaled),
        'test': (X_test_scaled, y_test_scaled)
    }