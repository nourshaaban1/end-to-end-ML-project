import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class ChurnClassifier(nn.Module):
    """
    Neural Network classifier for customer churn prediction.
    Input: Preprocessed features (num + encoded cat)
    Output: Churn probability
    """

    def __init__(self, input_features: int, hidden_layers: list = None, dropout: float = 0.3):
        super(ChurnClassifier, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        layers = []
        input_dim = input_features

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        layers.extend([
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def save(self, file_path: Path):
        """Save the model state dict."""
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load(cls, file_path: Path, input_features: int):
        """Load the model state dict."""
        model = cls(input_features=input_features)
        model.load_state_dict(torch.load(file_path, map_location='cpu'))
        model.eval()
        return model


def create_torch_dataloader(X: np.ndarray, y: np.ndarray = None, batch_size: int = 32, shuffle: bool = False):
    """
    Create a PyTorch DataLoader from numpy arrays.

    Args:
        X: Feature matrix
        y: Target vector (optional, for training)
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    X_tensor = torch.FloatTensor(X)

    if y is not None:
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(X_tensor)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
