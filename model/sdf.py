from typing import Optional, Callable

import numpy as np
import torch
from torch import Tensor
from torch import nn

from utils.logging import get_predefined_logger



logger = get_predefined_logger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding module which encode 3D coordinates to frequency domain.

    - Input tensor: Nx3
    - Output tensor: Nx(3+6F)

    Where N is number of points, F is number of frequencies.
    """
    def __init__(self):
        super().__init__()
        self.basis = [-3, -2, -1, 0, 1, 2]
        self.n_frequencies = len(self.basis)
        self.out_dim = 3 + 2 * 3 * self.n_frequencies

    def forward(self, coordinates: Tensor, alpha: float):
        # Compute weights for all frequency bands.
        device = coordinates.device
        freq_indices = torch.arange(self.n_frequencies, device=device)
        progress = torch.clamp(alpha * self.n_frequencies - freq_indices, 0, 1)
        weights = 0.5 * (1 - torch.cos(progress * torch.pi))
        # Compute input for triangular functions.
        basis_exps = torch.tensor(self.basis, dtype=torch.float32, device=device)
        scaled_coords = (2 ** basis_exps).view(1, self.n_frequencies, 1) * torch.pi * coordinates.unsqueeze(1)
        # Compute weighted sin/cos values.
        weights = weights.view(1, self.n_frequencies, 1)
        sin_values = weights * torch.sin(scaled_coords)
        cos_values = weights * torch.cos(scaled_coords)
        # Concatenate sin/cos values and flat each point's encoding result to a 6 * n_frequencies vector.
        frequencies = torch.cat((sin_values, cos_values), dim=-1).view(coordinates.shape[0], -1)
        return torch.cat((coordinates, frequencies), dim=-1)

class SignedDistancePredictor(nn.Module):
    """
    This network fits an SDF (signed distance field) of the object to be
    reconstructed.

    - Input: A tensor with shape of (Q, 3) containing all query points.
    - Output: A tensor with shape of (Q, 1) representing signed distance.
    """
    def __init__(self, radius, init_alpha: float, max_batches: int, n_dim_hidden: int=512):
        """
        :param radius: Radius of sphere created by GNI.
        :param init_alpha: Initialize value of alpha (controlling the progress
            of positional encoding).
        :param max_batches: Number of batches to train.
        :param n_dim_hidden: Number of dimensions of the hidden layers' output.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=False)
        self.max_batches = nn.Parameter(torch.tensor(max_batches), requires_grad=False)
        self.positional_encoding = PositionalEncoding()
        self.extract_feature_query = nn.Sequential(
            nn.Linear(self.positional_encoding.out_dim, n_dim_hidden),
            nn.Softplus(beta=100)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_dim_hidden, n_dim_hidden),
            nn.Softplus(beta=100)
        )
        for i in range(7):
            self.linear_relu_stack.append(nn.Linear(n_dim_hidden, n_dim_hidden))
            self.linear_relu_stack.append(nn.Softplus(beta=100))
        self.calculate_distance = nn.Linear(n_dim_hidden, 1)
        # Initialize the MLP with Geometric Network Initialization (GNI) technique.
        for layer in self.extract_feature_query.modules():
            if isinstance(layer, nn.Linear):
                # Initially, all the weights multiplied to positional encoding
                # results will be zero, because the GNI algorithm only specifies
                # how to initialize a network taking (x, y, z) as its input.
                # This configuration is referenced from HF-NeuS.
                nn.init.normal_(layer.weight[:, :3], mean=0.0, std=np.sqrt(2) / np.sqrt(n_dim_hidden))
                nn.init.zeros_(layer.weight[:, 3:])
                nn.init.zeros_(layer.bias)
        for layer in self.linear_relu_stack.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(n_dim_hidden))
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.calculate_distance.weight, np.sqrt(np.pi) / np.sqrt(n_dim_hidden), std=1e-6)
        nn.init.constant_(self.calculate_distance.bias, -radius)

    def forward(
            self, query_points: torch.Tensor
        ) -> torch.Tensor:
        """
        :param query_points: A tensor with shape of (Q, 3) containing all query
            points.
        :returns: A tensor SDF is  with shape of (Q, 1), the last dimension is
            retained only for alignment.
        """
        if self.training:
            self.alpha += 1 / self.max_batches
        encoded_points = self.positional_encoding(query_points, self.alpha.item())
        feature_query = self.extract_feature_query(encoded_points)
        features = self.linear_relu_stack(feature_query)
        signed_distance = self.calculate_distance(features)

        return signed_distance

class SignedDistancePredictorNoPE(nn.Module):
    """
    This network fits a SDF (signed distance field) of the object to be
    reconstructed.

    - Input: A tensor with shape of (Q, 3) containing all query
        points.
    - Output: A tensor SDF with shape of (Q, 1), the last dimension is
        retained only for alignment.
    """
    def __init__(self, radius, n_dim_hidden: int=512):
        """
        :param radius: Radius of sphere created by GNI.
        :param n_dim_hidden: Number of dimensions of the hidden layers' output.
        """
        super().__init__()
        self.extract_feature_query = nn.Sequential(
            nn.Linear(3, n_dim_hidden),
            nn.Softplus(beta=100)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_dim_hidden, n_dim_hidden),
            nn.Softplus(beta=100)
        )
        for i in range(7):
            self.linear_relu_stack.append(nn.Linear(n_dim_hidden, n_dim_hidden))
            self.linear_relu_stack.append(nn.Softplus(beta=100))
        self.calculate_SDF = nn.Linear(n_dim_hidden, 1)
        # Initialize the MLP with Geometric Network Initialization (GNI) technique.
        for i, layer in enumerate(self.extract_feature_query.modules()):
            if isinstance(layer, nn.Linear):
                # nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(n_dim_hidden))
                # nn.init.zeros_(layer.bias)
                nn.init.normal_(layer.weight[:, :3], mean=0.0, std=np.sqrt(2) / np.sqrt(n_dim_hidden))
                nn.init.zeros_(layer.weight[:, 3:])
                nn.init.zeros_(layer.bias)
        for layer in self.linear_relu_stack.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(n_dim_hidden))
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.calculate_SDF.weight, np.sqrt(np.pi) / np.sqrt(n_dim_hidden), std=1e-6)
        nn.init.constant_(self.calculate_SDF.bias, -radius)

    def forward(
            self, query_points: torch.Tensor
        ) -> torch.Tensor:
        """
        :param query_points: A tensor with shape of (Q, 3) containing all query
            points.
        :return: A tensor SDF is  with shape of (Q, 1), the last dimension is
            retained only for alignment.
        """
        feature_query = self.extract_feature_query(query_points)
        features = self.linear_relu_stack(feature_query)
        SDF = self.calculate_SDF(features)

        return SDF
