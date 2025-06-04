from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor



class MLPEncoder(nn.Module):
    """
    The encoder used by the ODF (On-surface Decision Function) network.
    This encoder takes KNN (K-Nearest Neighbor) of each query point to extract
    a feature for each query point.

    - Input tensor's shape: (Q, K, 3)
    - Output tensor's shape: (Q, 512)

    Where Q stands for number of query points and K stands for number of nearest
    neighbors.
    """
    def __init__(self, K: int, n_hidden_dims: int=512, n_layers: int=8) -> None:
        """
        :param K: Number of nearest neighbors.
        """
        super().__init__()
        self.K = K
        self.flatten = nn.Flatten(1, 2)
        self.ascend = nn.Sequential(
            nn.Linear(K * 3, n_hidden_dims),
            nn.ReLU()
        )
        self.linear_relu_stack = nn.Sequential()
        for i in range(n_layers):
            self.linear_relu_stack.append(nn.Linear(n_hidden_dims, n_hidden_dims))
            self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Linear(n_hidden_dims, n_hidden_dims))
    
    def forward(self, KNNs: Tensor) -> Tensor:
        """
        :param KNNs: A tensor with shape of (Q, K, 3), containing KNN of each
            query point. Meaning of Q and K is described in this class's docstring.
        """
        KNNs = self.flatten(KNNs)
        features = self.ascend(KNNs)
        features = self.linear_relu_stack(features)
        return features

class PointNetEncoder(nn.Module):
    def __init__(self, K: int, n_hidden_dims: int=512):
        super().__init__()
        self.K = K
        #self.extract_point_feature_1 = nn.Sequential(
        #    nn.Conv1d(4, n_hidden_dims // 4, 1),
        #    nn.ReLU(),
        #    nn.Conv1d(n_hidden_dims // 4, n_hidden_dims // 2, 1)
        #)
        #self.extract_point_feature_2 = nn.Sequential(
        #    nn.Conv1d(n_hidden_dims, n_hidden_dims, 1),
        #    nn.ReLU(),
        #    nn.Conv1d(n_hidden_dims, n_hidden_dims, 1)
        #)
        self.extract_point_feature = nn.Sequential(
            nn.Conv1d(4, n_hidden_dims // 2, 1),
            nn.ReLU(),
            nn.Conv1d(n_hidden_dims // 2, n_hidden_dims, 1)
        )
    
    def forward(self, KNNs: Tensor) -> Tensor:
        distances = (KNNs * KNNs).sum(axis=2, keepdim=True)
        KNNs = torch.cat([KNNs, distances], dim=2)
        # transpose KNNs to (Q, 3, K)
        KNNs = torch.transpose(KNNs, 1, 2)
        # extract point featrues to (Q, 256, K)
        #point_features = self.extract_point_feature_1(KNNs)
        ## perform max pooling to get KNN feature (Q, 256, 1)
        #KNN_features_1, _ = torch.max(point_features, dim=2, keepdim=True)
        ## repeat KNN feature and append it to point features to get (Q, 512, K)
        #point_features = torch.cat([point_features, torch.tile(KNN_features_1, (1, 1, self.K))], dim=1)
        ## extract point features to (Q, 512, K)
        #point_features = self.extract_point_feature_2(point_features)
        ## perform max pooling to get final KNN feature (Q, 512)
        #KNN_features_2, _ = torch.max(point_features, dim=2)
        #return KNN_features_2
        point_features = self.extract_point_feature(KNNs)
        KNN_features, _ = torch.max(point_features, dim=2)
        return KNN_features

class OnSurfaceDecoder(nn.Module):
    """
    The decoder used by the ODF (On-surface Decision Function) network. This
    decoder takes the feature produced by PointNetEncoder to calculate unsigned
    distance to the local surface for each query point.

    - Input tensor's shape: (Q, 512)
    - Output tensor's shape: (Q, 1)

    Where Q stands for number of query points.
    """
    def __init__(self, n_hidden_dims: int=512):
        super().__init__()
        self.linear_relu_stack = nn.Sequential()
        for i in range(9):
            self.linear_relu_stack.append(nn.Linear(n_hidden_dims, n_hidden_dims))
            self.linear_relu_stack.append(nn.ReLU())
        self.calculate_UDF = nn.Sequential(
            nn.Linear(n_hidden_dims, 1),
            nn.ReLU()
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: A tensor with shape of (Q, 512) representing features
            of each query point. Where Q is the number of query points.
        :return: A tensor with shape of (Q, 1) representing unsigned distance
            to the local surface for each query point.
        """
        features = self.linear_relu_stack(features)
        UDF = self.calculate_UDF(features)
        return UDF

class NearestPointDecoder(nn.Module):
    """
    This decoder takes the KNN feature to predict the nearest point on the local
    surface represented by KNN.

    - Input tensor's shape: (Q, 512)
    - Output tensor's shape: (Q, 3)

    Where Q stands for number of query points.
    """
    def __init__(self, n_hidden_dims: int=512, n_layers: int=8):
        super().__init__()
        self.linear_relu_stack = nn.Sequential()
        for i in range(n_layers):
            self.linear_relu_stack.append(nn.Linear(n_hidden_dims, n_hidden_dims))
            self.linear_relu_stack.append(nn.ReLU())
        self.calculate_pos = nn.Linear(n_hidden_dims, 3)

    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: A tensor with shape of (Q, 512) representing features
            of each query point. Where Q is the number of query points.
        :return: A tensor with shape of (Q, 3) representing the nearest point
            on the local surface for each query point.
        """
        features = self.linear_relu_stack(features)
        position = self.calculate_pos(features)
        return position

class OnSurfaceDecisionFunction(nn.Module):
    """
    The On-surface Decision Function network. This network should be pre-trained
    before reconstruction. Its output is unsigned distance from the query
    point to local surface, which will be used as loss value of the SDF network.
    """
    def __init__(self, K: int):
        """
        :param K: Number of nearest neighbors.
        """
        super().__init__()
        self.encoder = MLPEncoder(K, n_hidden_dims=512)
        self.decoder = OnSurfaceDecoder()

    def forward(self, KNNs: Tensor) -> Tensor:
        """
        :param KNNs: A tensor with shape of (Q, K, 3), containing KNN of each
            query point. Where Q is the number of query points and K is the
            number of nearest neighbors.
        """
        features = self.encoder(KNNs)
        UDF = self.decoder(features)
        return UDF

class NearestPointPredictor(nn.Module):
    """
    This network predict the nearest point on a local surface represented by
    KNN for each query point.
    """
    def __init__(self, K: int):
        """
        :param K: Number of nearest neighbors as input.
        """
        super().__init__()
        self.encoder = PointNetEncoder(K, n_hidden_dims=512)
        self.decoder = NearestPointDecoder(n_hidden_dims=512, n_layers=3)

    def forward(self, KNNs: Tensor) -> Tensor:
        """
        :param KNNs: A tensor with shape of (Q, K, 3), containing KNN of each
            query point. Where Q is the number of query points and K is the
            number of nearest neighbors.
        """
        features = self.encoder(KNNs)
        position = self.decoder(features)
        return position
