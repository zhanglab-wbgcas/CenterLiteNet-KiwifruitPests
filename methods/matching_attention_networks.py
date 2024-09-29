

from typing import Optional

import torch
from torch import Tensor, nn

from easyfsl.modules.predesigned_modules import (
    default_matching_networks_query_encoder,
    default_matching_networks_support_encoder,
)

from .few_shot_classifier import FewShotClassifier


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation, gamma):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = gamma

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out, attention


class Matching_attenNet(FewShotClassifier):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf

    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.

    Be careful: while some methods use Cross Entropy Loss for episodic training, Matching Networks
    output log-probabilities, so you'll want to use Negative Log Likelihood Loss.
    """

    def __init__(
        self,
        *args,
        feature_dimension: int,
        support_encoder: Optional[nn.Module] = None,
        query_encoder: Optional[nn.Module] = None,
        node_indices: list = [4, 19, 35, 51, 67],
        weightlearnable: bool = True,
        weight: list = [1.0, 1.0, 1.0, 1.0, 1.0],
        gamma: float = 1.0,
        **kwargs,
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: dimension of the feature vectors extracted by the backbone.
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.
        """
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension

        self.support_features_encoder = (
            support_encoder
            if support_encoder
            else default_matching_networks_support_encoder(self.feature_dimension)
        )
        self.query_features_encoding_cell = (
            query_encoder
            if query_encoder
            else default_matching_networks_query_encoder(self.feature_dimension)
        )

        self.softmax = nn.Softmax(dim=1)

        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

        self.node_indices = node_indices
        self.weightlearnable = weightlearnable
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor([w])) for w in weight])
        self.gamma1 = nn.Parameter(torch.Tensor([gamma]))
        self.gamma2 = nn.Parameter(torch.Tensor([gamma]))

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        support_features = self.compute_features(support_images)
        self._validate_features_shape(support_features)
        self.contextualized_support_features = self.encode_support_features(
            support_features
        )

        self.one_hot_support_labels = (
            nn.functional.one_hot(  # pylint: disable=not-callable
                support_labels
            ).float()
        )

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        self._validate_features_shape(query_features)
        contextualized_query_features = self.encode_query_features(query_features)

        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                nn.functional.normalize(self.contextualized_support_features).T
            )
        )

        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def encode_support_features(
        self,
        support_features: Tensor,
    ) -> Tensor:
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0)

        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

        return contextualized_support_features

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)

            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state

    def calculate(self, images: torch.Tensor, node_idx: int) -> torch.Tensor:
        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        feature_extractor = create_feature_extractor(self.backbone, return_nodes=[str(node_idx)])
        out = feature_extractor(images)['0']
        input_size = out  # Batch size, number of channels, sequence length
        n_channels = input_size.size()[1]
        self_attention = Self_Attn(in_dim=n_channels, activation='relu', gamma=self.gamma1)
        output, _ = self_attention(input_size)
        out_pooled = global_avg_pool(output)
        features = out_pooled.view(out_pooled.size(0), -1)
        return features

    def calculate_distances(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        distances = [self.calculate(support_images, idx) for idx in self.node_indices]
        query_distances = [self.calculate(query_images, idx) for idx in self.node_indices]
        return distances, query_distances

    def forward_with_attention(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor) -> torch.Tensor:
        support_distances, query_distances = self.calculate_distances(support_images, support_labels, query_images)
        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [support_distances[i][torch.nonzero(support_labels == label)].mean(0) for i in range(len(support_distances)) for label in range(n_way)]
        )
        dists = torch.cdist(query_distances, z_proto)
        if self.weightlearnable:
            scores = -sum(self.weights[i] * dists[i] for i in range(len(dists)))
        else:
            scores = -sum(self.weight[i] * dists[i] for i in range(len(dists)))
        return scores

    def _validate_features_shape(self, features: Tensor):
        self._raise_error_if_features_are_multi_dimensional(features)
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False
