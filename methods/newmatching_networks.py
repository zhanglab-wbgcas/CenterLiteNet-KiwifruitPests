#只增加了inter_class_distance_weight 参数：添加了一个用于控制类间距离最大化损失的权重参数 inter_class_distance_weight
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from easyfsl.modules.predesigned_modules import (
    default_matching_networks_query_encoder,
    default_matching_networks_support_encoder,
)
from .few_shot_classifier import FewShotClassifier

class NEWMatchingNetworks(FewShotClassifier):
    def __init__(
        self,
        *args,
        feature_dimension: int,
        support_encoder: Optional[nn.Module] = None,
        query_encoder: Optional[nn.Module] = None,
        inter_class_distance_weight: float = 0.1,  # 权重参数
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension
        self.inter_class_distance_weight = inter_class_distance_weight  # 权重初始化

        # These modules refine support and query feature vectors using information from the whole support set
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

        # Compute class centers
        self.class_centers = self.compute_class_centers(support_features, support_labels)

    def compute_class_centers(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the center of each class based on support features.
        Args:
            features: Tensor of shape (n_support, feature_dimension)
            labels: Tensor of shape (n_support,)
        Returns:
            Tensor of shape (n_classes, feature_dimension) representing the centers of each class.
        """
        unique_labels = labels.unique()
        centers = torch.stack([
            features[labels == label].mean(dim=0)
            for label in unique_labels
        ])
        return centers

    def inter_class_distance_maximization_loss(self) -> Tensor:
        """
        Compute the loss that maximizes the inter-class distance.
        Returns:
            A scalar Tensor representing the inter-class distance maximization loss.
        """
        n_classes = self.class_centers.size(0)
        distance_sum = 0.0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                distance_sum += torch.norm(self.class_centers[i] - self.class_centers[j])

        # Normalize by the number of pairs
        return distance_sum / (n_classes * (n_classes - 1) / 2)

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

        # Add inter-class distance maximization loss
        inter_class_loss = self.inter_class_distance_maximization_loss()
        total_loss = log_probabilities + self.inter_class_distance_weight * inter_class_loss

        return self.softmax_if_specified(total_loss)
    
    def encode_support_features(
        self,
        support_features: Tensor,
    ) -> Tensor:
        """
        Refine support set features by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_features: output of the backbone of shape (n_support, feature_dimension)

        Returns:
            contextualised support features, with the same shape as input features
        """

        # Since the LSTM is bidirectional, hidden_state is of the shape
        # [number_of_support_images, 2 * feature_dimension]
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0)

        # Following the paper, contextualized features are computed by adding original features, and
        # hidden state of both directions of the bidirectional LSTM.
        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

        return contextualized_support_features

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        """
        Refine query set features by putting them in the context of the whole support set,
        using attention over support set features.
        Args:
            query_features: output of the backbone of shape (n_query, feature_dimension)

        Returns:
            contextualized query features, with the same shape as input features
        """

        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        # We do as many iterations through the LSTM cell as there are query instances
        # Check out the paper for more details about this!
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

    def _validate_features_shape(self, features: Tensor):
        self._raise_error_if_features_are_multi_dimensional(features)
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False

