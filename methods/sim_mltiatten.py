from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

from easyfsl.modules import MultiHeadAttention
# from easyfsl.modules.feat_resnet12 import feat_resnet12

from .simple_shot import SimpleShot
from .utils import strip_prefix


class Simplechange(SimpleShot):
    """
    Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, Fei Sha.
    "Few-Shot Learning via Embedding Adaptation With Set-to-Set Functions" (CVPR 2020)
    https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.html

    This method uses an episodically trained attention module to improve the prototypes.
    Queries are then classified based on their euclidean distance to the prototypes,
    as in Prototypical Networks.
    This in an inductive method.

    The attention module must follow specific constraints described in the docstring of FEAT.__init__().
    We provide a default attention module following the one used in the original implementation.
    FEAT can be initialized in the default configuration from the authors, by calling FEAT.from_resnet12_checkpoint().
    """

    def __init__(self, *args, attention_module: nn.Module, **kwargs):
        """
        FEAT needs an additional attention module.
        Args:
            *args:
            attention_module: the forward method must accept 3 Tensor arguments of shape
                (1, num_classes, feature_dimension) and return a pair of Tensor, with the first
                one of shape (1, num_classes, feature_dimension).
                This follows the original implementation of https://github.com/Sha-Lab/FEAT
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.attention_module = attention_module

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract prototypes from support set and rectify them with the attention module.
        Args:
            support_images: support images of shape (n_support, **image_shape)
            support_labels: support labels of shape (n_support,)
        """
        super().process_support_set(support_images, support_labels)
        self.prototypes = self.attention_module(
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
        )[0][0]

    