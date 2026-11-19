"""Embedding models for face recognition and deep metric learning."""

from typing import Literal

import torch
import torch.nn as nn


class FaceEmbeddingNet(nn.Module):
    """Backbone neural network with projection head for hypersphere metric embeddings.

    Adapts a vision backbone (e.g. ResNet, MobileNet) to output an L2-normalized
    d-dimensional feature vector on the unit hypersphere.

    Attributes:
        backbone_name: Name of the convolutional architecture.
        embedding_dim: Dimensionality of the target metric space (d).
        backbone: The feature extraction network.
        projection_head: Linear layer mapping backbone features to embedding_dim.

    Note:
        The models backbone can be taken from torchvision,
        The following code can be used as example for resnet18.

        ```python
        import torchvision.models as models
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        ```
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "mobilenet_v3_small"] = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initializes FaceEmbeddingNet.

        Args:
            backbone: Choice of vision backbone ('resnet18' or 'mobilenet_v3_small').
            embedding_dim: Dimension of the output embedding vector.
            pretrained: If True, initializes with ImageNet pre-trained weights.
            freeze_backbone: If True, freezes backbone layers and trains only the head.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and L2-normalizes face embeddings.

        Args:
            x: Input face images of shape (B, 3, H, W).

        Returns:
            L2-normalized embeddings of shape (B, embedding_dim).
        """
        raise NotImplementedError
