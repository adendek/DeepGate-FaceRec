"""Dataset loaders, image transforms, and identity-aware PK batch sampling."""

from collections.abc import Iterator, Sequence
from typing import Protocol

import torchvision.transforms as transforms
from torch.utils.data import Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _HasTargets(Protocol):
    """Structural protocol for datasets exposing integer class labels via `.targets`."""

    targets: Sequence[int]


def get_train_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns data augmentation transforms for training face embeddings.

    Args:
        image_size: Target ``(height, width)`` for model input. Defaults to ``(128, 128)``.

    Returns:
        Composed torchvision transform pipeline producing normalized ``(3, H, W)`` tensors.

    Notes:
        - **Why horizontal flips are safe here:** Unlike digit or text classification, human
          faces are approximately bilaterally symmetric—a horizontally flipped face still
          belongs to the same identity, effectively doubling pose variety.
        - **Gentle photometric jitter:** ``ColorJitter(brightness=0.1, contrast=0.1)``
          simulates mild camera exposure and lighting shifts without distorting skin tones
          or facial geometry.
        - **ImageNet statistics:** Normalizing with ``IMAGENET_MEAN`` and ``IMAGENET_STD``
          aligns input pixel distributions with pretrained ``resnet18`` and
          ``mobilenet_v3_small`` backbones.
    """
    raise NotImplementedError


def get_eval_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns deterministic preprocessing transforms for validation, gallery, and probe.

    Args:
        image_size: Target ``(height, width)`` for model input. Defaults to ``(128, 128)``.

    Returns:
        Composed torchvision transform pipeline producing normalized ``(3, H, W)`` tensors.

    Notes:
        - **Determinism:** Random augmentations are omitted so gallery templates and probe
          queries map to reproducible points on the embedding hypersphere.
    """
    raise NotImplementedError


def build_identity_index_map(dataset: _HasTargets) -> dict[int, list[int]]:
    """Builds a mapping from identity class label to sample indices in the dataset.

    Args:
        dataset: Any dataset (such as ``torchvision.datasets.ImageFolder``) exposing
            integer identity labels in ``dataset.targets``.

    Returns:
        Dictionary mapping ``identity_id -> list[dataset_index]``.
    """
    raise NotImplementedError


class PKBatchSampler(Sampler[list[int]]):
    """Samples P distinct identities with K images each per mini-batch.

    Produces batches of size ``P * K``.

    Args:
        identity_to_indices: Mapping from identity ID to non-empty sample index list.
        p: Number of distinct identities per batch (must be ``> 1``).
        k: Number of images sampled per identity (must be ``> 1``).
        batches_per_epoch: Number of batches to yield per epoch. If ``None``,
            defaults to ``max(1, len(identity_to_indices) // p)``.
        seed: Optional random seed for reproducible batch sampling.

    Raises:
        ValueError: If ``p <= 1``, ``k <= 1``, ``batches_per_epoch <= 0``, fewer than ``p``
            identities are available, or any identity has an empty index list.

    Notes:
        - **Why standard random shuffling fails for triplet mining:** With thousands of
          identities, a uniform random batch of size 32 almost never contains two photos of
          the same person.
        - **Sampling with vs. without replacement:** If an identity has at least ``K``
          images, ``K`` distinct images are drawn without replacement. If an identity has
          fewer than ``K`` images (e.g., long-tail datasets where some people have only 1-2
          photos), images are sampled *with* replacement so the batch shape remains fixed
          at ``P * K``.
        - **Usage with PyTorch ``DataLoader``:** Pass this sampler via ``batch_sampler=...``
          (and leave ``batch_size``, ``shuffle``, and ``drop_last`` at their defaults).
    """

    def __init__(
        self,
        identity_to_indices: dict[int, list[int]],
        p: int,
        k: int,
        batches_per_epoch: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Init the PKBatchSampler."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the total number of batches in one epoch."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[list[int]]:
        """Yields batches of dataset indices of length ``P * K``."""
        raise NotImplementedError
