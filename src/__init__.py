"""DeepGate-FaceRec: Face Recognition via Deep Metric & Embedding Learning."""

from src import data, losses, metrics, mining, models
from src.data import (
    PKBatchSampler,
    build_identity_index_map,
    get_eval_transforms,
    get_train_transforms,
)
from src.dummy_functions import dummy_function
from src.losses import TripletMarginLoss
from src.models import FaceEmbeddingNet

__version__ = "0.1.0"

__all__ = [
    "FaceEmbeddingNet",
    "PKBatchSampler",
    "TripletMarginLoss",
    "__version__",
    "build_identity_index_map",
    "data",
    "dummy_function",
    "get_eval_transforms",
    "get_train_transforms",
    "losses",
    "metrics",
    "mining",
    "models",
]
