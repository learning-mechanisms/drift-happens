import hashlib
from functools import partial
from pathlib import Path
from typing import cast

from torch.utils.data import TensorDataset

from drift_happens.configs import ExperimentConfig
from drift_happens.configs.experiment import CacheReusePolicy, CacheSpec
from drift_happens.dataset.dataset import TensorDatasetCache
from drift_happens.model.dataset.image.transfer_learning.base import (
    TransferLearningModel,
)
from drift_happens.model.dataset.image.transfer_learning.embedding_head import (
    CachedEmbeddingHead,
)
from drift_happens.model.trainer.pytorch import PytorchTrainer
from drift_happens.pipeline.context import PipelineContext
from drift_happens.utils.log import get_logger

logger = get_logger()

# --------------------------------------- SETUP -------------------------------------- #


def _embedding_cache_id(
    dataset_id: str,
    key: str,
    model: TransferLearningModel,
    dataset: TensorDataset,
) -> str:
    """
    Content-addressed id for the cached embeddings.

    Folds the backbone identity (model class + config) and a hash of the exact input and
    label tensors into the cache id, so any change to the backbone, its config, or the
    rows — a relabeling included — yields a new cache file instead of silently reusing
    stale embeddings.
    """
    hasher = hashlib.sha256()
    for tensor in dataset.tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        # Length-framed like dataset.cache.content_fingerprint, so shifting
        # bytes across tensor boundaries cannot make different inputs collide.
        descriptor = f"{array.dtype}:{array.shape}".encode()
        hasher.update(len(descriptor).to_bytes(8, "big"))
        hasher.update(descriptor)
        hasher.update(array.data)
    cache_id = CacheSpec.cache_id_from_data(
        {
            "dataset": dataset_id,
            "input_version": hasher.hexdigest(),
            "kind": "image_embedding",
            "output": "pooled",
            "params": model.config.model_dump(mode="json"),
            # The class name is load-bearing: e.g. DINOv2 and DINOv3 share an
            # empty config subclass, so it is the only field that tells them
            # apart. Renaming a model class deliberately invalidates its cache.
            "producer": type(model).__name__,
            # Bump when a change the fields above cannot see alters the
            # embeddings: a backbone's model_map checkpoint mapping, its image
            # size / resize, or the forward_only_backend math.
            "schema_version": 1,
        }
    )
    return f"{key}_{cache_id}"


def reuse_policy_from_config(
    experiment_config: ExperimentConfig | None,
) -> CacheReusePolicy:
    """Cache reuse policy for the run, defaulting to ``reuse`` when unconfigured."""
    if experiment_config is None or experiment_config.preprocessing.cache is None:
        return "reuse"
    return experiment_config.preprocessing.cache.reuse_policy


def embed_dataset_if_needed(
    ctx: PipelineContext,
    trainer: PytorchTrainer,
    key: str,
    *,
    dataset_cache_dir: Path,
    dataset_id: str,
    reuse_policy: CacheReusePolicy = "reuse",
) -> TensorDataset:
    dataset_cache = TensorDatasetCache(
        cache_dir=dataset_cache_dir, dataset_id=dataset_id
    )
    base_dataset = cast(TensorDataset, ctx.tensor_dataset)

    model = trainer._model
    if (
        isinstance(model, TransferLearningModel)
        and not model.config.needs_backend_fw_pass
    ):
        logger.info(f"Embedding dataset for model '{key}'...")

        # Cache id is content-addressed; see _embedding_cache_id.
        tensor_dataset = dataset_cache.get_create_cache_embedding(
            embed_id=_embedding_cache_id(dataset_id, key, model, base_dataset),
            embedding_fn=partial(
                model.forward_only_backend_batched,
                dataset=base_dataset,
            ),
            reuse_policy=reuse_policy,
        )
        _use_cached_embedding_head(trainer, model, tensor_dataset)
    else:
        # Use the original
        logger.info(f"Using original dataset for model '{key}'...")
        tensor_dataset = base_dataset

    return tensor_dataset


def _use_cached_embedding_head(
    trainer: PytorchTrainer,
    transfer_model: TransferLearningModel,
    tensor_dataset: TensorDataset,
) -> None:
    features = tensor_dataset.tensors[0]
    if features.ndim != 2:
        raise ValueError(
            "cached transfer-learning embeddings must be a 2-D tensor; "
            f"got shape {tuple(features.shape)}"
        )

    classifier = getattr(transfer_model, "classifier", None)
    if classifier is None or not hasattr(classifier, "out_features"):
        raise TypeError(
            "cached transfer-learning models must expose a classifier with "
            "an out_features attribute"
        )

    input_dim = int(features.shape[1])
    num_classes = int(classifier.out_features)
    trainer.replace_model_factory(
        lambda: CachedEmbeddingHead(input_dim=input_dim, num_classes=num_classes)
    )
