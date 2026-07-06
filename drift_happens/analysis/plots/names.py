"""Display and filesystem-safe labels for models and slices."""

from __future__ import annotations

import re

from drift_happens.analysis.datasets import DatasetSpec


def slugify(text: str) -> str:
    """Filesystem-safe slug for a label."""
    return re.sub(r"[^0-9A-Za-z]+", "_", text)


FAMILY_LABELS = {
    "image-mlp": "MLP",
    "image-cnn": "CNN",
    "image-resnet": "ResNet",
    "image-vit": "ViT",
    "image-transfer": "Transfer",
    "text-ffn": "FFN",
    "text-ffn-regression": "FFN",
    "text-textcnn": "TextCNN",
    "text-textcnn-regression": "TextCNN",
    "text-rnn": "Recurrent",
    "text-rnn-regression": "Recurrent",
    "text-tx": "Transformer",
    "text-tx-regression": "Transformer",
    "text-frozen-head": "Frozen",
    "text-frozen-head-regression": "Frozen",
}

_TEXT_DATASETS = {"arxiv": "", "amazon_reviews_23": "-regression"}
_TEXT_RECURRENT = {"bigru", "bilstm", "bilstm_attn"}


def trainer_family(dataset: str, trainer: str) -> str:
    """Family key for ``trainer`` under ``dataset``; resolves to a ``FAMILY_LABELS``
    entry."""
    if dataset not in _TEXT_DATASETS:
        family = (
            "image-transfer"
            if trainer.endswith("_frozen")
            else f"image-{trainer.rpartition('_')[0]}"
        )
    else:
        suffix = _TEXT_DATASETS[dataset]
        if trainer.endswith("_frozen"):
            family = f"text-frozen-head{suffix}"
        else:
            architecture = trainer.rpartition("_")[0]
            base = "rnn" if architecture in _TEXT_RECURRENT else architecture
            family = f"text-{base}{suffix}"
    if family not in FAMILY_LABELS:
        raise ValueError(
            f"no family label for {dataset} trainer {trainer!r} ({family})"
        )
    return family


_ARCHITECTURES = {
    "ffn": "FFN",
    "textcnn": "TextCNN",
    "bigru": "BiGRU",
    "bilstm": "BiLSTM",
    "bilstm_attn": "BiLSTM-Attn",
    "tx": "TX",
    "mlp": "MLP",
    "cnn": "CNN",
    "resnet": "ResNet",
    "vit": "ViT",
}
_SIZES = {"s": "S", "m": "M", "l": "L"}
_BACKBONES = {
    "minilm_l6_frozen": "MiniLM-L6",
    "distilbert_base_frozen": "DistilBERT",
    "bert_base_frozen": "BERT",
    "roberta_base_frozen": "RoBERTa",
    "deberta_v3_base_frozen": "DeBERTa-v3",
    "electra_base_frozen": "ELECTRA",
    "mpnet_base_frozen": "MPNet",
    "modernbert_base_frozen": "ModernBERT",
    "resnet50_in_frozen": "ResNet50-IN",
    "vit_s16_in21k_frozen": "ViT-S16-IN21k",
    "dinov2_s_frozen": "DINOv2-S",
    "dinov3_s_frozen": "DINOv3-S",
    "convnext_s_frozen": "ConvNeXt-S",
    "mae_b_frozen": "MAE-B",
    "eva02_b_frozen": "EVA02-B",
    "clip_b32_frozen": "CLIP-B32",
    "siglip_b_frozen": "SigLIP-B",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_HALF_YEAR_EPOCH = 2000  # half-year index 0 is the first half of 2000


def get_display_name(model: str) -> str:
    """Human-readable label for a trainer key."""
    if model in _BACKBONES:
        return _BACKBONES[model]
    architecture, _, size = model.rpartition("_")
    if architecture in _ARCHITECTURES and size in _SIZES:
        return f"{_ARCHITECTURES[architecture]}-{_SIZES[size]}"
    return model.replace("_", " ").title()


def figure_name(model: str) -> str:
    return _UNSAFE.sub("-", get_display_name(model))


def slice_label(label: str, spec: DatasetSpec) -> str:
    if spec.slice_noun != "half-year":
        return label
    try:
        value = int(label)
    except ValueError:
        return label
    return half_year_label(value)


def half_year_year(value: int) -> int:
    """Calendar year of an Amazon half-year index (0 -> 2000, 2 -> 2001)."""
    return _HALF_YEAR_EPOCH + value // 2


def half_year_label(value: int) -> str:
    """Decode an Amazon half-year index: 0 -> '2000-H1', 37 -> '2018-H2'."""
    return f"{half_year_year(value)}-H{value % 2 + 1}"
