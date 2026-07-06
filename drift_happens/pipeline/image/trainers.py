from functools import partial

import torch
from pydantic import BaseModel
from torch import nn

from drift_happens.model.blocks.cnn import CNNConfig
from drift_happens.model.blocks.mlp import MLPConfig
from drift_happens.model.blocks.resnet import ResNetConfig
from drift_happens.model.blocks.vit import ViTConfig
from drift_happens.model.dataset.image.architectures import (
    CNNPreset,
    ImageModelFactory,
    MLPPreset,
    ResNetPreset,
    ViTPreset,
)
from drift_happens.model.dataset.image.transfer_learning.clip import ClipConfig
from drift_happens.model.dataset.image.transfer_learning.convnext import (
    ConvNeXtConfig,
)
from drift_happens.model.dataset.image.transfer_learning.dino_v2 import DinoV2Config
from drift_happens.model.dataset.image.transfer_learning.dino_v3 import DinoV3Config
from drift_happens.model.dataset.image.transfer_learning.eva02 import EVA02Config
from drift_happens.model.dataset.image.transfer_learning.siglip import SigLIPConfig
from drift_happens.model.dataset.image.transfer_learning.timm_backbone import (
    TimmBackboneConfig,
)
from drift_happens.model.trainer.pytorch import (
    EpochPrintMode,
    PytorchTrainer,
    PytorchTrainerConfig,
)
from drift_happens.pipeline._shared.conference_defaults import (
    CONFERENCE_IMAGE_TRAINING,
)
from drift_happens.pipeline._shared.optimizers import make_optimizer_factory
from drift_happens.pipeline.training_config import TrainingConfig
from drift_happens.utils.pytorch import device_manual_mps_or_cuda_if_available


def image_model_factory(
    architecture_specific_config: BaseModel,
    img_size: int = 32,
    num_channels: int = 3,
    num_classes: int = 2,
) -> nn.Module:
    # Defaults match yearbook/imdb_faces (32×32 RGB, 2-class); a new dataset must override all three.
    if isinstance(architecture_specific_config, MLPConfig):
        return ImageModelFactory.create_mlp(
            num_input_channels=num_channels,
            height=img_size,
            width=img_size,
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, CNNConfig):
        return ImageModelFactory.create_cnn(
            num_input_channels=num_channels,
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, ResNetConfig):
        return ImageModelFactory.create_resnet(
            num_input_channels=num_channels,
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, ViTConfig):
        return ImageModelFactory.create_vit(
            num_input_channels=num_channels,
            image_size=img_size,
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, DinoV2Config):
        return ImageModelFactory.create_dinov2(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, DinoV3Config):
        return ImageModelFactory.create_dinov3(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, ClipConfig):
        return ImageModelFactory.create_clip(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, ConvNeXtConfig):
        return ImageModelFactory.create_convnext(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, SigLIPConfig):
        return ImageModelFactory.create_siglip(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, EVA02Config):
        return ImageModelFactory.create_eva02(
            num_classes=num_classes,
            config=architecture_specific_config,
        )
    if isinstance(architecture_specific_config, TimmBackboneConfig):
        return ImageModelFactory.create_timm_backbone(
            num_classes=num_classes,
            config=architecture_specific_config,
        )

    raise ValueError(f"Unknown model config type: {type(architecture_specific_config)}")


def conference_image_model_configs() -> dict[str, BaseModel]:
    """Architecture configs for the 21-model conference image lineup."""
    configs: dict[str, BaseModel] = {}
    mlp_presets: tuple[MLPPreset, ...] = ("mlp_s", "mlp_m", "mlp_l")
    for mlp_preset in mlp_presets:
        configs[mlp_preset] = ImageModelFactory._mlp_preset(mlp_preset)
    cnn_presets: tuple[CNNPreset, ...] = ("cnn_s", "cnn_m", "cnn_l")
    for cnn_preset in cnn_presets:
        configs[cnn_preset] = ImageModelFactory._cnn_preset(cnn_preset)
    resnet_presets: tuple[ResNetPreset, ...] = ("resnet_s", "resnet_m", "resnet_l")
    for resnet_preset in resnet_presets:
        configs[resnet_preset] = ImageModelFactory._resnet_preset(resnet_preset)
    vit_presets: tuple[ViTPreset, ...] = ("vit_s", "vit_m", "vit_l")
    for vit_preset in vit_presets:
        configs[vit_preset] = ImageModelFactory._vit_preset(vit_preset)
    frozen = {
        "resnet50_in_frozen": ImageModelFactory._timm_preset("resnet50_in"),
        "vit_s16_in21k_frozen": ImageModelFactory._timm_preset("vit_s16_in21k"),
        "dinov2_s_frozen": ImageModelFactory._dinov2_preset("small"),
        "dinov3_s_frozen": ImageModelFactory._dinov3_preset("small"),
        "convnext_s_frozen": ImageModelFactory._convnext_preset("small"),
        "mae_b_frozen": ImageModelFactory._timm_preset("mae_b"),
        "eva02_b_frozen": ImageModelFactory._eva02_preset("base"),
        "clip_b32_frozen": ImageModelFactory._clip_preset("base-patch32"),
        "siglip_b_frozen": ImageModelFactory._siglip_preset("base"),
    }
    for key, model_config in frozen.items():
        configs[key] = model_config.model_copy(
            update={"fine_tune": False, "needs_backend_fw_pass": False}
        )
    return configs


def is_frozen_conference_model(key: str) -> bool:
    # Convention: frozen entries carry a _frozen suffix; fine_tune=False is the structural source of truth.
    return key.endswith("_frozen")


class ConferenceImageTrainingConfig(TrainingConfig):
    architecture_specific_config: BaseModel


def conference_image_trainer_configs() -> dict[str, ConferenceImageTrainingConfig]:
    """Training configs for the 21-model conference image lineup."""
    return {
        key: ConferenceImageTrainingConfig(
            architecture_specific_config=model_config,
            batch_size=CONFERENCE_IMAGE_TRAINING.batch_size,
            learning_rate=(
                CONFERENCE_IMAGE_TRAINING.frozen_learning_rate
                if is_frozen_conference_model(key)
                else CONFERENCE_IMAGE_TRAINING.scratch_learning_rate
            ),
            num_epochs=CONFERENCE_IMAGE_TRAINING.num_epochs,
        )
        for key, model_config in conference_image_model_configs().items()
    }


def build_image_trainers_from_configs(
    trainer_configs: dict[str, ConferenceImageTrainingConfig],
    print_mode: EpochPrintMode = False,
    *,
    device: str | torch.device | None = None,
) -> dict[str, PytorchTrainer]:
    # None keeps the auto-detected device for direct module-CLI runs; the staged
    # runtime passes the device it resolved and recorded as effective_device.
    resolved_device = (
        str(device) if device is not None else device_manual_mps_or_cuda_if_available()
    )
    trainers: dict[str, PytorchTrainer] = {}

    for key, config in trainer_configs.items():
        # figure out weight decay from the model config (if available)
        if isinstance(config.architecture_specific_config, (MLPConfig, CNNConfig)):
            weight_decay = config.architecture_specific_config.weight_decay
        else:
            weight_decay = 0.0

        trainers[key] = PytorchTrainer(
            model_factory=partial(
                image_model_factory, config.architecture_specific_config
            ),
            optimizer_factory=make_optimizer_factory(
                "adam",
                learning_rate=config.learning_rate,
                weight_decay=weight_decay,
            ),
            criterion=torch.nn.CrossEntropyLoss(),
            config=PytorchTrainerConfig(
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                device=resolved_device,
            ),
            print_mode=print_mode,
        )

    return trainers
