"""
Construct the conference image trainers for the yearbook dataset.

The yearbook lineup is exactly the shared conference image lineup; the aliases below
keep the per-dataset names that run.py, the contexts and the tests import.
"""

from drift_happens.pipeline.image.trainers import (
    ConferenceImageTrainingConfig,
    build_image_trainers_from_configs,
    conference_image_trainer_configs,
)

YearbookTrainingConfig = ConferenceImageTrainingConfig

yearbook_conference_trainer_configs = conference_image_trainer_configs

build_trainers_from_configs = build_image_trainers_from_configs
