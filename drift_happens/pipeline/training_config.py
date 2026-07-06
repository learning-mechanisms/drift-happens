from pydantic import BaseModel, ConfigDict


class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: int
    learning_rate: float
    num_epochs: int
