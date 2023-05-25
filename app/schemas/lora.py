from typing import Optional

from diffusers.loaders import AttnProcsLayers
from pydantic import BaseModel, Field
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


class LoraTrainingArguments(BaseModel):
    """
    A Pydantic model for handling LoRa training arguments.

    Attributes:
    dataset_name (str): The name of the dataset (from the HuggingFace hub) to train on or a path to a local dataset or folder.
    dataset_config_name (Optional[str]): The config of the dataset, leave as None if there's only one config.
    image_column (str): The column of the dataset containing an image.
    caption_column (str): The column of the dataset containing a caption or a list of captions.
    max_train_samples (Optional[int]): Maximum number of training examples to use for debugging purposes or quicker training.
    seed (Optional[int]): A seed for reproducible training.
    resolution (int): The resolution for input images, all images in the train/validation dataset will be resized to this resolution.
    center_crop (bool): Whether to center crop the input images to the resolution after resizing.
    random_flip (bool): Whether to randomly flip images horizontally.
    train_batch_size (int): Batch size (per device) for the training dataloader.
    num_train_epochs (int): Number of training epochs.
    max_train_steps (Optional[int]): Total number of training steps to perform, overrides num_train_epochs if provided.
    gradient_accumulation_steps (int): Number of update steps to accumulate before performing a backward/update pass.
    gradient_checkpointing (bool): Whether to use gradient checkpointing to save memory at the expense of slower backward pass.
    learning_rate (float): Initial learning rate (after the potential warmup period) to use.
    scale_lr (bool): Whether to scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    lr_scheduler (str): The scheduler type to use for learning rate scheduling.
    lr_warmup_steps (int): Number of steps for the warmup in the lr scheduler.
    allow_tf32 (bool): Whether to allow TF32 on Ampere GPUs to speed up training.
    use_ema (bool): Whether to use EMA model.
    non_ema_revision (Optional[str]): Revision of pretrained non-ema model identifier.
    dataloader_num_workers (int): Number of subprocesses to use for data loading.
    adam_beta1 (float): The beta1 parameter for the Adam optimizer.
    adam_beta2 (float): The beta2 parameter for the Adam optimizer.
    adam_weight_decay (float): Weight decay to use.
    adam_epsilon (float): Epsilon value for the Adam optimizer.
    max_grad_norm (float): Max gradient norm.
    local_rank (int): For distributed training: local_rank.
    checkpointing_steps (int): Save a checkpoint of the training state every X updates.
    checkpoints_total_limit (Optional[int]): Max number of checkpoints to store.
    resume_from_checkpoint (Optional[str]): Whether to resume training from a previous checkpoint.
    noise_offset (float): The scale of noise offset.
    validation_prompt (str): The prompt that will be used for training validation.
    validation_epochs (int): Number of validation epochs.
    num_validation_images (int): Number of validation images.
    """

    dataset_name: Optional[str] = Field(
        ...,
        description="The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly "
        "private, dataset). It can also be a path pointing to a local copy of a dataset in your "
        "filesystem, or to a folder containing files that 🤗 Datasets can understand.",
    )
    dataset_config_name: Optional[str] = Field(
        None,
        description="The config of the Dataset, leave as None if there's only one config.",
    )
    image_column: str = Field(
        "image", description="The column of the dataset containing an image."
    )
    caption_column: str = Field(
        "text",
        description="The column of the dataset containing a caption or a list of captions.",
    )
    max_train_samples: Optional[int] = Field(
        None,
        description="For debugging purposes or quicker training, truncate the number of training examples "
        "to this value if set.",
    )
    seed: Optional[int] = Field(None, description="A seed for reproducible training.")
    resolution: int = Field(
        512,
        description="The resolution for input images, all the images in the train/validation dataset will be resized "
        "to this resolution",
    )
    center_crop: bool = Field(
        False,
        description="Whether to center crop the input images to the resolution. If not set, the images will be "
        "randomly cropped. The images will be resized to the resolution first before cropping.",
    )
    random_flip: bool = Field(
        False, description="whether to randomly flip images horizontally"
    )
    train_batch_size: int = Field(
        16, description="Batch size (per device) for the training dataloader."
    )
    num_train_epochs: int = Field(100)
    max_train_steps: Optional[int] = Field(
        None,
        description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    gradient_checkpointing: bool = Field(
        False,
        description="Whether or not to use gradient checkpointing to save memory at the expense of slower "
        "backward pass.",
    )
    learning_rate: float = Field(
        1e-4,
        description="Initial learning rate (after the potential warmup period) to use.",
    )
    scale_lr: bool = Field(
        False,
        description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    lr_scheduler: str = Field(
        "constant",
        description="The scheduler type to use. Choose between "
        '["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    lr_warmup_steps: int = Field(
        500, description="Number of steps for the warmup in the lr scheduler."
    )
    use_ema: bool = Field(False, description="Whether to use EMA model.")
    non_ema_revision: Optional[str] = Field(
        None,
        description="Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier "
        "of the local or remote repository specified with --pretrained_model_name_or_path.",
    )
    dataloader_num_workers: int = Field(
        0,
        description="Number of subprocesses to use for data loading. 0 means that the data will be loaded "
        "in the main process.",
    )
    adam_beta1: float = Field(
        0.9, description="The beta1 parameter for the Adam optimizer."
    )
    adam_beta2: float = Field(
        0.999, description="The beta2 parameter for the Adam optimizer."
    )
    adam_weight_decay: float = Field(1e-2, description="Weight decay to use.")
    adam_epsilon: float = Field(
        1e-08, description="Epsilon value for the Adam optimizer"
    )
    max_grad_norm: float = Field(1.0, description="Max gradient norm.")
    local_rank: int = Field(-1, description="For distributed training: local_rank")
    checkpointing_steps: int = Field(
        500,
        description="Save a checkpoint of the training state every X updates. These checkpoints are only suitable "
        "for resuming training using --resume_from_checkpoint.",
    )
    checkpoints_total_limit: Optional[int] = Field(
        None,
        description="Max number of checkpoints to store. Passed as total_limit to the Accelerator "
        "ProjectConfiguration. See Accelerator::save_state "
        "https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate."
        "Accelerator.save_state for more docs",
    )
    resume_from_checkpoint: Optional[str] = Field(
        None,
        description=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' --checkpointing_steps, or "latest" to automatically select the last available checkpoint.'
        ),
    )
    noise_offset: float = Field(0, description="The scale of noise offset.")

    class Config:
        """
        Configuration class for pydantic model.
        """

        schema_extra = {
            "example": {
                "dataset_name": "lambdalabs/pokemon-blip-captions",
                "resolution": 512,
                "random_flip": True,
                "train_batch_size": 1,
                "num_train_epochs": 1,
                "checkpointing_steps": 1000,
                "learning_rate": 0.0001,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "seed": 42,
            }
        }


class LoraProgress(BaseModel):
    """
    Pydantic model for the progress of LoRA fine-tuning.

    This model represents the progress of LoRA fine-tuning process by storing
    the process name and a string representing the current percentage.

    Attributes:
        process_name (str): The name of the process.
        progress (str): A string describing the current progress as percentage.
    """

    process_name: str
    progress: str


class AcceleratorAttributes(BaseModel):
    """
    Pydantic model that contains attributes for LoRA training process.

    Attributes:
        lora_layers (AttnProcsLayers): An instance of the AttnProcsLayers class that defines
            the LoRA layers and their attention processing for the model.
        optimizer (AdamW): An instance of the AdamW optimizer from the PyTorch
            library, used for updating the model's parameters during training.
        train_dataloader (DataLoader): An instance of the DataLoader class from the PyTorch
            library, responsible for providing the training data in batches.
        lr_scheduler (LambdaLR): An instance of the LambdaLR learning rate scheduler from the
            PyTorch library, used for adjusting the learning rate during the training process.
    """

    lora_layers: AttnProcsLayers
    optimizer: AdamW
    train_dataloader: DataLoader
    lr_scheduler: LambdaLR

    class Config:
        arbitrary_types_allowed = True
