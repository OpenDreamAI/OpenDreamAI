import logging
import math
import os
import random
from typing import Any, Optional

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import ulid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import DatasetDict, load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from fastapi import HTTPException, status
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from app.attention_processor import LoRAAttnProcessor
from app.core.config import settings
from app.schemas.lora import AcceleratorAttributes, LoraProgress, LoraTrainingArguments
from app.services.base import lora_progress

logger = get_logger(__name__, log_level="INFO")


class LoraService:
    def __init__(self, args: LoraTrainingArguments):
        self.process_name = self.generate_process_name()
        self.args = args
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            settings.TXT2IMG_MODEL, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            settings.TXT2IMG_MODEL,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            settings.TXT2IMG_MODEL,
            subfolder="text_encoder",
        )
        self.vae = AutoencoderKL.from_pretrained(
            settings.TXT2IMG_MODEL, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            settings.TXT2IMG_MODEL, subfolder="unet"
        )
        self.directory_name = self.init_directory(self.process_name)
        self.accelerator = self.init_accelerator(args, self.directory_name)
        self.weight_dtype = self.set_weight_dtype(self.accelerator)
        self.override_max_train_steps = False

    def save_lora_layers(self) -> None:
        """
        Saves the attention process layers of the UNet model.
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet = self.unet.to(torch.float32)
            unet.save_attn_procs(self.directory_name)

    def freeze_parameters(self) -> None:
        """
        Freeze parameters of models to save more memory
        """
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def move_to_device(self) -> None:
        """
        Move unet, vae, and text_encoder to device and cast to weight_dtype

        Returns:
            weight data type
        """
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

    def scale_learning_rate(self) -> None:
        """
        Scales learning rate according to number of processes, batch size, and gradient accumulation steps.
        """
        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

    def validate_dataset_columns(self, column_names: list[str]) -> None:
        """
        Ensures that dataset has columns needed.

        Parameters:
            column_names: list of column names present in the dataset
        """
        if self.args.image_column not in column_names:
            raise ValueError(
                f"Dataset does not contain {self.args.image_column} column"
            )
        if self.args.caption_column not in column_names:
            raise ValueError(
                f"Dataset does not contain {self.args.caption_column} column"
            )

    def initialize_accelerator_attrs(self) -> AcceleratorAttributes:
        """
        Initializes accelerator attributes.

        Returns:
            AcceleratorAttributes object
        """
        lora_layers = AttnProcsLayers(self.unet.attn_processors)

        self.scale_learning_rate()

        optimizer = torch.optim.AdamW(
            lora_layers.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        train_dataloader = self.get_train_dataloader()

        # Scheduler and math around the number of training steps.

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            self.override_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
        )
        return AcceleratorAttributes(
            lora_layers=lora_layers,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
        )

    def fine_tune(self) -> None:
        """
        Fine-tunes a model using LoRA process.
        """
        self.init_logging(self.accelerator)
        self.freeze_parameters()

        if self.args.seed is not None:
            set_seed(self.args.seed)

        self.move_to_device()

        self.set_lora_attn_procs(self.unet)

        train_attrs = self.initialize_accelerator_attrs()

        # Prepare everything with our `accelerator`.
        (
            train_attrs.lora_layers,
            train_attrs.optimizer,
            train_attrs.train_dataloader,
            train_attrs.lr_scheduler,
        ) = self.accelerator.prepare(*train_attrs.dict().values())

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_attrs.train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.override_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )

        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "text2image-fine-tune", config=vars(self.args)
            )

        self.print_initial_logs()

        self.train(train_attrs, num_update_steps_per_epoch)

        self.save_lora_layers()

        self.accelerator.end_training()
        torch.cuda.empty_cache()

    def train(
        self, train_attrs: AcceleratorAttributes, num_update_steps_per_epoch: int
    ) -> None:
        """
        Train the LoRA model using the given arguments and configurations.

        Parameters:
            train_attrs: attributes needed for training process.
            num_update_steps_per_epoch: Number of update steps per epoch.
        """
        global_step = 0
        first_epoch = 0

        # If needed, load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            path = self.get_path()
            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.directory_name, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        progress_bar = self.init_progress_bar(global_step)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_attrs.train_dataloader):
                # Skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint:
                    if self.skip_step(
                        self.args, epoch, first_epoch, step, resume_step, progress_bar
                    ):
                        continue

                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.convert_to_latents(batch)
                    noise = self.sample_noise(self.args, latents)
                    timesteps = self.sample_timesteps(latents, self.noise_scheduler)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    target = self.get_target(
                        self.noise_scheduler, noise, latents, timesteps
                    )
                    loss = self.compute_loss(
                        self.unet,
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        target,
                    )
                    train_loss = self.update_train_loss(
                        self.args, self.accelerator, loss, train_loss
                    )

                    self.backpropagate(loss, train_attrs)

                # Checks if the accelerator has performed an optimization step behind the scenes
                global_step, train_loss = self.sync_gradients(
                    progress_bar,
                    global_step,
                    train_loss,
                )
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": train_attrs.lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                lora_progress[self.process_name] = (
                    global_step / self.args.max_train_steps * 100
                )
                if global_step >= self.args.max_train_steps:
                    break

    def get_path(self) -> str:
        """
        Gets the path for resuming from a checkpoint.

        Returns:
            str: Path for resuming from a checkpoint.
        """
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            path = self.get_most_recent_checkpoint()
        return path

    def get_most_recent_checkpoint(self) -> Optional[str]:
        """
        Gets the most recent checkpoint from the directory.

        Returns:
            str: Path to the most recent checkpoint.
        """
        dirs = os.listdir(self.directory_name)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        return path

    def sync_gradients(
        self, progress_bar: tqdm, global_step: int, train_loss: float
    ) -> tuple[int, float]:
        """
        Synchronizes gradients, updates progress bar, logs training loss, and saves state.

        Parameters:
            progress_bar: Progress bar for tracking training progress.
            global_step (int): Current global step.
            train_loss (float): Current accumulated train loss.

        Returns:
            int: Updated global step.
            float: Updated train loss.
        """
        if self.accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            self.accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

            if global_step % self.args.checkpointing_steps == 0:
                self.save_state(self.accelerator, global_step, self.directory_name)
        return global_step, train_loss

    @staticmethod
    def compute_loss(
        unet: UNet2DConditionModel,
        noisy_latents: Tensor,
        timesteps: Tensor,
        encoder_hidden_states: Any,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the loss using the predicted noise residual.

        Parameters:
            unet: UNet model for conditioning.
            noisy_latents: Noisy latent representations.
            timesteps: Timesteps for noise scheduler.
            encoder_hidden_states: Hidden states from the text encoder.
            target: Target for loss computation.

        Returns:
            loss: Computed loss.
        """
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    @staticmethod
    def update_train_loss(
        args: LoraTrainingArguments,
        accelerator: Accelerator,
        loss: Tensor,
        train_loss: float,
    ) -> float:
        """
        Updates the training loss by gathering the losses across all processes.

        Parameters:
            args: Training arguments.
            accelerator: Accelerator object for distributed training.
            loss: Computed loss.
            train_loss (float): Current accumulated train loss.

        Returns:
            float: Updated train loss.
        """
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        train_loss += avg_loss.item() / args.gradient_accumulation_steps

        return train_loss

    @staticmethod
    def get_target(noise_scheduler, noise: Tensor, latents: Tensor, timesteps: Tensor):
        """
        Gets the target for loss computation based on the prediction type.

        Parameters:
            noise_scheduler: Noise scheduler for adding noise to latent space.
            noise: Noise sampled for the latent representations.
            latents: Latent representations of the images.
            timesteps: Timesteps for noise scheduler.

        Returns:
            target: Target for loss computation.
        """
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        return target

    @staticmethod
    def skip_step(
        args: LoraTrainingArguments,
        epoch: int,
        first_epoch: int,
        step: int,
        resume_step: int,
        progress_bar: tqdm,
    ) -> bool:
        """
        Skips steps until the resumed step is reached.

        Parameters:
            args: Training arguments.
            epoch (int): Current epoch.
            first_epoch (int): First epoch in the resumed training.
            step (int): Current step in the training loop.
            resume_step (int): Step to resume training from.
            progress_bar: Progress bar for tracking training progress.

        Returns:
            bool: True if the current step should be skipped, False otherwise.
        """
        if epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            return True
        return False

    @staticmethod
    def save_state(
        accelerator: Accelerator, global_step: int, directory_name: str
    ) -> None:
        """
        Saves the state of the model during training.

        Parameters:
            accelerator: Accelerator object for distributed training.
            global_step (int): Current global step.
            directory_name (str): Directory name for saving the model.
        """
        if accelerator.is_main_process:
            save_path = os.path.join(directory_name, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    @staticmethod
    def sample_timesteps(latents: Tensor, noise_scheduler: DDPMScheduler) -> Tensor:
        """
        Samples random timesteps for noise scheduling.

        Parameters:
            latents: Latent representations of the images.
            noise_scheduler: Noise scheduler for adding noise to latent space.

        Returns:
            timesteps: Sampled timesteps.
        """
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        return timesteps

    def convert_to_latents(self, batch: dict) -> Tensor:
        """
        Converts images to latents using the VAE's encoder.

        Parameters:
            batch: Batch of images to be processed.

        Returns:
            latents: Latent representations of the images.
        """
        latents = self.vae.encode(
            batch["pixel_values"].to(dtype=self.weight_dtype)
        ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @staticmethod
    def sample_noise(args, latents: Tensor) -> Tensor:
        """
        Samples random noise for the latent representations.

        Parameters:
            args: Training arguments.
            latents: Latent representations of the images.

        Returns:
            noise: Sampled noise for the latents.
        """
        noise = torch.randn_like(latents)
        if args.noise_offset:
            noise += args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
            )
        return noise

    def backpropagate(self, loss: Tensor, train_attrs: AcceleratorAttributes) -> None:
        """
        Performs backpropagation and updates model parameters.

        Parameters:
            train_attrs: model containing attributes for training
            loss: Computed loss.
        """
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            params_to_clip = train_attrs.lora_layers.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
        train_attrs.optimizer.step()
        train_attrs.lr_scheduler.step()
        train_attrs.optimizer.zero_grad()

    def init_progress_bar(self, global_step: int) -> tqdm:
        """
        Initializes the progress bar for training.

        Parameters:
            global_step (int): Current global step.

        Returns:
            progress_bar: Initialized progress bar.
        """
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        return progress_bar

    def get_train_dataset(self, train_transforms: Compose, dataset: DatasetDict):
        """
        Preprocesses and gets the train dataset.

        Parameters:
            train_transforms: Image transformations for the train dataset.
            dataset: Dataset to be used for training.

        Returns:
            train_dataset: Preprocessed train dataset.
        """

        def preprocess_train(examples: DatasetDict):
            images = [
                image.convert("RGB") for image in examples[self.args.image_column]
            ]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = self.tokenize_captions(
                examples, self.tokenizer, self.args.caption_column
            )
            return examples

        with self.accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle(seed=self.args.seed)
                    .select(range(self.args.max_train_samples))
                )
            train_dataset = dataset["train"].with_transform(preprocess_train)
            return train_dataset

    @staticmethod
    def get_train_transforms(args: LoraTrainingArguments) -> Compose:
        """
        Gets the training data transforms.

        Parameters:
            args (LoraTrainingArguments): Arguments needed for the fine-tuning process.

        Returns:
            train_transforms: Training data transformations.
        """
        train_transforms = Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms

    @staticmethod
    def tokenize_captions(
        examples: DatasetDict,
        tokenizer: CLIPTokenizer,
        caption_column: str,
        is_train: bool = True,
    ):
        """
        Tokenizes captions using the given tokenizer.

        Parameters:
            examples: Examples containing captions to be tokenized.
            tokenizer: Tokenizer for encoding captions.
            caption_column (str): Column name for captions in the dataset.
            is_train (bool): Indicates if the dataset is for training or not.

        Returns:
            input_ids: Tokenized captions as input_ids.
        """
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def get_train_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the train dataset.

        Returns:
            train_dataloader: DataLoader for the train dataset.
        """

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        dataset = self.get_dataset(self.args)

        self.validate_dataset_columns(dataset["train"].column_names)

        # Preprocessing the datasets.
        train_transforms = self.get_train_transforms(self.args)

        train_dataset = self.get_train_dataset(train_transforms, dataset)

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )
        logger.info(f"Num examples = {len(train_dataset)}")
        return train_dataloader

    def print_initial_logs(self) -> None:
        """
        Logs initial information about the training process.
        """
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

    @staticmethod
    def init_directory(process_name: str) -> str:
        """
        Initializes the directory for the given process name.

        Parameters:
            process_name (str): Name of the process.

        Returns:
            str: The directory name.
        """
        directory_name = os.path.join(settings.LORA_FOLDER, process_name)
        os.makedirs(directory_name, exist_ok=True)
        return directory_name

    @staticmethod
    def init_accelerator(
        args: LoraTrainingArguments, directory_name: str
    ) -> Accelerator:
        """
        Initializes the accelerator for the given arguments and directory name.

        Parameters:
            args (LoraTrainingArguments): Arguments needed for the fine-tuning process.
            directory_name (str): Directory name for saving the model.

        Returns:
            accelerator: Accelerator object for distributed training.
        """
        logging_dir = os.path.join(directory_name, "logs")

        accelerator_project_config = ProjectConfiguration(
            total_limit=args.checkpoints_total_limit
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=settings.MIXED_PRECISION,
            logging_dir=logging_dir,
            project_config=accelerator_project_config,
        )
        return accelerator

    @staticmethod
    def init_logging(accelerator: Accelerator) -> None:
        """
        Initializes the logging for the accelerator.

        Parameters:
            accelerator: Accelerator object for distributed training.
        """
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    @staticmethod
    def set_lora_attn_procs(unet: Any) -> None:
        """
        Sets the LoRA attention processors for the UNet model.

        Parameters:
            unet: UNet model for conditioning.
        """
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        unet.set_attn_processor(lora_attn_procs)

    @staticmethod
    def get_dataset(args: LoraTrainingArguments) -> DatasetDict:
        """
        Loads the dataset for training.

        Parameters:
            args (LoraTrainingArguments): Arguments needed for the fine-tuning process.

        Returns:
            dataset: Loaded dataset.
        """
        if args.dataset_name is not None:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
            )
        else:
            data_files = {}
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
            )
        return dataset

    @staticmethod
    def set_weight_dtype(accelerator: Accelerator) -> torch.dtype:
        """
        Set the weight dtype based on the accelerator's mixed_precision attribute.

        Parameters:
            accelerator: The accelerator object.

        Returns:
            The weight dtype (torch.float32 or torch.float16).
        """
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16

        return weight_dtype

    @staticmethod
    def generate_process_name() -> str:
        """
        Generates new name for a directory as a new ULID.

        Returns:
            directory name
        """
        return str(ulid.ULID())

    @staticmethod
    async def get_lora_progress(process_name: str) -> LoraProgress:
        """
        Get LoRA fine-tuning progress by process name.

        Parameters:
            process_name (str): The name of the process to get progress for.

        Returns:
            LoraProgress: An instance of the LoraProgress schema with the process name and progress.
        """
        if process_name not in lora_progress:
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        return LoraProgress(
            process_name=process_name, progress=f"{lora_progress[process_name]:.0f}%"
        )

    @staticmethod
    async def list_lora_models() -> list[str]:
        """
        Lists all trained LoRA weights in the lora output directory.

        Returns:
            list of lora weight folder names
        """
        os.makedirs(settings.LORA_FOLDER, exist_ok=True)
        files = os.listdir(settings.LORA_FOLDER)
        return files
