import enum
import os

import dotenv

dotenv.load_dotenv()


class DeviceConfig(enum.Enum):
    """
    An enum class representing device configurations for running image generation models.

    Attributes:
        CPU (str): Value for running the model on CPU.
        GPU (str): Value for running the model on GPU using CUDA.
    """

    CPU = "cpu"
    GPU = "cuda"


class Settings:
    """
    A class containing settings for the project.

    Attributes:
        DEVICE (DeviceConfig): The device configuration for running the model.
        TXT2IMG_MODEL (str): The ID of the txt2img model to use.
        DEPTH2IMG_MODEL (str): The ID of the depth2img model to use.
        INPAINT_MODEL (str): The ID of the inpaint model to use.
        PROJECT_NAME (str): The name of the project.
        SERVER_NAME (str): The name of the server.
        OUTPUT_FOLDER (str): The name of the folder to store output files.
        IMAGE_GENERATION_TAG (str): A tag for image generation operations.
        OPENAPI_TAGS (list): A list of tags for OpenAPI operations.
    """

    DEVICE = DeviceConfig(os.getenv("DEVICE", "cpu"))
    MIXED_PRECISION = os.getenv("MIXED_PRECISION", None)
    TXT2IMG_MODEL = "stabilityai/stable-diffusion-2-1"
    DEPTH2IMG_MODEL = "stabilityai/stable-diffusion-2-depth"
    INPAINT_MODEL = "stabilityai/stable-diffusion-2-inpainting"
    PROJECT_NAME = "Open Dream AI"
    SERVER_NAME = "open-dream-ai"
    OUTPUT_FOLDER = "output"
    LORA_FOLDER = "lora"
    IMAGE_GENERATION_TAG = "Image generation"
    LORA_TAG = "Lora fine-tuning"
    HEALTH_CHECK = "Health check"
    OPENAPI_TAGS = [
        {
            "name": IMAGE_GENERATION_TAG,
            "description": "Operations with images.",
        },
        {"name": LORA_TAG, "description": "Operations with LoRA fine-tuning."},
    ]


settings = Settings()
