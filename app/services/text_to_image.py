from diffusers import StableDiffusionPipeline

from app.core.config import settings
from app.services.base import BaseService


class TextToImageService(BaseService):
    """
    A service class for generating images from text using the StableDiffusion pipeline.

    Notes:
        This service class provides methods for generating images from text using the StableDiffusion pipeline,
        which is a deep learning-based method for generating high-quality images from text. The class uses the
        `StableDiffusionPipeline` class from the `diffusers` library to generate images, and inherits from the
        `BaseService` class to reuse its methods for generating unique filenames and saving images to disk.
    """

    initial_step = 1

    @classmethod
    def initialize_pipeline(cls) -> StableDiffusionPipeline:
        """
        Initializes a StableDiffusionPipeline object from a pretrained model,
        sets the device configuration, and returns the initialized pipeline.

        Returns:
            A StableDiffusionPipeline object initialized with the given model ID and a DPMSolverMultistepScheduler
            as its scheduler.
        """
        pipeline = StableDiffusionPipeline.from_pretrained(
            settings.TXT2IMG_MODEL, torch_dtype=cls.get_torch_datatype()
        )
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline
