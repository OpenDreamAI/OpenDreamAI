from diffusers import StableDiffusionImg2ImgPipeline

from app.core.config import settings
from app.services.base import BaseService


class ImageToImageService(BaseService):
    """
    A service class for generating images from other images using the StableDiffusionImg2ImgPipeline.

    Notes:
        This service class provides methods for generating images from an input using StableDiffusionImg2ImgPipeline,
        which is a deep learning-based method for generating high-quality images from other images. The class uses the
        `StableDiffusionImg2ImgPipeline` class from the `diffusers` library to generate images, and inherits from the
        `BaseService` class to reuse its methods for generating unique filenames and saving images to disk.
    """

    initial_step = 1

    @classmethod
    def initialize_pipeline(cls) -> StableDiffusionImg2ImgPipeline:
        """
        Initializes a StableDiffusionImg2ImgPipeline object from a pretrained model, sets the device configuration,
        and returns the initialized pipeline.

        Returns:
            A StableDiffusionImg2ImgPipeline object initialized with the given model ID and configured for the
            specified device.
        """
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            settings.TXT2IMG_MODEL
        )
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline


image_to_image_service = ImageToImageService()
