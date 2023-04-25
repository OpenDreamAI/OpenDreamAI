from diffusers import StableDiffusionDepth2ImgPipeline

from app.core.config import settings
from app.services.base import BaseService


class DepthToImageService(BaseService):
    """
    A service class for converting depth images to RGB images using the StableDiffusionDepth2ImgPipeline.

    Notes:
        This service class provides methods for converting depth images to RGB images using the
        StableDiffusionDepth2ImgPipeline, which is a deep learning-based method for generating high-quality
        RGB images from depth images. The class uses the `StableDiffusionDepth2ImgPipeline` class from the
        `diffusers` library to process depth images, and inherits from the `BaseService` class to reuse its
        methods for generating unique filenames and saving images to disk.
    """

    initial_step = 0

    @classmethod
    def initialize_pipeline(cls) -> StableDiffusionDepth2ImgPipeline:
        """
        Initializes a StableDiffusionDepth2ImgPipeline object from a pretrained model,
        sets the device configuration, and returns the initialized pipeline.

        Returns:
            A StableDiffusionDepth2ImgPipeline object initialized with the given model ID and configured for the
            specified device.
        """
        pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
            settings.DEPTH2IMG_MODEL, torch_dtype=cls.get_torch_datatype()
        )
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline
