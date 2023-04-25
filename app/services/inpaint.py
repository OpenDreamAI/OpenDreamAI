from diffusers import StableDiffusionInpaintPipeline

from app.core.config import settings
from app.services.base import BaseService


class InpaintService(BaseService):
    """
    A service class for inpainting images using the StableDiffusionInpaintPipeline.

    Notes:
        This service class provides methods for inpainting images using the StableDiffusionInpaintPipeline,
        which is a deep learning-based method for generating high-quality inpainted images. The class uses the
        `StableDiffusionInpaintPipeline` class from the `diffusers` library to inpaint images, and inherits from the
        `BaseService` class to reuse its methods for generating unique filenames and saving images to disk.
    """

    initial_step = 0

    @classmethod
    def initialize_pipeline(cls) -> StableDiffusionInpaintPipeline:
        """
        Initializes a StableDiffusionInpaintPipeline object from a pretrained model,
        sets the device configuration, and returns the initialized pipeline.

        Returns:
            A StableDiffusionInpaintPipeline object initialized with the given model ID and configured for the
            specified device.
        """
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            settings.INPAINT_MODEL, torch_dtype=cls.get_torch_datatype()
        )
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline
