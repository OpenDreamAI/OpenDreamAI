from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from app.core.config import settings


class InitializePipeline:
    @staticmethod
    def text_to_image() -> StableDiffusionPipeline:
        """
        Initializes a StableDiffusionPipeline object from a pretrained model,
        sets the device configuration, and returns the initialized pipeline.

        Returns:
            A StableDiffusionPipeline object initialized with the given model ID and a DPMSolverMultistepScheduler
            as its scheduler.
        """
        pipeline = StableDiffusionPipeline.from_pretrained(settings.STABLE_DIFFUSION_MODEL_ID)
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline

    @staticmethod
    def image_to_image() -> StableDiffusionImg2ImgPipeline:
        """
        Initializes a StableDiffusionImg2ImgPipeline object from a pretrained model, sets the device configuration,
        and returns the initialized pipeline.

        Returns:
            A StableDiffusionImg2ImgPipeline object initialized with the given model ID and configured for the
            specified device.
        """
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(settings.STABLE_DIFFUSION_MODEL_ID)
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline

