from diffusers import StableDiffusionPipeline

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
        pipeline = StableDiffusionPipeline.from_pretrained(
            settings.STABLE_DIFFUSION_MODEL_ID
        )
        pipeline = pipeline.to(settings.DEVICE.value)
        return pipeline
