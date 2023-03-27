from diffusers import StableDiffusionPipeline
from PIL.Image import Image
from pytorch_lightning import seed_everything

from app.core.pipelines import InitializePipeline
from app.schemas.text_to_image import TextToImageRequest
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

    pipeline: StableDiffusionPipeline = InitializePipeline.text_to_image()

    def text_to_image(self, params: TextToImageRequest, filepaths: list[str]) -> None:
        """
        Generates images from the input text using the StableDiffusion pipeline and saves them to disk.

        This method generates a set of images from the input text using the StableDiffusion pipeline, and saves them
        to the specified filepaths using the `save_images` method inherited from the `BaseService` class.

        Parameters:
            params (TextToImageRequest): A Pydantic model containing the configuration for the image generation pipeline.
            filepaths (list): A list of filepaths where the generated images should be saved.
        """
        images = self.generate_images(params=params)
        self.save_images(filepaths=filepaths, images=images)

    def generate_images(self, params: TextToImageRequest) -> list[Image]:
        """
        Generates a set of images from the input text using the StableDiffusion pipeline.

        This method generates a set of images from the input text using the StableDiffusion pipeline, and returns
        them as a list of PIL `Image` objects.

        Parameters:
            params (TextToImageRequest): A Pydantic model containing the configuration for the image generation pipeline.

        Returns:
            list: A list of PIL `Image` objects generated from the input text.

        """
        seed_everything(seed=params.seed)
        return self.pipeline(**params.dict(exclude_none=True, exclude={"seed"})).images
