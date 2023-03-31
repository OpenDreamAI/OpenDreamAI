import os
from abc import ABC, abstractmethod
from typing import Callable

from PIL.Image import Image
from pytorch_lightning import seed_everything
from torch import FloatTensor
from ulid import ULID

from app.core.config import settings
from app.schemas.base import BaseRequestModel
from app.services.image_retrieval import progress


class BaseService(ABC):
    """
    A base class for image generation services.

    Notes:
        This class defines methods that are common to image generation services, such as generating unique filenames
        and saving images to disk. Other image generation services can inherit from this class to reuse its methods
        and add their own functionality.
    """

    def __init__(self):
        self.pipeline = self.initialize_pipeline()

    @property
    @abstractmethod
    def initial_step(self):
        pass

    @classmethod
    @abstractmethod
    def initialize_pipeline(cls):
        pass

    @staticmethod
    def generate_ulid_filename() -> str:
        """
        Generate a unique filename using the ULID algorithm.

        This method generates a unique filename by creating a new ULID (Universally Unique Lexicographically Sortable
        Identifier) using the `ULID` class from the `ulid` library. The ULID is a 128-bit identifier that is designed to
        be unique, lexicographically sortable, and URL-safe.

        The generated filename consists of the ULID value as a string, followed by the `.png` extension to indicate that
        the file is a PNG image.

        Returns:
            str: A unique filename in the format "<ulid>.png", where "<ulid>" is a ULID value as a string.
        """
        return f"{ULID()}.png"

    @staticmethod
    def calculate_number_of_images(params: BaseRequestModel) -> int:
        """
        Determine the number of filenames that need to be generated based on request parameters.

        Parameters:
             params: request parameters

        Returns:
            number of filenames
        """
        return len(params.prompt) * params.num_images_per_prompt

    def generate_filenames(self, params: BaseRequestModel) -> list[str]:
        """
        Generate filenames based on request parameters.

        Parameters:
             params: request parameters

        Returns:
            list of filenames
        """
        return [
            self.generate_ulid_filename()
            for _ in range(self.calculate_number_of_images(params))
        ]

    @staticmethod
    def create_filepaths(filenames: list[str]) -> list[str]:
        """
        Creates filepaths by joining output folder path with filenames.

        Parameters:
            filenames: list of filenames to generate filepaths from.

        Returns:
            A list of filepaths.
        """
        return [
            os.path.join(settings.OUTPUT_FOLDER, filename) for filename in filenames
        ]

    def save_images(self, filenames: list[str], images: list[Image]):
        """
        Save a list of images to disk.

        This method saves a list of PIL `Image` objects to disk using the filepaths provided in `filepaths`.
        The images are saved as PNG files.

        Parameters:
            filenames (list): A list of filepaths where the images should be saved.
            images (list): A list of PIL `Image` objects to save.
        """
        filenames = self.create_filepaths(filenames)
        os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)
        for image, filepath in zip(images, filenames):
            image.save(filepath)

    def create_callback(
        self, filenames: list[str], steps: int
    ) -> Callable[[int, int, FloatTensor], None]:
        """
        Creates a callback to pass into pipeline.

        Parameters:
            filenames: list of image filenames
            steps: total number of steps

        Returns:
            Callable to pass into pipeline as a callback function.
        """

        def custom_callback(step: int, timestep: int, latents: FloatTensor) -> None:
            images_progress = (step + self.initial_step) / steps * 100
            for filename in filenames:
                progress[filename] = images_progress

        return custom_callback

    def generate_images(
        self, params: BaseRequestModel, filenames: list[str]
    ) -> list[Image]:
        seed_everything(seed=params.seed)
        callback_function = self.create_callback(
            filenames=filenames, steps=params.total_steps
        )
        return self.pipeline(
            **params.dict(exclude_none=True, exclude={"seed", "total_steps"}),
            callback=callback_function,
        ).images

    def initialize_images_progress(self, filenames: list[str]):
        """
        Initializes progress for each filename as 0%.

        Parameters:
             filenames:
        """
        for filename in filenames:
            progress[filename] = 0

    def process(self, params: BaseRequestModel, filenames: list[str]) -> None:
        """
        Method to process request by passing request model into the generate_images method and saving result images.

        Parameters:
            params: request model containing information to process.
            filenames: list of filenames to save images.
        """
        self.initialize_images_progress(filenames)
        try:
            images = self.generate_images(filenames=filenames, params=params)
            self.save_images(filenames=filenames, images=images)
        finally:
            for filename in filenames:
                progress.pop(filename, None)
