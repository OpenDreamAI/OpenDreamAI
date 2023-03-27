import os

from PIL.Image import Image
from ulid import ULID

from app.core.config import settings


class BaseService:
    """
    A base class for image generation services.

    Notes:
        This class defines methods that are common to image generation services, such as generating unique filenames
        and saving images to disk. Other image generation services can inherit from this class to reuse its methods
        and add their own functionality.
    """

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
    def save_images(filepaths: list[str], images: list[Image]):
        """
        Save a list of images to disk.

        This method saves a list of PIL `Image` objects to disk using the filepaths provided in `filepaths`.
        The images are saved as PNG files.

        Parameters:
            filepaths (list): A list of filepaths where the images should be saved.
            images (list): A list of PIL `Image` objects to save.
        """
        os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)
        for image, filepath in zip(images, filepaths):
            image.save(filepath)
