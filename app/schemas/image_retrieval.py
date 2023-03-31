from pydantic import BaseModel


class ImageProgress(BaseModel):
    """
    Pydantic model for the progress of image generation or processing.

    This model represents the progress of an image generation or processing task by storing
    the filename and a string representing the current percentage.

    Attributes:
        filename (str): The filename of the image.
        progress (str): A string describing the current progress of the image generation or processing as percentage.
    """

    filename: str
    progress: str
