from typing import Union

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.image_retrieval import ImageProgress
from app.services.image_retrieval import ImageRetrievalService

router = APIRouter()


@router.get("/image")
async def get_image(filename: str):
    """
    A FastAPI router method that retrieves an image progress and returns it as a response.
    If the generation process is completed and the image is ready, returns generated image.

    Parameters:
        filename (str): The filename of the image file to retrieve.

    Response:
        If generation process is in progress, returns current status.
        If the image is ready, returns a FileResponse object that contains the retrieved image file.

    Raises:
        404 Not Found if the filename was not found in filesystem nor in progress dictionary.
    """
    service: ImageRetrievalService = ImageRetrievalService()
    return await service.get_image_progress(filename)


@router.get("/images")
async def get_list_of_images() -> list[str]:
    """
    Lists all generated images in the output directory.

    Returns:
        list of image filenames
    """
    service: ImageRetrievalService = ImageRetrievalService()

    return await service.get_list_of_filenames()
