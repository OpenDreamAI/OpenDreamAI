from fastapi import APIRouter

from app.services.image_retrieval import image_retrieval_service as service

router = APIRouter()


@router.get("/")
async def get_image(
    filename: str,
):
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

    return await service.get_image_progress(filename)
