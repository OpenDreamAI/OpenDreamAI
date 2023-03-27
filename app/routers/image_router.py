import os.path
from fastapi import HTTPException

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi import status

from app.core.config import settings

router = APIRouter()
@router.get("")
async def get_image(
        filename: str
):
    """
    A FastAPI router method that retrieves an image file from disk and returns it as a response.

    Parameters:
        filename (str): The filename of the image file to retrieve.

    Response:
        A FileResponse object that contains the retrieved image file.

    Raises:
        404 Not Found if the filename was not found.
    """
    files = os.listdir(settings.OUTPUT_FOLDER)
    if filename not in files:
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    return FileResponse(os.path.join(settings.OUTPUT_FOLDER, filename))
