from fastapi import APIRouter, BackgroundTasks

from app.schemas.text_to_image import TextToImageRequest
from app.services.text_to_image import TextToImageService

router = APIRouter()


@router.post("/", status_code=202, response_model=list[str])
async def generate_image_from_text(
    params: TextToImageRequest, background_tasks: BackgroundTasks
) -> list[str]:
    """
    A FastAPI router method that converts text to images using TextToImageService and saves the images to disk.

    Parameters:
        params (TextToImageRequest): A TextToImageRequest object containing the configuration for the text-to-image
            conversion.
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.

    Response:
        A TextToImageResponse object containing a list of filenames and an information message.
    """
    service: TextToImageService = TextToImageService()
    filenames = service.generate_filenames(params)
    background_tasks.add_task(service.process, params, filenames)
    return filenames
