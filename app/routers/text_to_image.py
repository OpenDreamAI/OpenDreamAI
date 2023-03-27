import os.path

from fastapi import APIRouter, BackgroundTasks, Depends

from app.context import get_text_to_image_service
from app.core.config import settings
from app.schemas.text_to_image import ImagesResponse, TextToImageRequest
from app.services.text_to_image import TextToImageService

router = APIRouter()


@router.post("/")
async def text_to_image(
    params: TextToImageRequest,
    background_tasks: BackgroundTasks,
    service: TextToImageService = Depends(get_text_to_image_service),
):
    """
    A FastAPI router method that converts text to images using TextToImageService and saves the images to disk.

    Parameters:
        params (TextToImageRequest): A TextToImageRequest object containing the configuration for the text-to-image
            conversion.
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.
        service (TextToImageService, optional): A TextToImageService object used to convert text to image.

    Response:
        A TextToImageResponse object containing a list of filenames and an information message.
    """
    filenames = [
        service.generate_ulid_filename()
        for _ in range(len(params.prompt) * params.num_images_per_prompt)
    ]
    filepath = [
        os.path.join(settings.OUTPUT_FOLDER, filename) for filename in filenames
    ]
    background_tasks.add_task(service.text_to_image, params, filepath)
    return ImagesResponse(
        images=filenames,
        info="Your request was placed in background. It will be available shortly.",
    )
