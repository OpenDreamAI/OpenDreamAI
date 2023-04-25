from fastapi import APIRouter, Depends
from starlette.background import BackgroundTasks

from app.schemas.image_to_image import ImageToImageRequest
from app.services.image_to_image import ImageToImageService

router = APIRouter()


@router.post("/", status_code=202, response_model=list[str])
async def generate_image_from_image(
    background_tasks: BackgroundTasks,
    params: ImageToImageRequest = Depends(ImageToImageRequest.as_form),
) -> list[str]:
    """
    A FastAPI router method that applies a transformation to an input image using an ImageToImageService,
    saves the output image to disk, and returns the filename of the saved image(s).

    Parameters:
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.
        params (ImageToImageRequest, optional): An ImageToImageRequest object containing the configuration for the
            image-to-image conversion.

    Response:
        A TextToImageResponse object containing a list of filenames and an information message.
    """
    service: ImageToImageService = ImageToImageService()
    filenames = service.generate_filenames(params)
    background_tasks.add_task(service.process, params, filenames)
    return filenames
