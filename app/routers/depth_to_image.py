from fastapi import APIRouter, BackgroundTasks, Depends

from app.schemas.depth_to_image import DepthToImageRequest
from app.services.depth_to_image import depth_to_image_service as service

router = APIRouter()


@router.post("/", status_code=202, response_model=list[str])
async def generate_depth_to_image(
    background_tasks: BackgroundTasks,
    params: DepthToImageRequest = Depends(DepthToImageRequest.as_form),
) -> list[str]:
    """
    A FastAPI router method that generates images from depth information using a DepthToImageService, saves the output
    images to disk, and returns the filename of the saved image(s).

    Parameters:
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.
        params (DepthToImageRequest, optional): A DepthToImageRequest object containing the configuration for the
            depth-to-image conversion.

    Response:
        An ImagesResponse object containing a list of filenames and an information message.
    """
    filenames = service.generate_filenames(params)
    background_tasks.add_task(service.process, params, filenames)

    return filenames
