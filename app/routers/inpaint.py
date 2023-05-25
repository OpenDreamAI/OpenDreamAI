from fastapi import APIRouter, BackgroundTasks, Depends

from app.schemas.inpaint import InpaintRequest
from app.services.inpaint import InpaintService

router = APIRouter()


@router.post("/", status_code=202, response_model=list[str])
async def inpaint_image(
    background_tasks: BackgroundTasks,
    params: InpaintRequest = Depends(InpaintRequest.as_form),
) -> list[str]:
    """
    A FastAPI router method that performs inpainting on an input image using an InpaintService, saves the output
    image to disk, and returns the filename of the saved image(s).

    Parameters:
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.
        params (InpaintRequest): An InpaintRequest object containing the configuration for the inpainting process.

    Response:
        An ImagesResponse object containing a list of filenames and an information message.
    """
    service: InpaintService = InpaintService()
    filenames = service.generate_filenames(params)

    background_tasks.add_task(service.process, params, filenames)

    return filenames
