from fastapi import APIRouter, BackgroundTasks, Depends

from app.schemas.lora import LoraTrainingArguments
from app.services.lora import LoraService

router = APIRouter()


@router.post("/", status_code=202, response_model=str)
async def start_lora_training(
    params: LoraTrainingArguments,
    background_tasks: BackgroundTasks,
):
    """
    A FastAPI router method that starts LoRA fine-tuning and saves resulting LoRA weights to disk.

    Parameters:
        params (TextToImageRequest): A LoraTrainingArguments object containing the configuration for the LoRA process.
        background_tasks (BackgroundTasks): A BackgroundTasks object used to add tasks to be executed in the background.

    Response:
        Fine-tuning process name
    """
    service = LoraService(params)
    background_tasks.add_task(service.fine_tune)
    return service.process_name


@router.get("/progress")
async def get_progress(process_name: str):
    """
    A FastAPI router method that retrieves lora progress and returns it as a response.

    Parameters:
        process_name (str): The process name to retrieve progress for.

    Response:
        Current status as percentage.

    Raises:
        404 Not Found if the process name was not found in progress dictionary.
    """
    return await LoraService.get_lora_progress(process_name)


@router.get("/lora_models")
async def list_lora_models():
    """
    Lists all trained LoRA weights in the lora output directory.

    Returns:
        list of lora weight folder names
    """
    return await LoraService.list_lora_models()
