from fastapi import APIRouter

from app.core.config import settings

from . import (
    depth_to_image,
    health_check,
    image_retrieval,
    image_to_image,
    inpaint,
    lora,
    text_to_image,
)

api_router = APIRouter()
api_router.include_router(
    health_check.router, tags=[settings.HEALTH_CHECK], prefix="/health_check"
)
api_router.include_router(
    text_to_image.router, tags=[settings.IMAGE_GENERATION_TAG], prefix="/txt2img"
)
api_router.include_router(
    image_to_image.router, tags=[settings.IMAGE_GENERATION_TAG], prefix="/img2img"
)
api_router.include_router(
    inpaint.router, tags=[settings.IMAGE_GENERATION_TAG], prefix="/inpaint"
)
api_router.include_router(
    depth_to_image.router, tags=[settings.IMAGE_GENERATION_TAG], prefix="/depth2img"
)
api_router.include_router(image_retrieval.router, tags=[settings.IMAGE_GENERATION_TAG])
api_router.include_router(lora.router, tags=[settings.LORA_TAG], prefix="/lora")
