from fastapi import APIRouter

from app.core.config import settings

from . import health_check, image_router, text_to_image

api_router = APIRouter()
api_router.include_router(
    text_to_image.router, tags=[settings.IMAGE_GENERATION_TAG], prefix="/txt2img"
)
api_router.include_router(
    image_router.router, tags=[settings.IMAGE_TAG], prefix="/image"
)
api_router.include_router(
    health_check.router, tags=[settings.HEALTH_CHECK], prefix="/health_check"
)
