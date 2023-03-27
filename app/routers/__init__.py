from fastapi import APIRouter

from app.core.config import settings
from . import text_to_image
from . import health_check
from . import image_router

api_router = APIRouter()
api_router.include_router(text_to_image.router, tags=[settings.GENERATION_TAG], prefix="/txt2img")
api_router.include_router(health_check.router, tags=[settings.HEALTH_CHECK], prefix="/health_check")
api_router.include_router(image_router.router, tags=[settings.IMAGE_TAG], prefix="/image")
