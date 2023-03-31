from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_health_check():
    """
    Healthcheck endpoint to check if the service is alive.
    """
    return {"message": "Hello World 2"}
