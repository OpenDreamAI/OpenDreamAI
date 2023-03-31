import os

from fastapi import HTTPException, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.image_retrieval import ImageProgress

progress: dict[str, float] = {}


class ImageRetrievalService:
    async def get_image_progress(self, filename: str):
        """
        Get image generation progress for the given filename.

        If the filename is not found in the progress dictionary, it tries to return the image file using `get_image`.

        Parameters:
            filename (str): The filename of the image to get progress for.

        Returns:
            ImageProgress: An instance of the ImageProgress schema with the filename and progress.
        """
        if filename not in progress:
            return await self.get_image(filename)

        return ImageProgress(filename=filename, progress=f"{progress[filename]:.0f}%")

    @staticmethod
    async def get_image(filename: str):
        """
        Retrieve an image by filename from the output folder.

        If the image is not found in the output folder, raise an HTTP 404 Not Found exception.

        Parameters:
            filename (str): The filename of the image to retrieve.

        Returns:
            FileResponse: An instance of FileResponse for the requested image file.

        Raises:
            404 Not Found If the filename is not found in the output folder.
        """
        os.makedirs(settings.OUTPUT_FOLDER, exist_ok=True)
        files = os.listdir(settings.OUTPUT_FOLDER)
        if filename not in files:
            raise HTTPException(status.HTTP_404_NOT_FOUND)

        return FileResponse(os.path.join(settings.OUTPUT_FOLDER, filename))


image_retrieval_service = ImageRetrievalService()
