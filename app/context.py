from app.services.text_to_image import TextToImageService


def get_text_to_image_service() -> TextToImageService:
    """
    Factory function for creating a new TextToImageService instance.

    This function creates a new instance of the `TextToImageService` class and returns it. The `TextToImageService` class
    provides methods for generating images from text using the StableDiffusion pipeline.

    Returns:
        TextToImageService: A new instance of the `TextToImageService` class.
    """
    return TextToImageService()