from typing import Optional

from fastapi import Form, UploadFile
from PIL.Image import Image
from pydantic import validator

from app.schemas.base import BaseRequestModel, prepare_image


class ImageToImageRequest(BaseRequestModel):
    """
    A Pydantic model representing a request for an image-to-image transformation.

    Attributes:
        prompt (list[str]): The prompt to use for the image-to-image transformation.
        image (PIL.Image.Image): The image to process.
        num_images_per_prompt (int, optional): The number of output images to generate per prompt. Defaults to 1.
        num_inference_steps (int, optional): The number of DDIM sampling steps to use for the transformation.
        Defaults to 50.
        guidance_scale (float, optional): The unconditional guidance scale for the transformation.
        Defaults to 7.5.
        strength (float, optional): The strength of the transformation. Must be between 0 and 1. Defaults to 0.7.
        seed (int, optional): The seed (for reproducible sampling).

    """

    num_inference_steps: Optional[int]
    guidance_scale: Optional[float]
    strength: Optional[float]
    image: Image
    _prepare_image = validator("image", allow_reuse=True, pre=True)(prepare_image)

    @classmethod
    def as_form(
        cls,
        image: UploadFile,
        prompt: list[str] = Form(
            ...,
            description="the prompt to render",
        ),
        num_images_per_prompt: Optional[int] = Form(
            1,
            description="how many samples to produce for each given prompt. A.k.a batch size",
            ge=1,
            le=10,
            alias="number_of_images",
        ),
        num_inference_steps: Optional[int] = Form(
            50, description="number of ddim sampling steps", ge=1, le=150, alias="steps"
        ),
        guidance_scale: Optional[float] = Form(
            7.5,
            description="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
            ge=0,
            le=20,
        ),
        strength: Optional[float] = Form(
            0.7,
            description="Conceptually, indicates how much to transform the reference image. Must be between 0 and 1. "
            "image will be used as a starting point, adding more noise to it the larger the strength. "
            "The number of denoising steps depends on the amount of noise initially added. "
            "When strength is 1, added noise will be maximum and the denoising process will run "
            "for the full number of iterations specified in steps. "
            "A value of 1, therefore, essentially ignores image.",
        ),
        seed: Optional[int] = Form(
            None, description="the seed (for reproducible sampling)"
        ),
    ):
        return cls(
            image=image,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
