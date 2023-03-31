from typing import Optional

from fastapi import Form, UploadFile
from PIL.Image import Image
from pydantic import validator

from app.schemas.base import BaseRequestModel, prepare_image


class InpaintRequest(BaseRequestModel):
    """
    A Pydantic model representing a request for an inpainting transformation.

    Attributes:
        prompt (list[str]): The prompt to use for the inpainting transformation.
        image (PIL.Image.Image): The image to process.
        mask_image (PIL.Image.Image): The image to use as a mask.
        num_images_per_prompt (Optional[int]): The number of output images to generate per prompt. Defaults to 1.
        num_inference_steps (Optional[int]): The number of DDIM sampling steps to use for the transformation.
            Defaults to 50.
        guidance_scale (Optional[float]): The unconditional guidance scale for the transformation.
            Defaults to 7.5.
        seed (int, optional): The seed (for reproducible sampling).

    """

    num_inference_steps: Optional[int]
    guidance_scale: Optional[float]
    image: Image
    _prepare_image = validator("image", allow_reuse=True, pre=True)(prepare_image)
    mask_image: Image
    _prepare_mask_image = validator("mask_image", allow_reuse=True, pre=True)(
        prepare_image
    )

    @classmethod
    def as_form(
        cls,
        image: UploadFile,
        mask_image: UploadFile,
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
        seed: Optional[int] = Form(
            None, description="the seed (for reproducible sampling)"
        ),
    ):
        return cls(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
