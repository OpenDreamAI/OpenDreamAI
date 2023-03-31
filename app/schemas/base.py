from io import BytesIO
from typing import Optional

from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from PIL.Image import Image
from PIL.Image import open as image_open
from pydantic import BaseModel, Field, root_validator


def prepare_image(image: UploadFile) -> Image:
    contents = image.file.read()
    image: Image = image_open(BytesIO(contents)).convert("RGB")
    return image


class BaseRequestModel(BaseModel):
    """
    Base model to inherit request models from.

    Attributes:
        prompt (list[str]): The prompt to use for the depth-to-image transformation.
        num_images_per_prompt (Optional[int]): The number of output images to generate per prompt. Defaults to 1.
    """

    prompt: list[str] = Field(
        [],
        description="the prompt to render",
    )
    num_images_per_prompt: Optional[int] = Field(
        1,
        description="how many samples to produce for each given prompt. A.k.a batch size",
        ge=1,
        le=10,
    )
    seed: Optional[int] = Field(
        None, description="the seed (for reproducible sampling)"
    )
    total_steps: int

    class Config:
        arbitrary_types_allowed = True

        exclude = {"total_steps"}

    @root_validator(pre=True)
    def calculate_total_steps(cls, values):
        values["total_steps"] = int(
            values.get("strength", 1) * values.get("num_inference_steps", 1)
        )
        if values["total_steps"] < 1:
            raise HTTPException(
                422, detail="(num_inference_steps * strength) should be greater than 1!"
            )
        return values
