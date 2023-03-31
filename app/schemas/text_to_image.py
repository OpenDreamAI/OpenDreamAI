from typing import Optional

from pydantic import Field, validator

from app.schemas.base import BaseRequestModel


class TextToImageRequest(BaseRequestModel):
    """
    Pydantic model for generating images from text using StableDiffusion.

    Attributes:
        prompt (list[str]): The prompts to render.
        num_inference_steps (int, optional): The number of ddim sampling steps. Default value is 50.
        height (int, optional): The image height in pixel space. Default value is 512.
        width (int, optional): The image width in pixel space. Default value is 512.
        num_images_per_prompt (int, optional): How many samples to produce for each given prompt. A.k.a batch size.
        Default value is 1.
        guidance_scale (float, optional): Unconditional guidance scale. Default value is 7.5.
        eta (float, optional): Ddim eta (eta=0.0 corresponds to deterministic sampling).
        seed (int, optional): The seed (for reproducible sampling).

    Raises:
        ValueError: If the height or width is not divisible by 8.

    Notes:
        This Pydantic model is used for generating images from text using StableDiffusion. It specifies the parameters
        for generating images, such as the prompt, the image size, and the number of samples to produce. The model also
        includes default values for some parameters and constraints on the allowed values of others.
    """

    num_inference_steps: Optional[int] = Field(
        50, description="number of ddim sampling steps", ge=1, le=150, alias="steps"
    )
    height: Optional[int] = Field(
        512,
        description="image height, in pixel space, must be divisible by 8",
        ge=8,
        le=768,
    )
    width: Optional[int] = Field(
        512,
        description="image width, in pixel space, must be divisible by 8",
        ge=8,
        le=768,
    )
    guidance_scale: Optional[float] = Field(
        7.5,
        description="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        ge=0,
        le=20,
    )
    eta: Optional[float] = Field(
        None, description="ddim eta (eta=0.0 corresponds to deterministic sampling"
    )

    class Config:
        """
        Configuration class for pydantic model.
        """

        schema_extra = {
            "example": {
                "prompt": [
                    "a professional photograph of an astronaut riding a triceratops"
                ],
                "steps": 50,
                "height": 512,
                "width": 512,
                "num_images_per_prompt": 1,
                "guidance_scale": 7.5,
                "eta": None,
                "seed": None,
            }
        }

    @validator("height", "width", pre=True)
    def check_dimensions(cls, value):
        """
        Validator to check correctness of height and width attributes.

        Parameters:
            value: height or width value

        Returns:
            Value if it's correct
        """
        if value % 8 != 0:
            raise ValueError("Dimensions must be divisible by 8.")
        return value

    @validator("prompt", pre=False)
    def check_prompts(cls, prompts):
        """
        Validator to check correctness of prompts.

        Parameters:
            prompts: list of prompts

        Returns:
            Value if it's correct

        Raises:
            ValueError if the list of prompts has incorrect values
        """
        if not prompts:
            raise ValueError("Prompts cannot be empty.")
        for prompt in prompts:
            if not prompt:
                raise ValueError("Prompts cannot be empty.")

        return prompts
