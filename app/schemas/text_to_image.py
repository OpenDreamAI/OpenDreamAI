from typing import List, Optional

from pydantic import BaseModel, Field, validator


class TextToImageRequest(BaseModel):
    """
    Pydantic model for generating images from text using StableDiffusion.

    Attributes:
        prompt (List[str]): The prompts to render.
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

    prompt: List[str] = Field(
        [],
        description="the prompt to render",
    )
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
    num_images_per_prompt: Optional[int] = Field(
        1,
        description="how many samples to produce for each given prompt. A.k.a batch size",
        ge=1,
        le=10,
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
    seed: Optional[int] = Field(
        None, description="the seed (for reproducible sampling)"
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": ["a professional photograph of an astronaut riding a triceratops"],
                "steps": 50,
                "height": 512,
                "width": 512,
                "num_images_per_prompt": 1,
                "guidance_scale": 7.5,
            }
        }
        exclude = {"eta", "seed"}

    @validator("height", "width", pre=True)
    def check_dimensions(cls, v):
        if v % 8 != 0:
            raise ValueError("Dimensions must be divisible by 8.")
        return v

    @validator("prompt", pre=False)
    def check_prompts(cls, prompts):
        if not prompts:
            raise ValueError("Prompts cannot be empty.")
        for prompt in prompts:
            if not prompt:
                raise ValueError("Prompts cannot be empty.")

        return prompts


class ImagesResponse(BaseModel):
    """
    Pydantic model for the response returned by the image generation API.

    Attributes:
        images (List[str]): A list of image filenames generated from the input prompt.
        info (str): A message providing information about the generation process.

    Notes:
        This Pydantic model is used to represent the response returned by image generation API.
        It contains a list of filenames for the generated images and a message providing information about
        generation process.
    """

    images: List[str]
    info: str
