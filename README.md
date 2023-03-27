# Open Dream AI

Open Dream AI is a deep learning-based image generation platform that uses StableDiffusion to generate high-quality images from text prompts. This repository contains the source code for the Open Dream AI platform.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the Open Dream AI platform, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Configure the platform by editing the `app/core/config.py` file. The configuration file contains settings for the device to use (CPU or GPU), the project name, the server name, and the output folder to use for storing generated images.
4. Start the API server by running `uvicorn app.main:app --reload`. This will start the API server on port 8000.

## Usage

To generate images from text prompts using the Open Dream AI platform, follow these steps:

1. Send a POST request to the `v1/open-dream-ai/txt2img/` endpoint with a JSON payload containing the text prompt and other configuration options.
2. The API will generate one or more images from the text prompt using the StableDiffusion pipeline.
3. The API will return a JSON response containing the filenames of the generated images.

For example, to generate a single image from the prompt "a professional photograph of an astronaut riding a triceratops", you could use the following `curl` command:

```curl
curl -X POST "http://localhost:8000/v1/open-dream-ai/txt2img/" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a professional photograph of an astronaut riding a triceratops"}'
```

This will generate an image and return a response like the following:

```json
{
"images": [
"01FEH7W9N9MH84QJN65F2C8Y26.png"
],
"info": "Your request was placed in background. It will be available shortly."
}
```

You can then retrieve the generated image by accessing the file at `output/01FEH7W9N9MH84QJN65F2C8Y26.png` on the server.

## Contributing

We welcome contributions to the Open Dream AI platform! To contribute, follow these steps:

1. Fork this repository to your own account.
2. Create a new feature branch for your changes.
3. Make your changes and commit them to your feature branch.
4. Push your feature branch to your fork of the repository.
5. Submit a pull request from your feature branch to the `main` branch of this repository.

Please make sure that your changes follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide and include appropriate unit tests.

## License

The Open Dream AI platform is released under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.
