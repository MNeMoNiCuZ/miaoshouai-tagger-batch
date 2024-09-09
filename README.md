# Miaoshouai Caption Batch
This tool uses the [MiaoshouAI/Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5) finetune based on the [Florence2 VLM](https://huggingface.co/microsoft/Florence-2-large) from Microsoft to caption images in an input folder. Thanks to their team for training this great model.

It's a very fast and fairly robust captioning model that can produce good outputs in 3 different levels of detail.

## Requirements
* Python 3.10 or above.
  * It's been tested with 3.10, 3.11 and 3.12.

* Cuda 12.1.
  * It may work with other versions. Untested.
 
To use CUDA / GPU speed captioning, you'll need ~6GB VRAM or more.

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and change the BATCH_SIZE = 7 value to match the level of your GPU.

>   For a 6gb VRAM GPU, use 1.
  
>   For a 24gb VRAM GPU, use 7.

## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Detail Mode
You can edit the variable `PROMPT` to be one of the options listed below:

> `<GENERATE_TAGS>` Generate prompt as danbooru style tags

> `<CAPTION>` A one line caption for the image
  
> `<DETAILED_CAPTION>` A structured caption format which detects the position of the subjects in the image

> `<MORE_DETAILED_CAPTION>` A very detailed description for the image

> `<MIXED_CAPTION>` A mixed caption style of more detailed caption and tags, this is extremely useful for FLUX model when using T5XXL and CLIP_L together. A new node in MiaoshouTagger ComfyUI is added to support this instruction.

> `<EMPTY>` Creates an empty text-file

> `<RANDOM>` Randomly picks one caption type from the ones above. The total amount will be evenly distributed.

Here's an example:

![2024-08-30 - 14 39 59](https://github.com/user-attachments/assets/868fe471-7d2d-4948-bf98-a022279ad923)

`<GENERATE_TAGS>`
> solo, gloves, 1boy, jacket, male focus, boots, outdoors, sky, cloud, motor vehicle, yellow jacket, blue pants, building, bob cut, motorcycle, orange background, motor bike


`<CAPTION>`
> a cartoon-style illustration of a humanoid robot riding a motorcycle in a rural setting


`<DETAILED_CAPTION>`
> a whimsical scene where a humanoid robot is positioned in the center of the frame, facing the viewer directly, the robot is dressed in a yellow jacket, blue pants, and brown boots, with a black mask covering its face and ears, he is seated on a orange motorcycle, positioned at an angle to the right of the image, in the background, a cartoonish character is positioned on the motorcycle, wearing a yellow coat and brown pants, the scene is set against a vibrant sunset backdrop with wooden buildings, power lines, and clouds, creating a sense of depth and atmosphere


`<MORE_DETAILED_CAPTION>`
> a digital illustration in a vibrant, cartoonish style, depicting a humanoid robot riding a motorcycle, the robot is wearing a yellow jacket and blue pants, with a black mask covering its face and ears, giving it a sleek, futuristic appearance, the motorcycle is positioned in the center of the image, with its front wheel facing the viewer, the background features a medieval-style village with wooden buildings and a clear sky with fluffy white clouds, the ground is a cobblestone path, with scattered debris and a utility pole on the left side, the overall color palette is warm, with shades of yellow, orange, and brown dominating the scene, the textures are smooth and polished, typical of high-quality digital art, with bold lines and vibrant colors, enhancing the sense of depth and realism


`<MIXED_CAPTION>`
> a digital illustration in a vibrant, cartoonish style, depicting a humanoid robot riding a motorcycle, the robot is wearing a yellow jacket and blue pants, with a black mask covering its face and eyes, giving it a sleek, futuristic appearance, the motorcycle is positioned in the center of the image, with its front wheel prominently displayed, the background features a rustic, medieval-style village with wooden buildings on either side, the sky is a gradient of warm oranges and pinks, with scattered white clouds and a few small stars, adding a whimsical touch to the scene, the textures are smooth and detailed, with sharp lines and vibrant colors, enhancing the sense of depth and realism, the overall mood is one of adventure and exploration, with the robot occupying the foreground, the illustration is highly polished, with attention to detail in the textures and shading, typical of high-quality digital art   \(solo, gloves, 1boy, jacket, male focus, boots, outdoors, sky, cloud, orange background, motor vehicle, building, bob cut, helmet, motor bike, hoop earrings, house, windmill

`<EMPTY>`
> &nbsp;

`<RANDOM>`
> [a random one from the ones above]
