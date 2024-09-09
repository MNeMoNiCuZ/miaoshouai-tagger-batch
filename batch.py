import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from pathlib import Path
from tqdm import tqdm
import argparse
import requests
import random

# Configuration options
PRINT_CAPTIONS = False  # Print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Print captioning file status to the console
OVERWRITE = True  # Allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
STRIP_LINEBREAKS = True  # Remove line breaks from generated captions before saving
DEFAULT_SAVE_FORMAT = ".txt"  # Default format for saving captions

# Image resizing options
MAX_WIDTH = 1024  # Set to 0 or less to ignore
MAX_HEIGHT = 1024  # Set to 0 or less to ignore

# Generation parameters
REPETITION_PENALTY = 1.3  # Penalty for repeating phrases, float ~1.5
TEMPERATURE = 0.7  # Sampling temperature to control randomness
TOP_K = 50  # Top-k sampling to limit number of potential next tokens
BATCH_SIZE = 7  # How many images to process at one time

# Default values for input folder, output folder, and prompt
DEFAULT_INPUT_FOLDER = Path(__file__).parent / "input"  # Folder containing input images
DEFAULT_OUTPUT_FOLDER = DEFAULT_INPUT_FOLDER  # Folder for saving captions
DEFAULT_PROMPT = "<RANDOM>"  # Default prompt for generating captions
# <GENERATE_TAGS> Generate prompt as danbooru style tags
# <CAPTION> A one line caption for the image
# <DETAILED_CAPTION> A structured caption format which detects the position of the subjects in the image
# <MORE_DETAILED_CAPTION> A very detailed description for the image
# <MIXED_CAPTION> A mixed caption style of more detailed caption and tags, this is extremely useful for FLUX model when using T5XXL and CLIP_L together. A new node in MiaoshouTagger ComfyUI is added to support this instruction.
# <EMPTY> Creates an empty text-file.
# <RANDOM> Randomly picks one caption type from the ones above (configured in the variable below). The total amount will be evenly distributed.

# Available prompt styles for the <RANDOM> mode
random_prompts_list = ["<GENERATE_TAGS>", "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<MIX_CAPTION>", "<EMPTY>"]

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch process images and generate captions.")
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER, help="Path to input folder containing images.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Path to output folder for saving captions.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt type from a list for generating the caption. Available prompt types are:<GENERATE_TAGS><CAPTION><DETAILED_CAPTION><MORE_DETAILED_CAPTION><MIXED_CAPTION><EMPTY><RANDOM>")
    parser.add_argument("--save_format", type=str, default=DEFAULT_SAVE_FORMAT, help="Format for saving captions (e.g., .txt, .md, .json).")
    parser.add_argument("--max_width", type=int, default=MAX_WIDTH, help="Maximum width for resizing images.")
    parser.add_argument("--max_height", type=int, default=MAX_HEIGHT, help="Maximum height for resizing images.")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, help="Penalty for repetition during generation.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature for generation.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k sampling during generation.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of images to process at one time.")
    return parser.parse_args()


# Filter images that don't have output files
def filter_images_without_output(input_folder, save_format):
    images_to_caption = []
    skipped_images = 0
    total_images = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                total_images += 1
                image_path = os.path.join(root, file)
                output_path = os.path.splitext(image_path)[0] + save_format
                if not OVERWRITE and os.path.exists(output_path):
                    skipped_images += 1
                else:
                    images_to_caption.append(image_path)

    return images_to_caption, total_images, skipped_images

# Resize the image proportionally based on max width and/or height
def resize_image_proportionally(image, max_width=None, max_height=None):
    if (max_width is None or max_width <= 0) and (max_height is None or max_height <= 0):
        return image

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if max_width and not max_height:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    elif max_height and not max_width:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_width
        new_height = max_height
        if new_width / aspect_ratio > new_height:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# Save caption to file
def save_caption_to_file(image_path, caption, save_format):
    txt_file_path = os.path.splitext(image_path)[0] + save_format
    caption = PREPEND_STRING + caption + APPEND_STRING

    with open(txt_file_path, "w") as txt_file:
        txt_file.write(caption)

    if PRINT_CAPTIONING_STATUS:
        print(f"Caption for {os.path.abspath(image_path)} saved in {save_format} format.")

# Process all images in a folder
def process_images_in_folder(images_to_caption, model, processor, prompt, save_format, max_width, max_height, repetition_penalty, temperature, top_k, batch_size):
    # Shuffle the images
    random.shuffle(images_to_caption)
    total_images = len(images_to_caption)
    num_batches = total_images // batch_size + (1 if total_images % batch_size > 0 else 0)

    # Handle the <RANDOM> option with balanced rotation starting from a random index
    if prompt == "<RANDOM>":
        start_index = random.randint(0, len(random_prompts_list) - 1)
        print(f"Starting balanced rotation at index: {start_index}")
    else:
        print(f"Using prompt: {prompt}")

    # Count how many images need resizing
    images_to_resize = 0
    for image_path in images_to_caption:
        image = Image.open(image_path).convert("RGB")
        if image.width > max_width or image.height > max_height:
            images_to_resize += 1

    if images_to_resize > 0:
        print(f"{images_to_resize} images will be resized to the maximum sizes ({max_width}x{max_height}).")
        print("This resizing will not save over the images; it is only applied during inference.")

    # Process images in batches
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch = images_to_caption[i * batch_size:(i + 1) * batch_size]

        for j, image_path in enumerate(batch):
            # Select the prompt in a balanced rotation if <RANDOM> is used
            if prompt == "<RANDOM>":
                current_prompt = random_prompts_list[(start_index + i * batch_size + j) % len(random_prompts_list)]
            else:
                current_prompt = prompt

            print(f"Using prompt: {current_prompt} for image {os.path.basename(image_path)}")

            if current_prompt == "<EMPTY>":
                # Save empty text files for <EMPTY> prompt
                save_caption_to_file(image_path, "", save_format)
                print(f"Generated empty caption for {os.path.abspath(image_path)}.")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                # Resize image if needed
                image = resize_image_proportionally(image, max_width, max_height)
                inputs = processor(text=current_prompt, images=image, return_tensors="pt").to(device)

                inputs_batch = {"input_ids": inputs["input_ids"], "pixel_values": inputs["pixel_values"]}

                generated_ids = model.generate(
                    input_ids=inputs_batch["input_ids"],
                    pixel_values=inputs_batch["pixel_values"],
                    max_new_tokens=1024,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_k=top_k,
                    num_beams=3
                )

                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                cleaned_text = generated_text.replace('<pad>', '').strip()

                parsed_answer = processor.post_process_generation(cleaned_text, task=current_prompt, image_size=(image.width, image.height))
                caption = parsed_answer.get(current_prompt, "")
                save_caption_to_file(image_path, caption, save_format)

                if PRINT_CAPTIONS:
                    print(f"Caption for {os.path.abspath(image_path)}: {caption}")

            except Exception as e:
                print(f"Error processing {os.path.abspath(image_path)}: {str(e)}")

        torch.cuda.empty_cache()
        
        
# Custom get_imports to remove flash_attn
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
    return imports

# Main function
if __name__ == "__main__":
    args = parse_arguments()
    input_folder = args.input_folder
    output_folder = args.output_folder
    prompt = args.prompt  # Now taken as an input argument
    save_format = args.save_format
    max_width = args.max_width
    max_height = args.max_height
    repetition_penalty = args.repetition_penalty
    temperature = args.temperature
    top_k = args.top_k
    batch_size = args.batch_size  # Now taken as an input argument

    images_to_caption, total_images, skipped_images = filter_images_without_output(input_folder, save_format)

    print(f"\nFound {total_images} images.")
    if not OVERWRITE:
        print(f"{skipped_images} images already have captions with format {save_format}, skipping.")
    print(f"\nCaptioning {len(images_to_caption)} images.\n\n")

    if len(images_to_caption) > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True).to(device).eval()
        
        processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True)

        process_images_in_folder(images_to_caption, model, processor, prompt, save_format, max_width, max_height, repetition_penalty, temperature, top_k, batch_size)
    else:
        print("No images to process. Exiting.")