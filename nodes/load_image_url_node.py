from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    # if multiple images and all are the same size, concatenate them
    # otherwise, return the first image
    if len(output_images) > 1:
        if all(img.shape == output_images[0].shape for img in output_images):
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.TooManyRedirects,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
        requests.exceptions.HTTPError
    )),
    reraise=True
)
def load_image(image_source):
    if image_source.startswith('http'):
        print(f"Fetching image from URL: {image_source}")
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        print(f"Loading image from path: {image_source}")
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


class LoadImageByUrlOrPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }


    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, url_or_path):
        print(url_or_path)
        img, name = load_image(url_or_path)
        img_out, mask_out = pil2tensor(img)
        return (img_out, mask_out)


if __name__ == "__main__":
    img, name = load_image("https://creativestorage.blob.core.chinacloudapi.cn/test/bird.png")
    img_out, mask_out = pil2tensor(img)

