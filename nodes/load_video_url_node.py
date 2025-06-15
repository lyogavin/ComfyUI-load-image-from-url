from PIL import Image, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import cv2
import tempfile


def frames2tensor(frames):
    """Convert a list of PIL images (frames) to tensor"""
    output_images = []
    
    for frame in frames:
        frame = ImageOps.exif_transpose(frame)
        if frame.mode == 'I':
            frame = frame.point(lambda i: i * (1 / 255))
        image = frame.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        output_images.append(image)
    
    if len(output_images) > 0:
        # Stack all frames along a new dimension (num_frames, height, width, channels)
        output_tensor = torch.stack(output_images, dim=0)
        return output_tensor
    else:
        # Return empty tensor if no frames
        return torch.zeros((0, 64, 64, 3), dtype=torch.float32)


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
def load_video_frames(video_source):
    """Load video from URL or file path and extract all frames"""
    temp_file = None
    
    try:
        if video_source.startswith('http'):
            print(f"Fetching video from URL: {video_source}")
            response = requests.get(video_source, timeout=60)  # Longer timeout for videos
            response.raise_for_status()
            
            # Create temporary file for video data
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(response.content)
            temp_file.close()
            video_path = temp_file.name
            file_name = video_source.split('/')[-1]
        else:
            print(f"Loading video from path: {video_source}")
            video_path = video_source
            file_name = os.path.basename(video_source)
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_source}")
        
        frames = []
        frame_count = 0
        
        print(f"Extracting frames from video: {file_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Extracted {frame_count} frames...")
        
        cap.release()
        print(f"Total frames extracted: {frame_count}")
        
        if frame_count == 0:
            raise Exception("No frames found in video")
        
        return frames, file_name
        
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


class LoadVideoByUrlOrPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "load"
    CATEGORY = "video"

    def load(self, url_or_path, max_frames=0):
        print(f"Loading video: {url_or_path}")
        frames, name = load_video_frames(url_or_path)
        
        # Limit frames if max_frames is specified and > 0
        if max_frames > 0 and len(frames) > max_frames:
            print(f"Limiting to {max_frames} frames (original: {len(frames)})")
            frames = frames[:max_frames]
        
        frames_tensor = frames2tensor(frames)
        frame_count = len(frames)
        
        print(f"Output tensor shape: {frames_tensor.shape}")
        return (frames_tensor, frame_count)

    @classmethod
    def IS_CHANGED(s, url_or_path, max_frames=0):
        return float("NaN")


if __name__ == "__main__":
    # Test with a sample video URL
    frames, name = load_video_frames("https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4")
    frames_tensor = frames2tensor(frames)
    print(f"Loaded {len(frames)} frames, tensor shape: {frames_tensor.shape}") 