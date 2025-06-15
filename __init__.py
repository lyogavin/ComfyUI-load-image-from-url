from .nodes.load_image_url_node import LoadImageByUrlOrPath
from .nodes.load_video_url_node import LoadVideoByUrlOrPath


NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrlOrPath": LoadImageByUrlOrPath,
    "LoadVideoFromUrlOrPath": LoadVideoByUrlOrPath,
}

