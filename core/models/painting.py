import PIL
from dataclasses import dataclass

@dataclass
class Painting:
    """Representation of painting
    """   
    image: PIL.JpegImagePlugin.JpegImageFile
    label: str