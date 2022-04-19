from dataclasses import dataclass

import numpy as np

@dataclass
class Painting:
    """Representation of painting
    """   
    image: np.array
    label: str