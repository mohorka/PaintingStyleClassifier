from typing import List
import os
from models.painting import Painting
from config.config import DATA_DIR
from keras.preprocessing.image import load_img

def data_load() -> List[Painting]:
    data = []
    for style in os.listdir(DATA_DIR):
        for painting in os.listdir(DATA_DIR+style):
            image = load_img(DATA_DIR+style+painting)
            data.append(Painting(image,style))
    return data


