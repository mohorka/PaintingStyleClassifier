from typing import List
import os
from core.models.painting import Painting
from core.config.config import DATA_DIR, RESIZE
from keras.preprocessing.image import load_img, img_to_array

def data_load() -> List[Painting]:
    data = []
    for style in os.listdir(DATA_DIR):
        print(style)
        for painting in os.listdir(DATA_DIR+style):
            image = load_img(DATA_DIR+style+'/'+painting,target_size=RESIZE)
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            data.append(Painting(image,style))
    return data


