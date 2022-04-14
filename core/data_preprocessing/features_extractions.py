from ast import In
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from config.config import VGG_RESIZE, INCEPTIONRESNET_RESIZE
from keras.models import Model
import cv2 as cv

def vgg16_extractor(img):
    img = cv.resize(img, VGG_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    model = VGG16()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def vgg19_extractor(img):
    img = cv.resize(img, VGG_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    model = VGG19()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def inception_resnet_extractor(img):
    img = cv.resize(img, INCEPTIONRESNET_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    model = InceptionResNetV2()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features




