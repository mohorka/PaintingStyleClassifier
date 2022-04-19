from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_input
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_input
from keras.applications.xception import Xception, preprocess_input as xception_input
from keras.applications.resnet import ResNet50,  preprocess_input as resnet50_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet_input
from core.config.config import VGG_RESIZE, INCEPTION_RESIZE, RESNET_RESIZE
from keras.models import Model
import cv2 as cv

def vgg16_extractor(img):
    img = cv.resize(img, VGG_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = vgg16_input(img)

    model = VGG16()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def vgg19_extractor(img):
    img = cv.resize(img, VGG_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = vgg19_input(img)

    model = VGG19()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def xception_extractor(img):
    img = cv.resize(img, INCEPTION_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = xception_input(img)

    model = Xception()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def resnet50_extractor(img):
    img = cv.resize(img, RESNET_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = resnet50_input(img)

    model = ResNet50()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def inceptionv3_extractor(img):
    img = cv.resize(img, INCEPTION_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = inception_input(img)

    model = InceptionV3()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features




def inception_resnet_extractor(img):
    img = cv.resize(img, INCEPTION_RESIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = inception_resnet_input(img)

    model = InceptionResNetV2()
    model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
    features = model.predict(img)
    return features

def feature_extract(img, model):
    img = inception_resnet_input(img)
    features = model.predict(img)
    return features



