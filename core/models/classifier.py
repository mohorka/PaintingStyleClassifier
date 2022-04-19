from core.models.painting import Painting
from typing import List
import numpy as np
from core.data_preprocessing.features_extractions import feature_extract
from core.data_preprocessing.load_data import data_load
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from core.config.config import RESIZE

class Classifier:
    def __init__(self, method) -> None:
        if method == 'vgg16':
            self.model = VGG16(include_top=False, input_shape=RESIZE)
        elif method == 'vgg19':
            self.model = VGG19(include_top=False, input_shape=RESIZE)
        elif method == 'xception':
            self.model = Xception(include_top=False, input_shape=RESIZE)
        elif method == 'resnet50':
            self.model = ResNet50(include_top=False, input_shape=RESIZE)           
        elif method == 'inception_v3':
            self.model = InceptionV3(include_top=False, input_shape=RESIZE)
        elif method == 'inception_resnet_v2':
            self.model = InceptionResNetV2(include_top=False, input_shape=RESIZE)
        else: self.model = None
        self.method = method
        self.references: List[Painting] = []
    

    def __get_features(self, image):
        print(self.method)
        return feature_extract(image,self.model)
        
    def __loss(self,  reference_feature:np.array, test_feature:np.array):
        return np.linalg.norm(np.subtract(reference_feature,test_feature))
    
    def __search(self, test:np.array):
        min_loss = np.inf
        reference = None
        #TO_DO -- change cause our class has non array field image
        for ref in (self.references):
            if (loss := self.__loss(ref.image,test)) < min_loss:
                min_loss = loss
                reference = ref

        return reference
    
    def fit(self, data: List[Painting]):
        references =[]
        for painting in data:
            references.append(Painting(self.__get_features(painting.image),painting.label))
        self.references = references
    
    def predict(self, paintings:List[np.array]):
        pred_labels = []
        pred_features =[]
        for painting in paintings:
            paint_feature = self.__get_features(painting)
            ref = self.__search(paint_feature)
            pred_labels.append(ref.label)
            pred_features.append(ref.image)
        return pred_labels, pred_features
    
    def score(self, true_labels:np.array, pred_labels:np.array) -> float:
        if len(true_labels) != len(pred_labels):
            raise Exception("Different length of true and pred labels")
        true = 0
        for idx, true_label in enumerate(true_labels):
            if true_label == pred_labels[idx]:
                true += 1
        
        return true / len(pred_labels)





    



        