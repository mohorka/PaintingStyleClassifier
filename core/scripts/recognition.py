from typing import Counter
from core.config.config import NETS
from core.data_preprocessing.load_data import data_load
from core.models.classifier import Classifier
from core.plots.plot import plot_mask



def recognition(image):
    paintings = data_load()

    classifiers = [Classifier(net) for net in NETS]
    print(classifiers)

    for classifier in classifiers:
        classifier.fit(paintings)
    
    features = []
    labels = Counter()
    for classifier in classifiers:
        pred_label, pred_features = classifier.predict(image)
        labels[pred_label[0]] += 1
        features.append(pred_features[0])
        plot_mask(pred_features[0],classifier.method)
    
    
    label = labels.most_common(1)[0][0]
    return label






