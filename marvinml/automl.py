from abc import ABCMeta, abstractmethod

from autosklearn.classification import AutoSklearnClassifier,AutoSklearnRegressor

class MarvinAutomlClassifier(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class MarvinAutomlRegressor(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass

class AutoskClassifierBackend(AutoSklearnClassifier,MarvinAutomlClassifier):

    def __init__(self):
        super()


class AutoskRegressorBackend(AutoSklearnClassifier,MarvinAutomlClassifier):

    def __init__(self):
        super()


BACKENDS_CLASSIFIER = {
    'autosk': AutoskClassifierBackend()
}

BACKENDS_REGRESSOR = {
    'autosk': AutoskRegressorBackend()
}



def get_automl_classifier(backend):
    return BACKENDS_CLASSIFIER.get(backend)


def get_automl_regressor(backend):
    return BACKENDS_REGRESSOR.get(backend)