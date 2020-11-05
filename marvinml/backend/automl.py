from abc import ABCMeta, abstractmethod

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

