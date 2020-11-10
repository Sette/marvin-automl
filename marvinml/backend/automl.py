from abc import ABCMeta, abstractmethod

class MarvinH2oAutoml(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

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

