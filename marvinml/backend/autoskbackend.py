
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from automl import MarvinAutomlClassifier, MarvinAutomlRegressor


class AutoskClassifierBackend(AutoSklearnClassifier,MarvinAutomlClassifier):
    
    def __init__(self, *args,**kwargs):
        super(AutoskClassifierBackend,self). __init__( *args,**kwargs)


class AutoskRegressorBackend(AutoSklearnClassifier,MarvinAutomlClassifier):

    def __init__(self, *args,**kwargs):
        super(AutoskRegressorBackend,self). __init__( *args,**kwargs)