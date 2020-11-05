
from tpot import TPOTClassifier
from tpot import TPOTRegressor

from marvinml.backend.automl import MarvinAutomlClassifier, MarvinAutomlRegressor


class TpotClassifierBackend(TPOTClassifier,MarvinAutomlClassifier):

    def __init__(self, *args,**kwargs):
        super(TpotClassifierBackend,self). __init__( *args,**kwargs)


class TpotRegressorBackend(TPOTRegressor,MarvinAutomlClassifier):

    def __init__(self, *args,**kwargs):
        super(TpotRegressorBackend,self). __init__( *args,**kwargs)