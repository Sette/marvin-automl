import h2o
from h2o.automl import H2OAutoML
#from h2o.sklearn import H2OAutoMLClassifier
#from h2o.sklearn import H2OAutoMLRegressor

from marvinml.backend.automl import MarvinH2oAutoml


class H2OClassifierBackend(H2OAutoML, MarvinH2oAutoml):

    def __init__(self, *args,**kwargs):
        super(H2OClassifierBackend, self).__init__(*args, **kwargs)


class H2ORegressorBackend(H2OAutoML, MarvinH2oAutoml):

    def __init__(self, *args,**kwargs):
        super(H2ORegressorBackend, self).__init__(*args, **kwargs)