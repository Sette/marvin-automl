import h2o
import pandas as pd
from h2o.automl import H2OAutoML
#from h2o.sklearn import H2OAutoMLClassifier
#from h2o.sklearn import H2OAutoMLRegressor

from marvinml.backend.automl import MarvinAutomlClassifier, MarvinAutomlRegressor


class H2OClassifierBackend(H2OAutoML, MarvinAutomlClassifier):

    def __init__(self, *args,**kwargs):
        super(H2OClassifierBackend, self).__init__(*args, **kwargs)
    
    def fit(self, x, y, training_frame, *args, **kwargs):
        x = x.tolist()
        training_frame = pd.concat([training_frame[0], training_frame[1]], axis=1, sort=False)
        training_frame = h2o.H2OFrame.from_python(training_frame)
        self.train(x=x, y=y, training_frame=training_frame, **kwargs)

class H2ORegressorBackend(H2OAutoML, MarvinAutomlRegressor):

    def __init__(self, *args,**kwargs):
        super(H2ORegressorBackend, self).__init__(*args, **kwargs)
    
    def fit(self, x, y, training_frame, *args, **kwargs):
        x = x.tolist()
        training_frame = pd.concat([training_frame[0], training_frame[1]], axis=1, sort=False)
        training_frame = h2o.H2OFrame.from_python(training_frame)
        self.train(x=x, y=y, training_frame=training_frame, **kwargs)