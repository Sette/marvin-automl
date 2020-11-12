from autoviz.AutoViz_Class import AutoViz_Class
from marvinml.backend.automl import MarvinAutoVizEDA


class AutoVizEDABackend(AutoViz_Class, MarvinAutoVizEDA):

    def __init__(self, *args,**kwargs):
        super(AutoVizEDABackend, self).__init__(*args, **kwargs)
    
    def fit(self, *args,**kwargs):
        self.AutoViz(args[0])