def tpot_classifier(**kwargs):
    from marvinml.backend.tpotbackend import TpotClassifierBackend
    return TpotClassifierBackend(**kwargs)

def tpot_regressor(**kwargs):
    from marvinml.backend.tpotbackend import TpotRegressorBackend
    return TpotRegressorBackend(**kwargs)

def h2o(**kwargs):
    from marvinml.backend.h2obackend import H2OClassifierBackend
    return H2OClassifierBackend(**kwargs)

def autosk_classifier(**kwargs):
    from marvinml.backend.autoskbackend import AutoskClassifierBackend
    return AutoskClassifierBackend(**kwargs)

def autosk_regressor(**kwargs):
    from marvinml.backend.autoskbackend import AutoskRegressorBackend
    return AutoskRegressorBackend(**kwargs)

def autoviz_eda(**kwargs):
    from marvinml.backend.autovizbackend import AutoVizEDABackend
    return AutoVizEDABackend(**kwargs)

def prep(**kwargs):
    from marvinml.backend.datapreparation import preprocess
    return preprocess(**kwargs)

def trans(**kwargs):
    from marvinml.backend.datapreparation import transform
    return transform(**kwargs)

def data_prep(**kwargs):
    from marvinml.backend.datapreparation import data_preparation
    return data_preparation(**kwargs)

BACKENDS_CLASSIFIER = {
    'autosk': autosk_classifier,
    'tpot': tpot_classifier,
    'h2o': h2o
}

BACKENDS_REGRESSOR = {
    'autosk': autosk_regressor,
    'tpot': tpot_regressor,
    'h2o': h2o
}

BACKENDS_EDA = {
    'autoviz': autoviz_eda
}

BACKENDS_DATA_PREP = {
    'preprocess': prep,
    'data_prep': data_prep,
    'trans': trans
}

def get_automl_classifier(backend,**kwargs):
    return BACKENDS_CLASSIFIER.get(backend)(**kwargs)

def get_automl_regressor(backend,**kwargs):
    return BACKENDS_REGRESSOR.get(backend)(**kwargs)

def get_auto_eda(backend,**kwargs):
    return BACKENDS_EDA.get(backend)(**kwargs)

def get_data_preparation(backend, **kwargs):
    return BACKENDS_DATA_PREP.get(backend)(**kwargs)