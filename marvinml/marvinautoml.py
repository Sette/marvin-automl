
def tpot_classifier(**kwargs):
    from marvinml.backend.tpotbackend import TpotClassifierBackend
    return TpotClassifierBackend(**kwargs)

def tpot_regressor(**kwargs):
    from marvinml.backend.tpotbackend import TpotRegressorBackend
    return TpotRegressorBackend(**kwargs)

def autosk_classifier(**kwargs):
    from marvinml.backend.autoskbackend import AutoskClassifierBackend
    return AutoskClassifierBackend(**kwargs)

def autosk_regressor(**kwargs):
    from marvinml.backend.autoskbackend import AutoskRegressorBackend
    return AutoskRegressorBackend(**kwargs)


BACKENDS_CLASSIFIER = {
    'autosk': tpot_classifier,
    'tpot':tpot_regressor
}

BACKENDS_REGRESSOR = {
    'autosk': autosk_classifier,
    'tpot':autosk_regressor
}


def get_automl_classifier(backend,**kwargs):
    return BACKENDS_CLASSIFIER.get(backend)(**kwargs)


def get_automl_regressor(backend,**kwargs):
    return BACKENDS_REGRESSOR.get(backend)(**kwargs)