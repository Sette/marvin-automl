
from marvinml.marvinautoml import get_automl_classifier,get_automl_regressor

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)


tpot = get_automl_classifier(backend="tpot",generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)






