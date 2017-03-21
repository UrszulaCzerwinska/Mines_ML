from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn import ensemble


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = ensemble.GradientBoostingClassifier(n_estimators=70, learning_rate=1.0, max_depth=1, random_state=0)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
