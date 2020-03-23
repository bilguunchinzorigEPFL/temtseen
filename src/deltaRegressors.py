from sklearn import base
from sklearn.utils.testing import all_estimators
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import pickle as pkl

allRegressorClasses = {name: class_ for name, class_ in all_estimators() if issubclass(class_, base.RegressorMixin)}

class Regressor:
    def __init__(self, class_):
        self.class_ = class_
        self.numOutput = 0
        self.regressor = None

    def train(self, X, Y):
        validX = X.loc[Y.index]
        if len(Y.shape) == 1:
            self.regressor = self.class_()
            self.numOutput = 1
            self.regressor.fit(validX, Y)
        elif Y.shape[1] > 1:
            self.regressor = MultiOutputRegressor(self.class_())
            self.numOutput = Y.shape[1]
            self.regressor.fit(validX, Y)
        else:
            raise ValueError

    def predict(self, X):
        if self.regressor is None:
            raise AttributeError
        as_np = self.regressor.predict(X)
        return pd.Series(as_np, index=X.index)

    def save(self, filePath):
        pkl.dump(self.regressor, open(filePath, "wb"))

    def load(self, filePath):
        self.regressor = pkl.load(open(filePath, "rb"))


def GetRegressor(modelName):
    if modelName in GetAllRegressorNames():
        return Regressor(allRegressorClasses[modelName])
    else:
        raise NotImplementedError


def GetAllRegressorNames():
    return allRegressorClasses.keys()
