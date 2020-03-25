import pickle as pkl

import pandas as pd
from sklearn import base
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.testing import all_estimators

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


def GetRegressor(modelName, mode="normal"):
    if mode == "normal":
        if modelName in GetAllRegressorNames():
            return Regressor(allRegressorClasses[modelName])
    elif mode == "unit":
        if modelName in GetAllRegressorNames():
            return UnitRegressor(allRegressorClasses[modelName])
    raise NotImplementedError


def GetAllRegressorNames():
    return allRegressorClasses.keys()


class UnitRegressor(Regressor):
    def __init__(self, class_):
        super(UnitRegressor, self).__init__(class_)
        self.regressor = {}

    def unitTrainer(self, X, Y):
        if len(Y.shape) == 1:
            regressor = self.class_()
            self.numOutput = 1
            regressor.fit(X, Y)
        elif Y.shape[1] > 1:
            regressor = MultiOutputRegressor(self.class_())
            self.numOutput = Y.shape[1]
            regressor.fit(X, Y)
        else:
            raise ValueError
        return regressor

    def train(self, X, Y):
        validX = X.loc[Y.index]
        keys = validX["BUSSTOP_ID"] + validX["DEST_BUSSTOP_ID"]
        self.regressor = {key: self.unitTrainer(validX[keys == key], Y[keys == key]) for key in keys.unique()}

    def predict(self, X):
        keys = X["BUSSTOP_ID"] + X["DEST_BUSSTOP_ID"]
        preds = pd.concat([self.regressor[key].predict(X[keys == key]) for key in keys.unique()])
        return preds.loc[X.index]


class ReturnMean:
    def __init__(self):
        self.means = None
        self.columns = ["BUSROUTE_ID", "BUSSTOP_ID", "DEST_BUSSTOP_ID"]

    def getkeys(self, X):
        keys = X[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            keys += "-" + X[col].astype(str)
        return keys

    def fit(self, X, Y):
        featureCombinations = self.getkeys(X)
        tmp = pd.concat([featureCombinations, Y], axis=1)
        tmp.columns = ["feature", "time"]
        self.means = tmp.groupby("feature").mean()["time"]
        self.means["general"] = Y.mean()

    def predict(self, X):
        featureCombinations = pd.DataFrame(self.getkeys(X))
        featureCombinations.columns = ["feature"]
        preds = pd.merge(left=featureCombinations, right=self.means.reset_index(), how='left', left_on="feature",
                         right_on="feature").fillna(self.means["general"])
        return preds["time"].values


allRegressorClasses["MeanRegressor"] = ReturnMean
