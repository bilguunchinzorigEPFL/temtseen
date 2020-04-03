import pickle as pkl

import numpy as np
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
        self.givesConfidence = hasattr(class_, 'confidence')

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
        if self.givesConfidence:
            as_np, conf = self.regressor.predict(X)
            self.confidence = pd.Series(conf, index=X.index)
        else:
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
    elif mode == "double":
        if modelName in GetAllRegressorNames():
            return DoubleRegressor(allRegressorClasses[modelName])
    raise NotImplementedError


def GetAllRegressorNames():
    return allRegressorClasses.keys()


class UnitRegressor(Regressor):
    def __init__(self, class_):
        super(UnitRegressor, self).__init__(class_)
        self.regressor = {}
        self.minCount = 2
        self.givesConfidence = True

    def unitTrainer(self, X, Y):
        key = len(X["keys"].unique())
        n_est = 20 if key == 1 else 100
        depth = 5 if key == 1 else 10
        if len(X) < self.minCount:
            return None
        if len(Y.shape) == 1:
            regressor = self.class_(criterion="mae", max_depth=depth, n_estimators=n_est)
            self.numOutput = 1
            regressor.fit(X[X.columns[:-1]], Y)
        elif Y.shape[1] > 1:
            regressor = MultiOutputRegressor(self.class_())
            self.numOutput = Y.shape[1]
            regressor.fit(X[X.columns[:-1]], Y)
        else:
            raise ValueError
        return regressor

    def getKey(self, X):
        return X["BUSSTOP_ID"].astype(str) + "-" + X["DEST_BUSSTOP_ID"].astype(str)

    def train(self, X_, Y):
        X = X_.loc[Y.index]
        X["keys"] = self.getKey(X)
        tmp = pd.concat([X, Y], axis=1)
        group = tmp.groupby("keys")
        counts = group.count()["BUS_ID"]
        x_cols = tmp.columns[:-1]
        y_col = tmp.columns[-1]
        self.regressor = group.apply(lambda data: self.unitTrainer(data[x_cols], data[y_col])).dropna()

        trickyFeatures = counts[counts < self.minCount].index.values
        trickyVals = tmp[tmp["keys"].isin(trickyFeatures)]
        generalY = trickyVals[y_col]  # np.array([Y.median()]*len(trickyVals)) + np.random.normal(size=len(trickyVals))
        self.regressor["general"] = self.unitTrainer(trickyVals[x_cols], generalY)

    def unitPredictor(self, X):
        key = X["keys"].iloc[0]
        cols = X.columns[:-1]
        if key in self.regressor.keys():
            p = self.regressor[key].predict(X[cols])
        else:
            p = self.regressor["general"].predict(X[cols])
        return pd.Series(p, index=X.index)

    def unitConfidence(self, X):
        key = X["keys"].iloc[0]
        cols = X.columns[:-1]
        if key in self.regressor.keys():
            reg = self.regressor[key]
        else:
            reg = self.regressor["general"]
        c = np.stack([est.predict(X[cols]) for est in reg.estimators_], axis=1).std(axis=1)
        return pd.Series(c, index=X.index)

    def predict(self, X):
        X["keys"] = self.getKey(X).apply(lambda key: key if key in self.regressor.keys() else "sexyback")
        group = X.groupby("keys")
        self.confidence = group.apply(lambda x: self.unitConfidence(x)).reset_index(level=0, drop=True)[X.index]
        preds = group.apply(lambda x: self.unitPredictor(x))
        return preds.reset_index(level=0, drop=True)[X.index]


from sklearn.neural_network import MLPClassifier


class DoubleRegressor(Regressor):
    def __init__(self, class_):
        super(DoubleRegressor, self).__init__(class_)
        self.splitThres = 363
        self.regressorSmall = None
        self.regressorBig = None

    def train(self, X, Y):
        smallY = Y[Y < self.splitThres]
        tmpY = pd.concat([Y, Y.shift(1)], axis=1)
        tmpY = tmpY[tmpY.sum(axis=1) > 600]
        tmpY = tmpY / tmpY.sum(axis=1)
        self.regressorSmall = self.class_()
        self.regressorBig = MLPClassifier()
        self.numOutput = 1
        self.regressorSmall.fit(X.loc[smallY.index], smallY)
        self.regressorBig.fit(X.loc[tmpY.index], tmpY)

    def predict(self, X):
        if self.givesConfidence:
            as_np, conf = self.regressor.predict(X)
            self.confidence = pd.Series(conf, index=X.index)
        else:
            as_np = self.regressorSmall.predict(X)
            self.bigPreds = pd.Series(self.regressorBig.predict(X), index=X.index)
        return pd.Series(as_np, index=X.index)


class ReturnMean:
    def __init__(self):
        self.means = None
        self.columns = ["BUSROUTE_ID", "BUSSTOP_ID", "DEST_BUSSTOP_ID"]
        self.stableMinLimit = 2

    def getkeys(self, X):
        keys = X[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            keys += "-" + X[col].astype(str)
        return keys

    def fit(self, X, Y):
        featureCombinations = self.getkeys(X)
        tmp = pd.concat([featureCombinations, Y], axis=1)
        tmp.columns = ["feature", "time"]
        counts = tmp.groupby("feature").count()["time"]
        self.means = tmp.groupby("feature").mean()["time"][counts >= self.stableMinLimit]
        trickyFeatures = counts[counts < self.stableMinLimit].index.values
        trickyVals = tmp[tmp["feature"].isin(trickyFeatures)]["time"]
        self.means["general"] = Y.median()

    def predict(self, X):
        featureCombinations = pd.DataFrame(self.getkeys(X))
        featureCombinations.columns = ["feature"]
        preds = pd.merge(left=featureCombinations, right=self.means.reset_index(), how='left', left_on="feature",
                         right_on="feature").fillna(self.means["general"])
        return preds["time"].values


class ReturnMeanConf:
    confidence = True

    def __init__(self):
        self.means = None
        self.columns = ["BUSROUTE_ID", "BUSSTOP_ID", "DEST_BUSSTOP_ID"]  # "BUSROUTE_ID",
        self.stableMinLimit = 2

    def getkeys(self, X):
        keys = X[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            keys += "-" + X[col].astype(str)
        return keys

    def fit(self, X, Y):
        featureCombinations = self.getkeys(X)
        tmp = pd.concat([featureCombinations, Y], axis=1)
        tmp.columns = ["feature", "time"]
        group = tmp.groupby("feature")
        means = group.mean()["time"]
        confs = group.std()["time"]
        counts = group.count()

        self.means = pd.concat([means, confs], axis=1)[counts >= self.stableMinLimit]
        self.means.columns = ["mean", "conf"]

        trickyFeatures = counts[counts < self.stableMinLimit].index.values
        trickyVals = tmp[tmp["feature"].isin(trickyFeatures)]["time"]
        self.means["mean"]["unknown"] = Y.median()
        self.means["conf"]["unknown"] = np.mean(np.abs(trickyVals - Y.median()))

    def predict(self, X):
        featureCombinations = pd.DataFrame(self.getkeys(X))
        featureCombinations.columns = ["feature"]
        preds = pd.merge(left=featureCombinations, right=self.means.reset_index(), how='left', left_on="feature",
                         right_on="feature")
        preds["mean"][pd.isnull(preds["mean"])] = self.means["mean"]["unknown"]
        preds["conf"][pd.isnull(preds["conf"])] = self.means["conf"]["unknown"]

        return preds["mean"].values, preds["conf"].values


allRegressorClasses["MeanRegressor"] = ReturnMean
allRegressorClasses["MeanConfRegressor"] = ReturnMeanConf
