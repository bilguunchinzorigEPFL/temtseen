import os

import pandas as pd

dataName = "na"


def loadX(featureType):
    return pd.read_csv(os.path.join("../data", dataName, featureType + ".csv"), index_col=0)


def calcTimeDelta(X):
    diff = -X["TIMESTAMP"].diff(-1)
    return diff[X["BUS_ID"].diff(-1) == 0]
