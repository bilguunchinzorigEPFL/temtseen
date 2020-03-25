import os

import pandas as pd

dataName = "na"


def loadX(featureType):
    return pd.read_csv(os.path.join("../data", dataName, featureType + ".csv"), index_col=0)


def calcTimeDelta(X):
    diff = -X["TIMESTAMP"].diff(-1)
    skip = (X["BUS_ID"].diff(-1) == 0) & (X["BUSSTOP_SEQ"].diff(-1) < 0) & (diff < 3600)
    diff = diff[skip]
    return diff
