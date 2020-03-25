import os
import resource
import sys
import time
from datetime import datetime

import pandas as pd

import dataHandler as data
from deltaRegressors import GetRegressor
from evaluators import evaluate
from featureGenerators import GetFeatureGroup, GetAllFeatureGroupNames


def memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (int(soft*0.8), hard))

memory_limit()

# configs
runName = "testRun"
timestamp = str(datetime.now())
featureGroupNames = GetAllFeatureGroupNames()
deltaRegressorNames = [
    'MeanRegressor',
    # 'LinearRegression',
    # 'RandomForestRegressor',
    # 'LinearSVR',
    # 'MLPRegressor',
    # 'GaussianProcessRegressor',
    # 'ElasticNet',
    # 'LassoLars'
]
regressorMode = "normal"
savePath = "../model/" + runName + "/"
data.dataName = "full"

# init
print(f"Loading Data From: {data.dataName}")
featureGroups = [GetFeatureGroup(fg) for fg in featureGroupNames]
deltaRegressors = [GetRegressor(dr) for dr in deltaRegressorNames]
if not os.path.exists(savePath):
    os.mkdir(savePath)
    with open(savePath + "results.csv", "w") as myfile:
        myfile.write(
            f"path, timestamp, trainTdError, trainTsError, evalTdError, evalTsError, featureGroupNames, data\n")

# training
trainFeatures = pd.concat([fg.getFeatures("train") for fg in featureGroups], axis=1)
trainX = data.loadX('train')
trainTimeDelta = data.calcTimeDelta(trainX)

evalFeatures = pd.concat([fg.getFeatures("eval") for fg in featureGroups], axis=1)
evalX = data.loadX('eval')
evalTimeDelta = data.calcTimeDelta(evalX)

for i, deltaRegressorName in enumerate(deltaRegressorNames):
    sys.stdout.write(f"Training Regressor: {deltaRegressorName} : ")
    sys.stdout.flush()
    s = time.time()
    deltaRegressors[i].train(trainFeatures, trainTimeDelta)
    print(f"Time : {time.time() - s}")

    trainTdError, trainTsError = evaluate(
        "Train " + deltaRegressorName,
        deltaRegressors[i],
        trainFeatures,
        trainTimeDelta,
        trainX
    )

    evalTdError, evalTsError = evaluate(
        "Eval " + deltaRegressorName,
        deltaRegressors[i],
        evalFeatures,
        evalTimeDelta,
        evalX
    )

    deltaRegressors[i].save(savePath + deltaRegressorName + "T" + timestamp)
    with open(savePath + "results.csv", "a") as myfile:
        myfile.write(
            f"{savePath + deltaRegressorName} T {timestamp}, {timestamp}, {trainTdError}, {trainTsError}, {evalTdError}, {evalTsError}, {featureGroupNames}, {data.dataName}\n")
