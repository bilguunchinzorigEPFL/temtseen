import os
import resource
import sys
import time
import warnings
from datetime import datetime

import pandas as pd

import dataHandler as data
from deltaRegressors import GetRegressor
from evaluators import evaluate
from featureGenerators import GetFeatureGroup, GetAllFeatureGroupNames

warnings.filterwarnings("ignore")

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * 0.9), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


memory_limit()

# configs
runName = "testRun"
timestamp = str(datetime.now())
featureGroupNames = GetAllFeatureGroupNames()
deltaRegressorNames = [
    # 'MeanRegressor',
    # 'MeanConfRegressor',
    # 'LassoLars',
    # 'GaussianProcessRegressor',
    # 'LinearRegression',
    'RandomForestRegressor',
    # 'GradientBoostingRegressor',
    # 'LinearSVR',
    # 'MLPRegressor',
    # 'ElasticNet',
]
regressorMode = "unit"
savePath = "../model/" + runName + "/"
data.dataName = "testM"

# init
print(f"Loading Data From: {data.dataName}")
featureGroups = [GetFeatureGroup(fg) for fg in featureGroupNames]
deltaRegressors = [GetRegressor(dr, regressorMode) for dr in deltaRegressorNames]
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
    #deltaRegressors[i].load(savePath + deltaRegressorName + "T2020-03-26 17:18:13.117775")
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
