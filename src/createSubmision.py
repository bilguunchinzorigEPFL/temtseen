import resource
from datetime import datetime

import pandas as pd

import dataHandler as data
from deltaRegressors import GetRegressor
from evaluators import timeStampPredictor, evaluate, timeStampPredictorFast
from featureGenerators import GetFeatureGroup, GetAllFeatureGroupNames


def memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (int(soft * 0.8), hard))


memory_limit()

# configs
runName = "submission RF"
timestamp = str(datetime.now())
featureGroupNames = GetAllFeatureGroupNames()
deltaRegressorName = 'MeanRegressor'
mode = "normal"
savePath = "../submission/" + runName + " T " + timestamp
data.dataName = "submission"

# init
print(f"Loading Data From: {data.dataName}")
featureGroups = [GetFeatureGroup(fg) for fg in featureGroupNames]
model = GetRegressor(deltaRegressorName, mode)

testFeatures = pd.concat([fg.getFeatures("test") for fg in featureGroups], axis=1)
testX = data.loadX('test')

# training
trainFeatures = pd.concat([fg.getFeatures("train") for fg in featureGroups], axis=1)
trainX = data.loadX('train')
trainTimeDelta = data.calcTimeDelta(trainX)

print("Training")
model.train(trainFeatures, trainTimeDelta)

trainTdError, trainTsError = evaluate(
    "Train " + deltaRegressorName,
    model,
    trainFeatures,
    trainTimeDelta,
    trainX
)

predTd = model.predict(testFeatures)
print("Joining")
y = testX["TIMESTAMP"].copy()
predTs = pd.DataFrame(timeStampPredictor(y, predTd))
print("Saving")
predTs.columns = ["TIMESTAMP"]
predTs.index.name = "index"
predTs.to_csv(savePath)
