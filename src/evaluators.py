import pickle as pkl
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import dataHandler as data


def timestampHider(X, availableIndices):
    endOfJourney = pd.Series([True] * len(X), index=X.index)
    endOfJourney[X.index.difference(availableIndices)] = False
    endOfJourney = endOfJourney & endOfJourney.shift(1)

    mvd = data.loadX("test")["TIMESTAMP"].isnull().values
    mvd = np.concatenate([mvd] * (int(len(X) / len(mvd)) + 1))[:len(X)]
    mvd = pd.Series(mvd, index=X.index)

    finalIndices = endOfJourney & mvd
    t = X["TIMESTAMP"].copy()
    t[finalIndices] = np.nan
    return t


def mae(a, b):
    return np.mean(np.abs(a - b))


def timeStampPredictorFast(tsHidden, predTd):
    prevVal = None
    for i, v in tqdm(list(tsHidden.iteritems())):
        if np.isnan(v):
            tsHidden[i] = prevVal + predTd[i]
            prevVal = tsHidden[i]
        else:
            prevVal = tsHidden[i]
    return tsHidden


def timeStampPredictor(tsHidden, predTd):
    prevVal = None
    unkIdx = []
    lastKnownIdx = None
    changes = []
    for i, v in tqdm(list(tsHidden.iteritems())):
        if np.isnan(v):
            tsHidden[i] = prevVal + predTd[i]
            prevVal = tsHidden[i]
            unkIdx.append(i)
        else:
            if len(unkIdx) > 0:
                if i in predTd.index:
                    # Within one movement
                    realDiff = tsHidden[i] - tsHidden[lastKnownIdx]
                    predDiff = tsHidden[unkIdx[-1]] + predTd[i] - tsHidden[lastKnownIdx]
                    diff = (realDiff - predDiff) / predDiff
                    for j in unkIdx:
                        tsHidden[j] += diff * (tsHidden[j] - tsHidden[lastKnownIdx])
            prevVal = tsHidden[i]
            lastKnownIdx = i
            unkIdx = []
    return tsHidden


def timeStampPredictorConf(tsHidden, predTd, conf):
    confCumSum = conf.cumsum()
    prevVal = None
    unkIdx = []
    lastKnownIdx = None
    for i, v in tqdm(list(tsHidden.iteritems())):
        if np.isnan(v):
            tsHidden[i] = prevVal + predTd[i]
            prevVal = tsHidden[i]
            unkIdx.append(i)
        else:
            if len(unkIdx) > 0:
                if i in predTd.index:
                    # Within one movement
                    z = confCumSum[lastKnownIdx]
                    diff = (tsHidden[i] - tsHidden[unkIdx[-1]] - predTd[i]) / (confCumSum[i] - z)
                    for j in unkIdx:
                        tsHidden[j] += diff * (confCumSum[j] - z)
            prevVal = tsHidden[i]
            lastKnownIdx = i
            unkIdx = []

    return tsHidden


def timeStampPredictorDouble(tsHidden, predSmall, predBig):
    prevVal = None
    unkIdx = []
    lastKnownIdx = None
    predBigCumSum = predBig.cumsum()
    for i, v in tqdm(list(tsHidden.iteritems())):
        if np.isnan(v):
            tsHidden[i] = prevVal + predSmall[i]
            prevVal = tsHidden[i]
            unkIdx.append(i)
        else:
            if len(unkIdx) > 0:
                if i in predSmall.index:
                    # Within one movement
                    realDiff = tsHidden[i] - tsHidden[lastKnownIdx]
                    predDiff = tsHidden[unkIdx[-1]] + predSmall[i] - tsHidden[lastKnownIdx]
                    if realDiff > 600:
                        z = predBigCumSum[lastKnownIdx]
                        diff = (realDiff - predDiff) / (predBigCumSum[i] - z)
                        for j in unkIdx:
                            tsHidden[j] += diff * (predBigCumSum[j] - z)
                    else:
                        diff = (realDiff - predDiff) / predDiff
                        for j in unkIdx:
                            tsHidden[j] += diff * (tsHidden[j] - tsHidden[lastKnownIdx])
            prevVal = tsHidden[i]
            lastKnownIdx = i
            unkIdx = []

    return tsHidden


def evaluate(name, model, inputFeatures, realTimeDelta, X):
    sys.stdout.write(f"    Calculating {name} Error: ")
    sys.stdout.flush()
    s = time.time()
    predTimeDeltas = model.predict(inputFeatures)
    tdError = mae(predTimeDeltas, realTimeDelta)
    confs = None
    if model.givesConfidence:
        confs = model.confidence

    ts = X["TIMESTAMP"]
    tsHidden = timestampHider(X, realTimeDelta.index)
    if confs is None:
        tsPredicted = timeStampPredictorDouble(tsHidden, predTimeDeltas, model.bigPreds)
    else:
        tsPredicted = timeStampPredictorConf(tsHidden, predTimeDeltas, confs)
    tsError = mae(tsPredicted, ts)

    print(f"\tdelta {tdError}, stamp {tsError}, time {time.time() - s}")
    return tdError, tsError
