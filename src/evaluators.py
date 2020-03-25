import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import dataHandler as data


def timestampHider(X, availableIndices):
    endOfJourney = pd.Series([True]*len(X), index=X.index)
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


def timeDeltaError(pred, real):
    return mae(pred, real)


def timeStampPredictor(tsHidden, predTd):
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
                    realDiff = tsHidden[i] - tsHidden[lastKnownIdx]
                    predDiff = tsHidden[unkIdx[-1]] + predTd[i] - tsHidden[lastKnownIdx]
                    diff = (realDiff - predDiff) / predDiff
                    if abs(diff) < 1:
                        for j in unkIdx:
                            tsHidden[j] += diff * (tsHidden[j] - tsHidden[lastKnownIdx])
                    else:
                        print(f"Too big Error {diff}")
            prevVal = tsHidden[i]
            lastKnownIdx = i
            unkIdx = []
    return tsHidden

from multiprocessing import Pool, cpu_count

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def supPredictor(data):
    if len(data) > 1:
        finalVal = data["realDiff"].iloc[0]
        if not np.isnan(finalVal):
            totalPred = data["Pred"].sum()
            unitErr = (finalVal - totalPred) / totalPred
        else:
            unitErr = 1
        preds = (data["Pred"].cumsum().shift(1) * (1 + unitErr)).dropna()
        preds.index = data["index"][1:]
        return preds

def timeStampPredictor2(tsHidden, predTd):
    tmpHidden = pd.DataFrame(tsHidden)
    tmpHidden["nullGroup"] = (~pd.isnull(tsHidden)).cumsum()
    tmpHidden["realDiff"] = tmpHidden["TIMESTAMP"].shift(-1).fillna(method="bfill") - tmpHidden["TIMESTAMP"]
    tmpHidden["Pred"] = predTd
    tmpHidden = tmpHidden.reset_index()

    finalPreds = tsHidden.copy().fillna(method="ffill")

    with Pool(cpu_count()) as p:
        ret_list = p.map(supPredictor, [group for name, group in tmpHidden.groupby("nullGroup")])
    return finalPreds.add(pd.concat([l for l in ret_list if l is not None]), fill_value=0)


def timeStampError(predTimeDelta, realTimeDelta, X):
    ts = X["TIMESTAMP"]
    tsHidden = timestampHider(X, realTimeDelta.index)
    tsPredicted = timeStampPredictor(tsHidden, predTimeDelta)
    return mae(tsPredicted, ts)


def evaluate(name, model, inputFeatures, realTimeDelta, X):
    sys.stdout.write(f"    Calculating {name} Error: ")
    sys.stdout.flush()
    s = time.time()
    predictedTimeDeltas = model.predict(inputFeatures)
    tdError = timeDeltaError(predictedTimeDeltas, realTimeDelta)
    tsError = timeStampError(predictedTimeDeltas, realTimeDelta, X)
    print(f"delta {tdError}, stamp {tsError}, time {time.time() - s}")
    return tdError, tsError
