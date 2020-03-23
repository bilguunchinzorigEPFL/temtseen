import numpy as np
import sys
import time
import dataHandler as data

def timestampHider(X):
    mvd = data.loadX("test")["TIMESTAMP"].isnull().values
    mvd = np.concatenate([mvd]*(int(len(X)/len(mvd))+1))[:len(X)]
    #TODO add exception for start of new bus travel
    mvd[0] = 0
    t = X["TIMESTAMP"].copy()
    t.iloc[mvd] = np.nan
    return t

def mae(a, b):
    return np.mean(np.abs(a - b))

def timeDeltaError(pred, real):
    return mae(pred, real)

def timeStampError(predTimeDelta, X):
    ts = X["TIMESTAMP"]
    tsHidden = timestampHider(X)
    prevVal = 0
    for i, v in tsHidden.iteritems():
        if np.isnan(v):
            tsHidden[i] = prevVal + predTimeDelta[i]
            prevVal = tsHidden[i]
        else:
            prevVal = tsHidden[i]
    return mae(tsHidden, ts)


def evaluate(name, model, inputFeatures, realTimeDelta, X):
    sys.stdout.write(f"    Calculating {name} Error: ")
    sys.stdout.flush()
    s = time.time()
    predictedTimeDeltas = model.predict(inputFeatures)
    tdError = timeDeltaError(predictedTimeDeltas, realTimeDelta)
    tsError = timeStampError(predictedTimeDeltas, X)
    print(f"delta {tdError}, stamp {tsError}, time {time.time() - s}")
    return tdError, tsError
