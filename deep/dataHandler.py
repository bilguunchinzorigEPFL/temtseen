import os
import pickle as pkl
import time

import numpy as np
import pandas as pd
import torch

featureSaveDir = "../features/"


class DataHandler:
    def __init__(self, name, dirname):
        self.dirName = dirname
        self.loadFromFile = True
        self.Name = name

    def _getPath(self, featureType=None):
        if featureType is None:
            if not os.path.exists(os.path.join(featureSaveDir, self.dirName)):
                os.mkdir(os.path.join(featureSaveDir, self.dirName))
            return os.path.join(featureSaveDir, self.dirName, self.Name)
        else:
            return os.path.join(featureSaveDir, self.dirName, self.Name, featureType)

    def _load(self, featureType):
        return pd.read_csv(os.path.join("../data", self.dirName, featureType + ".csv"), index_col=0)

    def _timestampHider(self, X, availableIndices):
        endOfJourney = pd.Series([True] * len(X), index=X.index)
        endOfJourney[X.index.difference(availableIndices)] = False
        endOfJourney = endOfJourney & endOfJourney.shift(1)

        mvd = pd.read_csv("../data/main/test.csv", index_col=0)["TIMESTAMP"].isnull().values
        mvd = np.concatenate([mvd] * (int(len(X) / len(mvd)) + 1))[:len(X)]
        mvd = pd.Series(mvd, index=X.index)

        finalIndices = endOfJourney & mvd
        t = X["TIMESTAMP"].copy()
        t[finalIndices] = np.nan
        return t

    def _generateSamples(self, df):
        diffs = calcTimeDelta(df)
        hidden = self._timestampHider(df, diffs.index)
        real = df["TIMESTAMP"]
        return hidden

    def getTensors(self, splitType):
        if self.loadFromFile:
            if os.path.exists(self._getPath(splitType)):
                print(time.time(), f"Loading features from: " + self._getPath(splitType))
                return pkl.load(open(self._getPath(splitType), "rb"))

        print(time.time(), f"Calculating {splitType} features: " + self.Name)
        features, D, Y = self._calc(splitType)
        if not os.path.exists(self._getPath()):
            os.mkdir(self._getPath())
        pkl.dump((features, D, Y), open(self._getPath(splitType), "wb"))
        return features, D, Y

    def _dfFeature(self, X, featureType):
        d = X[['BUSSTOP_ID', 'BUSSTOP_SEQ']].shift(-1).fillna(method="ffill")
        X['DEST_BUSSTOP_ID'] = d['BUSSTOP_ID']
        X['DEST_BUSSTOP_SEQ'] = d['BUSSTOP_SEQ']

        def normalizer(s):
            return (s - s.mean()) / s.std()

        if featureType == "train":
            print(time.time(), "\t \t Calculating GPS MAP")
            stops = pd.read_csv("../data/main/stops.csv", index_col=0)
            self.gps_x = normalizer(stops["GPS_COORDX"])
            self.gps_y = normalizer(stops["GPS_COORDY"])
            print(time.time(), "\t \t Calculating STOP MAPS")
            diff = calcTimeDelta(X)
            diffForward = pd.concat([X["BUSSTOP_ID"], diff], axis=1).groupby("BUSSTOP_ID")
            diffBackward = pd.concat([X["DEST_BUSSTOP_ID"], diff], axis=1).groupby("DEST_BUSSTOP_ID")
            self.dfFmean = diffForward.mean()["TIMESTAMP"]
            self.dfFstd = diffForward.std()["TIMESTAMP"]
            self.dfBmean = diffBackward.mean()["TIMESTAMP"]
            self.dfBstd = diffBackward.std()["TIMESTAMP"]
            print(time.time(), "\t \t Calculating TRANSACTION Maps")
            tDiff = pd.concat([X["BUSSTOP_ID"].astype(str) + "-" + X["DEST_BUSSTOP_ID"].astype(str), diff],
                              axis=1).groupby(0)
            self.dfTmean = tDiff.mean()["TIMESTAMP"]
            self.dfTstd = tDiff.std()["TIMESTAMP"]
            self.dfTcount = np.sqrt(tDiff.count()["TIMESTAMP"])

        def apply(d, map):
            map.index.name = "key"
            map.name = "value"
            return pd.merge(
                left=d.rename("key").reset_index(),
                right=map.reset_index(),
                how='left',
                left_on="key",
                right_on="key"
            )["value"].fillna(map.median())

        print(time.time(), "\t \t Applying GPS MAP")
        X['BUSSTOP_X'] = apply(X["BUSSTOP_ID"], self.gps_x)
        X['BUSSTOP_Y'] = apply(X["BUSSTOP_ID"], self.gps_y)
        X['DEST_BUSSTOP_X'] = apply(X["DEST_BUSSTOP_ID"], self.gps_x)
        X['DEST_BUSSTOP_Y'] = apply(X["DEST_BUSSTOP_ID"], self.gps_y)
        print(time.time(), "\t \t Applying STOP MAP")
        X['BUSSTOP_MEAN'] = apply(X["BUSSTOP_ID"], self.dfFmean)
        X['BUSSTOP_STD'] = apply(X["BUSSTOP_ID"], self.dfFstd)
        X['DEST_BUSSTOP_MEAN'] = apply(X["DEST_BUSSTOP_ID"], self.dfBmean)
        X['DEST_BUSSTOP_STD'] = apply(X["DEST_BUSSTOP_ID"], self.dfBstd)
        print(time.time(), "\t \t Applying Transaction MAP")
        s = X["BUSSTOP_ID"].astype(str) + "-" + X["DEST_BUSSTOP_ID"].astype(str)
        X['T_MEAN'] = apply(s, self.dfTmean)
        X['T_STD'] = apply(s, self.dfTstd)
        X['T_CNT'] = apply(s, self.dfTcount)

        print(time.time(), "\t \t Applying TIME Features")
        d = pd.to_datetime(X["TIMESTAMP"].fillna(method="ffill"), unit='s')
        X['HOUR'] = d.dt.hour
        X['DAY'] = d.dt.weekday

        return X[[
            "BUSSTOP_SEQ", "DEST_BUSSTOP_SEQ",  # todo "BUS_ID", "BUSROUTE_ID", "BUSSTOP_ID", "DEST_BUSSTOP_ID"
            "BUSSTOP_X", "BUSSTOP_Y", "DEST_BUSSTOP_X", "DEST_BUSSTOP_Y",
            "BUSSTOP_MEAN", "BUSSTOP_STD", "DEST_BUSSTOP_MEAN", "DEST_BUSSTOP_STD",
            "T_MEAN", "T_STD", "T_CNT", "HOUR", "DAY"
        ]], X["TIMESTAMP"]

    def _dfFeatureSimple(self, X, featureType):
        if featureType == "train":
            self.stopIdMap = pd.Series({v: i for i, v in enumerate(X["BUSSTOP_ID"].unique())})
            print(len(self.stopIdMap))

        def apply(d, map):
            map.index.name = "key"
            map.name = "value"
            return pd.merge(
                left=d.rename("key").reset_index(),
                right=map.reset_index(),
                how='left',
                left_on="key",
                right_on="key"
            )["value"].fillna(map.median())

        d = pd.to_datetime(X["TIMESTAMP"].fillna(method="ffill"), unit='s')
        F = pd.concat([
            apply(X["BUSSTOP_ID"], self.stopIdMap),
            d.dt.hour,
            d.dt.minute,
            d.dt.weekday
        ],axis=1)
        F.columns = ["BUSSTOP_ID", "HOUR", "MINUTE", "DAY"]
        return F, X["TIMESTAMP"]

    def _calc(self, splitType):
        df = self._load(splitType)
        if splitType == "test":
            hidden = df["TIMESTAMP"]
        else:
            hidden = self._generateSamples(df)
        print(time.time(), "\t Calculating Features")
        df, Y = self._dfFeatureSimple(df, splitType)
        print(time.time(), "\t Calculating Journeys")
        journeyIdx = (~pd.isnull(hidden)).cumsum()
        journeyEndIdx = journeyIdx[journeyIdx.diff(1) == 1] - 1
        journeyIdx = pd.concat([journeyIdx, journeyEndIdx]).rename("journey")
        print(time.time(), "\t Calculating Input Embeds")
        dfgrouped = pd.concat([df.loc[journeyIdx.index], journeyIdx], axis=1).groupby("journey")
        inputEmbeds = dfgrouped.apply(lambda journey: self._calcEmbedding(journey)).dropna()
        # print(time.time(), "\t \t Padding Input Embeds")
        # inputTensors = self._2dpadder(inputEmbeds)
        print(time.time(), "\t Calculating Diff Embeds")
        startEnd = pd.concat([hidden.loc[journeyIdx.index], journeyIdx], axis=1).groupby("journey")
        diffEmbeds = startEnd.apply(lambda journey: self._calcDiff(journey)).dropna()
        # diffTensors = torch.tensor(diffEmbeds.dropna().values)
        print(time.time(), "\t Calculating Output Embeds")
        if splitType == "test":
            outputEmbeds = None
            # outputTensors = None
        else:
            timestamps = pd.concat([Y.loc[journeyIdx.index], journeyIdx], axis=1).groupby("journey")
            outputEmbeds = timestamps.apply(lambda journey: self._calcOutEmbedding(journey)).dropna()
            # print(time.time(), "\t \t Padding Output Embeds")
            # outputTensors = self._1dpadder(outputEmbeds)

        return inputEmbeds, diffEmbeds, outputEmbeds  # inputTensors, diffTensors, outputTensors

    def _calcEmbedding(self, df):
        if len(df) > 2:
            return df.values

    def _calcDiff(self, df):
        if len(df) > 2:
            return df["TIMESTAMP"].dropna().diff().values[1]

    def _calcOutEmbedding(self, df):
        if len(df) > 2:
            return (-df["TIMESTAMP"].diff(-1).dropna()).values


def calcTimeDelta(X):
    diff = -X["TIMESTAMP"].diff(-1)
    skip = (X["BUS_ID"].diff(-1) == 0) & (X["BUSSTOP_SEQ"].diff(-1) < 0) & (diff < 3600)
    diff = diff[skip]
    return diff

if __name__ == '__main__':
    a = DataHandler("DeepSimple", "full")
    a.getTensors("train")
    a.getTensors("eval")
    # a.getTensors("test")
    a = ""

import random


class DataLoader:
    def __init__(self, dataHandler, type, batch_size=32, method="Random"):
        self.inputEmbeds, self.diffEmbeds, self.outputEmbeds = dataHandler.getTensors(type)
        self.inputEmbeds.reset_index(drop=True, inplace=True)
        self.diffEmbeds.reset_index(drop=True, inplace=True)
        self.outputEmbeds.reset_index(drop=True, inplace=True)
        okIdx = self.inputEmbeds.apply(lambda v: ~np.isnan(v.sum()))
        self.inputEmbeds = self.inputEmbeds[okIdx]
        self.diffEmbeds = self.diffEmbeds[okIdx]
        self.outputEmbeds = self.outputEmbeds[okIdx]
        self.inputEmbeds.reset_index(drop=True, inplace=True)
        self.diffEmbeds.reset_index(drop=True, inplace=True)
        self.outputEmbeds.reset_index(drop=True, inplace=True)
        print(len([i for i, v in self.outputEmbeds.iteritems() if v.mean() > 300]), len(self.outputEmbeds))
        self.size = len(self.inputEmbeds)
        self.batchSize = batch_size
        self.indices = self.inputEmbeds.index.copy().values
        self.method = method

    def __iter__(self):
        self.i = 0
        random.shuffle(self.indices)
        return self

    def __next__(self):
        s = self.i * self.batchSize
        e = (self.i + 1) * self.batchSize
        if e >= self.size:
            raise StopIteration
        indices = self.indices[s:e]
        iTensor = self._2dpadder(self.inputEmbeds[indices])
        dTensor = torch.tensor(self.diffEmbeds[indices].values, dtype=torch.float32)
        oTensor = self._1dpadder(self.outputEmbeds[indices])
        self.i += 1
        # mean = dTensor / (oTensor != 0).sum(dim =1)
        # iTensor = iTensor[mean > 300]
        # dTensor = dTensor[mean > 300]
        # oTensor = oTensor[mean > 300]
        return iTensor[:,:,:-1], dTensor, oTensor

    def __len__(self):
        return self.size // self.batchSize

    def _2dpadder(self, listEmbeds):
        padSize = max([len(embed) for i, embed in listEmbeds.iteritems()]) + 1
        paddedList = [np.pad(embed, [(0, padSize - embed.shape[0]), (0, 0)]) for i, embed in listEmbeds.iteritems()]
        return torch.tensor(paddedList, dtype=torch.float32)

    def _1dpadder(self, listEmbeds):
        padSize = max([len(listEmbeds[i]) for i, embed in listEmbeds.iteritems()]) + 2
        paddedList = [np.pad(embed, (0, padSize - embed.shape[0])) for i, embed in listEmbeds.iteritems()]
        return torch.tensor(paddedList, dtype=torch.float32)
