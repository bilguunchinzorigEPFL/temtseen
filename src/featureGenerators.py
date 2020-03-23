import os
import pickle as pkl

import dataHandler as data

featureSaveDir = "../features/"
loadFromFile = True


class FeatureGroup:
    def __init__(self, name):
        self.Name = name

    def getFeatures(self, featureType):
        if loadFromFile:
            if os.path.exists(self.getPath(featureType)):
                print(f"Loading features from: " + self.getPath(featureType))
                return self.load(featureType)

        print(f"Calculating {featureType} features: " + self.Name)
        features = self.calc(data.loadX(featureType))
        self.save(features, featureType)
        return features

    def save(self, features, featureType):
        if not os.path.exists(self.getPath()):
            os.mkdir(self.getPath())
        pkl.dump(features, open(self.getPath(featureType), "wb"))

    def load(self, featureType):
        return pkl.load(open(self.getPath(featureType), "rb"))

    def getPath(self, featureType=None):
        if featureType is None:
            if not os.path.exists(os.path.join(featureSaveDir, data.dataName)):
                os.mkdir(os.path.join(featureSaveDir, data.dataName))
            return os.path.join(featureSaveDir, data.dataName, self.Name)
        else:
            return os.path.join(featureSaveDir, data.dataName, self.Name, featureType)

    def calc(self, X):
        raise NotImplementedError


def GetFeatureGroup(featureGroupName):
    if featureGroupName == "Base":
        return Base("Base")
    raise NotImplementedError


def GetAllFeatureGroupNames():
    return ["Base"]


# Feature implementations
class Base(FeatureGroup):
    def calc(self, X):
        return X.loc[:, ~X.columns.isin(['TIMESTAMP', 'RECORD_DATE'])]
