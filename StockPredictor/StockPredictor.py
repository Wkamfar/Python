import csv
import numpy as np
from pandas import *
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
def IsPriceIncreased(startPrice, endPrice):
    boolResults = []
    for currentStartPrice, currentEndPrice in zip(startPrice, endPrice):
        boolResults.append(currentEndPrice - currentStartPrice >= 0)
    return boolResults
def accuracy(predictedResults, knownResults):
    total = 0
    sum = 0
    for pr, kr in zip(predictedResults, knownResults):
        if pr == kr:
            sum += 1
        total += 1
    return sum / total
fileName = "FINAL_FROM_DF.csv"
data = read_csv(fileName)
symbol = data['SYMBOL'].tolist()
series = data['SERIES'].tolist()
open = data['OPEN'].tolist()
high = data['HIGH'].tolist()
low = data['LOW'].tolist()
close = data['CLOSE'].tolist()
last = data['LAST'].tolist()
prevClose = data['PREVCLOSE'].tolist()
tottrdqty = data['TOTTRDQTY'].tolist()
tottrdval = data['TOTTRDVAL'].tolist()
totalTrades = data['TOTALTRADES'].tolist()
isin = data['ISIN'].tolist()

isPriceIncreased = IsPriceIncreased(open, close)
features = []
for _symbol, _open, _high, _low, _totalTrades in zip(symbol, open, high, low, totalTrades):
    features.append([_open, _high, _low, _totalTrades])
results = isPriceIncreased
#trainingFeatures = features[0:int(len(features) * 0.7)]
#testingFeatures = features[int(len(features) * 0.7) + 1:]
#trainingResults = results[0:int(len(results) * 0.7)]
#testingResults = results[int(len(results) * 0.7) + 1:]
endTraining = 10000
endTesting = 15000
trainingFeatures = features[0:endTraining]
testingFeatures = features[endTraining + 1:endTesting]
trainingResults = results[0:endTraining]
testingResults = results[endTraining + 1:endTesting]

clf = svm.SVC()
clf.fit(trainingFeatures, trainingResults)
predictedResults = clf.predict(testingFeatures)
print(accuracy(predictedResults, testingResults) * 100, "%")
#This is now a neural network
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(trainingFeatures, trainingResults)
predictedResults = clf.predict(testingFeatures)
print(accuracy(predictedResults, testingResults) * 100, "%")
#This is random forest
clf = RandomForestClassifier(n_estimators=10)
clf.fit(trainingFeatures, trainingResults)
predictedResults = clf.predict(testingFeatures)
print(accuracy(predictedResults, testingResults) * 100, "%")