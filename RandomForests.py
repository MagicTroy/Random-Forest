'''
Author name: Sixun Ouyang
'''

'''
Guide:
this program have been achieve
ID3 decision tree classifier, random boostrap option, random feature selection and random forest with majority Voting
basicly, this program have three modules:
	[1] math tools, i.e. calculate frequency, entropy and information gain
	[2] ID3 classifier
	[3] random forest
besides that, "main" function if the entrency to this program
'''

''' ----------------------------------------[1] Math tools--------------------------------------
calculate frequency, entropy and information gain
------------------------------------------------------------------------------------------------------------------------
'''
import math
'''
calculate the frequency of an attribute
return: frequency of an attribute
'''
def calculateFrequency(data, attrID):
	frequency = {}
	for row in data:
		if(frequency.has_key(row[attrID])):
			frequency[row[attrID]] += 1.0
		else:
			frequency[row[attrID]] = 1.0
	return frequency

'''
calculate the entropy of an attribute or subattribute
return: entropy
'''
def calculateEntropy(data, attrID):
	frequency = calculateFrequency(data, attrID)
	entropy = 0.0
	for freq in frequency.values():
		percentage = freq / len(data)
		entropy += (-percentage) * math.log(percentage, 2)
	return entropy

'''
calculate the information gain of an attribute
return: the information gain of this attribute
'''
def calculateInfoGain(data, attrID, attributes):
	attrFrequency = calculateFrequency(data, attrID)
	subsetEntropy = 0.0
	for attr in attrFrequency.keys():
		attrPercentage = attrFrequency[attr] / sum(attrFrequency.values())
		subsetData = [row for row in data if row[attrID] == attr]
		subsetEntropy += attrPercentage * calculateEntropy(subsetData, len(attributes))
	return calculateEntropy(data, attrID) - subsetEntropy

'''----------------------------------------[2] ID3 Classifier----------------------------------------
steps: 
	generate a ID3 decision tree at the first step
	then predict results of test set
------------------------------------------------------------------------------------------------------------------------
'''
'''
the entrence of ID3 classifier
return: predict results
'''
def ID3classifier(trainSet, testSet, attributes, verifyAttributes):
	ID3tree = createID3Tree(trainSet, attributes)
	finalRults = []
	for i in range(len(testSet)):
		results = predict(ID3tree, verifyAttributes, testSet[i])
		predictResults = 0
		if(len(results) > 1 or len(results) == 1):
			predictResults = results[-1]
		finalRults.append(predictResults)
	return finalRults

'''
generate an ID3 tree
return: the ID3 tree (dict)
'''
def createID3Tree(data, attributes):
	#print(len(attributes))
	results = [i for i in data]
	if(results.count(results[0]) == len(results)):
		return results[0]
	if(len(data[0]) == 1):
		frequency = calculateFrequency(data, 0)
		return max(frequency)
	bestAttrID, bestAttr = chooseAttr(data, attributes)
	tree = {bestAttr:{}}
	tempAttrs = attributes
	attributes.remove(bestAttr)
	bestValues = [row[bestAttrID] for row in data]
	uniqueValues = set(bestValues)
	for value in uniqueValues:
		subAttr = attributes[:]
		tree[bestAttr][value] = createID3Tree(splitTree(data, bestAttrID, value), subAttr)
	return tree

'''
make a predict of a singal row in test set
return: predict of this row
'''
def predict(tree, featureset, testRow):
	firstNode = tree.keys()[0]
	nextNode = tree[firstNode]
	firstNodeIndex = featureset.index(firstNode)   
	classLabel = ""
	for key in nextNode.keys():
		if testRow[firstNodeIndex] == key:
			if isinstance(nextNode[key],dict):
				classLabel = predict(nextNode[key], featureset, testRow)
			else:
				classLabel = nextNode[key]
	return classLabel

'''
choose the best attributes based on information gain
return: the index of the selected attributes and the selected attributes itself
'''
def chooseAttr(data, attributes):
	attrInfoGain = {}
	for attrID in range(len(attributes)):
		attrInfoGain[attrID] = calculateInfoGain(data, attrID, attributes)
	return max(attrInfoGain), attributes[max(attrInfoGain)]

'''
split a tree or sub tree based on the previous selected attributes
that is to say, removing the selected attributes from data set
return: the sub tree
'''
def splitTree(data, bestAttrID, value):
	subTree = []
	for row in data:
		if row[bestAttrID] == value:
			subRow  = row[:bestAttrID]
			subRow.extend(row[bestAttrID+1:])
			subTree.append(subRow)
	return subTree

'''----------------------------------------[3] random forest----------------------------------------
achieve radom boostrap, random feature selecte, ensemble size and majority Voting

Steps: 
	first it will prepaer the traing set, test set, attributes to build a tree, verifing attributes to test
	then following the ensemble size to build a forest 
	(warnning: if the ensemble size goes to high (e.g. >=50), python will change the data format of these trees to list, which bug will show)
	after that, each tree will using boostrap and random feature by user permite
	finaly, the the accuracy will show
------------------------------------------------------------------------------------------------------------------------
'''
'''
Start training and test
Generally, the default set of boostrap and random feature are all False, also, ensemble size default to 1
these can be changed at main functinon by manually typing 'True' at each side
return: accuracy
'''
def trainAndTest(split, reader, ifBootstrapSet = False, ifRandomFeature = False, ensembleSize = 1):
	trainingSet, testSet, attributes, verifyAttributes = prepareInitialParameter(split, reader)
	tS, tT, attr, vAttr = [], [], [], []
	forest = []
	for i in range(ensembleSize):
		if(ifBootstrapSet == True):
			bootstrapSet = randomBootstrap(trainingSet)
			trainingSet = bootstrapSet
		tS, tT, attr, vAttr = trainingSet, testSet, attributes, verifyAttributes
		if(ifRandomFeature == True):
			tS, tT, attr, vAttr = randomFeature(trainingSet, testSet, attributes, verifyAttributes, True)
		else:
			tS, tT, attr, vAttr = randomFeature(trainingSet, testSet, attributes, verifyAttributes, False)
		results = ID3classifier(tS, tT, attr, vAttr)
		forest.append(results)
	
	accuracy  =calculateAccuracy(forest, testSet)
	print("In this ID3 decision tree classifier we use the split is " + repr(split) + " getting the accuracy: " + repr(accuracy))

'''
initally prepaer the traing set, test set, attributes to build a tree, verifing attributes to test
return: traing set, test set, attributes, verifing attributes
'''
def prepareInitialParameter(split, reader):
	dataSet, attributes = generateDataset(reader)
	trainingSet = dataSet[int(split * len(dataSet)):]
	testSet = dataSet[:int(split * len(dataSet))]
	verifyAttributes = []
	for attribute in attributes:
		verifyAttributes.append(attribute)
	return trainingSet, testSet, attributes, verifyAttributes
'''
generate a dataset based on the file that loaded
return: the whole data set and attributes
'''
import csv
def generateDataset(read):
	dataSet = [row for row in read]
	attributes = [attr for attr in dataSet[0]]
	return dataSet, attributes[:-1]

'''
add an option to the classifier to support classification
using a random boostrap sample of examples with replacement.
return: boostrap training set
'''
import random
def randomBootstrap(trainData):
	lenTrainData = len(trainData)
	boostrapSet = []
	for i in trainData:
		randomIndex = random.randint(0, lenTrainData - 1)
		choosenRow = trainData[randomIndex]
		boostrapSet.append(choosenRow)
	return boostrapSet

'''
add an option to the classifier to also support random feature selection
return modified training set, modified test set, modified attributes, modified verify attributes
'''
def randomFeature(trainingSet, testSet, attributes, verifyAttributes, ifTrue = True):
	#randomly choose attributes and verify attributes by randomly length
	lenAttr = len(attributes)
	listAttr = range(lenAttr)
	if(ifTrue == True):
		sliceAttr = sorted(random.sample(listAttr, random.randint(3, lenAttr)))
	else:
		sliceAttr = listAttr
	modAttr = [i for i in attributes if attributes.index(i) in sliceAttr]
	modVerifyAttr = [i for i in verifyAttributes if verifyAttributes.index(i) in sliceAttr]
	modTrainSet = []
	modTestSet = []
	for row in trainingSet:
		xRow = []
		for i in range(len(row)):
			for j in sliceAttr:
				if i == j:
					xRow.append(row[i])
		xRow.append(row[len(row) - 1])
		modTrainSet.append(xRow)
	for row in testSet:
		xRow = []
		for i in range(len(row)):
			for j in sliceAttr:
				if i == j:
					xRow.append(row[i])
		xRow.append(row[len(row) - 1])
		modTestSet.append(xRow)
	return modTrainSet, modTestSet, modAttr, modVerifyAttr

'''
load the forest and test set to calculate the accuracy
return:accuracy
'''
def calculateAccuracy(forest, testSet):
	accurateNum = 0
	finalTree = []
	for i in range(len(testSet)):
		tempRowPredict = []
		for tree in forest:
			tempRowPredict.append(tree[i])
		votedResults = majorityVote(tempRowPredict)
		finalTree.append(votedResults)

	for i in range(len(testSet)):
		if(finalTree[i] == testSet[i][len(testSet[i]) - 1]):
	  		accurateNum += 1
	return float(accurateNum) / float(len(testSet))

'''
using majority voting for each resul that forest made
return: the majority result
'''
def majorityVote(results):
	frequency = {}
	for i in results:
		if(frequency.has_key(i)):
			frequency[i] += 1.0
		else:
			frequency[i] = 1.0
	return max(frequency)


''' ----------------------------------------Main functinon--------------------------------------
calculate frequency, entropy and information gain
Steps:
	first parameter is to split test and train
	second parameter is a read file
	third parameter is control the booststrap
	fourth parameter is control the randomfeature
	fivith parameter is to control the number of trees
------------------------------------------------------------------------------------------------------------------------
'''
def main():
	fileReader = file("/home/troy/PycharmProjects/Advanced Machine learning/banks.csv", "r")
	reader = csv.reader(fileReader)
	trainAndTest(0.6, reader, True, False, 100)

	fileReader = file("/home/troy/PycharmProjects/Advanced Machine learning/politics.csv", "r")
	reader = csv.reader(fileReader)
	trainAndTest(0.6, reader, True, False, 20)

	fileReader = file("/home/troy/PycharmProjects/Advanced Machine learning/tennis.csv", "r")
	reader = csv.reader(fileReader)
	trainAndTest(0.6, reader, True, True, 1)

if __name__ == "__main__":
	main()
