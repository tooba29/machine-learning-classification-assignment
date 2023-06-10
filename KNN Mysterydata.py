# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:34:14 2022

@author: tooba_29
"""
#for knn defining funcions
import math
import operator 

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances.append((trainingSet.iloc[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet.iloc[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
  
def knn(X_tr, X_te):
  k = [1,3,5,7,9,11,13]
  for i in range (len(k)):
    predictions=[]
    for x in range(len(X_te)):
        neighbors = getNeighbors(X_tr, X_te.iloc[x], k[i])
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(X_te.iloc[x][-1]))
    accuracy = getAccuracy(X_te, predictions)
    print('accuracy of k = ', k[i], " is ", accuracy)

#FOR PRECEPTRON
def weights(X_in):
        b=np.ones((X_in.shape[0],1)) #adding one column in X_input vector 
        X_in['b'] = b
        W = np.zeros(X_in.shape[1])# weight vector has similar size of X_input
        new_X = np.array(X_in)
        return new_X, W

def perceptron(new_X, W, y):
        while True:
            missClassification = 0
            for x in range(len(new_X)):
                result = W.T.dot(new_X[x])
                #activation = 1.0 if result > 0.0 else 0.0
               
                if y[x]*result<=0:
                  W = W + y[x]*(new_X[x])
                  missClassification +=1
                  plt.plot(W*new_X[x])
                  plt.show
                  print('missclassified')
                  input("Press Enter to continue...")
                    
            if missClassification==0:
                break
        return W
    
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns 
# from sklearn import preprocessing


dfmyst= pd.read_csv(r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Assignment Datasets\mystery.csv")
print(dfmyst)

print(dfmyst.shape)

#dropping class column
dfmyst2 = dfmyst['class']
dfmyst_f= dfmyst.drop("class",axis=1)
print(dfmyst_f)

print(dfmyst_f.head(15))

#normalization

# value=np.array(dfmyst["feature 1"])
# print(value)

print("Mystery Data after normilization")
normaldfmyst=(dfmyst_f-dfmyst_f.min())/(dfmyst_f.max()-dfmyst_f.min())
print(normaldfmyst.shape)
print(normaldfmyst.head(15))

# def split_train_test(data, test_ratio):
#     #to not compromise the test daya
#   np.random.seed(45)  
#   shuffled =np.random.permutation(len(data)) #to ran
#   test_set_size= int(len(data)*test_ratio)
#   test_indices = shuffled[:test_set_size]
#   train_indices =shuffled[test_set_size:]
#   return data.iloc[train_indices],data.iloc[test_indices]

# print("Spliting data into test and train")
# train_set_m, test_set_m = split_train_test(dfmyst_f,0.2)
# print(f"Rows in test set: {len(test_set_m)}\nRows in train set: {len(train_set_m)}")


# print("\n\n String values into binary format: ")
# dfmyst2  = (dfmyst2!='Yes').astype(int)
# print(dfmyst2)

# print("\n\nApplying KNN on Splitting data into test and train: ")
# train_set_my, test_set_my = split_train_test(dfmyst,test_ratio=0.2)
# knn(train_set_my,test_set_my)


# print("\n\n\Applying KNN on nromalized data splitting data into test and train:")
# train_set_my, test_set_my = split_train_test(normaldfmyst,test_ratio=0.2)
# knn(train_set_my,test_set_my)
print('For myster Data')
new_X_m, W_m = weights(dfmyst_f)
perceptron(new_X_m, W_m, dfmyst2)








    