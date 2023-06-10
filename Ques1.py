# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:41:04 2022

@author: tooba_29
"""

import numpy as np
#import plotly.express as px    
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
#from sklearn import preprocessing
import pandas as pd

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
def split_train_test(data, test_ratio):
  np.random.seed(42)   #to stop train and test data merging 
  shuffled =np.random.permutation(len(data)) #to ran
  test_set_size= int(len(data)*test_ratio)
  test_indices = shuffled[:test_set_size]
  train_indices =shuffled[test_set_size:]
  return data.iloc[train_indices],data.iloc[test_indices]

#GIVING PATH OF FILE AND NAMING THE COLUMNS
mydata = pd.read_csv (r"C:\Users\square\Desktop\TOOBA\UNI\SEM6\ML\Assignment Datasets\german data.csv")
mydata.columns=["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21"]

mydata["A1"] = mydata["A1"].map({'A14':"0",'A11':"<0 DM", 'A12': "0 <= <200 DM",
                                                   'A13':">= 200 DM "})

mydata["A3"] = mydata["A3"].map({"A34":"critical account","A33":"delay in paying off",
                                                     "A32":"existing credits paid",
                                                     "A31":"all credits at this bank paid back duly",
                                                     "A30":"no credits taken"})

mydata["A4"] = mydata["A4"].map({ "A40": "car-new", "A41": "car-used", "A42": "furniture", "A43": "television",
                                       "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                                      "A48": "retraining", "A49": "business", "A410": "others"})

mydata["A6"] = mydata["A6"].map({"A65" : "no savings account","A61" :"<100 DM","A62" : "100 <= <500 DM",
                                                     "A63" :"500 <= < 1000 DM", "A64" :">= 1000 DM"})


mydata["A7"] = mydata["A7"].map({'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 
                                               'A72':"<1 years",'A71':"unemployed"})

mydata["A9"] = mydata["A9"].map({"A93": "male", "A92": "female", "A91": "male", "A94": "male", "A95": "female"})

mydata["A10"] = mydata["A10"].map({'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"})

mydata["A12"] = mydata["A12"].map({'A121':"real estate", 'A122':"savings agreement/life insurance", 
                                         'A123':"car or other", 'A124':"unknown / no property"})

mydata["A14"] = mydata["A14"].map({'A143':"none", 'A142':"store", 'A141':"bank"})

mydata["A15"] = mydata["A15"].map({"A151" : "rent", "A152" : "own", "A153" : "for free"})

mydata["A17"] = mydata["A17"].map({'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 
                               'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"})

mydata["A19"] = mydata["A19"].map({'A192':"yes", 'A191':"none"})
mydata["A20"] = mydata["A20"].map({'A201':"yes", 'A202':"no"})

print(mydata)




#TO CALCULATE STD,MIN,MAX OF THE DATAFRAME

print(mydata.describe())


#data not following normal distribution
#foroutliers
cf = pd.DataFrame(data = mydata, columns = ["A8","A11","A16","A21"])
cf.boxplot()

Q1 = mydata['A16'].quantile(0.25)
Q3 = mydata['A16'].quantile(0.75)

IQR = Q3 - Q1    #IQR is interquartile range. 

upper_limit = Q3+ 1.5 * IQR
lower_limit = Q1- 1.5 * IQR

mydata[mydata['A16'] > upper_limit]
mydata[mydata['A16'] < lower_limit]

newmydata= mydata[mydata['A16'] < upper_limit]
print(newmydata.shape)

sns.boxplot(newmydata['A16'])
plt.show()

print(newmydata)

numvars = newmydata.drop(["A1","A3", "A4","A6",
             "A7", "A9","A10",
             "A12","A14","A15","A17",
              "A19","A20","A21"], axis = 1)
print(numvars)

qualitative = newmydata.drop(['A2', 'A5', 'A8','A11', 'A13', 'A16', 'A18'], axis = 1)
print(qualitative)

normal_df =(numvars-numvars.min())/(numvars.max()-numvars.min())
print(normal_df)


y = newmydata["A21"]
X_train, X_test = split_train_test(numvars,0.2)
print(f"Rows in test set: {len(X_test)}\nRows in train set: {len(X_train)}")

Y_train, Y_test = split_train_test(y,0.2)
print(f"Rows in test set: {len(Y_test)}\nRows in train set: {len(Y_train)}")
              
X_tr = pd.concat([X_train, Y_train],axis=1)
X_te = pd.concat([X_test, Y_test],axis=1)
 
# knn(X_tr,X_te)




# #UNDERSTANING A ROUGH IDEA OF DATA DIVIDING INTO CLASS
# target = mydata.values[:,-1]
# counter = Counter(target)
# for k,v in counter.items():
#  	per = v / len(target) * 100
#  	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

# #CHECK FOR NULL VALUES IN DATA, NO NULL VALUES FOUND
# print(newmydata.isnull().sum())
     
#NOW CREATING DUMMIES TO CONVERT CATEGORICAL DATA INTO ONE HOT
dummies = pd.get_dummies(data = newmydata, columns =["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"])
print(dummies.head())
merged = pd.concat([newmydata,dummies], axis ='columns')

# #Merging newly formed data frame into exsisting and droping the qualitative values 
final = merged.drop(["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"], axis='columns')
numeric = newmydata.drop(["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"],axis=1)
dummies.head(15)


merged = pd.concat([numvars,dummies], axis ='columns')
X_train2, X_test2 = split_train_test(merged,0.2)
print(f"Rows in test set: {len(X_test)}\nRows in train set: {len(X_train)}")

# X_tr2 = pd.concat([X_train2, Y_train],axis=1)
# X_te2 = pd.concat([X_test2, Y_test],axis=1)

# knn(X_tr2,X_te2)





   


# X = newmydata.iloc[:,0:20]  #independent columns
# y = newmydata.iloc[:,-1]    #target column i.e A21
# #get correlations of each features in dataset
# corrmat = newmydata.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(newmydata[top_corr_features].corr(),annot=True,cmap="RdYlGn")

newmydata1=newmydata.drop('A13',axis=1)
newmydata2=newmydata1.drop('A16',axis=1)
newmydata_f=newmydata2.drop('A18',axis=1)

print(newmydata_f)

dummies_useless = pd.get_dummies(data = newmydata_f, columns =["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"])
print(dummies_useless.head())
merged_useless = pd.concat([newmydata_f,dummies], axis ='columns')

# #Merging newly formed data frame into exsisting and droping the qualitative values 
final_useless = merged_useless.drop(["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"], axis='columns')
numeric_useless = newmydata_f.drop(["A1","A3","A4","A6","A7","A9","A10","A12","A14","A15","A17","A19","A20"],axis=1)
dummies_useless.head(15)


merged_useless = pd.concat([numvars,dummies_useless], axis ='columns')
X_train2, X_test2 = split_train_test(merged,0.2)
print(f"Rows in test set: {len(X_test)}\nRows in train set: {len(X_train)}")

X_tr3 = pd.concat([X_train2, Y_train],axis=1)
X_te3 = pd.concat([X_test2, Y_test],axis=1)

knn(X_tr3,X_te3)





# newmydata_f2 = newmydata_f['A21']
# newmydata_ff= newmydata_f.drop("A21",axis=1)
# print(newmydata_ff)

# print(newmydata_ff.head(15))


