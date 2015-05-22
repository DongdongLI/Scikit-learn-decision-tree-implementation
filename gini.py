# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

''' load the data '''
y=[]
data = []
labels = []
with open('iris.data') as ifile:
	content=ifile.readlines()
	for line in content:
		if(line!='\n'):
			tokens = line.rstrip().split(',')
			data.append([float(tk) for tk in tokens[:-1]])
			labels.append(tokens[-1])


for i in range(0,len(labels)):
	if(labels[i]=='Iris-virginica'):
		y.append(0)
	elif(labels[i]=='Iris-setosa'):
		y.append(1)
	elif(labels[i]=='Iris-versicolor'):
		y.append(2)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size = 0.1)



clf = tree.DecisionTreeClassifier(criterion='gini')
print(clf)
clf.fit(x_train, y_train)

answer = clf.predict(x_test)  
#print(x_train)
print("The Prediction:")  
print(answer)
print("The answer:")  
print(y_test)
print("The Acuraccy:")  
print(np.mean( answer == y_test))
#precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))  
#answer = clf.predict_proba(x)[:,1]  
print(classification_report(y_test, answer, target_names = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']))



