# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:45:53 2023

@author: Noreen
"""


# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
# define dataset
import numpy as np
from matplotlib import pyplot
import pandas as pd
#from keras.layers import optimizers

from sklearn.model_selection import train_test_split
#load data
data = pd.read_csv('E:/maryadata/newbinaryclass.csv')
#data.drop(['Unnamed: 0'],axis=1,inplace=True)

data.columns


data=data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9',
       'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Class' ]]
data.dropna(inplace=True)
# ONE HOT ENCODER
data['Q1'],level=pd.factorize(data['Q1'])
data['Q1'].unique()
data['Q2'],level=pd.factorize(data['Q2'])
data['Q2'].unique()
data['Q3'],level=pd.factorize(data['Q3'])
data['Q3'].unique()
data['Q4'],level=pd.factorize(data['Q4'])
data['Q4'].unique()
data['Q5'],level=pd.factorize(data['Q5'])
data['Q5'].unique()
data['Q6'],level=pd.factorize(data['Q6'])
data['Q6'].unique()
data['Q7'],level=pd.factorize(data['Q7'])
data['Q7'].unique()
data['Q8'],level=pd.factorize(data['Q8'])
data['Q8'].unique()
data['Q9'],level=pd.factorize(data['Q9'])
data['Q9'].unique()
data['Q10'],level=pd.factorize(data['Q10'])
data['Q10'].unique()
data['Q11'],level=pd.factorize(data['Q11'])
data['Q11'].unique()
data['Q12'],level=pd.factorize(data['Q12'])
data['Q12'].unique()
data['Q13'],level=pd.factorize(data['Q13'])
data['Q13'].unique()
data['Q14'],level=pd.factorize(data['Q14'])
data['Q14'].unique()
data['Q15'],level=pd.factorize(data['Q15'])
data['Q15'].unique()
data['Q16'],level=pd.factorize(data['Q16'])
data['Q16'].unique()
data['Q17'],level=pd.factorize(data['Q17'])
data['Q17'].unique()
data['Class'],level=pd.factorize(data['Class'])
data['Class'].unique()

# label encoding using pandas
# label encoding using pandas

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import accuracy_score, roc_curve, precision_score, precision_recall_curve,classification_report,recall_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
X = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9',
       'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17' ]]
y = data[['Class' ]]
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.1)
#sc_X = StandardScaler()
#X_trainscaled=sc_X.fit_transform(X_train)
#X_testscaled=sc_X.transform(X_test)
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset

y_pred=clf.predict(X_test)
print(clf.score(X_test, y_test))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
fig=plot_confusion_matrix(clf, X_test, y_test,display_labels=["0","1"])
fig.figure_.suptitle("Confusion Matrix")
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, y_pred)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('AGILE: ROC AUC=%.3f' % (ns_auc))
print('OTHERS: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, y_pred)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='AGILE')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='OTHERS')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
import pickle
# save the model to disk
filename = 'decisiontree_model.sav'
pickle.dump(clf, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
