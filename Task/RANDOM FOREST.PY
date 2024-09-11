# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:35:14 2022

@author: Noreen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:26:58 2022

@author: Noreen
"""


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve,classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


dataset = pd.read_csv('E:/ainee/AINEE DATA+CODE/Isolated_retinal_dytrophies_vs_syndrome.csv')

dataset.dropna()
dataset.columns


X = dataset[['GENDER', 'Age', 'ETHNICITY',
       'PARENT_COUSIN_MARRIAGE', 'RELATION_OF_PARENTS', 'NUMBER_OF_AFFECTED','Weight_(KGs) ','height_inches',
       'Age_of_Onset_of_diagnosed_disease',
       'CURRENT_VISION_STATUS', 'Any_diagnostic_tests_of_eye_done',
       'Any_medical_surgical_treatments_done', 'Night_Blindness ',
       'Onset_of_Night_Blindness', 'PHOTOPHOBIA', 'aa', 'NYSTAGMUS',
       'Color_Perception', 'Light_Perception', 'Cataract', 'Microphthalmia',
       'Anophthalmia', 'Corneal_Opacity', 'Blindness', 'Onset_of_Blindness',
       'Hearing', 'Speaking', 'Facial_dysmorphism',
       'Teeth_Structure_and_Count', 'Bone_Deformities',
       'Infertile_Fertile_Unmarried', 'NUMBER_AND_STRUCTURE_OF_HANDS_FINGERS',
       'NUMBER_AND_STRUCTURE_OF_FOOT_FINGERS', 'ANY_OTHER_DISEASES',
       'OTHER_ASSESSMENTS', 'VISUAL_ACUITY', 'Visual_Field',
       'ANY_OTHER_FINDINGS']]
#X = dataset[['start_node','end_node','agg_minutes', 'Condition_Weather', 'holiday', 'peak hour','hour', 'day', 'month', 'year']]
y = dataset['DIAGNOSED_DISEASE']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle='true')
model= RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)

model.fit(X_train, y_train)
y_pred = model.predict(X_test).round()
#y_score = model.predict_proba(X_test)
# test last record in test.csv manually and the answer is correct which is 4 minutes i.e. 240 seconds
print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
print('accuracy score', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print(metrics.confusion_matrix(y_test, y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_pred, digits=2))

values = y_test
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#precision, recall, thresholds = precision_recall_curve(y_test,y_pred)
## precision recall curve
precision = dict()
recall = dict()
for i in range(2):
    precision[i], recall[i], _ = precision_recall_curve(onehot_encoded[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

print("Accuracy score (training): {0:.3f}".format(model.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(model.score(X_test, y_test)))

