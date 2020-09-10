# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:06:03 2020

@author: Gan Jia Jiat
"""

# Import the model we are using
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# clf = svm.SVC(kernel='linear', C = 1.0)

bcrl_study_data = pd.read_csv("BCRL Study UPM_Monash Uni.csv")
bcrl_study_data


# # bcrl_study_data = bcrl_study_data.dropna()

bcrl_study_data["Education"] = bcrl_study_data["Education"].replace(np.NaN, 4) # not answered
bcrl_study_data["Occupation"] = bcrl_study_data["Occupation"].replace(np.NaN, 5) # not answered
bcrl_study_data["Breast biopsy"] = bcrl_study_data["Breast biopsy"].replace(np.NaN, 3) # not answered
bcrl_study_data["Lymph nodes removal"] = bcrl_study_data["Lymph nodes removal"].replace(np.NaN, 3) # not answered
bcrl_study_data["Presence of chronic diseases"] = bcrl_study_data["Presence of chronic diseases"].replace(np.NaN, 4) # not answered
bcrl_study_data["SS1: Heaviness/tightness"] = bcrl_study_data["SS1: Heaviness/tightness"].replace(np.NaN, 3) # not answered
bcrl_study_data["SS2: Hardness/ difficulty finding shirts that fits"] = bcrl_study_data["SS2: Hardness/ difficulty finding shirts that fits"].replace(np.NaN, 3) # not answered
bcrl_study_data["SS3: Pain at any part of the hands/body"] = bcrl_study_data["SS3: Pain at any part of the hands/body"].replace(np.NaN, 3) # not answered
bcrl_study_data["Extra supplementation"] = bcrl_study_data["Extra supplementation"].replace(np.NaN, 3) # not answered
bcrl_study_data["Part of breast cancer"] = bcrl_study_data["Part of breast cancer"].replace(np.NaN, 4)
bcrl_study_data["Stage of BC"] = bcrl_study_data["Stage of BC"].replace(np.NaN, 5)

print(bcrl_study_data.info())
bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
bcrl_y = bcrl_study_data["Group"]

bcrl_x_train, bcrl_x_test, bcrl_y_train, bcrl_y_test = train_test_split(bcrl_x, bcrl_y, test_size=0.3, random_state = 42)

# bcrl_matrix =  bcrl_study_data.corr()


# # Train the model on training data
# clf.fit(bcrl_x_train, bcrl_y_train);

# # Use the forest's predict method on the test data
# predictions = clf.predict(bcrl_x_test)
# print("Accuracy:",metrics.accuracy_score(bcrl_y_test, predictions))
# print("Precision:",metrics.precision_score(bcrl_y_test, predictions))
# print("Recall:",metrics.recall_score(bcrl_y_test, predictions))

# scores = cross_val_score(clf, bcrl_x, bcrl_y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# # We defined “sensitivity” as the rate of the true positive lymphedema cases which measures the proportion of patients who were correctly identified as having lymphedema among those who do have it. 
# # “Specificity” refers to the true non-lymphedema cases which measures the proportion of patients who were correctly identified to have non-lymphedema among those who do not have it.

# tn, fp, fn, tp = confusion_matrix(bcrl_y_test, predictions).ravel()
# sensitivity = tp / (tp+fn)
# specificity = tn / (tn+fp)

# print(sensitivity)
# print(specificity)