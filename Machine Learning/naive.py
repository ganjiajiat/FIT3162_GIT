
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn import metrics

bcrl_study_data = pd.read_csv("BCRL Study UPM_Monash Uni.csv")
bcrl_study_data = bcrl_study_data.dropna()

bcrl_x = bcrl_study_data.drop(["ID", "Group"], axis=1)
bcrl_y = bcrl_study_data["Group"]

bcrl_x_train, bcrl_x_test, bcrl_y_train, bcrl_y_test = train_test_split(bcrl_x, bcrl_y, test_size=0.25)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(bcrl_x_train, bcrl_y_train)

#Predict the response for test dataset
y_pred = gnb.predict(bcrl_x_test)

print("Accuracy:",metrics.accuracy_score(bcrl_y_test, y_pred))