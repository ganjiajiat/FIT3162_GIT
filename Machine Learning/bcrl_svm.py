# Import library
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Creating a Runs experiments given the predictor and outcome datasets, the classifier,
# and, optinally, the number of stratified K-fold repetitions to perform.
def run_experiment(x, y, clf, n_folding_repeats=250):
    cumm_accu, cumm_sens, cumm_spec, iterations = 0, 0, 0, 0
    rskf = RepeatedStratifiedKFold(n_repeats=n_folding_repeats, random_state=0)
    for train_indices, test_indices in rskf.split(x, y.values.reshape((y.shape[0],))):
        clf.fit(x.iloc[train_indices,:], y.iloc[train_indices,:].values.reshape((len(train_indices))))
        y_pred = clf.predict(x.iloc[test_indices,:])
        (tp, fn), (fp, tn) = confusion_matrix(y.iloc[test_indices,:].values.reshape((len(test_indices))), y_pred)
        cumm_accu += (tp + tn) / (tp + fn + fp + tn)
        cumm_sens += tp / (tp + fn)
        cumm_spec += tn / (tn + fp)
        iterations += 1
    return (cumm_accu / iterations, cumm_sens / iterations, cumm_spec / iterations)

# Step 1: Import Dataset
bcrl_study_data = pd.read_csv("BCRL Study UPM_Monash Uni.csv")

# Step 2: Data preprocessing
# Data imputation
bcrl_study_data['BC receptor'].replace(3,np.NaN, inplace = True)
bcrl_study_data['Number of lymph nodes removed'].replace(3,np.NaN, inplace = True)
bcrl_study_data = bcrl_study_data.drop(["ID"], axis=1)

for feature in bcrl_study_data.columns.tolist():
    bcrl_study_data[feature].fillna(bcrl_study_data[feature].mode()[0], inplace=True)

bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
bcrl_y = bcrl_study_data[["Group"]]

method_names = ["SVM (Linear Kernel)"]
classifiers = [SVC(kernel='linear')]
metrics_format_str = "\n[{0}]\nAccuracy: {1}\nSensitivity: {2}\nSpecificity: {3}\n"

for method_name, clf in zip(method_names, classifiers):
    accu, sens, spec = run_experiment(bcrl_x,bcrl_y, clf)
    print("Test 1: Filling null columns with mode.")
    print(metrics_format_str.format(method_name, accu, sens, spec))

accu_list = []
sens_list = []
spec_list = []

bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
bcrl_y = bcrl_study_data[["Group"]]

i = 12
X = bcrl_x
y = bcrl_y['Group'] == 1
num_feats = i
X_norm = MinMaxScaler().fit_transform(X)
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()

for feature in bcrl_x.columns.tolist():
    if feature not in rfe_feature:
        bcrl_x = bcrl_x.drop(feature, axis = 1)

for method_name, clf in zip(method_names, classifiers):
    accu, sens, spec = run_experiment(bcrl_x,bcrl_y, clf)
    print("Test 2: Feature selection (RFE): ")
    print(metrics_format_str.format(method_name, accu, sens, spec))
    accu_list.append(accu)
    sens_list.append(sens)
    spec_list.append(spec)