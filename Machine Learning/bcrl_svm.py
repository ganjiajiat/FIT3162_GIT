# Import library
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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

for feature in bcrl_study_data.columns.tolist():
    bcrl_study_data[feature].fillna(bcrl_study_data[feature].mode()[0], inplace=True)

bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
bcrl_y = bcrl_study_data[["Group"]]

method_names = ["SVM (Linear Kernel)", "SVM (Polynomial Kernel)", "SVM (Gaussian Kernel)"]
classifiers = [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')]
metrics_format_str = "\n[{0}]\nAccuracy: {1}\nSensitivity: {2}\nSpecificity: {3}\n"

for method_name, clf in zip(method_names, classifiers):
    accu, sens, spec = run_experiment(bcrl_x,bcrl_y, clf)
    print("Test 1: Filling null columns with mode.")
    print(metrics_format_str.format(method_name, accu, sens, spec))

# Drop Unrelated Columns and choose important features using chi squared method
bcrl_study_data = bcrl_study_data.drop(["Age", "Race", "Religion", "Education", "Occupation", "Children"], axis=1)

accu_list1 = []
sens_list1 = []
spec_lsit1 = []

accu_list2 = []
sens_list2 = []
spec_lsit2 = []

accu_list3 = []
sens_list3 = []
spec_lsit3 = []

for i in range(1,15):
    bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
    bcrl_y = bcrl_study_data[["Group"]]
    
    X = bcrl_x
    y = bcrl_y['Group'] == 1
    num_feats = i
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    
    for feature in bcrl_x.columns.tolist():
        if feature not in chi_feature:
            bcrl_x = bcrl_x.drop(feature, axis = 1)
    
    for method_name, clf in zip(method_names, classifiers):
        accu, sens, spec = run_experiment(bcrl_x,bcrl_y, clf)
        print("Test 2: Feature selection (Chi-squared): ")
        print("iteration {}".format(i))
        print(metrics_format_str.format(method_names, accu, sens, spec))
        if method_name == "SVM (Linear Kernel)":
            accu_list1.append(accu)
            sens_list1.append(sens)
            spec_lsit1.append(spec)
        elif method_name == "SVM (Polynomial Kernel)":
            accu_list2.append(accu)
            sens_list2.append(sens)
            spec_lsit2.append(spec)
        else:
            accu_list3.append(accu)
            sens_list3.append(sens)
            spec_lsit3.append(spec)