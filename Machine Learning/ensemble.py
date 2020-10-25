import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
 
# get a voting ensemble of models
def get_voting():
    # define the base models
    models = list()
    models.append(('Logistic Regression', LogisticRegression(random_state=1, max_iter=1000)))
    models.append(('SVC - Linear', SVC(kernel='linear', probability=True)))
    models.append(('Naive Bayes', GaussianNB()))
     # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble
 
# get a list of models to evaluate
def get_models():
    models = dict()
    models['Logistic Regression'] = LogisticRegression(random_state=1, max_iter=1000)
    models['SVC - Linear'] = SVC(kernel='linear', probability=True)
    models['Naive Bayes'] = GaussianNB()
    models['Soft Voting'] = get_voting()
    return models
 

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

accu_list1 = []
sens_list1 = []
spec_list1 = []

accu_list2 = []
sens_list2 = []
spec_list2 = []

accu_list3 = []
sens_list3 = []
spec_list3 = []

accu_list4 = []
sens_list4 = []
spec_list4 = []

for i in range(1, 20):
    bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
    bcrl_y = bcrl_study_data[["Group"]]
    
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
            
    metrics_format_str = "\n[{0}]\nAccuracy: {1}\nSensitivity: {2}\nSpecificity: {3}\n"
    
    X, y = bcrl_x, bcrl_y
    
    # get the models to evaluate
    models = get_models()
    for name, model in models.items():
        scores = run_experiment(X, y, model)
        print(metrics_format_str.format(name, scores[0], scores[1], scores[2]))
        if name == "Logistic Regression":
            accu_list1.append(scores[0])
            sens_list1.append(scores[1])
            spec_list1.append(scores[2])
        if name == "SVC - Linear":
            accu_list2.append(scores[0])
            sens_list2.append(scores[1])
            spec_list2.append(scores[2])
        if name == "Naive Bayes":
            accu_list3.append(scores[0])
            sens_list3.append(scores[1])
            spec_list3.append(scores[2])
        else:
            accu_list4.append(scores[0])
            sens_list4.append(scores[1])
            spec_list4.append(scores[2])
