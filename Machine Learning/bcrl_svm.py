import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def run_experiment(x, y, clf, n_folding_repeats=250):
    """
    Runs experiment and produce results to evaluate the strength of the model (accuracy, sensitivity, specificity). 
    
    :param x: The predictors of the dataset
    :type x: pandas.DataFrame
    :param y: The outcomes of the dataset
    :type y: pandas.DataFrame
    :param clf: The type of machine learning classifier 
    :type clf: classifier of sklearn
    :param n_folding_repeats: Number of folds for Stratified K-Fold
    :type n_folding_repeats: int
    :returns: a tuple representing the average accuracy, average sensitivity, average specificity
    :rtype: tuple  
    """
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

def replace_null_with_mode(dataset):
    """
    In-place replacement of null values with mode.
    
    :param dataset: The initial dataset with some null values
    :type dataset: pandas.DataFrame
    """
    for feature in dataset.columns.tolist():
        dataset[feature].fillna(dataset[feature].mode()[0], inplace = True)

def print_result(clf, accu, sens, spec):
    """Print results by specifying its classifier, accuracy, sensitivity and specificity"""
    metrics_format_str = "\n[{0}]\nAccuracy: {1}\nSensitivity: {2}\nSpecificity: {3}\n" # Specify result output format
    print(metrics_format_str.format(clf, accu, sens, spec))

if __name__ == "__main__":
    # Step 1: Import Dataset
    bcrl_study_data = pd.read_csv("BCRL Study UPM_Monash Uni.csv")
    
    # Step 2: Data preprocessing
    
    # Replace "Not answered" with Null value for processing
    bcrl_study_data['BC receptor'].replace(3,np.NaN, inplace = True) 
    bcrl_study_data['Number of lymph nodes removed'].replace(3,np.NaN, inplace = True)
    
    bcrl_study_data = bcrl_study_data.drop(["ID"], axis=1) # Drop identification column
    replace_null_with_mode(bcrl_study_data) # Data imputation: Replace null values with mode
    
    # Step 3: Training and Testing
    
    # Split dataset into predictors and estimators
    bcrl_x = bcrl_study_data.drop(["Group"], axis=1)
    bcrl_y = bcrl_study_data[["Group"]]
    
    method_name = "SVM (Linear Kernel)" # Specify classifier name
    classifier = SVC(kernel='linear') # Specify Scikit Learn machine learning classifiers
    
    ###########################################################################
    ##### Run test for Scenario 1: Support Vector Machine (Linear Kernel) #####
    ###########################################################################
    accu, sens, spec = run_experiment(bcrl_x,bcrl_y, classifier)
    print("Scenario 1: Support Vector Machine (Linear Kernel)")
    print_result(method_name, accu, sens, spec)
    
    ##########################################################################################################
    #####  Run test for Scenario 2: Recursive Feature Selection + Support Vector Machine (Linear Kernel) #####
    ##########################################################################################################
   
    # Run feature selection (RFE) to select top 12 features from 30 features
    X = bcrl_x
    y = bcrl_y['Group'] == 1
    num_feats = 12 # Choose the top 12 features using RFE
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    
    # Remove features that are not selected
    for feature in bcrl_x.columns.tolist():
        if feature not in rfe_feature:
            bcrl_x = bcrl_x.drop(feature, axis = 1)
    
    accu, sens, spec = run_experiment(bcrl_x,bcrl_y, classifier)
    print("Scenario 2: Recursive Feature Selection + Support Vector Machine (Linear Kernel)")
    print_result(method_name, accu, sens, spec)