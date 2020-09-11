import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix

# Preproccessing
bcrl = pd.read_csv("BCRL Study UPM_Monash Uni.csv")
for column_name in ["Children", "Radiotherapy", "Chemotherapy", "Hormonal therapy", "Hypertension", "Diabetes", "Hypertensin & diabetes", "SS1: Heaviness/tightness", "SS2: Hardness/ difficulty finding shirts that fits", "SS3: Pain at any part of the hands/body", "Extra supplementation"]:
  for i in range(bcrl.shape[0]):
    b = bcrl.iloc[i][column_name]
    if not pd.isnull(b):
      bcrl.at[i, column_name] = 2 - int(b)
for i in range(bcrl.shape[0]):
    b = bcrl.iloc[i]["Group"]
    if not pd.isnull(b):
      bcrl.at[i, "Group"] = int(b) - 1
bcrl = bcrl.drop(["ID"], axis=1).dropna()
bcrl_no_cat = pd.get_dummies(bcrl, columns=["Race", "Religion", "Education", "Occupation", "Part of breast cancer", "Breast biopsy", "Stage of BC", "BC receptor", "Types of surgery", "Lymph nodes removal", "BC treatment", "Presence of chronic diseases", "SS1: Heaviness/tightness", "SS2: Hardness/ difficulty finding shirts that fits", "SS3: Pain at any part of the hands/body", "Extra supplementation"])

# Computing PCA components
bcrl_std_nrm_x = StandardScaler().fit_transform(bcrl_no_cat.drop(["Group"], axis=1))
pca = PCA(n_components=bcrl_std_nrm_x.shape[1]).fit_transform(bcrl_std_nrm_x)
pcdf_x = pd.DataFrame(data=pca, columns=['PC' + str(i) for i in range(1, bcrl_std_nrm_x.shape[1] + 1)])
pcdf = pd.concat([pcdf_x, bcrl_no_cat[['Group']]], axis=1)

# Reducing dimensionatility by checking cummalative variance from PC1 to PCk, 
# where k is the component after which the cummalitive variance is equal to or exceeds 99%.
min_preserved_var = 0.99
vars = pcdf.values.var(axis=0)
var_ratios = vars / vars.sum()
cumm_var_ratio, pc_count = 0, 0
for vr in var_ratios:
    cumm_var_ratio += vr
    pc_count += 1
    if cumm_var_ratio >= min_preserved_var:
        break
print("Number of preserved principle components: {0}\nProportion of preserved variance {1}\n".format(pc_count, cumm_var_ratio))
pcdf = pcdf.iloc[:, [i for i in range(pc_count)] + [pcdf.shape[1] - 1]]

# Creating a Runs experiments given the predictor and outcome datasets, the classifier,
# and, optinally, the number of stratified K-fold repetitions to perform.
def run_experiment(x, y, clf, n_folding_repeats=500):
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

# Running the experiments
method_names = ["Logistic Regression", "SVM (Linear Kernel)", "SVM (Polynomial Kernel)", "SVM (Gaussian Kernel)", "Random Forest", "Gradient Boost"]
classifiers = [LogisticRegression(solver='liblinear'), SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf'), RandomForestClassifier(), GradientBoostingClassifier()]
metrics_format_str = "\n[{0}]\nAccuracy: {1}\nSensitivity: {2}\nSpecificity: {3}"

print("WITHOUT PCA")
for method_name, clf in zip(method_names, classifiers):
    accu, sens, spec = run_experiment(bcrl.drop(["Group"], axis=1), bcrl[["Group"]], clf, n_folding_repeats=250)
    print(metrics_format_str.format(method_name, accu, sens, spec))

print("\nWITH PCA")
for method_name, clf in zip(method_names, classifiers):
    accu, sens, spec = run_experiment(pcdf.drop(["Group"], axis=1), pcdf[["Group"]], clf)
    print(metrics_format_str.format(method_name, accu, sens, spec))