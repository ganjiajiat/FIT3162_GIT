import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

def getInfoGainImportances(x, y, n_iters=1000, prop_threshold=0.0):
      importances = np.zeros(x.shape[1], dtype=np.int64)
  for _ in range(n_iters):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    importances += model.get_support()*1
  importances = (importances - importances.min()) / (importances.max() - importances.min())
  data = pd.DataFrame(data=[[c, v] for c, v in zip(x.columns, importances)], 
                      columns=['Feature', 'Proportion of Occurences'])
  return data[data['Proportion of Occurences'] >= prop_threshold]