import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

class Experiment:
    def __init__(self, prelearn_processor_names, prelearn_processors, model_learner_names, model_learners, x, y, num_cv_folds=5, num_exp_iter=50):
        self.prelearn_processors = [o for o in zip(prelearn_processor_names, prelearn_processors)]
        self.model_learners = [o for o in zip(model_learner_names, model_learners)]
        self.experiment_results = np.zeros((len(model_learners), len(prelearn_processors), 3))
        self.x = x
        self.y = y
        self.cv_folder = RepeatedStratifiedKFold(n_splits=num_cv_folds, n_repeats=int(num_exp_iter/num_cv_folds))
  
    def getFeatureSelectorIdx(self, name):
        for i, (prelearn_processor_name, _) in enumerate(self.prelearn_processors):
            if prelearn_processor_name == name:
                return i
        return -1
  
    def getModelLearnerIdx(self, name):
        for i, (model_learner_name, _) in enumerate(self.model_learners):
            if model_learner_name == name:
                return i
        return -1

    def addFeatureSelectors(self, prelearn_processor_names, prelearn_processors):
        self.prelearn_processors += zip(prelearn_processor_names, prelearn_processors)
        shape = self.experiment_results.shape
        self.experiment_results = np.concatenate([self.experiment_results, np.zeros((shape[0], len(prelearn_processors), shape[2]))], axis=1)
  
    def addModelLearners(self, model_selector_names, model_learners):
        self.model_learners += zip(model_selector_names, model_learners)
        shape = self.experiment_results.shape
        self.experiment_results = np.concatenate([self.experiment_results, np.zeros((len(model_learners), shape[1], shape[2]))], axis=0)
  
    def runExperimentOnCombination(self, fsIdx, mlIdx):
        new_x, new_y = self.prelearn_processors[fsIdx][1](self.x, self.y)
        clf = self.model_learners[mlIdx][1]
        cumm_accu, cumm_sens, cumm_spec, iterations = 0, 0, 0, 0
        reshaped_y = new_y.values.reshape(new_y.shape[0])
        for train_indices, test_indices in self.cv_folder.split(new_x, reshaped_y):
            clf.fit(new_x.iloc[train_indices], reshaped_y[train_indices])
            (tp, fn), (fp, tn) = confusion_matrix(reshaped_y[test_indices], clf.predict(new_x.iloc[test_indices]))
            cumm_accu += (tp + tn) / (tp + fn + fp + tn)
            cumm_sens += tp / (tp + fn)
            cumm_spec += tn / (tn + fp)
            iterations += 1
        self.experiment_results[mlIdx, fsIdx, 0] = cumm_accu / iterations
        self.experiment_results[mlIdx, fsIdx, 1] = cumm_sens / iterations
        self.experiment_results[mlIdx, fsIdx, 2] = cumm_spec / iterations
  
    def runAll(self):
        for pl_idx in range(len(self.prelearn_processors)):
            for ml_idx in range(len(self.model_learners)):
                self.runExperimentOnCombination(pl_idx, ml_idx)

    def getResultsString(self):
        result_str = ""
        for pl_idx, (pl_name, _) in enumerate(self.prelearn_processors):
            for ml_idx, (ml_name, _) in enumerate(self.model_learners):
                accu = self.experiment_results[ml_idx, pl_idx, 0]
                sens = self.experiment_results[ml_idx, pl_idx, 1]
                spec = self.experiment_results[ml_idx, pl_idx, 2]
                result_str += "[{}] -> [{}]:\n\taccuracy = {}\n\tsensitivity = {}\n\tspecificity = {}\n\n".format(pl_name, ml_name, accu, sens, spec)
        return result_str

    def getSpecificResultsString(self, prelearn_processor_names, model_learner_names):
        result_str = ""
        for pl_name in prelearn_processor_names:
            for ml_name in model_learner_names:
                pl_idx = self.getFeatureSelectorIdx(pl_name)
                ml_idx = self.getModelLearnerIdx(ml_name)
                accu = self.experiment_results[ml_idx, pl_idx, 0]
                sens = self.experiment_results[ml_idx, pl_idx, 1]
                spec = self.experiment_results[ml_idx, pl_idx, 2]
                result_str += "[{}] -> [{}]:\n\taccuracy = {}\n\tsensitivity = {}\n\tspecificity = {}\n\n".format(pl_name, ml_name, accu, sens, spec)
        return result_str
