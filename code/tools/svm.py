import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score


class SVMClassifier:
    def __init__(self, X_train, X_valid, y_train, y_valid):

        y_train, y_valid = np.asarray(y_train), np.asarray(y_valid)

        assert X_train.shape[1] == X_valid.shape[1], "Number of features do not match"
        assert X_train.shape[0] == len(
            y_train
        ), "Number of training data points and labels do not match"
        assert X_valid.shape[0] == len(
            y_valid
        ), "Number of validation data points and labels do not match"

        self.X_train, self.X_valid = X_train, X_valid
        self.y_train, self.y_valid = y_train, y_valid

    def classify(self):

        # Scale Features: # Features should come pre-scaled

        X_train_sc = self.X_train
        X_valid_sc = self.X_valid

        self.cls = SVC(class_weight="balanced")
        params = {"C": [0.1, 0.5, 1, 10, 50]}
        self.cls = RandomizedSearchCV(self.cls, params, n_iter=5, n_jobs=5).fit(
            X_train_sc, self.y_train
        )

        y_pred_train = self.cls.predict(X_train_sc)
        y_pred_valid = self.cls.predict(X_valid_sc)

        return balanced_accuracy_score(
            self.y_train, y_pred_train
        ), balanced_accuracy_score(self.y_valid, y_pred_valid)
