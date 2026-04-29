# rf_wrapper.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from .auc_utils import auc, split_xy


class RFWrapper:
    def __init__(self, n_estimators=100, random_state=42):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit_and_score(self, df, target_col='target'):
        X, y = split_xy(df, target_col)
        probs = cross_val_predict(self.clf, X, y, cv=5, method='predict_proba')[:, 1]
        return auc(y, probs)

    def train_and_score(self, train_df, test_df, target_col='target'):
        X_train, y_train = split_xy(train_df, target_col)
        X_test, y_test = split_xy(test_df, target_col)
        self.clf.fit(X_train, y_train)
        probs = self.clf.predict_proba(X_test)[:, 1]
        return auc(y_test, probs)

    def stratified_score(self, df, target_col='target', test_size=0.3, n_repeats=10):
        from sklearn.model_selection import StratifiedShuffleSplit
        import numpy as np
        X, y = split_xy(df, target_col)
        splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=self.clf.random_state)
        scores = []
        for train_idx, test_idx in splitter.split(X, y):
            self.clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = self.clf.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(auc(y.iloc[test_idx], probs))
        return np.mean(scores), np.min(scores), np.max(scores)

    @classmethod
    def from_combined(cls, df, target_col='target', test_size=0.3, n_repeats=10, **kwargs):
        return cls(**kwargs).stratified_score(df, target_col, test_size, n_repeats)

    @classmethod
    def from_dataframe(cls, df, target_col='target', **kwargs):
        return cls(**kwargs).fit_and_score(df, target_col)

    @classmethod
    def from_pair(cls, train_df, test_df, target_col='target', **kwargs):
        return cls(**kwargs).train_and_score(train_df, test_df, target_col)