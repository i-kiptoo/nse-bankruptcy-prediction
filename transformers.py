from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataFrameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer
        
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X_t = self.transformer.transform(X)
        return pd.DataFrame(X_t, columns=self.columns_, index=X.index)