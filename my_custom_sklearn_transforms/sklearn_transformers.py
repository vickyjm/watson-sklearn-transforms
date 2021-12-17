from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from catboost import CatBoostClassifier

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class DistTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns, transform_type):
        self.columns = columns
        self.transform_type = transform_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()

        if transform_type == 'log':
            for c in self.columns:
                data[c+'_LOG'] = np.log(data[c])
        elif transform_type == 'cbrt':
            for c in self.columns:
                data[c+'_CBRT'] = np.cbrt(data[c])
        elif transform_type == 'sqrt':
            for c in self.columns:
                data[c+'_SQRT'] = np.sqrt(data[c])
        elif transform_type == 'square':
            for c in self.columns:
                data[c+'_SQUARE'] = np.square(data[c])
        elif transform_type == 'abs':
            for c in self.columns:
                data[c+'_ABS'] = np.abs(data[c])

        return data

class CatBoostModel(BaseEstimator, TransformerMixin):
    def __init__(self, scale_pos_weight=1):
        self.model = CatBoostClassifier(scale_pos_weight=scale_pos_weight)

    def fit(self, X, y=None):
        self.model.fit(X,y)
        return self

    def transform(self, X):
        return self.model.predict(X)

