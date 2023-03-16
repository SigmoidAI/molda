'''
Created with love by Sigmoid
@Author - Butucea Adelina - butucea.adelina@gmail.com
'''

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.feature_extraction.text import CountVectorizer
FLOAT_DTYPES = (np.float16, np.float32, np.float64)


class ObservedExpectedTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, epsilon = 3e-10):
    self.epsilon = epsilon
  
  def fit(self, X, y=None):
    if not sp.issparse(X):
      X = sp.csr_matrix(X)
    return self
  
  def transform(self, X, y=None):
    if not sp.issparse(X):
      sp.csr_matrix(X)

    allsum = np.sum(X)
    colsum = np.sum(X, axis = 0)
    rowsum = np.sum(X, axis = 1)

    for i in range(len(colsum)):
      for j in range(len(rowsum)):
        expected = colsum[i] * rowsum[j]/(allsum+self.epsilon)
        X[j][i] /= (expected+self.epsilon)
    
    return X
  
  def fit_transform(self, X, y=None):
    return self.fit(X).transform(X)
    

class ObservedExpectedVectorizer(CountVectorizer):
    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_cf=1.0,
        min_cf=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        epsilon=3e-10,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_cf,
            min_df=min_cf,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.epsilon = epsilon
  
    def _check_params(self):
      if self.dtype not in FLOAT_DTYPES:
        warnings.warn(
                  "Only {} 'dtype' should be used. {} 'dtype' will "
                  "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                  UserWarning,
        )

    def fit(self, raw, y=None):
      X = super().fit_transform(raw)
      self._obexp = ObservedExpectedTransformer()
      self._obexp.fit(X)
      return self
    

    def transform(self, raw):
      self._check_params()

      check_is_fitted(self, msg="The ObservedExpected instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute.")
      
      X = super().transform(raw)
      return self._obexp.transform(X.toarray())

    def fit_transform(self, raw_documents, y=None):
      self._check_params()
      X = super().fit_transform(raw_documents)
      self._obexp = ObservedExpectedTransformer(epsilon=self.epsilon)

      return self._obexp.fit(X).transform(X.toarray())