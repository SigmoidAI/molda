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


class TficfTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, *, norm="l2", use_icf=True, smooth_icf=True, sublinear_tf=False):
    self.norm = norm
    self.use_icf = use_icf
    self.smooth_icf = smooth_icf
    self.sublinear_tf = sublinear_tf


  def _category_frequency(self, X):
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)

  def fit(self, X, y=None):
    if not sp.issparse(X):
      X = sp.csr_matrix(X)
    dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

    if self.use_icf:
      n_samples, n_features = X.shape
      cf = self._category_frequency(X)
      cf = cf.astype(dtype, copy=False)
      cf += int(self.smooth_icf)
      n_samples += int(self.smooth_icf)

      icf = np.log(n_samples/cf) + 1
    
      self._icf_diag = sp.diags(
          icf,
          offsets=0,
          shape = (n_features, n_features),
          format = 'csr',
          dtype = dtype
      )
      return self

  def transform(self, X, copy=True):
    X = self._validate_data(
        X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
    )
    if not sp.issparse(X):
      sp.csr_matrix(X)
    
    if self.sublinear_tf:
      np.log(X.data, X.data)
      X.data += 1

    if self.use_icf:
      check_is_fitted(self, attributes=["icf_"], msg="icf vector is not fitted") 
      X = X * self._icf_diag

    if self.norm is not None:
      X = normalize(X, norm=self.norm, copy=False)
    return X

  def fit_transform(self, X, y=None):
    return self.fit(X).transform(X)


  @property
  def icf_(self):
    return np.ravel(self._icf_diag.sum(axis=0))
    
  @icf_.setter
  def icf_(self, value):
      value = np.asarray(value, dtype=np.float64)
      n_features = value.shape[0]
      self._icf_diag = sp.spdiags(
          value, diags=0, m=n_features, n=n_features, format="csr"
      )


class TficfVectorizer(CountVectorizer):
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
        norm="l2",
        use_icf=True,
        smooth_icf=True,
        sublinear_tf=False,
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
        self.norm = norm
        self.use_icf = use_icf
        self.smooth_icf = smooth_icf
        self.sublinear_tf = sublinear_tf

  def _check_params(self):
    if self.dtype not in FLOAT_DTYPES:
      warnings.warn(
        "Only {} 'dtype' should be used. {} 'dtype' will "
        "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
        UserWarning,
            )
  
  def fit(self, raw, y=None):

    self._tficf = TficfTransformer(
        norm = self.norm,
        use_icf = self.use_icf,
        smooth_icf = self.smooth_icf,
        sublinear_tf = self.sublinear_tf
    )
    X = super().fit_transform(raw)

    self._tficf.fit(X)
    return self
  
  def transform(self, raw):
    check_is_fitted(self, msg="The TF-ICF vectorizer is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute.")
    X = super().transform(raw)
    return self._tficf.transform(X, copy=False)
  
  def fit_transform(self, raw, y=None):
    return self.fit(raw).transform(raw)
  
  @property
  def icf_(self):
    if not hasattr(self, "_tficf"):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute."
            )
    return self._tficf.icf_

  @icf_.setter
  def icf_(self, value):
      if not self.use_icf:
          raise ValueError("`icf_` cannot be set when `user_icf=False`.")
      if not hasattr(self, "_tficf"):
          self._tficf = TficfTransformer(
              norm=self.norm,
              use_icf=self.use_icf,
              smooth_icf=self.smooth_icf,
              sublinear_tf=self.sublinear_tf,
          )
      # self._validate_vocabulary()
      if hasattr(self, "vocabulary_"):
          if len(self.vocabulary_) != len(value):
              raise ValueError(
                  "idf length = %d must be equal to vocabulary size = %d"
                  % (len(value), len(self.vocabulary))
              )
      self._tficf.icf_ = value