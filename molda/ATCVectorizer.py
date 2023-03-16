'''
Created with love by Sigmoid
@Author - Smocvin Denis - denis.smocvin@isa.utm.md
'''

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin #, _OneToOneFeatureMixin
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted

class ATCTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        '''
        The constructor of the class.
        :param norm: string ({'l1', 'l2', default='l2'})
            Each output row will have a unit norm.
        :param use_idf: bool
            the distance metric to use for finding the nearest neighbor
        :param smooth_idf: bool
            Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every
            term in the collection exactly once. Prevents division by zero.
        :param sublinear_tf: bool
            Apply sublinear TF scaling, i.e. replace TF with 1 + log(TF).
        '''

        self.norm = norm

    def fit(self, X, y=None, **fit_params):
        '''
        :param X:
            Sparse matrix of term/token counts
        :param y:
            The classes of the classification problem
        '''
        return self

    def transform(self, X, y=None, **fit_params):
        '''
        :param X:
            Sparse matrix of term/token counts
        :param y:
            The classes of the classification problem
        '''

        X = np.array(X.toarray())

        N = X.shape[0]
        max_f = np.max(X)

        # number of documents in which a term appears
        n = np.count_nonzero(X, axis=0)

        # calculate intermediate values
        log_nn = np.log(N / n)
        f_max_f = X / max_f
        left = .5 + f_max_f / 2

        numerator = left * log_nn
        denominator = np.sum(np.square(numerator), axis=1)

        X = np.array(numerator / np.reshape(denominator, (denominator.shape[0], -1)))

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)


class ATCVectorizer:
    def __init__(self, *, input='content', encoding='utf-8', decode_error='strict', trip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern=r'(?u)\b\w\w+\b',
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64):
        '''
        Initialize the vectorizer object.
        '''

        # initialize countVectorizer
        self.count_vectorizer = CountVectorizer(input=input, lowercase=lowercase, preprocessor=preprocessor,
                                                tokenizer=tokenizer,
                                                stop_words=stop_words, token_pattern=token_pattern,
                                                ngram_range=ngram_range,
                                                analyzer=analyzer, min_df=min_df, max_df=max_df,
                                                max_features=max_features,
                                                vocabulary=vocabulary, binary=binary, dtype=dtype)

        self.transformer, self.BOW, self._y = y = None, None, None

    def fit(self, raw_documents, y=None):
        '''
        :param raw_documents: list
            The list of documents representing the corpus
        :param y:
            The classes of the documents
        '''
        self.transformer = ATCTransformer()

        self.BOW = self.count_vectorizer.fit_transform(raw_documents)
        self.BOW = self.BOW.astype(np.single)
        self.transformer.fit(self.BOW)
        self._y = y

        return self

    def transform(self, raw_documents):
        '''
        :param raw_documents: list
            The list of documents representing the corpus
        :return:
            The transformed matrix
        '''

        if self.BOW is None or self.transformer is None:
            raise NotFittedError

        return self.transformer.transform(self.BOW, self._y)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)
