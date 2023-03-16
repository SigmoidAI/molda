'''
Created with love by Sigmoid
@Author - Smocvin Denis - denis.smocvin@isa.utm.md
'''

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin #, _OneToOneFeatureMixin
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# for SparseEfficiencyWarning refer to this thread: https://stackoverflow.com/questions/45436741/changing-sparsity-structure-for-a-single-operation

class TtestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=3e-10):
        '''
        The constructor of the class.
        :param epsilon: float, default = 3e-10
            The smoothing parameter to escape division by zero.
        '''

        self.epsilon = epsilon

    def fit(self, X, y=None, **fit_params):
        '''
        :param X:
            Sparse matrix of term/token counts.
        :param y:
            The classes of the classification problem.
        :return: self
        '''

        return self

    def transform(self, X, y=None, **fit_params):
        '''
        Using the T-Test formula calculate the weights of the matrix to be transformed and return it.
        The formula for T-Test is: (P(t_ij | c_j) - P(t_ij)P(c_j)) / sqrt(P(t_ij)P(c_j))
        :param X:
            Sparse matrix of term/token counts.
        :param y:
            The classes of the classification problem.
        :return:
        '''

        classes = np.unique(y)
        num_samples = len(y)

        # find probability for each class
        p_c = np.array([(X[y == cls].toarray()).shape[0] / num_samples for cls in classes])
        p_c = p_c.reshape(-1, 1)

        # find the probability for each token
        p_t = np.sum(X, axis=0) / num_samples

        # calculate P(c)P(t) and its root
        prod_prob = p_c * p_t

        rooted_prob = np.sqrt(prod_prob)

        # calculate the conditional probabilities
        X_cls = np.array([X[y == cls] for cls in classes])
        cond_prob = np.squeeze(np.array([np.sum(X_cls[cls], axis=0) / X_cls[cls].shape[0] for cls in classes]))

        # calculate the final probabilities
        probs = (cond_prob - prod_prob) / rooted_prob

        for cls in classes:
            X[y == cls] = probs[cls]

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)


class TtestVectorizer:
    def __init__(self, *, input='content', encoding='utf-8', decode_error='strict', trip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern=r'(?u)\b\w\w+\b',
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64):
        '''
        Initialize the vectorizer object.
        '''

        # initialize countVectorizer
        self.count_vectorizer = CountVectorizer(input=input, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                                          stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range,
                                          analyzer=analyzer, min_df=min_df, max_df=max_df, max_features=max_features,
                                          vocabulary=vocabulary, binary=binary, dtype=dtype)

        self.transformer, self.BOW, self._y = None, None, None

    def fit(self, raw_documents, y=None):
        '''
        :param raw_documents: list
            the list of documents representing the corpus
        :param y:
            the classes of the documents
        :return:
        '''

        self.transformer = TtestTransformer()

        self.BOW = self.count_vectorizer.fit_transform(raw_documents)
        self.BOW = self.BOW.astype(np.single)
        self.transformer.fit(self.BOW)
        self._y = y

        return self

    def transform(self, raw_documents):
        '''
        Vectorize the corpus.
        :param raw_documents: list
            the list of documents representing the corpus
        :return:
        '''

        if self.BOW is None or self.transformer is None:
            raise NotFittedError

        return self.transformer.transform(self.BOW, self._y)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)