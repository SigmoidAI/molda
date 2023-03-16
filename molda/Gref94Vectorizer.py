'''
Created with love by Sigmoid

@Author - Sclifos Tudor - tudor.sclifos@isa.utm.md
'''

from sklearn.base import BaseEstimator, TransformerMixin#, _OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES, check_scalar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

import re
import unicodedata

import numpy as np
import scipy.sparse as sp
import struct

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

_IS_32BIT = 8 * struct.calcsize("P") == 32


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


def _preprocess(doc, accent_function=None, lower=False):
    """Chain together an optional series of text preprocessing steps to
    apply to a document.
    Parameters
    ----------
    doc: str
        The string to preprocess
    accent_function: callable, default=None
        Function for handling accented characters. Common strategies include
        normalizing and removing.
    lower: bool, default=False
        Whether to use str.lower to lowercase all of the text
    Returns
    -------
    doc: str
        preprocessed string
    """
    if lower:
        doc = doc.lower()
    if accent_function is not None:
        doc = accent_function(doc)
    return doc


def _analyze(
        doc,
        analyzer=None,
        tokenizer=None,
        ngrams=None,
        preprocessor=None,
        decoder=None,
        stop_words=None,
):
    """Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing or preprocessing.
    If analyzer is used, only the decoder argument is used, as the analyzer is
    intended to replace the preprocessor, tokenizer, and ngrams steps.
    Parameters
    ----------
    analyzer: callable, default=None
    tokenizer: callable, default=None
    ngrams: callable, default=None
    preprocessor: callable, default=None
    decoder: callable, default=None
    stop_words: list, default=None
    Returns
    -------
    ngrams: list
        A sequence of tokens, possibly with pairs, triples, etc.
    """

    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc


def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart
    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.
    Parameters
    ----------
    s : string
        The string to strip
    See Also
    --------
    strip_accents_ascii : Remove accentuated char for any unicode symbol that
        has a direct ASCII equivalent.
    """
    try:
        # If `s` is ASCII-compatible, then it does not contain any accented
        # characters and we can avoid an expensive list comprehension
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join([c for c in normalized if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    """Transform accentuated unicode symbols into ascii or nothing
    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.
    Parameters
    ----------
    s : str
        The string to strip
    See Also
    --------
    strip_accents_unicode : Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize("NFKD", s)
    return nkfd_form.encode("ASCII", "ignore").decode("ASCII")


def strip_tags(s):
    """Basic regexp based HTML / XML tag stripper function
    For serious HTML/XML preprocessing you should rather use an external
    library such as lxml or BeautifulSoup.
    Parameters
    ----------
    s : str
        The string to strip
    """
    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, str):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:  # assume it's a collection
        return frozenset(stop)


class Gref94Transformer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            *,
            epsilon=3e-10
    ):
        '''
            Setting up the algorithm
        :param epsilon: default = 3e-10

        '''
        self.epsilon = epsilon

    def fit(self, X, y=None, **fit_params):
        '''
            Fit the data
        :param X: Sparse matrix of term/token counts
        :param y: The classes of the classification problem

        :return self
        '''

        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), accept_large_sparse=not _IS_32BIT
        )

        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        n_samples, n_features = X.shape
        df = _document_frequency(X)
        self.df = df.astype(dtype, copy=False)

        idf = np.log(1 / (df + self.epsilon) + 1)

        self._idf_diag = sp.diags(
            idf,
            offsets=0,
            shape=(n_features, n_features),
            format="csr",
            dtype=dtype,
        )

        return self

    def transform(self, X, y=None, **fit_params):
        '''
            Use the Gref94 formula
        :param X: Sparse matrix of term/token counts
        :param y: The classes of the classification problem

        If smooth_idf is True, apply document frequency smoothing

        :return vectors : sparse matrix
            Gref94-weighted document-term matrix.
        '''

        X = self._validate_data(
            X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=False, reset=False
        )

        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        X = X * self._idf_diag

        return X

    def fit_transform(self, X, y=None, **fit_params):

        '''
            Fit and transform the passed data
        :param X: Sparse matrix of term/token counts
        :param y: The classes of the classification problem

        :return vectors : sparse matrix
            Gref94-weighted document-term matrix.
        '''

        return self.fit(X).transform(X)

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}


class Gref94Vectorizer(CountVectorizer):
    r"""Convert a collection of raw documents to a matrix of Gref94 features.
    Equivalent to :class:`CountVectorizer` followed by
    :class:`Gref94Transformer`.

    Parameters
    ----------
    input : {'filename', 'file', 'content'}, default='content'
        - If `'filename'`, the sequence passed as an argument to fit is
          expected to be a list of filenames that need reading to fetch
          the raw content to analyze.
        - If `'file'`, the sequence items must have a 'read' method (file-like
          object) that is called to fetch the bytes in memory.
        - If `'content'`, the input is expected to be a sequence of items that
          can be of type string or byte.
    encoding : str, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode'}, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer`` is not callable.
    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
        .. versionchanged:: 0.21
            Since v0.21, if ``input`` is ``'filename'`` or ``'file'``, the data
            is first read from the file and then passed to the given callable
            analyzer.
    stop_words : {'english'}, list, default=None
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
        If there is a capturing group in token_pattern then the
        captured group content, not the entire match, becomes the token.
        At most one capturing group is permitted.
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.
        Only applies if ``analyzer`` is not callable.
    max_df : float or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float in range [0.0, 1.0], the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float in range of [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    binary : bool, default=False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs).
    dtype : dtype, default=float64
        Type of the matrix returned by fit_transform() or transform().
    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        - 'l2': Sum of squares of vector elements is 1. The cosine
          similarity between two vectors is their dot product when l2 norm has
          been applied.
        - 'l1': Sum of absolute values of vector elements is 1.
          See :func:`preprocessing.normalize`.
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    fixed_vocabulary_ : bool
        True if a fixed vocabulary of term to indices mapping
        is provided by the user.
    idf_ : array of shape (n_features,)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.

    """

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
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
            epsilon=3e-10
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
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.epsilon = epsilon

    # Broadcast the Gref94 parameters to the underlying transformer instance
    # for easy grid search and repr

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : None
            This parameter is not needed to compute tfidf.
        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self._check_params()
        self._warn_for_unused_params()
        self._tfidf = Gref94Transformer(
            epsilon=self.epsilon
        )
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        y : None
            This parameter is ignored.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        self._tfidf = Gref94Transformer(
            epsilon=self.epsilon
        )
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, msg="The Gref94 vectorizer is not fitted")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {"X_types": ["string"], "_skip_test": True}
