# Molda

Molda is a sci-kit learn inspired Python library for text vectorization of corpora. It is adapted to work in pipelines and numpy arrays.

The current version supports many algorithms denoted by the following classes:

* TTestVectorizer
* TficfVectorizer
* ObservedExpectedVectorizer
* LTUVectorizer
* Gref94Vectorizer
* ATCVectorizer

These classes are based on the sci-kit learn's CountVectorizer.

You need to instantiate the vectorizer with the parameters you need, fit and apply the transformations. Here is an example:

```python
from Tficf import TficfVectorizer

corpus = np.array([
    "Even though I enjoyed watching that, This is bullshit",
    "I really enjoyed watching that",
    "I resent watching this video"
])

y = [1, 0, 1]

v = TficfVectorizer()
v.fit(corpus, y)
v.transform(['Hello, there'])
```

Also, you can include the vectorizer in a pipeline, like in the following example:

```python
pipe = Pipeline([
            ('vectorizer', TficfVectorizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('estimator', SVC())
        ])
pipe.fit(corpus, y)
pipe.score(corpus, y)
pipe.predict(['This is wonderful'])
```

Molda works with Pandas DataFrames too:
```python
df = pd.read_csv('../irony-labeled.csv')
df = df.dropna()

corpus_ = df['comment_text'].to_numpy()
y_ = df['label'].to_numpy()

v = TficfVectorizer()
v.fit(corpus_, y_)
v.transform(['Hello, there', 'Goodbye'])
```

With love from Sigmoid.

We are open for feedback. Please send your impression to vladimir.stojoc@gmail.com
