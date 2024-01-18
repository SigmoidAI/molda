from setuptools import setup
long_description = '''
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

'''
setup(
  name = 'molda',
  packages = ['molda'],
  version = '0.1.1',
  license='MIT',
  description = 'Molda is a sci-kit learn inspired Python library for text vectorization of corpora. It is adapted to work in pipelines and numpy arrays.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'SigmoidAI - Stojoc Vladimir, Smocvin Denis, Butucea Adelina, Sclifos Tudor',
  author_email = 'vladimir.stojoc@gmail.com',
  url = 'https://github.com/SigmoidAI/molda',
  download_url = 'https://github.com/ScienceKot/kydavra/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'natural language processing', 'python'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scikit-learn',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
  ],
)
