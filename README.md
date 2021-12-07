# `httrees`

`httrees` is a Python module for hierarchical topic modeling. It implements an algorithm that constructs a topic hierarchy tree through successive application of flat topic models. It also contains several text vectorizer implementations, including support for fine-tuning deep word embeddings.

This project was started in 2021 as part of CS410 at the University of Illinois Urbana-Champaign.

# Dependencies

`httrees` requires:

-NumPy
-SciPy
-Pandas
-Gensim

It does not strictly require scikit-learn, but is intended to be used alongside sklearn flat clustering models, though any clustering model following the sklearn API will be compatible.

# Installation

`httrees` can be installed from git:

`pip install git+git://github.com/bllguo/CourseProject`

# Usage

A broad written overview of the implementation can be found in <TODO>
An accompanying IPython notebook with examples is also available <TODO>
