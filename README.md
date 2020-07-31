# Localized Feature Selection (LFS)


Full documentation can be found at: lfspy.readthedocs.io

Localized feature selection (LFS) is a supervised machine learning approach for embedding localized feature selection in classification. The sample space is partitioned into overlapping regions, and subsets of features are selected that are optimal for classification within each local region. As the size 
and membership of the feature subsets 
can vary across regions, LFS is able to adapt to local variation across the entire sample space.

### Installation
It is recommended to install in a [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

```
pip install lfspy
```

### Dependancies
* Python 3
* [NumPy](https://numpy.org/)>=1.14
* [SciPy](https://www.scipy.org/)>=1.1
* [Scikit-learn](https://scikit-learn.org/stable/index.html)>=0.18.2
* [pytest](https://docs.pytest.org/en/latest/)>=5.0.0