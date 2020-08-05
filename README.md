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
### Install requirements
``pip install -r requirements.txt``

### Dependancies
* Python 3
* [NumPy](https://numpy.org/)>=1.14
* [SciPy](https://www.scipy.org/)>=1.1
* [Scikit-learn](https://scikit-learn.org/stable/index.html)>=0.18.2
* [pytest](https://docs.pytest.org/en/latest/)>=5.0.0
* [Seaborn] (https://seaborn.pydata.org/)

#### Core File descriptions
* main.py: This is the main python file that does the comparison and generate the artifacts for the report
* Final Project.ipynb: This is the jupyter version of main.py
* Makefile: Generates the report using pre-existing training data
* Makefile_New: Retrains the model, does a comparison and generates a report
* report.tex: Contains the latex template for the report
