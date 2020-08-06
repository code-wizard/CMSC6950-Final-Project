# CMSC6950 Localized Feature Selection (LFS)
### Chukwuebuka Amaefula

In this project I compared the classification performance of 3 Scikit-learn 
classifiers ( K-Nearest Neighbors, Gaussian Bayes, 
and Decision Tree) with another third party software Local Feature Selection (LFS). 

#####software: (https://github.com/McMasterRS/LFSpy/tree/master/LFSpy)
#####Data source: (https://archive.ics.uci.edu/ml/datasets/iris)
To reproduce this report, clone this repo and type `make`. 
### Steps to Reproduce Report
 * clone the repo `git clone https://github.com/code-wizard/CMSC6950-Final-Project.git`
 * It is recommended to install in a [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (optional).
 * Install dependencies `pip install -r requirements.txt`
 * Run `make` to redo the comparison and create a new report
 * Run `make clean` to clean old report
 * To re-train the models before comparison run `make -f Makefile_New` (optional)

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
