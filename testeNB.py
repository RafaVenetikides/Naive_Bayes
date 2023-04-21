from sklearn.datasets import load_breast_cancer
from tabulate import tabulate
import pandas as pd
from NaiveBayes import Naive_bayes

X, y=load_breast_cancer(return_X_y=True)
nb = Naive_bayes()

nb.Py(y)
