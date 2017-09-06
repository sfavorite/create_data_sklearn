#!/usr/bin/env python

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Create a simulated feature matrix and output vector with 100 samples,
X, y = make_classification(n_samples = 100,
                                       # ten features
                                       n_features = 10,
                                       # five features that actually predict the output's classes
                                       n_informative = 5,
                                       # five features that are random and unrelated to the output's classes
                                       n_redundant = 5,
                                       # three output classes
                                       n_classes = 3,
                                       # with 20% of observations in the first class, 30% in the second class,
                                       # and 50% in the third class. ('None' makes balanced classes)
                                       weights = [.2, .3, .8])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(np.mean(predictions == y_test) * 100)
