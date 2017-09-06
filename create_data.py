#!/usr/bin/env python

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create a simulated feature matrix and output vector with 100 samples,
features, output = make_classification(n_samples = 100,
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
