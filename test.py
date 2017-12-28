# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
from run_helper import *
from data_processing import *

# Acquire data from the csv file
print('Load Data')
y_data, x_data, ids = load_csv_data('train.csv', sub_sample=False)

#%% FEATURE BUILDING
""" Add features to the model combining existing ones"""

# Isolate primitive features from all the others
primes = np.arange(13,30)
allfeat = np.arange(0,30)

# Decide starting degrees
degrees = [0,1]

x_data = normalize_invalid_data(x_data)

print('Starting cross building on train data')
tx_data = build_cross_features(x_data, primes)

print('Starting polynomial building on train data')
# Decide the degrees used in the feature augmentation and calculate polynomial
tx_data = build_poly(tx_data, degrees, allfeat)

#%% FIND BEST SET OF DEGREES

# This function compares scores for different set of degrees in order to
# understand which are the most important and when new degrees 
# start to become useless

# Create a list of degrees to test
degrees_test = [2,3,4,5,6,7,8,9,10,1/2,1/3,1/4]
# Create a list of lambdas
lambdas = np.logspace(-9,-3,5)
# Decide number of folds in cross_validation
k_fold = 8

w_star, score = iterate_over_degrees(y_data, tx_data, degrees_test, allfeat, lambdas, k_fold)

print('Best score over degrees is: {}'.format(score))


#%% FIND BEST LAMBDAS FOR A GIVEN DEGREE AND PLOT SCORES
# Iterate cross validation test for some values of lambda and return 
# the one with the higher score on test folds. Plot train and test 
# scores over lambdas."""
"""
lambdas = np.logspace(-10, 0,20)
k_fold = 8
w_star, score = iterate_over_lambdas(y_data, tx_data, lambdas, k_fold)

print('Best score over lambdas is: {}'.format(score))
"""