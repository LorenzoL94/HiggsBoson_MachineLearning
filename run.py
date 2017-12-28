# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from proj1_helpers import *
from run_helper import *
from data_processing import *

# Acquire data from the csv file
print('Load Data')
y_data, x_data, ids = load_csv_data('data/train.csv', sub_sample=False)

#%% HARD CODED BEST PARAMS

degrees = [1/2,1,2,3,4,5,6,7,8,9,10,11]

lambda_ = 3.162277660168379e-07

k_fold = 8      ## 16.7% fo testing, 83.3% fo training each fold

#%% FEATURE BUILDING
""" Add features to the model combining existing ones"""

# Differentiate primitive features from all the others
primes = np.arange(13,30)
allfeat = np.arange(0,30)

x_data, indices_to_keep = cleanup_train_data(y_data, x_data)

print('Starting cross building on train data')
# Add cross features, use only primitive features for cross terms generation
tx_data = build_cross_features(x_data, primes)

# Clear useless variable
x_data = None

print('Starting polynomial building on train data')
# Build polynomial data with given degreees
tx_data = build_poly(tx_data, degrees, allfeat)

#%% CALCULATE WEIGHTS WITH RIDGE REGRESSION

w_star, _, score, _ = cross_validation(y_data, tx_data, lambda_, k_fold)

print('Score on cross validation test is: {}'.format(score))

#%% WRITE FINAL PREDICTIONS

# Clear old heavy variables to have some free memory
x_data = None
y_data = None
ids = None
tx_data = None

# Acquire final test data
_, x_final, ids_final = load_csv_data('data/test.csv')

x_final = cleanup_test_data(x_final,indices_to_keep)

print('Starting cross building on final test data')
tx_final = build_cross_features(x_final, primes)
print('Starting polynomial building on final test data')
tx_final = build_poly(tx_final, degrees, allfeat)

# Evaluate predictions
y_pred = predict_labels(w_star, tx_final)

# Write predictions to csv file
create_csv_submission(ids_final, y_pred, 'data/submission_with_cross_validation.csv')
