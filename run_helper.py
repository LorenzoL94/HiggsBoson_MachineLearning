# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from plots import *

########################## DATA PREPROCESSING #################################
	
def split_data(x, y, ratio, seed=1):
    """ Split the dataset in training and testing data, 
		following the given ratio (train/test)."""
	
    # set seed
    np.random.seed(seed)
	
    N = len(y)
    n_train = int(round(N*ratio))
    # Permute indices to pick them randomly
    shuffle_ind = np.random.permutation(N)
    
    x_train = x[shuffle_ind[:n_train]]
    y_train = y[shuffle_ind[:n_train]]
    
    x_test = x[shuffle_ind[n_train:]]
    y_test = y[shuffle_ind[n_train:]]
    
    return [x_train, y_train], [x_test, y_test]

def normalize_invalid_data(x):
    """" Set the -999 values equal to mean of their column, in order to 
         not affect the regression """
    mask = (x == -999)
    x[mask] = np.nan
    
    # Use mask.T to iterate over columns of the boolean matrix
    for i, bools in enumerate(mask.T):
        x[bools,i] = np.nanmean(x[:,i])
    
    return x

def build_poly(x, degrees, features):
    """ Performe polynomial feature augmentation.
        degrees: a list of the degree which I want
        features: list of features of which I want to make power"""
    
    # Always add ones features (grade 0)
    tx = np.column_stack((x, np.ones(x.shape[0])))
    
    # Remove 0 degree from the list because it is always added separately
    if 0 in degrees:
        degrees.remove(0)
        
    # Remove 1 degree from the list because they are already included
    if 1 in degrees:
        degrees.remove(1)
        
    for degree in degrees:
        if degree < 1 and degree > -1:
			# Take the root of the absolute value, 
			# in order to include negative numbers
            tx = np.column_stack([tx, np.abs(x[:, features])**degree])
        else:
            tx = np.column_stack([tx, x[:, features]**degree])
    
    return tx

def build_cross_features(x, features):
    """	Add cross products to train data
		x: starting data
		features: index of original features taken to calculate cross features"""
    
    # Separate selected features
    x_sel = x[:,features]
    
    for i in range(x_sel.shape[1]):
        for j in range(i+1, x_sel.shape[1]):
			# Calculate product for each pair if features
            prod = x_sel[:,i] * x_sel[:,j]
            # Add cross products among features
            x = np.column_stack((x, prod))
            # Add squared cross products
            x = np.column_stack((x, prod**2))
            # Add root of cross products
            x = np.column_stack((x, np.sqrt(np.abs(prod))))
			
    return x
   

################### CROSS VALIDATION ##########################################

def build_kfold_indices(N, k_fold, seed=1):
    """Randomly calculate indices of k_folds for cross_validation"""
	
    fold_size = int(N / k_fold)
    
    np.random.seed(seed)
    rand_indices = np.random.permutation(N)
    
    # Return a list of indices for each k-fold.
    # These indices will build test data and all the others train data
    k_indices = [rand_indices[k * fold_size : (k + 1) * fold_size]
                 for k in range(k_fold)]
				 
    return np.array(k_indices)

def cross_validation_split(y, x, k_indices, k):
    """	Split data for cross validation. Use k'th subgroup in test
		for testing and all others for training."""
    
    # Get k'th subgroup for testing
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    
    # Build iterator over folds excluding k value
    train_indices = [i for i in range(len(k_indices)) if i != k]
	
    # Use iterator to extract training data
    x = x[k_indices[train_indices].ravel()]
    y = y[k_indices[train_indices].ravel()]
    # ravel() needed because k_indices is a bidimensional array
    # and here I am selecting several rows, containing indices for x.
    # I need only one dimension for x because I am extracting rows (datapoints)
    
    return [x,y], [x_te, y_te]

def cross_validation(y_data, tx_data, lambda_, k_fold):
    """ Build folds for cross validation and iterate ridge regression
		over them, saving w_star and score obtained on test data. At the
		end, return the means of w_star and score and the standard deviation
		of scores. """
    
    N = len(y_data)
    # Retrive random incides for dividing data
    k_indices = build_kfold_indices(N, k_fold)
    
    ws_star = []
    scores_te = []
    scores_tr = []
    # Loop over number of k_folds
    for k in range(k_fold):
	
        # Split data for testing and training
        [tx, y], [tx_te, y_te] = cross_validation_split(y_data, tx_data, k_indices, k)     
        # Compute best weights (ridge)
        ws_star.append( ridge_regression(y, tx, lambda_) )
        # Compute scores
        scores_tr.append( compute_score(y, tx, ws_star[k]) )
        scores_te.append( compute_score(y_te, tx_te, ws_star[k]) )
    
    return np.mean(ws_star, axis=0), np.mean(scores_tr), np.mean(scores_te), np.std(scores_te)
	

##################### TEST ROUTINES ##########################################

def iterate_over_degrees(y_data, tx_data, degrees_test, features, lambdas, k_fold):
    """ Iterate cross validation adding a new degree each iteration. For each
        degree find the ridge regression with the best score over some values 
        of lambda. Print standard deviation of scores in order to check a 
        possible overfitting. At the end plot computed performances."""
    
    all_ws_star = []
    all_scores = []
    
    for degree in degrees_test:
        # Add degree to the data
        tx_data = build_poly(tx_data, [degree], features)
        
        # Look for the best lambda with the given degrees
        scores = []
        stds = []
        ws_star = []
        
        for l, lambda_ in enumerate(lambdas):
            w_star, _, score, std = cross_validation(y_data, tx_data, lambda_,k_fold)
            scores.append(score)
            ws_star.append(w_star)
            stds.append(std)
    
        # Find the iteration that maximized the score and return its results and parameters
        best = np.argmax(scores)
        w_star = ws_star[best]
        score = scores[best]
        lambda_ = lambdas[best]
        std = stds[best]
        
        all_scores.append(score)
        all_ws_star.append(w_star)
    
        print('Deg: {}, Score: {}, lambda:{} std:{}'.format(degree, score, lambda_, std))
	
    best = np.argmax(all_scores)
    score = all_scores[best]
    w_star = all_ws_star[best]
    degree = degrees_test[best]
    print('Max score: {} with deg {}'.format(score, degree))
    
    # Plot performances
    degree_performance_visualization(degrees_test, all_scores)
	
    return w_star, score

def iterate_over_lambdas(y_data, tx_data, lambdas, k_fold):
    """ Iterate cross validation test for some values of lambda and return 
        the one with the higher score on test folds. Plot train and test 
        scores over lambdas."""
        
    # Look for the lambda which gives the best score
    scores_tr = []
    scores_te = []
    stds = []
    ws_star = []
    
    for l, lambda_ in enumerate(lambdas):
        w_star, score_tr, score_te, std = cross_validation(y_data, tx_data, lambda_,k_fold)
        scores_tr.append(score_tr)
        scores_te.append(score_te)
        ws_star.append(w_star)
        stds.append(std)
        print('Iteration: {}'.format(l))
        
    # Find the iteration that maximized the score and return its results and parameters
    best = np.argmax(scores_te)
    w_star = ws_star[best]
    score = scores_te[best]
    lambda_ = lambdas[best]
    std = stds[best]
    print('Best score: {}, best lambda: {}, std: {}'.format(score,lambda_,std))
    
    # Plot test and train performances
    print('Preparing plot')
    cross_validation_visualization(lambdas, scores_tr, scores_te)
    
    return w_star, score
    

def iterate_on_splitted_data(y_data, x_data, ratio, degrees, features, lambdas):              ##check!!!!!!!!!!!!!!!
    
    [x,y], [x_te, y_te] = split_data(x_data, y_data, ratio)
    
    scores = []
    for d, degree in enumerate(degrees):
        for l, lambda_ in enumerate(lambdas):
            tx = build_poly(x, degree, features)
            tx_te = build_poly(x_te, degree, features)
            
            w_star = ridge_regression(y, tx, lambda_)
            score = compute_score(y_te, tx_te, w_star)
            scores.append(score)
    ind = np.argmin(scores)
    return scores[ind], np.unravel_index(ind, (len(degrees), len(lambdas)))
