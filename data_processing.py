import numpy as np

 
def cleanup_test_data(x,features_to_use, cleanup_method = 0,  normalize = False):
    """Filters out all coumns that are not in features to use, 
    and then replaces -999 with mean (median) depending on the cleanup_method:
        0: -999 is replaced with the mean (ignoring -999 values) 
        of its correspoding feature in 'x'.
        1: -999 is replaced with the median (ignoring -999 values) 
        of its correspoding feature in 'x'.
    """
    
    tx = x.copy()
    tx[np.where(tx == -999)] = np.nan
    
    if(normalize):
        tx,_,_ = standardize(tx)
        
    if(cleanup_method==0):
        tx = replace_nan_with_mean(tx)
    else:
        tx = replace_nan_with_median(tx)
        
    return tx[: , features_to_use]

def cleanup_train_data(y, x, cleanup_method = 0, normalize = False, filter_column = False, filter_ratio = 0.65):
    """Filters out all coumns that have -999 in at least `filter_ratio` number 
    of their entries, and then replaces -999 with appropriate values depending 
    on the cleanup_method). If the `cleanup_method` is set to:
    0 : -999 is replaced with the mean (ignoring -999 values) of its 
    correspoding feature in 'x'.
    1 : -999 is replaced with the median (ignoring -999 values) of its 
    correspoding feature in 'x'.
    2 : -999 in ith feature and jth data point is replaced with the mean
    (ignoring -999 values) of the ith feature values corresponding to the 
    background (signal) data points if the jth data point label is 'b' ('s').
    3 : -999 in ith feature and jth data point is replaced with the median of the 
    ith feature values corresponding to the background (signal) data points 
    if the jth data point label is 'b' ('s').
    """ 
    
    #Replace the -999 values with Nan
    tx = x.copy()
    tx[np.where(tx == -999)] = np.nan
  
    if(normalize):
        tx,_,_ = standardize(tx)
        
    #Filters out all coumns that have Nan in at least `filter_ratio` number of their entries    
    if(filter_column):
        indices_to_keep, tx = filter_features(tx, filter_ratio)
    else:
        num_of_features = tx.shape[1]
        indices_to_keep = np.arange(num_of_features)
       
    if(cleanup_method == 0):
        return replace_nan_with_mean(tx), indices_to_keep
    elif(cleanup_method == 1):
        return replace_nan_with_median(tx), indices_to_keep
    elif(cleanup_method == 2):
        indices_signal = np.array([i for i,j in enumerate(y) if j==1])
        indices_background = np.array([i for i,j in enumerate(y) if j==-1])
        if(indices_signal != np.array([])):
            tx[indices_signal, : ]=replace_nan_with_mean(tx[indices_signal, : ])
        if(indices_background != np.array([])):
            tx[indices_background, : ] = replace_nan_with_mean(tx[indices_background, : ])
        return tx, indices_to_keep
    else:    
        indices_signal = np.array([i for i,j in enumerate(y) if j==1])
        indices_background = np.array([i for i,j in enumerate(y) if j==-1])
        if(indices_signal != np.array([])):
            tx[indices_signal, : ]=replace_nan_with_median(tx[indices_signal, : ])
        if(indices_background != np.array([])):
            tx[indices_background, : ] = replace_nan_with_median(tx[indices_background, : ])
        return tx, indices_to_keep
    

def standardize(x):
    
    #Compute the mean of each column ignoring Nan.
    mean_x = np.nanmean(x, axis = 0)
    x = x - mean_x
    
    #Compute the standard deviation of each column ignoring Nan
    std_x = np.nanstd(x, axis = 0)
    
    # To avoid dividing by zero if the data in a certain column
    # doesn't change, we replace the 0 values with 1.
    modified_std_x = np.array([1 if s==0 else s for s in std_x])
    x = np.divide(x, modified_std_x) 
    
    return x, mean_x, std_x

def filter_features(x, threshold = 0.65):
    """Filters out columns that have Nan in at least 
    `threshold` number of their entries"""
    # Number of data points.
    len = x.shape[0]
    
    # Number of Nan values in each column.
    num_Nan = np.sum(np.isnan(x), axis = 0)
    
    #Indices of features to keep.
    indices_to_keep = np.array([i for i,x in enumerate(num_Nan) if x<threshold*len])
    filtered_features = x[: , indices_to_keep]
    
    return indices_to_keep, filtered_features

def replace_nan_with_mean(x):
    #Compute the mean of each column ignoring Nan.
    mean_x = np.nanmean(x, axis = 0)
    y = replace_nan_with_value(x, mean_x)
    return y

def replace_nan_with_median(x):
    #Compute the mean of each column ignoring Nan.
    median_x = np.nanmedian(x, axis = 0)
    y = replace_nan_with_value(x, median_x)
    return y

def replace_nan_with_value(x, val):
    y = np.where(~np.isnan(x), x, val)
    return y