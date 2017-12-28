# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import predict_labels

## Loss and score Functions

def compute_score(y, tx, w):
    """
        Compute percentage of well predicted labels.
        
            INPUT:
                y           - Labels vector
                tx          - Samples
                w           - Weights
                
            OUTPUT:
                score       - Percentage obtained
    """
    # Predict labels
    y_pred = predict_labels(w, tx)
    # Calculate the percentage of correct predictions
    score = np.sum(y_pred == y) / len(y)
    return score

def compute_loss(y, tx, w, mode='MSE'):
    """
        Compute the MSE or MAE or RMSE cost.
        
            INPUT:
                y           - Labels vector
                tx          - Samples
                w           - Weights
                
            OUTPUT:
                loss        - Loss of the selected mode
    """
    
    N = len(y)
    e = y - np.dot(tx, w)
    
    if mode == 'MAE':
        loss = np.sum(np.abs(e)) / N
    elif mode == 'RMSE':
        loss = np.sqrt((1/N) * np.dot(e.T, e))
    else:
        loss = 1/(2*N) * np.dot(e.T, e)
        
    return loss

def compute_loss_lr(y, tx, w):
    """
        Computes the loss using the negative log likelihood function. Used for 
        logistic regression
        
            INPUT:
                y           - Labels vector
                tx          - Samples
                w           - Weights
                
            OUTPUT:
                loss        - Loss computed
    """
    
    y_expected = tx.dot(w)
    log = np.empty(y.shape)
    
    # Create mask for approximation of big number
    mask = y_expected>20
    log[mask] = y_expected[mask]    # Approximation
    log[~mask] = np.log(1 + np.exp(y_expected[~mask]))
    return np.sum( log - y*y_expected )

def compute_loss_plr(y, tx, w, lambda_):
    """Computes the loss using the penalized negative log likelihood function, where `y` is an array of labels, 
    tx is an array of features and `w` is an array of the parameters of the linear model. """
    
    return compute_loss_lr(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))

## Gradient Functions

def compute_gradient_mse(y, tx, w):
    """
        Compute the gradient for the Gradient Descent method
        
            INPUT:
                y           - Predictions vector
                tx          - Samples
                w           - Weights
                
            OUTPUT:
                Return the gradient for the given input
    """
    error = y - tx.dot(w)
    num_of_samples = y.shape[0]
    
    return (-tx.T.dot(error))/num_of_samples

def compute_gradient_lr(y, tx, w):
    """
        Compute the gradient for the negative log likelihood loss function
        
            INPUT:
                y           - Predictions vector
                tx          - Samples
                w           - Weights
                
            OUTPUT:
                Return the gradient for the given input
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)

def compute_gradient_plr(y, tx, w, lambda_):
    """Computes the graident of the penalized negative log likelihood loss function"""
    return compute_gradient_lr(y, tx, w) + 2 * lambda_ * w

## Helper Functions

def sigmoid(z):
    """Apply sigmoid function on t. It is used as logistic function"""
    
    sigma = np.zeros(z.shape[0])
    # mask returns the indices of the element of z which are greater than 15
    mask = z>15
    # for the element of z with value greater than 15,
    # use an approximation for sigmoid: x->inf sigma(x) -> 1-e^(-x)
    sigma[mask] = np.ones(len(z[mask])) - np.exp(-z[mask])
    
    exp = np.exp(z[~mask])
    sigma[~mask] = exp/(np.ones(len(exp)) + exp)
    
    return sigma

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

## Regression Functions

def least_squares(y, tx):
    """
        Use the Least Square method to find the best weights
        
            INPUT:
                y           - Predictions
                tx          - Samples
                
            OUTPUT:
                w           - Best weights
                loss        - Minimum loss
    """
   
    #solves the linear equation Aw = b
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss = compute_loss(y,tx,w)
    return w, loss
                     
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
        Use the Gradient Descent method to find the best weights
        
            INPUT:
                y           - Predictions
                tx          - Samples
                initial_w   - Initial weights
                max_iters   - Maximum number of iterations
                gamma       - Step size
                
            OUTPUT:
                w           - Best weights
                loss        - Minimum loss
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Compute the gradient and the loss (See helpers.py for the functions)
        loss = compute_loss(y, tx, w)
        grad = compute_gradient_mse(y, tx, w)

        # Update w by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)

        if n_iter % 128 == 0:
            print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))
            last_loss = loss

            # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -8:
            break

    print("  Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))
    # Get the latest loss and weights
    return ws[-1], losses[-1]

    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
        Use the Stochastic Gradient Descent (batch size 1) method to find the best weights
        
            INPUT:
                y           - Predictions
                tx          - Samples
                initial_w   - Initial weights
                max_iters   - Maximum number of iterations
                gamma       - Step size
                
            OUTPUT:
                w           - Best weights
                loss        - Minimum loss
    """   
    w = initial_w
    for minibatch_y, minibatch_tx in  batch_iter(y, tx, 1, max_iters):
        gradient    = compute_gradient_mse(minibatch_y , minibatch_tx, w)
        w -= gamma*gradient
    loss =  compute_loss(y,tx,w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
        Use the Ridge Regression method to find the best weights
        
            INPUT:
                y           - Predictions
                tx          - Samples
				lambda_		- Coefficient of the norm
                
            OUTPUT:
                w_star           - Best weights
    """
    num_of_samples = tx.shape[0]
    num_of_features = tx.shape[1]
    
    if lambda_==0:
        # If lambda = 0 perform a least square regression
        w_star, loss = least_squares(y, tx)
    else:
		#solves the linear equation Aw = b
        b = tx.T.dot(y)
        A = tx.T.dot(tx) + 2*lambda_* num_of_samples * np.identity(num_of_features)
        w_star = np.linalg.solve(A, b)   
    
    return w_star

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        Use the Logistic Regression method to find the best weights
            
            INPUT:
                y           - Predictions
                tx          - Samples
                initial_w   - Initial weights
                max_iters   - Maximum number of iterations
                gamma       - Step size
                
            OUTPUT:
                w           - Best weights
                loss        - Minimum loss
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Gradient descent method
        loss = compute_loss_lr(y, tx, w)
        grad = compute_gradient_lr(y, tx, w)
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if n_iter % 128 == 0:
            print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))
            last_loss = loss

        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break

    print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))

    return ws[-1], losses[-1]
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    """
        Use the Logistic Regression method to find the best weights
        
            INPUT:
                y           - Predictions
                tx          - Samples
                initial_w   - Initial weights
                max_iters   - Maximum number of iterations
                gamma       - Step size
                
            OUTPUT:
                w           - Best weights
                loss        - Minimum loss
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Gradient descent method
        loss = compute_loss_plr(y, tx,w)
        gradient = compute_gradient_plr(y, tx, w, lambda_)
        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if n_iter % 128 == 0:
            print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))
            last_loss = loss

        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break

    print("Iter={it}, loss={ll}, diff={dff}".format(it=n_iter, ll=loss, dff=(loss - last_loss)))

    return ws[-1], losses[-1]

############ BACKUPS OF OLD CODE ###########
### (((TODO: CANCEL IT IF EVERYTHING IN IMPLEMENTATIONS.PY WORKS))))

def logistic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    grad_norms = 0
    past_loss = calculate_logistic_loss(y, tx, w)
    print("GD({bi}/{ti}): loss={l} grad={g}".format(
                bi=0, ti=max_iters, l=calculate_logistic_loss(y, tx, w), 
                g=np.linalg.norm(calculate_logistic_gradient(y, tx, w))))
    grad = calculate_logistic_gradient(y, tx, w)
    grad_norm = np.linalg.norm(grad)
    grad_norms = grad_norms + grad_norm
    for n_iter in range(max_iters):
        
        # Compute gradient and update w
        grad = calculate_logistic_gradient(y, tx, w)
        grad_norm = np.linalg.norm(grad)
        loss = calculate_logistic_loss(y, tx, w)
        if n_iter>0:
            loss_ratio = (past_loss-loss)/past_loss
#            grad_norms = grad_norms + grad_norm
            if loss_ratio<0:
                gamma = gamma*(0.5)
            elif loss_ratio < 0.01:
                gamma = gamma*(1.5)
#                grad_norms = max(0, grad_norms - grad_norm)
        w = w - gamma * grad / grad_norm
            
        
        # Compute loss
        past_loss = loss
        
        if (n_iter % 1) == 0:
            print("GD({bi}/{ti}): loss={l} gamma={g}".format(
                    bi=n_iter+1, ti=max_iters, l=loss, g=gamma))

    return w, loss

def simple_log_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    print("GD({bi}/{ti}): loss={l} grad={g}".format(
                bi=0, ti=max_iters, l=calculate_logistic_loss(y, tx, w), 
                g=np.linalg.norm(calculate_logistic_gradient(y, tx, w))))
    for n_iter in range(max_iters):
        
        # Compute gradient and update w
        grad = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * grad / np.sqrt(np.linalg.norm(grad))
        loss = calculate_logistic_loss(y, tx, w)
        
        if (n_iter % 1) == 0:
            print("GD({bi}/{ti}): loss={l} gamma={g}".format(
                    bi=n_iter+1, ti=max_iters, l=loss, g=gamma))

    return w

def stoch_log_gd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    n_iter = 0
    for yn, txn in batch_iter(y, tx, max_iters):
        
		# Compute gradient and update w
        w = w - gamma*calculate_stoch_logistic_grad(yn, txn, w)
		# Compute loss
        #loss = calculate_logistic_loss(y, tx, w);
		
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #        bi=n_iter, ti=max_iters - 1, l=loss))
        #n_iter = n_iter + 1

    return w

def compute_stoch_gradient(y, tx, w):
	np.random.choice(tx)

def stoch_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        
		# Compute gradient and update w
        w = w - gamma*compute_stoch_gradient(y, tx, w)
		# Compute loss
        loss = compute_loss(y, tx, w, mode='RMSE');
		
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
         #       bi=n_iter, ti=max_iters - 1, l=loss))

    return w