# Higgs Boson Machine Learning Project

We use machine learning techniques to detect the precense of  the Higgs Boson particle.

## Getting started

* Data Sets:
    * The train and the test data sets should be named as follows: train.csv and test.csv, and should be placed in a folder named data.
    * The prediction file (submission.csv) will be created in the data folder.

* Files
     * The implementations of Least Squares, Ridge Regression, Gradient Descent, Stochastic Gradient Descent, Logistic Regression and Penalized Logistic Regression can be all found in implementations.py.
     * To obtian the best result we achieved on Kaggle: Run run.py and this will generate the prediction file. The data processing functions and the helper functions used in run.py  can be found in data_processing.py and run_helper.py, respectively.
     * For our chosen model: Use test.py to choose the lambda and degrees, to generate the score vs lambda  graph and score vs degrees graph.
     
     * Model Chosen in run.py attains a 82% prediction score. To achieve that we use Ridge Regression with:
          1. lambda_ = 3.162277660168379e-07
          2. We augment the data with cross terms of the primitive features, their squares, and their square root.
          3. We augment the data with powers of the primitive and the derived features. The degrees we chose are [1/2,1,2,3,4,5,6,7,8,9,10,11].
 

