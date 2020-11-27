#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 4: Supervised learning
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab4-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The IRIS dataset
# 
# The Iris flower data set is a data set describing the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa_, _Iris virginica and _Iris versicolor_). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.
# 
# In your work, you will use the Iris dataset, considering only two of the three species of Iris.
# 
# ---
# 
# We start by loading the dataset. The Iris dataset is available directly as part of `scikit-learn`and we use the `scikit-learn` command `load_iris` to retrieve the data. For example, you can run the code
# 
# ```python
# import numpy as np
# import matplotlib.pyplot as plt
# 
# from sklearn import datasets as data
# 
# # Load dataset
# iris = data.load_iris()
# 
# data_X = iris.data[50:,:]          # Select only 2 classes
# data_A = 2 * iris.target[50:] - 3  # Set output to {-1, +1}
# 
# # Get dimensions 
# nE = data_X.shape[0]
# nF = data_X.shape[1]
# 
# print('Number of examples:', nE)
# print('Number of features per example:', nE)
# ```
# 
# to get the output
# 
# ```
# Number of examples: 100
# Number of features per example: 4
# ```
# 
# In the code above, the dataset contains a total of `nE` examples (100); each example is described by `nF` features (4). The input data is stored in the numpy array `data_X`, while the output data is stored in `data_A`. 

# ---
# 
# #### Activity 1.        
# 
# Write a function named `preprocess` that takes as input two numpy arrays, `input_data` and `output_data`, with shapes, respectively, `(nE, nF)` and `(nE,)`. Your function should:
# 
# * Split the data into a "train set" and a "test set". The test set should correspond to 10% of the whole data. To this purpose, you can use the function `train_test_split` from the module `model_selection` of `scikit-learn`. For reproducibility, set `random_state` to some fixed value (e.g., 30).
# * Standardize the training data to have 0-mean and unitary standard deviation. You may find useful the `StandardScaler` from `sklearn.preprocessing`. You should also standardize the test data, but using the scaler fit to the training data.
# 
# The function should return, as output, a tuple of 5 elements, where
# 
# * The first element is a numpy array containing the (standardized) input data for the training set;
# * The second element is a numpy array containing the output data for the training set;
# * The third element is a numpy array containing the (standardizes) input data for the test set;
# * The fourth element is a numpy array containing the output data for the test set;
# * The fifth element is the scaler model.
# 
# Note that your function should work for *any* binary classification dataset, where `output_data` contains only the labels `-1` and `1`.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(input_data, output_data):
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=30)
    
    scaler = StandardScaler()
    #print(np.mean(X_train))
    #print(np.std(X_train))
    
    X_train = scaler.fit_transform(X_train)
    
    #print(np.mean(X_train))
    #print(np.std(X_train))
    
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler
    
# Load dataset
iris = data.load_iris()

features = 'Sepal length\n(normalized)', 'Sepal width\n(normalized)', 'Petal length\n(normalized)', 'Petal width\n(normalized)'

data_X = iris.data[50:,:]
data_A = 2 * iris.target[50:] - 3

train_x, train_a, test_x, test_a, _ = preprocess(data_X, data_A)

# Plot data
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

idx1 = np.where(train_a == -1)[0]
idx2 = np.where(train_a == 1)[0]

for ax in axes.flat:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

# Plot the data
for i, j in zip(*np.triu_indices_from(axes, k=1)):
    for x, y in [(i,j), (j,i)]:
        axes[x,y].plot(train_x[idx1, x], train_x[idx1, y], 'bx', label='Versicolor')
        axes[x,y].plot(train_x[idx2, x], train_x[idx2, y], 'ro', label='Virginica')

# Label the diagonal subplots
for i in range(4):
    axes[i, i].annotate(features[i], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

plt.tight_layout()


# As an example, you can execute the code
# 
# ```Python
# # Load dataset
# iris = data.load_iris()
# 
# features = 'Sepal length\n(normalized)', 'Sepal width\n(normalized)', 'Petal length\n(normalized)', 'Petal width\n(normalized)'
# 
# data_X = iris.data[50:,:]
# data_A = 2 * iris.target[50:] - 3
# 
# train_x, train_a, test_x, test_a, _ = preprocess(data_X, data_A)
# 
# # Plot data
# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
# fig.subplots_adjust(hspace=0.05, wspace=0.05)
# 
# idx1 = np.where(train_a == -1)[0]
# idx2 = np.where(train_a == 1)[0]
# 
# for ax in axes.flat:
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
# 
# # Plot the data
# for i, j in zip(*np.triu_indices_from(axes, k=1)):
#     for x, y in [(i,j), (j,i)]:
#         axes[x,y].plot(train_x[idx1, x], train_x[idx1, y], 'bx', label='Versicolor')
#         axes[x,y].plot(train_x[idx2, x], train_x[idx2, y], 'ro', label='Virginica')
# 
# # Label the diagonal subplots
# for i in range(4):
#     axes[i, i].annotate(features[i], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
# 
# plt.tight_layout()
# ```
# 
# To get as output the plot:
# 
# <img src="feature-matrix.png" align="left">

# In the example above, the samples are represented by four features that have been normalized to have zero mean and unit standard deviation. However, as can be seen in the plots above, the training examples do not cover the full feature space. In fact, the plots suggest a strong correlation between some of the features, meaning that there may be some redundance in the four features, and a more compact representation may be possible. 
# 
# As such, in the next activity, you will perform an additional pre-processing step to reduce the dimensionality of the data from four to two dimensions, while maintaining most of the information. Specifically, you will transform your data so that each sample is now represented by two artificial features that capture most of the variability in the data, using a technique called [_principal component analysis_]( https://en.wikipedia.org/wiki/Principal_component_analysis).

# ---
# 
# #### Activity 2.
# 
# Write a function called `get_components` that receives, as input, standardized input data and performs principal component analysis on this data. Your function takes as input two numpy arrays, `train_x` and `test_x`, corresponding to the standardized training and test data obtained from the application of the function `preprocess` from **Activity 1** to a binary classification dataset, and an integer `n_comp`. Your function should
# 
# * Apply PCA to the training data (in `train_x`) to obtain a representation of the data with only `n_comp` features. You may find useful the function `PCA` from `sklearn.decomposition`. You should also apply PCA decomposition to the test data (in `test_x`), but using the analyzer fit to the training data.
# 
# The function should return, as output, a tuple of 3 elements, where
# 
# * The first element is a numpy array containing the transformed training set;
# * The second element is a numpy array containing the transformed test set;
# * The third element is the analyzer model.
# 
# Note that your function should work for *any* binary classification dataset.  
# 
# ---

# In[2]:


from sklearn.decomposition import PCA

def get_components(train_x, test_x, n_comp):
    pca = PCA(n_components=n_comp)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    
    return train_x,test_x,pca

pca_train_x, pca_test_x, pca = get_components(train_x, test_x, 2)

# Plot data
plt.figure()

plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')

plt.tight_layout()


# As an example, you can run the following code on the data from the previous example.
# 
# ```python
# pca_train_x, pca_test_x, pca = get_components(train_x, test_x, 2)
# 
# # Plot data
# plt.figure()
# 
# plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
# plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')
# 
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(loc='best')
# 
# plt.tight_layout()
# ```
# 
# Output:
# 
# <img src="pca.png" align="left">

# As you may observe, the data now covers the feature space much more evenly, and the correlation between features is much less evident. In the continuation, you will use the gradient derivations from the homework to train a simple logistic regression model to classify the two types of flowers, "Versicolor" and "Virginica".

# ---
# 
# #### Activity 3.
# 
# Write a function named `my_train_lr` that trains a logistic regression classifier in Python using Newton-Raphson's method. The method is described by the update:
# 
# $$\mathbf{w}^{(k+1)}\leftarrow\mathbf{w}^{(k)}-\mathbf{H}^{-1}\mathbf{g},$$
# 
# where $\mathbf{H}$ and $\mathbf{g}$ are the _Hessian matrix_ and _gradient vector_ that you computed in your homework. To train the classifier, initialize the parameter vector, $\mathbf{w}$, to zeros; then, run a cycle that repeatedly updates the parameter vector according to the rule above until the difference between two iterations is sufficiently small (e.g., smaller than $10^{-5}$).
# 
# Your function should receive, as input, two numpy arrays:
# 
# * A numpy array `train_x`, corresponding to the input training data, after being standardized and principal components have been extracted. Your array should have a shape `(nE, nF)`, where `nE` is the number of examples in the training set, and `nF` is the number of features after PCA.
# * A numpy array `train_a`, corresponding to the output training data, i.e., the labels corresponding to the examples in `train_x`. The array should have a shape `(nE,)` and all its elements should be either `1` or `-1`.
# 
# Make sure to add to your input an additional feature corresponding to the bias term, i.e., an all-ones feature. 
# 
# Your function should return a numpy array `w` of shape `(nF + 1, 1)` corresponding to the learned logistic regression parameters. Parameters `w[0:nF, 0]` should correspond to the weights of the features, while `w[nF]` should correspond to the bias term.
# 
# Your function should work for *any* binary classification dataset.  
# 
# **Note:** You may find useful to define a function `lr` that takes as inputs a sample `x`, an action `a` and a parameter vector `w` and computes
# 
# $$\mathtt{lr}(\mathbf{x},a,\mathbf{w})=\frac{1}{1+\exp(-a~\mathbf{w}^\top\mathbf{x})}.$$
# 
# ---

# In[3]:


def lr(x,a,w):    
    a = a.reshape((a.shape[0], 1))     
    return 1/(1+np.exp(-a * (w.T @ x.T).T))

def my_train_lr(train_x, train_a):
    #print(train_x.shape)
    
    # Add bias :)
    col_of_ones = np.ones((train_x.shape[0],1))
    train_x = np.append(train_x, col_of_ones , axis=1)

    K = train_x.shape[1]
    N = train_x.shape[0]
    
    w = np.zeros((K, 1))    
    
    train_a = train_a.reshape((train_a.shape[0],1))

    while True:
        pi = lr(train_x,train_a,w)
        g = 1/N * ((train_a * train_x).T @ (pi - 1))

        H = 1/N * (((train_x @ train_x.T).T @ pi).T @ (1-pi) )
              
        new_W = w - np.linalg.inv(H) * g
        if(abs(np.linalg.norm(w-new_W)) < 1e-12):
            break
        w = new_W
              
    return w
    
    

# Train model
w = my_train_lr(pca_train_x, train_a)

# Print learned parameters
print('Learned vector of parameters:')
print(w)

idx1 = np.where(train_a == -1)[0]
idx2 = np.where(train_a == 1)[0]

plt.figure()

plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')

# Plot decision boundary
x_ax = np.arange(pca_train_x[:, 0].min(), pca_train_x[:, 0].max(), .01)
a_ax = - (w[0, 0] * x_ax + w[2, 0]) / w[1, 0]
plt.plot(x_ax, a_ax, 'k--')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis([pca_train_x[:, 0].min() - 0.1, 
          pca_train_x[:, 0].max() + 0.1, 
          pca_train_x[:, 1].min() - 0.1, 
          pca_train_x[:, 1].max() + 0.1])

plt.legend(loc='best')

plt.tight_layout()


# As an example, you can run the following code to plot the decision boundary learned by your function.
# 
# ```python
# # Train model
# w = my_train_lr(pca_train_x, train_a)
# 
# # Print learned parameters
# print('Learned vector of parameters:')
# print(w)
# 
# idx1 = np.where(train_a == -1)[0]
# idx2 = np.where(train_a == 1)[0]
# 
# plt.figure()
# 
# plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
# plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')
# 
# # Plot decision boundary
# x_ax = np.arange(pca_train_x[:, 0].min(), pca_train_x[:, 0].max(), .01)
# a_ax = - (w[0, 0] * x_ax + w[2, 0]) / w[1, 0]
# plt.plot(x_ax, a_ax, 'k--')
# 
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.axis([pca_train_x[:, 0].min() - 0.1, 
#           pca_train_x[:, 0].max() + 0.1, 
#           pca_train_x[:, 1].min() - 0.1, 
#           pca_train_x[:, 1].max() + 0.1])
# 
# plt.legend(loc='best')
# 
# plt.tight_layout()
# ```
# 
# As output, you should get something like:
# ````
# Learned vector of parameters:
# [[ 3.14501957]
#  [ 3.39615668]
#  [-0.13782688]]
# ```
# 
# <img src="lr.png" align="left">

# As you can see, the classifier does a fairly good job in separating the two classes. 
# 
# You will now compare the classifier from **Activity 3** with a logistic regression classifier implemented in `sci-kit learn`.

# ---
# 
# #### Activity 4
# 
# Write a function `skl_train_lr` that trains a logistic regression classifier in Python using Newton-Raphson's method, but now using the implementation in `scikit-learn`. You can import the logistic regression model from `sklearn.linear_model` under the name `LogisticRegression`. 
# 
# Your function should receive, as input:
# 
# * A numpy array `train_x`, corresponding to the input training data, after being standardized and principal components have been extracted. Your array should have a shape `(nE, nF)`, where `nE` is the number of examples in the training set, and `nF` is the number of features after PCA.
# * A numpy array `train_y`, corresponding to the output training data, i.e., the labels corresponding to the examples in `train_x`. The array should have a shape `(nE,)` and all its elements should be either `1` or `-1`.
# * A numerical constant, `C`.
# 
# Fit the logistic regression model to the provided data using the `newton-cg` solver and a regularization coefficient `C`.
# 
# Your function should return a numpy array `w` of shape `(nF + 1, 1)` corresponding to the learned logistic regression parameters. Parameters `w[0:nF]` should correspond to the weights of the features, which you can access as the attribute `coef_` of the LR model, after training; `w[nF]` should correspond to the bias term, which you can access as the attribute `intercept_` of the LR model, after training. Your function should work for *any* binary classification dataset.  
# 
# Compare the parameter vector and decision boundary obtained with those from Activity 3 on the IRIS data.
# 
# ---

# In[4]:


from sklearn.linear_model import LogisticRegression

def skl_train_lr(train_x, train_y, C):
    
    clf = LogisticRegression(C=C, solver="newton-cg").fit(train_x, train_y)
    
    w = np.append(clf.coef_[0], clf.intercept_, axis=0)
    w = w.reshape((w.shape[0],1))
    return w

# Train model
walt = skl_train_lr(pca_train_x, train_a, 1e40)

# Print learned parameters
print('Learned vector of parameters (my_train_lr):')
print(w)

print('Learned vector of parameters (skl_train_lr):')
print(walt)

plt.figure()

plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')

# Plot custom decision boundary
x_ax = np.arange(pca_train_x[:, 0].min(), pca_train_x[:, 0].max(), .01)
a_ax = - (w[0, 0] * x_ax + w[2, 0]) / w[1, 0]
plt.plot(x_ax, a_ax, 'k--', label='Custom LR')

# Plot Scikit-learn decision boundary
a_alt = - (walt[0, 0] * x_ax + walt[2, 0]) / walt[1, 0]
plt.plot(x_ax, a_alt, 'g:', label='Scikit-learn LR')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis([pca_train_x[:, 0].min() - 0.1, 
          pca_train_x[:, 0].max() + 0.1, 
          pca_train_x[:, 1].min() - 0.1, 
          pca_train_x[:, 1].max() + 0.1])

plt.legend(loc='best')

plt.tight_layout()


# <span style="color:blue">We immediately notice that the values obtained are vehemently similar and the decision boundaries overlap.
#     The parameter vector from Activity 3 was computed using the first and second partial derivatives in order of w. In other words, the NLL expression was derived twice and, using those partial derivatives, we computed the updated weights until the difference between two subsequent sets of parameters was minimal.
#     As for the sklearn approach, the Newton-CG solver consists of a different Gradient Descent, in the sense that it makes use of the Hessian matrix to update the model's parameters. The Hessian matrix, as we've learned in class, is the second partial derivative of the NLL expression.
#     Hence, we can see why the results obtained by the "explicit" implementation of Logistic Regression and the ones obtained through sklearn's are so similar as the underlying logic is the same.
# </span>

# As an example, you can run the following code to plot the two decision boundaries, from **Activity 3** and **Activity 4**.
# 
# ```python
# # Train model
# walt = skl_train_lr(pca_train_x, train_a, 1e40)
# 
# # Print learned parameters
# print('Learned vector of parameters (my_train_lr):')
# print(w)
# 
# print('Learned vector of parameters (skl_train_lr):')
# print(walt)
# 
# plt.figure()
# 
# plt.plot(pca_train_x[idx1, 0], pca_train_x[idx1, 1], 'bx', label='Versicolor')
# plt.plot(pca_train_x[idx2, 0], pca_train_x[idx2, 1], 'ro', label='Virginica')
# 
# # Plot custom decision boundary
# x_ax = np.arange(pca_train_x[:, 0].min(), pca_train_x[:, 0].max(), .01)
# a_ax = - (w[0, 0] * x_ax + w[2, 0]) / w[1, 0]
# plt.plot(x_ax, a_ax, 'k--', label='Custom LR')
# 
# # Plot Scikit-learn decision boundary
# a_alt = - (walt[0, 0] * x_ax + walt[2, 0]) / walt[1, 0]
# plt.plot(x_ax, a_alt, 'g:', label='Scikit-learn LR')
# 
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.axis([pca_train_x[:, 0].min() - 0.1, 
#           pca_train_x[:, 0].max() + 0.1, 
#           pca_train_x[:, 1].min() - 0.1, 
#           pca_train_x[:, 1].max() + 0.1])
# 
# plt.legend(loc='best')
# 
# plt.tight_layout()
# ```
# 
# As output, you should get something like:
# ```
# Learned vector of parameters (my_train_lr):
# [[ 3.14501957]
#  [ 3.39615668]
#  [-0.13782688]]
# Learned vector of parameters (skl_train_lr):
# [[ 3.14501955]
#  [ 3.39615666]
#  [-0.13782687]]
# ```
# 
# <img src="comparison.png" align="left">

# Although the learned classifiers seem to provide a reasonable classification of the two species of Iris flowers, we are only looking at the training data, so it is possible that the classifiers may actually be overfitting (i.e., they perform rather well on the training data, but would not perform so well in "real world" data). 
# 
# In the next activities, you will investigate whether overfitting occurs, by observing how the performance changes when limiting the representation power of the classifier. To do so, you will impose soft restrictions on the parameters that the classifier is allowed to use. You will then observe how the performance changes as you tighten those restrictions. To conduct such study, you first need to compute the performance of your LR classifier.

# ---
# 
# #### Activity 5
# 
# Write a function `nll` that computes the *negative log likelihood* of an LR classifier for a given dataset. Your function should receive, as input,
# 
# * A numpy array `x`, corresponding to the inputs where the classifier will be evaluated. The array should have a shape `(nE, nF)`, where `nE` is the number of examples in the set, and `nF` is the number of input features describing each example (bias feature not included).
# 
# * A numpy array `y`, corresponding to the respective outputs. The array should have a shape `(nE,)` and all its elements should be either `1` or `-1`.
# 
# * A numpy array `w`, such as those computed in **Activity 3** and **Activity 4**. The array should have a shape `(nF + 1, 1)` and corresponds to the parameters of your LR classifier.
# 
# Your function should return a numerical value corresponding to the average negative log likelihood of the data according to the classifier with parameters `w`. Recall that the average negative log likelihood of the data is given by
# 
# $$\hat{L}_N(\mathbf{w})=-\frac{1}{N}\sum_{n=1}^N\log\pi_{\mathbf{w}}(a_n\mid x_n)$$,
# 
# where $\pi_{\mathbf{w}}(a\mid x)$ is the probability of $a\in\{-1,1\}$ given input $x$.
# 
# **Note:** You main find useful the function `lr` suggested in **Activity 3**.
# 
# ---

# In[5]:


def nll(x,y,w):
    # Add bias :)
    col_of_ones = np.ones((x.shape[0],1))
    x = np.append(x, col_of_ones , axis=1)
    
    N = x.shape[0]
    
    value = -1/N * np.sum((np.log(lr(x,y,w))), axis=0)
    return value
    
COEFFS = [1e4,  9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3,
          1000, 900, 800, 700, 600, 500, 400, 300, 200, 
          100,  90,  80,  70,  60,  50,  40,  30,  20, 
          10,   9,   8,   7,   6,   5,   4,   3,   2, 
          1,    .9,  .8,  .7,  .6,  .5,  .4,  .3,  .2,  
          .1,   .09, .08, .07, .06, .05, .04, .03, .02, .01]

RUNS = 100

# Investigate overfitting
err_train = np.zeros((len(COEFFS), 1)) # Error in training set
err_valid = np.zeros((len(COEFFS), 1)) # Error in validation set

np.random.seed(45)

print('Completed: ', end='\r')

for run in range(RUNS):

    # Split training data in "train" and "validation"
    x_train, x_valid, a_train, a_valid = train_test_split(pca_train_x, train_a, test_size=0.15)

    for n in range(len(COEFFS)):

        # Print progress
        print('Completed: %i %%' % int((run * len(COEFFS) + n) / (RUNS * len(COEFFS)) * 100), end='\r')

        # Train classifier
        w = skl_train_lr(x_train, a_train, COEFFS[n])

        # Compute train and test loss
        nll_train = nll(x_train, a_train, w)
        nll_valid = nll(x_valid, a_valid, w)

        err_train[n] += nll_train / RUNS
        err_valid[n] += nll_valid / RUNS

print('Completed: 100%')

plt.figure()
plt.semilogx(COEFFS, err_train, 'k:', linewidth=3, label='Training')
plt.semilogx(COEFFS, err_valid, 'k-', linewidth=3, label='Test')
plt.xlabel('Regularization parameter')
plt.ylabel('Negative log-likelihood')
plt.legend(loc = 'best')

plt.tight_layout()


# You can now use your function to run the following interaction.
# 
# ```python
# COEFFS = [1e4,  9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3,
#           1000, 900, 800, 700, 600, 500, 400, 300, 200, 
#           100,  90,  80,  70,  60,  50,  40,  30,  20, 
#           10,   9,   8,   7,   6,   5,   4,   3,   2, 
#           1,    .9,  .8,  .7,  .6,  .5,  .4,  .3,  .2,  
#           .1,   .09, .08, .07, .06, .05, .04, .03, .02, .01]
# 
# RUNS = 100
# 
# # Investigate overfitting
# err_train = np.zeros((len(COEFFS), 1)) # Error in training set
# err_valid = np.zeros((len(COEFFS), 1)) # Error in validation set
# 
# np.random.seed(45)
# 
# print('Completed: ', end='\r')
# 
# for run in range(RUNS):
#     
#     # Split training data in "train" and "validation"
#     x_train, x_valid, a_train, a_valid = train_test_split(pca_train_x, train_a, test_size=0.15)
#     
#     for n in range(len(COEFFS)):
#         
#         # Print progress
#         print('Completed: %i %%' % int((run * len(COEFFS) + n) / (RUNS * len(COEFFS)) * 100), end='\r')
# 
#         # Train classifier
#         w = skl_train_lr(x_train, a_train, COEFFS[n])
# 
#         # Compute train and test loss
#         nll_train = nll(x_train, a_train, w)
#         nll_valid = nll(x_valid, a_valid, w)
# 
#         err_train[n] += nll_train / RUNS
#         err_valid[n] += nll_valid / RUNS
#         
# print('Completed: 100%')
# 
# plt.figure()
# plt.semilogx(COEFFS, err_train, 'k:', linewidth=3, label='Training')
# plt.semilogx(COEFFS, err_valid, 'k-', linewidth=3, label='Test')
# plt.xlabel('Regularization parameter')
# plt.ylabel('Negative log-likelihood')
# plt.legend(loc = 'best')
# 
# plt.tight_layout()
# ```
# 
# and observe the following output:
# 
# ```
# Completed: 100%
# ```
# 
# <img src="overfitting.png" align="left">

# ---
# 
# #### Activity 6.
# 
# Looking at the results in the example from **Activity 5**, indicate whether you observe overfitting or not, and explain your conclusion. Based on this conclusion, select a value for the parameter `C` that you find is more adequate for the Iris data, and explain your selection. Then, for your selected value of `C`,
# 
# * Train a logistic regression model using the whole training data, using the function from **Activity 4**.
# * Compute the negative log likelihood of the resulting classifier both in the training data and the test data.
# 
# Do the results match what you expected? Explain.
# 
# ---

# In[6]:


C = 0.9

# Split training data in "train" and "validation"
x_train, x_valid, a_train, a_valid = train_test_split(pca_train_x, train_a, test_size=0.15)

# Train classifier
w = skl_train_lr(x_train, a_train, C)

# Compute train and test loss
nll_train = nll(x_train, a_train, w)
nll_valid = nll(x_valid, a_valid, w)

print("Negative Log Likelihood (TRAIN):",nll_train)
print("Negative Log Likelihood (CV):",nll_valid)


# <span style="color:blue">We can detect overfitting by looking at the way the Negative Log Likelihood changes in function of the regularization parameter comparatively between our training and test sets. As we can see in the graph, after a certain value of C (~10^0) the NLL value for our train set continues diminishing, whilst for our test set it starts increasing quite noticeably. As such, we can confidently say that after that C value, the model overfits the training data, explaining the graph's appearance.</span>
