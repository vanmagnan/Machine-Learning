# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#linear regression with gradient descent function

#INPUTS: feature matrix X, target y, learning rate lr,
# number of iterations epochs

#OUTPUTS: theta vector which stores coefficients of our fit, mse vector
# storing our mean squared error at each iteration

def linregression_GD(X, y, learning_rate, epochs, momentum=0,testing_data=None):
    
    # number of points, number of features
    m,n = X.shape
    
    # initialize vector theta
    theta = np.random.randn(n)
    
    #initialize mse vector
    mse = np.zeros(epochs)
    
    if testing_data:
            mse_test = np.zeros(epochs)
            X_test,y_test = testing_data
            
    # initialize direction
    d = np.zeros(n)
    
    
    # gradient descent iterations
    
    for epoch in range(epochs):
        gradient = 2*X.T.dot(X.dot(theta)-y)
        d = gradient + momentum*d # could combine this line and that above
        theta = theta - learning_rate*d
        mse[epoch] = np.mean((y-X.dot(theta))**2)
        if testing_data:
            mse_test[epoch] = np.mean((y_test - X_test.dot(theta))**2)
        
        
    if testing_data:
        return theta, mse, mse_test
    else:
        return theta,mse
    
    
# BUILD POLYNOMIAL FEATURES FUNCTION

#INPUTS:
    
#OUTPUTS:
    
    
def build_poly_features(X,degree=1):
    from itertools import combinations_with_replacement as combwr
    from itertools import chain
    
    # number of data points (rows), number of features (columns)
    
    try:
        m,n = X.shape # will not work when x is a vector
    except:
        m = len(X)
        n = 1
        X = X.reshape(m,1)
    
    # number of polynomial features
    
    combinations = chain.from_iterable(combwr(range(n), i) for i in range(degree + 1))
    n_poly = sum(1 for combination in combinations)
    
    # polynomial feature matrix
    combinations = chain.from_iterable(combwr(range(n), i) for i in range(degree + 1))
    X_poly = np.ones((m,n_poly))
    for column_index, combination in enumerate(combinations):
        X_poly[:,column_index] = np.product(X[:,combination],axis = 1)
        
    return X_poly






# LOGISTIC REGRESSION WITH GRADIENT DESCENT

def logregression_GD(X,y,learning_rate,epochs):
    # sigmoid function
    def sigmoid(t):
        return 1/(1+np.exp(-t))
    # cost function E(theta)
    def logregression_cost(X,y,theta):
        p = sigmoid(X.dot(theta))
        return -np.sum(y*np.log(p)+(1-p)*np.log(1-p))
    
    m,n = X.shape
    theta = np.random.randn(n)
    cost = np.zeros(epochs)
    
    for epoch in range(epochs):
        gradient = X.T.dot(sigmoid(X.dot(theta))-y)
        cost[epoch] = logregression_cost(X,y,theta)
        theta = theta - learning_rate*gradient
    
    return theta, cost
    
    
# PLOT LOGISTIC REGRESSION CLASSIFICATION REGIONS    
def plot_logregression_regions(X, y, theta, degree=0):
    import numpy as np
    import matplotlib.pyplot as plt
    def sigmoid(t):
        return 1/(1+np.exp(-t))
    
    from matplotlib.colors import ListedColormap
    # create a 500x500 meshgrid
    m_plot = 500
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, m_plot)
    x2 = np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5, m_plot)
    X1, X2 = np.meshgrid(x1, x2) 
    X_plot = np.c_[X1.ravel(), X2.ravel()]
    
    X_plot_poly = build_poly_features(X_plot,degree=degree)

    # evaluate the logistic regression model at each point of the mesh grid    
    y_plot = sigmoid(X_plot_poly.dot(theta))
        
    # class prediction    
    y_plot[y_plot>=0.5]=1
    y_plot[y_plot<0.5]=0
    y_plot = y_plot.reshape(X1.shape)

    custom_cmap = ListedColormap(['C0','C1'])
    # contour map
    plt.figure(figsize=(12,5))
    plt.contourf(X1, X2, y_plot, alpha=0.3, cmap=custom_cmap)
    
    
    plt.plot(X[y==0,0],X[y==0,1],'o',label = 'class '+str(0))
    plt.plot(X[y==1,0],X[y==1,1],'o',label = 'class '+str(1))
    plt.legend(fontsize=15)
    
    
    
    
def softmax(X,theta):
    import numpy as np
    P = np.exp(X.dot(theta))
    row_sum = np.sum(P,axis=1).reshape(-1,1)
    return P/row_sum
    
# cost function E(theta)
def softmax_cost(X,Y,theta):
    import numpy as np
    P = softmax(X,theta)
    return -np.sum(Y*np.log(P))

# confusion matrix

# C[i,j] is the number of class-i points classified as class-j

def confusion_matrix(y,y_pred,labels):
    import numpy as np
    C = np.zeros((len(labels),len(labels)))
    
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            C[i,j] = np.sum(y_pred[y==label_i]==label_j)
            
    return C

# gradient descent
def softmaxregression_GD(X,y,learning_rate,epochs):
    import numpy as np
    def softmax(X,theta):
        P = np.exp(X.dot(theta))
        row_sum = np.sum(P,axis=1).reshape(-1,1)
        return P/row_sum
    def softmax_cost(X,Y,theta):
        P = softmax(X,theta)
        return -np.sum(Y*np.log(P))
    
    def one_hot_encoding(v):
        cat = np.unique(v)
        dic_cat = {cat[i] : i for i in range(len(cat))}
        v_ord = [dic_cat[v[i]] for i in range(len(v))]
        V = np.zeros((len(v),len(cat)))
        V[np.arange(len(v)),v_ord]=1
        return V,cat
    
    # shape of X
    m,n = X.shape
    
    #one-hot encoding
    Y,labels = one_hot_encoding(y)
    
    # number of classes
    k = len(labels)
    
    #initialize theta vector
    theta = np.random.randn(n,k)
    
    # initialize cost vector
    cost = np.zeros(epochs)
    
    # gradient descent iterations
    for epoch in range(epochs):
        gradient = X.T.dot(softmax(X,theta)-Y)
        theta = theta - learning_rate*gradient
        cost[epoch] = softmax_cost(X,Y,theta)
    return theta, cost, labels

def plot_softmax_regions(X, y, theta, labels, degree=1):
    from matplotlib.colors import ListedColormap
    
    # softmax predictor function
    def predictor_softmax(X,theta,labels):
        P = softmax(X,theta)
        return labels[np.argmax(P,axis=1)]
    
    # create a 500x500 meshgrid
    m_plot = 500
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, m_plot)
    x2 = np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5, m_plot)
    X1, X2 = np.meshgrid(x1, x2) 
    X_plot = np.c_[X1.ravel(), X2.ravel()]
    
    # add polynomial features
    X_plot_poly = build_poly_features(X_plot,degree=degree)

    # evaluate the softmax regression model at each point of the mesh grid    
    y_plot = predictor_softmax(X_plot_poly,theta,labels).reshape(X1.shape)        


    # custom color map
    k = len(labels)
    custom_cmap = ListedColormap(['C'+str(i) for i in range(k)])
    
    # softmax classification regions
    plt.figure(figsize=(12,5))
    plt.contourf(X1, X2, y_plot, alpha=0.3, cmap=custom_cmap)
    
    # plot data points
    for label in labels:
        plt.scatter(X[y==label,0],X[y==label,1], label=label)
        
    plt.legend(fontsize=15)