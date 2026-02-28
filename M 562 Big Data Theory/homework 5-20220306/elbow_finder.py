#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:29:34 2022

@author: vanmagnan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sympy 


# Define k-means algorithm

def kmeans(X,k,max_iterations=1000):
    from sklearn.metrics import pairwise_distances
    # initialize iteration counter
    it = 0
    repeat = True
    
    # number of datapoints
    m = X.shape[0] 
    
    # initialization
    means = X[np.random.choice(m,k,replace=False)] 
    dist = pairwise_distances(X,means)
    clusters = np.argmin(dist, axis=1)
 
    
    while repeat and it<max_iterations:
        
        # update means; # if a cluster has no data points associated with it, replace it with a random data point
        means = np.array([np.mean(X[clusters==i], axis=0) 
                          if np.sum(clusters==i)!=0
                          else  X[np.random.randint(m)]
                          for i in range(k)])
        
        # update clusters
        dist = pairwise_distances(X,means)
        new_clusters = np.argmin(dist, axis=1)
        
        # check if the new clusters are equal to the previous clusters
        if np.sum(clusters!=new_clusters)==0: 
            repeat = False
        clusters = new_clusters
                
        it += 1 # increment iteration counter by 1
    
    return clusters, means


# define inertia for given means/clusters 

def inertia(X,means,clusters,k):
    m = len(clusters)
    return np.sum([np.sum((X[clusters==i]-means[i])**2) for i in range(k)])/m



def k_optimizer(X,k_min=1,k_max=20):
    
   # generate k-value array
   
   k_vals = np.arange(k_min,k_max+1)
    
    #generate inertias array
    
   inertias = np.zeros(k_max+1-k_min)
   for i,k in enumerate(range(k_min,k_max=1)):
        clusters,means = kmeans(X,k)
        inertias[i] = inertia(X,means,clusters,k)
        
   # plot inertia values

   plt.plot(k_vals,inertias,label = 'Inertias')
   plt.legend()
   
   # fit curve of form 1/p(x) (with p(x) a degree 6
   # polynomial ) to inertia data
   
   m = len(k_vals)
   
   X = np.ones((m,7))
   for i in range(1,7):
       X[:,i] = k_vals**i
       
   
   theta = sp.linalg.solve(X,1/inertia)[0]
   
   # now define p(x) for sympy
   
   x = sympy.symbols('x')
   
   poly1 = theta[0]+theta[1]*x+theta[2]*x**2 + theta[3]*x**3 
   
   poly2 = theta[4]*x**4 + theta[5]*x**5 + theta[6]*x**6
   
   poly = poly1 +poly2
   
   func = 1/poly
   
   deriv = sympy.diff(func,x,2)
   
   
   equation = sympy.Eq(deriv,0)
   
   opt = sympy.solveset(equation, x)
   
   if opt>k_min or opt>k_max:
       print('no elbow between kmin and kmax' )
       return
   else:
       return opt
    