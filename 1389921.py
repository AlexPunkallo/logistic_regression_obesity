
# Alessandro Gallo

import pandas as pd
import numpy as np
import csv

data=pd.read_csv("nhanes.csv", delimiter= ',')
data.set_index("ID", inplace=True)
data.drop("BMI", axis=1, inplace=True)
data.fillna(data.median(), inplace=True)

def normalizeFeatures(X):
    X_norm=X
    mu=np.mean(X)
    sigma=np.std(X)
    X_norm=(X_norm-mu)/sigma
    return X_norm

data_new=normalizeFeatures(data[data.columns[:-1]])
X=data_new.values
m=np.ones((len(data_new),1 ), dtype=int)
X=np.concatenate((m, X), axis=1)
y=data['OBESE'].tolist()
theta= np.ones(19, dtype = float)

#---------------------------------------------------------------------------------#

def sigmoid(x, theta):
    d = 1.0 + np.e ** (sum([-theta[i] * x[i] for i in range (len(theta))]))
    sigmoid_func = 1.0 / d
    return sigmoid_func

def compute_cost(theta,X,y): 
    numb_X = X.shape[0]
    J=0
    for i in range(numb_X):
        J+=y[i]*np.log(sigmoid(X[i], theta))+(1-y[i])*np.log(1-sigmoid(X[i],theta))
    return J/numb_X

def compute_grad(theta, X, y):
    numb_X=X.shape[0]
    grad=np.zeros(len(theta))
    for i in range(numb_X):
        grad+=(y[i]-sigmoid(X[i],theta))*X[i]
    return  grad/numb_X

#--------------------------------------------------------------------------------#

def gradient_descent(X,y,theta,alpha,max_iter,eps):
    iter=0
    converged=False
    print "\n#----------------------------------------------------#"
    while not converged:
        old_error=compute_cost(theta, X, y)
        theta=theta+alpha*(compute_grad(theta,X,y))
        new_error=compute_cost(theta,X,y)
        print "  Iteration n:", iter, " with an error of:", new_error
        if abs(old_error-new_error)<=eps:
            converged=True
	    print "#----------------------------------------------------#"
	    print "\n 			Converged.\n"
            print "			Theta:\n", theta
	    print ""
        else:
            old_error=new_error
        if iter==max_iter:
            converged=True
        iter+=1
    return theta

def obese_results(theta, X):
    m, n = X.shape
    r = np.zeros(shape=(m, 1))
    h = sigmoid(X, theta)
    for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            r[i, 0] = 1
        else:
            r[i, 0] = 0
    return r

#--------------------------------------------------------------------------------# 

# 1)
m, n = X.shape
it = np.ones(shape=(m, 19))
theta_fin=gradient_descent(X,y,theta,20,50,0.002)

#--------------------------------------------------------------------------------#

# 2)

r = obese_results(theta, X)
print "		   Predicted results of obese:\n", r

