from scipy.stats import norm
import GPy
import GPyOpt
import numpy as np
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
from scipy.stats import mvn
from datacomplex import print_fancy
from parameters import *


def ctKernel(no,dInput,*arg):
    print_fancy('Initializing Kernel',0.1)
    names_ = {}
    for i in range(no):
        text_ = "ker" + str(i)
        names_[text_] = GPy.kern.RBF(input_dim=dInput,variance=VARIANCE_,lengthscale=arg[i],ARD=False)#+GPy.kern.White(input_dim=dInput)
    return names_

def trainModel(X,Y,Ker,eVal):
    global Kernels
    model_ = GPy.models.GPRegression(X,Y,Kernels[Ker])
    #model_.optimize_restarts(num_restarts = eVal)
    #print('______________________________')
    #print(Kernels['ker0'].variance)
    #model_.optimize(max_f_eval = eVal)
    return model_,Kernels[Ker]

def testModel(model_,x):
    [mu_per,sig_per] = model_.predict(x,full_cov=1)
    return mu_per[0,0],sig_per[0,0]

def dvt_mu(xs,Data,Kernels,yReal,Ker_,Kinv):
    #len_matrix = -np.linalg.inv(np.identity(INPUT_DIM)*LEN_SCALE**2)
    len_matrix = -(np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    XsT = (Data - xs).T
    tmpI = np.dot(len_matrix,XsT)
    KxsX = Kernels[Ker_].K(xs,Data).T
    #tmpII = np.dot(np.linalg.pinv(Kernels[Ker_].K(Data,Data)),np.matrix(yReal).T)
    tmpII = np.dot(Kinv,np.matrix(yReal).T)
    tmpIII = np.multiply(KxsX,tmpII)
    res_ = np.dot(tmpI,tmpIII)
    return res_

def dvt_var(xs,Data,Kernels,Ker_,Kinv):
    sizeD = Data.shape[0]
    len_matrix = (np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    KXX_m1 = Kinv#np.linalg.pinv(Kernels[Ker_].K(Data,Data))
    KxsX = Kernels[Ker_].K(xs,Data)
    X_xs = (Data-xs).T

    Alis_ = 0
    for i in range(0,sizeD):
        for j in range(0,sizeD):
            tmpI = np.dot(np.dot(np.matrix(X_xs[:,i]).T,np.matrix(X_xs[:,j])),len_matrix**2)
            tmp1 = KXX_m1[i,j]*KxsX[0,i]*KxsX[0,j]
            Alis_ += tmp1*tmpI

    return (len_matrix + Alis_)


def improved_dvt_var(xs,Data,Kernels,Ker_,Kinv):
    len_matrix = (np.identity(INPUT_DIM)*(1/(LEN_SCALE**2)))
    first_ = len_matrix
    second_ = np.dot((Kernels[Ker_].K(xs,Data)*(Data-xs).T).T,len_matrix)
    third_ = Kinv
    
    result = first_ + np.dot(np.dot(second_.T,third_),second_)
    return result


Kernels = ctKernel(OUTPUT_DIM,INPUT_DIM,LEN_SCALE,LEN_SCALE)