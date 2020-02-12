from __future__ import division
import sys
import time 
import random
import numpy as np
from pareto import *
from function import *
from gpstuff import *
from weights import *
from copy import deepcopy
from pygmo import hypervolume
import pygmo as pg
import math

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s --->  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def Generate_bounded(x1,x2,y1,y2,num):
    first_ = np.random.uniform(low = x1, high = x2, size=(num,))
    second_ = np.random.uniform(low = y1, high = y2, size=(num,))
    return np.vstack((first_,second_)).T

def SampleNonDom(num,yPareto,bnd):
    tmp_ = np.zeros(shape = (num,OUTPUT_DIM))
    indx = set(list(range(num)))
    indx_ = []
    for i in range(num):
        tmp_[i,0] = random.uniform(bnd['min'][0],bnd['max'][0])
        tmp_[i,1] = random.uniform(bnd['min'][1],bnd['max'][1])
        for val in yPareto:
            if (tmp_[i,0] > val[0]) and (tmp_[i,1] > val[1]):
                indx_.append(i)
    return tmp_[list(indx - set(indx_))]

def Cfunc(x,t):
    Wgs = np.sort(np.random.dirichlet(np.ones(INPUT_DIM),size=1))[0]
    Csum = []
    for i in range(0,INPUT_DIM):
        Lambda = 1/(Wgs[i]*t+1)
        f_x_t = Lambda*np.exp(-(x[i]*Lambda))
        Csum.append(1-f_x_t)
    return np.product(np.array(Csum))

def MakeW():
    d_ = [1]*OUTPUT_DIM
    return np.sort(np.random.dirichlet((d_)))

def logit(val):
    return 1/(1+np.exp(-val))

def AQFunc(X,dataset,cnt):
    ################# TRAIN THE MODEL
    mod1,Kernels['ker0'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,0]).T,'ker0',40)
    mod2,Kernels['ker1'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,1]).T,'ker1',40)
    Kinv_0 = np.linalg.pinv(Kernels['ker0'].K(dataset.data,dataset.data))
    Kinv_1 = np.linalg.pinv(Kernels['ker1'].K(dataset.data,dataset.data))

    #################  FIND THE PARETO & CHECK VALIDITY
    start_time = time.time()
    yPareto = mPareto(dataset.outputs)
    xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
    if (len(xPareto)!=len(yPareto)):
        sys.exit("Abort! Size of X pareto is not same as Y pareto!")

    #################  LAUNCH THE LOOP
    indices_ = [random.randint(0, X.shape[1]-1) for p in range(0, MAX_POINTS)]
    optimizerX = X.T[indices_]
    x_log = []
    imp_log = []
    imp_log_reg = []
    Cost_ = []
    BETA = 0.125*np.log(2*cnt+1)
    wgts = MakeW()
    Total_HVI_diff = 0
    Total_HVI_diff_r = 0
    
    for i in (range(0,MAX_POINTS)):
        progress(i, len(optimizerX), status = "")
        x_log.append([optimizerX[i,:]])
        y1,Sigy1 = testModel(mod1,np.array([optimizerX[i,:]]))
        y2,Sigy2 = testModel(mod2,np.array([optimizerX[i,:]]))       
        yreg = function(np.array([optimizerX[i,:]]))
        Mu_= np.array([y1,y2])
        Sigma_ = np.sqrt(BETA)*np.array([Sigy1,Sigy2])
        Total_HVI_diff = np.min([wgts[0]*(Mu_+Sigma_)[0] , wgts[1]*(Mu_+Sigma_)[1]])
        Total_HVI_diff_r = np.min([wgts[0]*yreg[0,0] , wgts[1]*yreg[0,1]])
        Cost_.append(Cfunc(optimizerX[i,:],cnt))
        Total_HVI_diff = Total_HVI_diff * (1-Cost_[-1])
        imp_log.append(Total_HVI_diff)
        Total_HVI_diff_r = Total_HVI_diff_r * (1-Cost_[-1])
        imp_log_reg.append(Total_HVI_diff_r)
        
    indx = imp_log.index(max(imp_log))
    Best_x = x_log[indx][0]
    tmp_y = function(np.array([Best_x]))
    
    indx_reg = imp_log_reg.index(max(imp_log_reg))
    Best_x_reg = x_log[indx_reg][0]
    tmp_y_reg = function(np.array([Best_x_reg]))
    
    cprint("\nLaunching regret core successfully!","red")
    Best_y = tmp_y[0,0],tmp_y[0,1]

    return Best_x,Best_y,(np.min([wgts[0]*tmp_y_reg[0,0] , wgts[1]*tmp_y_reg[0,1]])*(1-Cost_[indx_reg])
                            -    np.min([wgts[0]*tmp_y[0,0] , wgts[1]*tmp_y[0,1]])*(1-Cost_[indx]))