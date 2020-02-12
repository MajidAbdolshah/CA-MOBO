import numpy as np
from parameters import *

def mPareto(y):
    Px,Py = y.shape[0],y.shape[1]
    fPareto =  np.empty((0,Py))
    for i in range(0,Px):
        flag = np.zeros((Py))
        cflag = 1
        ind_sorted = np.argsort(y[i,:])
        for j in range(0,Px):
                if i!=j:
                    for k in range(0,Py):
                        if y[i,k] < y[j,k]:
                            flag[k] = 1
                    '''
                    print(flag)
                    print(np.sum(flag),Py)
                    print(cflag)
                    '''
                    if np.sum(flag) >= Py:
                        cflag = 0
                        break
                    else:
                        flag = np.zeros((Py))
        if cflag:
            fPareto = np.vstack((fPareto,y[i,:]))       
    return fPareto

def parY_X(y,ypar):

    size_y = y.shape[0]
    size_yp = ypar.shape[0]
    indx = []
    
    for i in range(size_yp):
        for j in range(size_y):
            if (y[j,0] == ypar[i,0] and y[j,1] == ypar[i,1]):
                indx.append(j)
    return indx

def findXpareto(xData,yData,yPareto):
    indx = parY_X(yData,yPareto)
    Xres = np.empty([INPUT_DIM,])
    for val in indx:
        Xres = np.vstack((Xres,xData[val,:]))
        

    return Xres[1:]


def samplePareto(par):
    infoDic = {}
    uPartsX = np.sort(par[:,0])
    uMapX = np.concatenate([[0],uPartsX,[REF[0]]])
    uPartsY = np.sort(par[:,1])
    uMapY = np.concatenate([[0],uPartsY,[REF[1]]])

    cells_ = np.empty((0,OUTPUT_DIM*2))
    Jparet_ = np.empty((0,OUTPUT_DIM*2))
    not_Jparet_ = np.empty((0,OUTPUT_DIM*2))
    for i in range(0,len(uMapX)-1):
        for j in range(0,len(uMapY)-1):

            pos_st = np.array([uMapX[i],uMapY[j]])
            pos_en = np.array([uMapX[i+1],uMapY[j+1]])

            pos_ = np.matrix(np.append(pos_st,pos_en,axis=0))
            mid_point = np.array([(pos_[0,0]+pos_[0,2])/2,
                                  (pos_[0,1]+pos_[0,3])/2])
            cells_ = np.vstack([cells_,pos_])
            infoDic[repr(pos_)] = ruPareto(mid_point,par)
            if ruPareto(mid_point,par):
                Jparet_ = np.vstack([Jparet_,pos_])
            else:
                not_Jparet_ = np.vstack([not_Jparet_,pos_])


    return cells_,Jparet_,not_Jparet_,infoDic


def ruPareto(x,par):
    for val in par:
        if(x[0]>val[0] and x[1]>val[1]):
            return False
    return True