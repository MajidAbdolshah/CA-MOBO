import numpy as np

def f(x,fNum):
    def firstfun(x):
        return x[:,0]
    def g_x(x):
        return 1 + ((9/29)*(x[:,1]+x[:,2]+x[:,3]+x[:,4]))
    def secondfun(x):
        return g_x(x)*(1-np.sqrt(x[:,0]/g_x(x))-(x[:,0]/g_x(x))*
                       np.sin(10*np.pi*x[:,0]))+1
    
    options = {0 : firstfun,
               1 : secondfun,}
    return options[fNum](x)

def function(Xp):
    tmp_1 = np.reshape(f(Xp,0),(len(f(Xp,0)),1))*-1 +1 
    tmp_2 = (np.reshape(f(Xp,1),(len(f(Xp,1)),1))*-1 + 2)/1.78
    res_ = np.hstack((tmp_1,tmp_2))
    return res_
