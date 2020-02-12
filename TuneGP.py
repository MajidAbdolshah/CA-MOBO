from datacomplex import *
from pareto import *
from function import *
from gpstuff import *
from optacquisition import *
from parameters import *
import random

MAXSAMPLE = 100
def progress(count, total, status=''):
    bar_len = 10
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s --->  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

################# Load
'''
xData = np.loadtxt('Data/xData.txt', dtype=float)
xData = xData.reshape(len(xData),1)
yData = np.loadtxt('Data/yData.txt', dtype=float)
'''

################# Parameters
NUM = 800
Lscale = [0.001,10]
Variance = [0,10]
scale_ = var_ = np.array([])
TEST = 256
for i in range(0,NUM):
    scale_ = np.append(scale_,random.uniform(Lscale[0],Lscale[1]))
    var_ = np.append(var_,random.uniform(Variance[0],Variance[1]))

################# Generate
xData,yData = initvals_(bounds,1000,1)
dataset = DataComplex(xData,yData)
yPareto = mPareto(dataset.outputs)
xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
initial_pareto = yPareto
X,Y = initvals_(bounds,MAXSAMPLE,1)
xData_test,yData_test = initvals_(bounds,TEST,1)
Sum = []

################# GP
for i in range(NUM):
    progress(i, NUM)
    kernel = GPy.kern.RBF(input_dim=1,lengthscale=scale_[i],variance=var_[i])
    m1 = GPy.models.GPRegression(X, np.reshape(Y[:,0],(len(Y[:,0]),1)),kernel)
    m2 = GPy.models.GPRegression(X, np.reshape(Y[:,1],(len(Y[:,1]),1)),kernel)
    stemp = 0
    for j in range(TEST):
        y1,s1 = m1.predict(np.array([xData_test[j]]),kern=kernel)
        y2,s2 = m2.predict(np.array([xData_test[j]]),kern=kernel)
        stemp += abs(y1 - yData_test[j,0]) + abs(y2 - yData_test[j,1])
    Sum.extend(stemp)       
plt.plot(Sum)
print(scale_[Sum.index(min(Sum))],var_[Sum.index(min(Sum))])

