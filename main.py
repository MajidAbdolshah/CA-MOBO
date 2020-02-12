from __future__ import division
from datacomplex import *
from pareto import *
from function import *
from gpstuff import *
from optacquisition import *
from parameters import *
import os

if __name__ == '__main__':
    
    ################# BO Main Loop
    path = "data"
    CreatePath(path)
    while rpt_ < RPT:
        tmp_ = "figlet OptimiSation round " + str(rpt_)
        os.system(tmp_)
        Regret = [10**6]
        start_time = time.time()
        xData,yData = initvals_no_balls(bounds,INITIAL,1)
        dataset = DataComplex(xData,yData)
        yPareto = mPareto(dataset.outputs)
        xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
        X = initvals_(bounds,MAXSAMPLE,0)
        print("Data Initialized in %s seconds; OK!\n" % round(time.time() - start_time,5))
        ctn = 0
        while ctn < COUNTER:
            cprint("\n_____________ INFO ("+str(rpt_)+")"+"("+str(ctn)+")"+" ________________",'green')
            info(dataset.data,dataset.outputs,yPareto,xPareto)
            Best_x,Best_y,Best_r = AQFunc(X,dataset,ctn+1)       
            Regret.append(Best_r)
            
            if Best_x not in dataset.data:
                dataset.newData(Best_x)
                dataset.newOut(Best_y)
            else:
                print("Abort! Hint: Check the parameters")
    
            if ctn%1 == 0:
                cprint("\nSaving...","blue")
                yPareto = mPareto(dataset.outputs)
                xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
                A_ = path+'/xPareto'+str(ctn)+"-"+str(rpt_)+'.txt'
                B_ = path+'/yPareto'+str(ctn)+"-"+str(rpt_)+'.txt'
                np.savetxt(A_, xPareto, fmt='%.100f')
                np.savetxt(B_, yPareto, fmt='%.100f')
            ctn += 1
        Regret = np.array(Regret[1:])
        str_ = path+'/Regret_' + str(rpt_)
        np.savetxt(str_, Regret, fmt='%.100f')
        rpt_ += 1
    