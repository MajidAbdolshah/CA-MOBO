from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from matplotlib import rc
from parameters import *
from weights import *
from pareto import *
from datacomplex import *
import os  
from matplotlib.patches import Rectangle
import itertools
from pygmo import hypervolume
import pygmo as pg
import matplotlib

marker = itertools.cycle(('x', '+', '>', 'o', '*')) 
Colors = itertools.cycle(('#FF8000', '#00FF40', '#00BFFF','#8000FF','#0A0A0A')) 
Fcolors = itertools.cycle(('#F5D0A9', '#A9F5BC', '#A9D0F5','#BCA9F5','#B2B2B2'))
Colors_ = itertools.cycle(('#00FF40', '#00BFFF','#8000FF','#0A0A0A')) 
Fcolors_ = itertools.cycle(('#A9F5BC', '#A9D0F5','#BCA9F5','#B2B2B2'))
linestyles = [':','-', '--', '-.', ':']
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size'   : 20})
rc('text', usetex=True)

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s --->  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def main():    
    ################## initialise
    input1 = int(sys.argv[1])
    input2 = int(sys.argv[2])
    input3 = int(sys.argv[3])

    logX = np.zeros(shape=(INPUT_DIM))
    Myregret = np.empty(shape=(input1))
    hvi_ = []
    path = os.getcwd()  
    print ("The current working directory is %s" % path) 
    path = str(input1)
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
        
    all_inputs = (r'$x_1$', r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$')
    y_pos = np.arange(len(all_inputs))
    Main_Matrix_HV = np.zeros(shape=(input2,input1))
    Main_Matrix_CST = {}

    ################## Load
    for cc in range(0,input2):
        progress(cc, input2, status = "Creating the plots...")
        total_cost = [0]*INPUT_DIM
        logX = np.zeros(shape=(INPUT_DIM))
        str_3 = "data/Regret_"+str(cc)
        tmp_reg = np.loadtxt(str_3)
        tmp_reg = tmp_reg[:input1]
        Myregret = np.vstack((Myregret,np.array([tmp_reg])))
        for c_ in range(0,input1):
            str_1 = "data/yPareto"+str(c_)+"-"+str(cc)+".txt"
            str_2 = "data/xPareto"+str(c_)+"-"+str(cc)+".txt"
            yPareto = np.loadtxt(str_1)
            xPareto = np.loadtxt(str_2)
            
            if (yPareto.ndim == 1):
                xPareto = xPareto.reshape((1,INPUT_DIM))
                yPareto = yPareto.reshape((1,OUTPUT_DIM))    
            
            ind1 = yPareto[:,0] > 1
            ind2 = yPareto[:,1] > 1
            yPareto[ind1,0] = 1 - EPS
            yPareto[ind2,1] = 1 - EPS
            ind1 = yPareto[:,0] < 0
            ind2 = yPareto[:,1] < 0
            yPareto[ind1,0] = 0 + EPS
            yPareto[ind2,1] = 0 + EPS

            try:
                hv = hypervolume(yPareto*-1)
                Main_Matrix_HV[cc,c_] = hv.compute(REF)
            except:
                print("Exception = 0") 
                #Main_Matrix_HV[cc,c_] = np.max(Main_Matrix_HV[cc,:])
            for j in range(0,INPUT_DIM):
                total_cost[j] += xPareto[-1][j]
            logX = np.vstack((logX,np.array(total_cost)))
            if input3:
                plt.bar(y_pos, total_cost, align='center', alpha=0.5)
                plt.xticks(y_pos, all_inputs)
                plt.ylabel(r'$Usage$')
                plt.title(r'$Resources '+str(c_)+' $')
                str_ = path + '/Resources-' + str(c_) + '.png'
                plt.savefig(str_, format='png', dpi=300)
                plt.show()

        Main_Matrix_CST[cc] = logX

    
    ################## PREPARE THE NUMS
    HVI_mean = []
    HVI_var = []
    Cost_Proc = {}
    
    Myregret= Myregret[1:,:]
    
    Myregret_ = np.empty(shape=(input1))
    for i in range(len(Myregret)):
        Myregret_ = np.vstack((Myregret_,np.cumsum(Myregret[i,:])))   
    Myregret_ = Myregret_[1:,:]
    
    #Myregret_ = Myregret
    Regret_mean = Myregret_.mean(axis=0)
    Regret_var = Myregret_.std(axis=0)
    
    Main_Matrix_HV = (Main_Matrix_HV/MAX_HVI)*100
    for i in range(input1):
        HVI_mean.append(np.mean(Main_Matrix_HV[:,i]))
        HVI_var.append(np.std(Main_Matrix_HV[:,i]))
    HVI_mean = np.array(HVI_mean)
    HVI_var = np.array(HVI_var)
    np.savetxt('HVI_mean.txt',HVI_mean)
    np.savetxt('HVI_var.txt',HVI_var)
    for i in range(0,INPUT_DIM):
        tmp = np.empty((0,(input1+1)), float)
        for j in range(0,input2):
            tmp = np.vstack((tmp,Main_Matrix_CST[j][:,i]))

        Cost_Proc[i] = np.empty((0,(input1+1)), float)
        Cost_Proc[i] = np.vstack((Cost_Proc[i],tmp[:,].mean(axis=0)))
        Cost_Proc[i] = np.vstack((Cost_Proc[i],tmp[:,].std(axis=0)))
        
        

    ################## PLOT PARETO
    #yPareto_Main = np.loadtxt('yPareto.txt')
    plt.figure()
    #plt.plot(yPareto_Main[:,0],yPareto_Main[:,1],'.c',label=r'$Main\ Pareto\ Points$')
    plt.scatter(yPareto[:,0],yPareto[:,1],c='#030b52',s = 50,label=r'$\mathrm{CA-MOBO\ Pareto\ Frontier}$') 
    str_ = path + '/Me-Pareto-' + str(input1) + '.pdf'
    plt.xlabel(r'$f_1$',fontsize=25)
    plt.ylabel(r'$f_2$',fontsize=25)
    plt.xlim([0,1.])
    plt.ylim([0,1.])
    #plt.xlim([ybounds['min'][0],ybounds['max'][0]])
    #plt.ylim([ybounds['min'][1],ybounds['max'][1]])
    plt.legend(prop={'size': 17})
    plt.savefig(str_, format='pdf', dpi=300,bbox_inches = 'tight')
    plt.show()
    
    ################## PLOT USAGE
    fig, axs = plt.subplots()
    plt.figure()
    for i in range(0,INPUT_DIM):
        plt.plot(np.arange(input1+1)+1,Cost_Proc[i][0,:],label=all_inputs[i], linestyle = linestyles[i],color=next(Colors))
        plt.fill_between(np.arange(input1+1)+1, Cost_Proc[i][0,:]-Cost_Proc[i][1,:], Cost_Proc[i][0,:]+Cost_Proc[i][1,:],
                         alpha=((i/10)+0.2),facecolor = next(Fcolors)) 
    str_ = path + '/Me-Resources--' + str(input1) + '.pdf'
    plt.xlabel(r'$t$',fontsize=25)
    plt.ylabel(r'$\sum_{i=1}^{t}\ \ x_i$',fontsize=25)
    plt.legend(prop={'size': 17})
    Start_ = int(input1*3.8/4)
    plt.arrow(Start_+20,215, -20,-65,head_width=6,color="red")
    plt.gca().add_patch(Rectangle((Start_-10,220),40,40,linewidth=1,edgecolor='r',facecolor='none'))
    
    
    sub_axes = plt.axes([0.64, .28, .25, .25])
    for j in range(1,INPUT_DIM):
        plt.plot((np.arange(input1+1)+1)[Start_:],Cost_Proc[j][0,Start_:], linestyle = linestyles[j],color=next(Colors_))
        plt.fill_between((np.arange(input1+1)+1)[Start_:], Cost_Proc[j][0,Start_:]-Cost_Proc[j][1,Start_:], Cost_Proc[j][0,Start_:]+Cost_Proc[j][1,Start_:],
                             alpha=((j/10)+0.2),facecolor = next(Fcolors_)) 
    plt.setp(sub_axes, xticks=[], yticks=[])
    plt.savefig(str_, format='pdf', dpi=300,bbox_inches = 'tight')
    plt.show()
    '''
    ################## PLOT HVI
    plt.figure()
    plt.plot(np.arange(input1)+1,HVI_mean,label=r'$CA-MOBO$', linewidth=4,color= '#2e86c1')
    plt.fill_between(np.arange(input1)+1, HVI_mean-HVI_var, HVI_mean+HVI_var,alpha=0.5,facecolor='#aed6f1') 
    str_ = path + '/HVI-' + str(input1) + '.pdf'
    plt.xlabel(r'$t$')
    plt.ylabel(r'$Hypervolume (\%)$')
    plt.legend(prop={'size': 15})
    plt.savefig(str_, format='pdf', dpi=300,bbox_inches = 'tight')
    plt.show()    

    ################## PLOT Regret
    plt.figure()
    plt.plot(np.arange(input1)+1,Regret_mean,'o',label=r'$CA-MOBO$', linewidth=4,color= '#2e86c1')
    plt.fill_between(np.arange(input1)+1, Regret_mean-Regret_var, Regret_mean+Regret_var,alpha=0.5,facecolor='#aed6f1') 
    str_ = path + '/Regret-' + str(input1) + '.pdf'
    plt.xlabel(r'$t$')
    plt.ylabel(r'$Regret$')
    plt.legend(prop={'size': 15})
    plt.savefig(str_, format='pdf', dpi=300,bbox_inches = 'tight')
    plt.show()    
    '''
if __name__ == "__main__":
   main()
