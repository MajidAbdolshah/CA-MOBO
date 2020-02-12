from __future__ import division
import numpy as np
import time
import sys
import random
from termcolor import *
from function import *
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import math
import copy
from termcolor import colored
from parameters import *

class DataComplex:
    data = np.empty((0,INPUT_DIM))
    outputs = np.empty((0,OUTPUT_DIM))
    def __init__(self, iData,iOut):
        self.data = iData
        self.outputs = iOut
    def newData(self,newPoint):
        self.data = np.append(self.data,[newPoint],axis=0)
    def newOut(self,newPoint):
        self.outputs = np.append(self.outputs,[newPoint],axis=0)

def loadData(add):
    xad = add + 'xData.txt' 
    yad = add + 'yData.txt' 
    xData = np.loadtxt(xad, dtype=float)
    xData = xData.reshape(len(xData),1)
    yData = np.loadtxt(yad, dtype=float)
    return xData,yData

def CreatePath(str_):
    path = os.getcwd()  
    print ("The current working directory is %s" % path) 
    path = str_
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
    print("_____________________________")
    
    
def print_dots(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def print_fancy(string,val):
    print_dots(string)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print()

######################################## DEF I, works good, decided to be revised tho
def createPoints_(deepness,breadth):
    points_ = {}
    for i in range(1,deepness+1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, i/ANGLE, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_[i] = temp_merge

    for key in points_:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_[key]))
    return points_[1].T

######################################## DEF II, works good, decided to be revised tho
def createPoints(deepness,breadth):
    points_ = {}
    for i in range(1,deepness+1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, -i/ANGLE, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_[i] = temp_merge

    for key in points_:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_[key]))

    points_2 = {}
    for i in range(1,-deepness-1,-1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, -i/ANGLE, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_2[i] = temp_merge
        
    for key in points_2:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_2[key]))
    
    return points_[1].T

def info(*arg):
    for i in range(len(arg)):
        cprint("Shape of "+str(i)+" is "+str(arg[i].shape),"green")

def initvals_(bounds,INITIAL,flag_):
    str_ = "Initializing " + str(INITIAL) + " Data"
    print_fancy(str_,0.1)
    gData = np.zeros([INITIAL,INPUT_DIM])
    for i in range(0,INITIAL):
        for j in range(0,INPUT_DIM):
            gData[i,j] = random.uniform(bounds['min'][j], bounds['max'][j])
    gDataB = copy.deepcopy(gData) 
    if flag_:
        gDataY = function(gDataB)
        return gData,gDataY
    else:
        return gData.T

def initvals_no_balls(bounds,INITIAL,flag_):
    str_ = "Initializing " + str(INITIAL) + " Data"
    print_fancy(str_,0.1)
    gData = np.zeros([INITIAL,INPUT_DIM])
    for i in range(0,INITIAL):
        for j in range(0,INPUT_DIM):
            gData[i,j] = random.uniform(bounds['min'][j], bounds['max'][j]/100)
    gDataB = copy.deepcopy(gData) 
    if flag_:
        gDataY = function(gDataB)
        return gData,gDataY
    else:
        return gData.T