import ctypes as ct
import os
import time
import json
import datetime
import argparse
import mcc_get_settings as mcc
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

def check_equal_hold(m):
    current = GetHolding(m)
    out = m.GetMeterValue(1)*1000000000000
    if np.abs(current-out)<1:
        equal=True
    else:
        equal=False
    return equal

def GetHolding(m): # in amp/volt
    return m.GetHolding() * 1000000000000

def SetHolding(m, val=0):
    func = getattr(m, "SetHolding")
    test = func(val) # in amp/volt
    return test

def GetVolt(m):
    return m.GetSecondarySignal()* 1000

 


# Driver code 
if __name__ == "__main__" :  
    m = mcc.MultiClampControl()
    UIDs = m.getUIDs()  # output all found amplifiers
    m.selectUniqueID(next(iter(UIDs)))  # select the first one (implicitly done by __init__)
    clampMode = m.GetMode() # return the clamp modei
    
    hold = GetHolding(m)
    if m._handleError()[0]:
    # handle error
        pass
    holding_ = []
    time_stamps = []

    for x in np.arange(500):
        holding_ = np.hstack((holding_, GetVolt(m)))
        time_stamps = np.hstack((time_stamps, time.time()))


    time_zero = np.diff(time_stamps) * 1000
    plt.plot(np.cumsum(time_zero), time_stamps[:-1])
    plt.show()
    plt.plot(1/time_zero)
    plt.show()
    print("d")
