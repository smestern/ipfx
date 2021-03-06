import ctypes as ct
import os
import time
import json
import datetime
import argparse
import mcc_get_settings as mcc
import numpy as np
from tkinter import *

def check_equal_hold(m):
    current = GetHolding(m)
    out = m.GetMeterValue(1)*1000000000000
    if np.abs(current-out)<1:
        equal=True
    else:
        equal=False
    return equal


def GetHolding(m):
    func = getattr(m, "GetHolding")
    test = func() # in amp/volt
    return test * 1000000000000

def SetHolding(m, val=0):
    func = getattr(m, "SetHolding")
    test = func(val) # in amp/volt
    return test

def seek_current(m, target, limits):
    #Get current values
    holding = GetHolding(m)
    volt = m.GetMeterValue(0) * 1000
    lower, upper = limits
    success = False
    up_down = False
    if volt > target:
        seeking_steps = np.arange(holding, lower, 1)
        up_down = False
    elif volt <= target:
        seeking_steps = np.arange(holding, upper, 1)
        up_down = True
    pairs = []
    for x in seeking_steps:
        SetHolding(m, x/1000000000000)
        time.sleep(0.1)
        volt = m.GetMeterValue(0) * 1000
        
        pairs.append([x, volt])
        
        if np.abs(volt - target)<0.5:
            
            time.sleep(.5)
            if np.abs(volt - target)<0.5:
                success = True
                break
            else:
                success = False
                break
        if up_down:
            test = (volt-target)
            if test>0.5:
                success = False
                break
        elif up_down == False:
            if ((target-volt))>0.5:
                success = False
                break
    if x == seeking_steps[-1]:
        success = True
    return success

def compute_ir(m, amp=5):
    resp, stim = testpulse(m, amp=amp)
    steady_state = np.nanmean(resp[50:73])/1000
    stim_steady =  np.nanmean(stim[50:73])
    #compute r as v/i in
    resist = steady_state / (stim_steady/1000000000000)
    return resist

def compute_adjust(m, target, amp=15):
    resist = compute_ir(m, amp=amp)
    volt = m.GetMeterValue(0)
    #Compute likely current
    I = (target/1000) / resist #in amp
    I_pA = I  * 1000000000000
    SetHolding(m, I)
    return







def testpulse(m, amp=5):
    holding = GetHolding(m)
    
    pulse = np.full(100,holding)
    pulse[25:75] = holding-5
    resp = []
    for x in pulse:
        SetHolding(m, x/1000000000000)
        time.sleep(0.01)
        temp = m.GetMeterValue(0) * 1000
        resp = np.hstack((resp, temp))
    return resp, pulse


def _adjust(m, target, limits, threshold=0.5):
    adjust = True
    t_start = time.time()
    while adjust:
        print(f"Adjusting test")
        compute_adjust(m, target)
        time.sleep(0.50)
        volt = m.GetMeterValue(0) * 1000
        if np.abs(volt-target)<threshold:
            print(volt)
            break
        else:
            print(f"Adjust Failed found voltage of {volt} - trying again")
            success= seek_current(m, target, limits)
            if success:
               break
        if (time.time()-t_start) > 2:
            print(f"Took too long to adjust - stopping")
        
    return

def auto_adjust(m, target, limits, threshold=0.5, polling_time=0.5):
    test = m.GetMeterValue(2)
    adjust = True
    while adjust:
        time.sleep(polling_time)
        if check_equal_hold(m):
            print("Equal Holding, Adjusting")
            volt = m.GetMeterValue(0) * 1000
            if np.abs(volt-target)<threshold:
                print(volt)
             
            else:
                success= seek_current(m, target, limits)
        else:
            print("unEqual Holding, Stim being applied?")
            continue
        
    return


def adjust_button():
    _target = int(target_field.get()) 
      
    _threshold = float(threshold_field.get())
  
    _upper = int(upper_field.get())
    _lower = int(lower_field.get())
    #compute_adjust(m, _target)
    _adjust(m, _target, [_lower,_upper], threshold=_threshold)



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
    # Create a GUI window  
    root = Tk()  
    
    # Set the background colour of GUI window  
    root.configure()  
    
    # Set the configuration of GUI window  
    root.geometry("400x250")  
    
    # set the name of tkinter GUI window  
    root.title("Auto Bias")   
        
    # Create a Principle Amount : label  
    label1 = Label(root, text = "Target (mV) : ")  
    
    # Create a Rate : label  
    label2 = Label(root, text = "Threshold (mV) : ")  
        
    # Create a Time : label  
    label3 = Label(root, text = "Upper Limit (pA): ") 
  
    # Create a Compound Interest : label  
    label4 = Label(root, text = "Lower Limit (pA): ")  
  
    # grid method is used for placing   
    # the widgets at respective positions   
    # in table like structure . 
  
    # padx keyword argument used to set paading along x-axis . 
    # pady keyword argument used to set paading along y-axis .  
    label1.grid(row = 1, column = 0, padx = 10, pady = 10)   
    label2.grid(row = 2, column = 0, padx = 10, pady = 10)   
    label3.grid(row = 3, column = 0, padx = 10, pady = 10) 
    label4.grid(row = 5, column = 0, padx = 10, pady = 10) 
  
    # Create a entry box   
    # for filling or typing the information. 
    target_field = Entry(root)   
    threshold_field = Entry(root)   
    upper_field = Entry(root) 
    lower_field = Entry(root) 

    target_field.insert(0,"-70")
    threshold_field.insert(0, "5")
    upper_field.insert(0,100)
    lower_field.insert(0,-250)

    # grid method is used for placing   
    # the widgets at respective positions   
    # in table like structure . 
      
    # padx keyword argument used to set paading along x-axis . 
    # pady keyword argument used to set paading along y-axis .  
    target_field.grid(row = 1, column = 1, padx = 10, pady = 10)   
    threshold_field.grid(row = 2, column = 1, padx = 10, pady = 10)   
    upper_field.grid(row = 3, column = 1, padx = 10, pady = 10) 
    lower_field.grid(row = 5, column = 1, padx = 10, pady = 10) 
  
    # Create a Submit Button and attached   
    # to calculate_ci function   
    button1 = Button(root, text = "Submit",
                     fg = "black", command = adjust_button)  
    
    # Create a Clear Button and attached   
    # to clear_all function   
    
    button1.grid(row = 6, column = 1, pady = 10)  
  
    # Start the GUI   
    root.mainloop() 
