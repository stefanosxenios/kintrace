import numpy as np
from .feed_fun import get_feed_cached

def model1(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP=k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000


    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = mumax2 * S2 / (KP + S2)
    
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V)*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                #S2=XYLOSE
    dPdt= mu2*X1-(dVdt/V)*P                                         #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model2(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KPI = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000


    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1))/(1+P/KPI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)      #S2=XYLOSE
    dPdt= mu2*X1-(dVdt/V)*P                               #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model3(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model4(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model5(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 +(S1**2)/KSI))/(1+P/KPI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model6(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1))/(1+P/KPI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model7(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model8(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 +(S1**2)/KSI))/(1+P/KPI)
    mu2 = mumax2 * S2 / (KP + S2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model9(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = (mumax2 * S2 / (KP + S2))/(1+P/KPI)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model10(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = mumax2 * S2 / (KP + S2 + (S2**2)/KSI)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model11(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = (mumax2 * S2 / (KP + S2 + (S2**2)/KSI))/(1+P/KPI)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model12(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI,KPI= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1)
    mu2 = (mumax2 * S2 / (KP + S2 + (S2**2)/KSI))/(1+P/KPI)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model13(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1, S1, S2, P, V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model14(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model15(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model16(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI1,KSI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI1)
    mu2 = mumax2 * S2 / (KP + S2 + (S2**2)/KSI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model17(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI1,KSI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI1)
    mu2 = mumax2 * S2 / (KP + S2 + (S2**2)/KSI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model18(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model19(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI1,KSI2,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI1))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2 + (S2**2)/KSI2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX

def model20(X, t, k, exter):
    #svm training
    #print("Mean Squared Error:", mse)
    # Parameters

    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI1,KSI2,KPI1,KPI2= k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)
    # Variables
    X = np.maximum(X,0)
    X1,S1,S2,P,V = X
    # Externals
    # Flow of Glucose solution
    Fs=feed_func(t)/1000
    
    # Kinetics
    mu1 = (mumax1 * S1 / (KS1 + S1 + (S1**2)/KSI1))/(1+P/KPI1)
    mu2 = (mumax2 * S2 / (KP + S2 + (S2**2)/KSI2))/(1+P/KPI2)
    # State variables
    dVdt = Fs
    rX = (mu1) * X1
    dXdt = rX - (dVdt/V) * X1 - kd*X1
    dS1dt = -(1/YXS1)*mu1*X1 + (dVdt/V)*(S1_feed-S1)                                      #S1=GLUCOSE
    dS2dt = -(1/YPS2)*mu2*X1 + (dVdt/V)*(S2_feed-S2)                                      #S2=XYLOSE
    dPdt =  mu2*X1 - (dVdt/V)*P                                                           #P=Xylitol
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model21(X, t, k, exter):
    # kd -, KSI1 -, KPI1 -, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model22(X, t, k, exter):
    # kd -, KSI1 -, KPI1 +, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KPI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = ((mumax1 * S1) / (KS1 + S1)) / (1 + P / KPI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model23(X, t, k, exter):
    # kd -, KSI1 +, KPI1 -, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1 + (S2**2) / KSI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model24(X, t, k, exter):
    # kd -, KSI1 +, KPI1 +, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI1,KPI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = ((mumax1 * S1) / (KS1 + S1 + (S2**2) / KSI1)) / (1 + P / KPI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model25(X, t, k, exter):
    # kd +, KSI1 -, KPI1 +, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KPI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = ((mumax1 * S1) / (KS1 + S1)) / (1 + P / KPI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1 - kd * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    return [dXdt, dS1dt, dS2dt, dPdt, dVdt]


def model26(X, t, k, exter):
    # kd +, KSI1 +, KPI1 -, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1 + (S2**2) / KSI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1 - kd * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model27(X, t, k, exter):
    # kd +, KSI1 +, KPI1 +, KSI2 -, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI1,KPI1 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = ((mumax1 * S1) / (KS1 + S1 + (S2**2) / KSI1)) / (1 + P / KPI1)
    mu2 = (mumax2 * S2) / (KP + S2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1 - kd * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P
    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model28(X, t, k, exter):
    # kd -, KSI1 -, KPI1 -, KSI2 -, KPI2 +
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KPI2 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1)
    mu2 = ((mumax2 * S2) / (KP + S2)) / (1 + P / KPI2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model29(X, t, k, exter):
    # kd -, KSI1 -, KPI1 -, KSI2 +, KPI2 -
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI2 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1)
    mu2 = (mumax2 * S2) / (KP + S2 + (S2**2) / KSI2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model30(X, t, k, exter):
    # kd -, KSI1 -, KPI1 -, KSI2 +, KPI2 +
    mumax1,mumax2,YXS1,YPS2,KS1,KP,KSI2,KPI2 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1)
    mu2 = ((mumax2 * S2) / (KP + S2 + (S2**2) / KSI2)) / (1 + P / KPI2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX


def model31(X, t, k, exter):
    # kd +, KSI1 -, KPI1 -, KSI2 +, KPI2 +
    mumax1,mumax2,YXS1,YPS2,KS1,KP,kd,KSI2,KPI2 = k
    S1_feed, S2_feed, feed_func = get_feed_cached(exter)

    X = np.maximum(X, 0)
    X1, S1, S2, P, V = X

    Fs = feed_func(t) / 1000.0
    dVdt = Fs

    mu1 = (mumax1 * S1) / (KS1 + S1)
    mu2 = ((mumax2 * S2) / (KP + S2 + (S2**2) / KSI2)) / (1 + P / KPI2)

    rX = mu1 * X1
    dXdt  = rX - (dVdt / V) * X1 - kd * X1
    dS1dt = -(1 / YXS1) * mu1 * X1 + (dVdt / V) * (S1_feed - S1)
    dS2dt = -(1 / YPS2) * mu2 * X1 + (dVdt / V) * (S2_feed - S2)
    dPdt  = mu2 * X1 - (dVdt / V) * P

    dX = [dXdt, dS1dt, dS2dt, dPdt, dVdt]
    return dX





def get_model(name):
    models = {
    "model1": model1,
    "model2": model2,
    "model3": model3,
    "model4": model4,
    "model5": model5,
    "model6": model6,
    "model7": model7,
    "model8": model8,
    "model9": model9,
    "model10": model10,
    "model11": model11,
    "model12": model12,
    "model13": model13,
    "model14": model14,
    "model15": model15,
    "model16": model16,
    "model17": model17,
    "model18": model18,
    "model19": model19,
    "model20": model20,
    "model21": model21,
    "model22": model22,
    "model23": model23,
    "model24": model24,
    "model25": model25,
    "model26": model26,
    "model27": model27,
    "model28": model28,
    "model29": model29,
    "model30": model30,
    "model31": model31,
    }
    
    if name not in models:
        raise ValueError(f"Model '{name}' not found.")
    return models[name]



