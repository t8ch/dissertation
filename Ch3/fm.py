###
### generation of mean and variance modulated input currents
###

import numpy as np
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import davies_harte as dh

#initiation values
N_max = 1032258 #max samples pClamp
N = 1e5
dt = 1e-1 #in ms
dt2 = .1
N2 = int(dt*N/dt2)
alphaAM = 1. #1/60.
alphaFM = 1/1000. #4/450.
cutoff = None

#n evenly distributed slices of l (for later purpose)
""" Yield successive n-sized chunks from l.
"""
def chunks(seq, num):
    avg = len(seq) / float(num)
    out = np.random.random(N)
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def white_noise(sigma):
    np.random.seed()
    #print 'white noise'
    #return np.array([np.random.normal(0,1) for x in range(int(N))])*sigma/np.sqrt(dt)
    return np.random.normal(0,1,int(N))*sigma/np.sqrt(dt)

def OU(tau, sig):
    if tau == 0:
    #NOTE: it's not the strict limit tau->0 here !
        return white_noise(sig)
    else:
        return dh.OUP(tau, sig, N, dt)

def sqrt_spec(tau, sig, w0):
    return dh.sq_exp(tau, sig, w0, N, dt)

def mono_spec(tau, sig, w0):
    return dh.mono_exp(tau, sig, w0, N, dt)


#ornstein uhlenbeck process
def OU2(tau0,sigma):
    np.random.seed() #in order to work parallel
    if tau0 == 0:
    #NOTE: it's not the strict limit tau->0 here !
        return white_noise(sigma)
    else:
        seq = np.zeros(N)
        e_dT= np.exp(-dt/tau0)
        e_dT2 = sigma*np.sqrt(1.-np.exp (-2./tau0*dt))
        for i in np.arange(N-1):
        #seq[i+1] = seq[i] * np.exp(-dt/tau0) + sigma* np.sqrt ((1.-np.exp (-2./tau0*dt)))*np.random.normal(0,1.)
            seq[i+1] = seq[i] * e_dT + e_dT2*np.random.normal(0,1.)
        return seq    

#function that creates mean modulation (given signal and noise)
def MM (signal, noise):
    return np.array(signal)+np.array(noise)

#function that generates AM
def AM (signal,noise):
    return (1+alphaAM*np.array(signal))*np.array(noise)


# actual variance modulation
def AM_var(signal,noise):
    return np.nan_to_num(np.sqrt(1+alphaAM*np.array(signal)))*np.array(noise)

#functions with single argument
def FM2 (s):
    return FM(s[0],s[1])

def AM2 (s):
    return AM (s[0],s[1])

def MM2 (s):
    return MM (s[0],s[1])

def effectiveTau(signal):
    #corr = np.correlate(signal,signal,mode='full')
    a = np.fft.rfft(signal)
    corr = np.conjugate(a)*a
    corr = np.fft.irfft(corr)
    #result = corr
    result = corr[:corr.size/2]
    result  /= result[result.argmax()]
    for i in np.arange(N/2.):
        if result[i] <= 1/np.exp(1):
            return (result,i*dt)

def effectiveSigma(signal):
    return np.std(signal[1:])

#output of parameters
def params(sigmaS,sigmaN,tauS,tauN,alphaAM, alphaFM):
    print 'N, dt: ' , N,', ', dt
    print 'sigma s: ' , sigmaS, ' tau s: ', tauS
    print 'sigma n: ' , sigmaN, ' tau n: ', tauN
    print 'alphaAM: ', alphaAM
    print 'alphaFM: ', alphaFM
    snrMM = float(sigmaS)/(sigmaS+sigmaN)
    #snrAM = sigmaS*alphaAM
    snrAM = sigmaS*alphaAM/(1+sigmaS*alphaAM)
    snrFM = alphaFM*sigmaS/(alphaFM*sigmaS+1/tauN)
    varMM, varAM, varFM = sigmaS**2 + sigmaN**2, sigmaN**2*(1+alphaAM**2*sigmaS**2), sigmaN**2
    print 'SNR MM: ', snrMM
    print 'SNR AM: ', snrAM
    print 'SNR FM: ', snrFM
    print 'var MM: ', varMM
    print 'var AM: ', varAM
    print 'var FM: ', varFM
        

#output of parameters in a string
def params_str(sigmaS,sigmaN,tauS,tauN,alphaAM, alphaFM):
    snrMM = sigmaS/(sigmaS+sigmaN)
    #snrAM = sigmaS*alphaAM
    snrAM = sigmaS*alphaAM/(1+sigmaS*alphaAM)
    snrFM = alphaFM*sigmaS/(alphaFM*sigmaS+1/tauN)
    a =  'N: ' +str(N)+'\n'
    b = 'sigma s: %.2f ' % sigmaS + ' tau s: %.2f' % tauS +'\n'
    c ='sigma n: ' + str(sigmaN) + ' tau n: '+ str(tauN) +'\n'
    d ='alphaAM: '+ str(alphaAM) +'\n'
    e =  'alphaFM: '+ str(alphaFM)+ '\n'
    f = 'SNR MM: '+ str( snrMM)+ '\n'
    g = 'SNR AM: '+ str( snrAM)+ '\n'
    h = 'SNR FM: '+ str( snrFM)+ '\n'
    return a+b+c+d+e+f+g+h

#create signal traces for clampex
def make_signal(sigmaS, tauS, sigmaN, tauN, SNR, mod = 0, trials = 1, pause = 0, scale = 1/1000.):
    #mod: 0,1,2 => MM,AM,FM
    #trials: number of signals
    #pause: number of zeros between signals
    #scale: rescale to other units than pA when needed (nA for clampex is default)
    s, n = OU(tauS,sigmaS), OU(tauN,sigmaN)
    signal = np.zeros([N+pause,trials])

    #MM
    if mod==0:
        for i in range(trials):
            if trials>1:
                s, n = OU(tauS,sigmaS), OU(tauN,sigmaN)   
            signal[:N,i] = MM(s,n)
        print 'created signal: FM'
    #AM
    if mod==1:
        alphaAM = SNR/sigmaS
        for i in range(trials):
            if trials>1:
                s, n = OU(tauS,sigmaS), OU(tauN,sigmaN)   
            signal[:N,i] = AM(s,n)
        print 'created signal: AM'
        print 'alpha= ', alphaAM
    #FM
    if mod==2:
        alphaFM = SNR/(tauN*sigmaS*(1-SNR))
        for i in range(trials):
            if trials>1:
                s, n = OU(tauS,sigmaS), OU(tauN,sigmaN)   
            signal[:N,i] = FM_n(s,n)
        print 'created signal: FM'
        print 'alpha= ', alphaFM

    print 'number of trials: ', trials
    signal =  scale*np.array([signal[:,x] for x in range(trials)]).flatten()
    #insert 0s to compensate for 'holding'
    return signal
    #signal =  scale*signal

