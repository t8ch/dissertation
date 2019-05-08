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
'''
def OU2(a):
    return OU(a[0],a[1])

'''

###Fourier decomposition
#this function reconstructs time domain by giving frequencies from rfft
def reconFT(frequencies):
    w0 = 2*np.pi/N
    summands = [2./N*np.sum(np.array([x*np.cos(w0*k*n)-y*np.sin(w0*k*n) for x,y,k in zip(np.real(frequencies), np.imag(frequencies),np.arange(N/2+1))])) for n in np.arange(N)]
    return summands

#function that generates modulated FM signal for given sequences of s and n
#NO parallel computation
def FM (signal,noise):
    w0 = 2*np.pi/(N)
    frequencies = np.fft.rfft (noise) 
    tInt =[0]
    tInt.extend(integrate.cumtrapz (signal,dx=dt))
    #tInt = integrate.cumtrapz (signal,dx=dt,initial=0)
    modulated = [2./N*np.sum(np.array([x*np.cos(w0*k*n+alphaFM*tInt[n])-y*np.sin(w0*k*n+alphaFM*tInt[n]) for x,y,k in zip(np.real(frequencies), np.imag(frequencies),np.arange(N/2+1))])) for n in np.arange(N,dtype=np.int16)]
    return modulated

#sliced computation of FM (due to pool() it needs to be defined at top level)
def mod (frequencies,tInt,range):
    w0 = 2*np.pi/(N)
    modulated_n = [2./N*np.sum(np.array([x*np.cos(w0*k*n+alphaFM*tInt[n])-y*np.sin(w0*k*n+alphaFM*tInt[n]) for x,y,k in zip(np.real(frequencies), np.imag(frequencies),np.arange(N/2+1))])) for n in range]
    print 'FM range %f.1 to %f.1' % (range[0],range[-1])
    return modulated_n

#includes multiprocessing
#that's what we use
def FM_n_old (signal,noise,cores=cpu_count()): #cpu is number of available cores
    slices = chunks(np.arange(N,dtype=int),cores)    
    frequencies = np.fft.rfft (noise)
    #tInt =[0]
    #tInt.extend(integrate.cumtrapz (signal,dx=dt))
    tInt = integrate.cumtrapz (signal,dx=dt,initial= 0)
    pool = Pool (cores)
    modulated = pool.map (partial(mod,frequencies,tInt),slices)
    pool.close ()
    pool.join ()
    return sum(modulated, []) #flattens the list

def FM_n(signal, noise):
    w0 = 2*np.pi/(N)
    frequencies = np.fft.rfft(noise)[:cutoff]
    l = len(frequencies)
    tInt= np.r_[0,integrate.cumtrapz (signal, dx = dt)]
    omega = w0*np.arange(l)
    t = np.arange(int(N))
    ret = np.zeros(N)
    re, im  = np.real(frequencies), np.imag(frequencies)
    for i in range(int(N)):
        #phase = omega*i+alphaFM*tInt[i]*omega #when alpha_k~ k
        #phase = omega*i+alphaFM*tInt[i]*w0 #here perfect modulation at alpha = k/s_min and k just index of frequency (found by inclusion of dt)
        phase = omega*i+alphaFM*tInt[i] #perfect mod at w_0*k/s_min;     
        ret[i] = np.dot(np.r_[re,-im], np.r_[np.cos(phase), np.sin(phase)])
        #if i%20000 == 0:
        #    print i, ' done'
    return 2./N*ret

def FM_n2(signal, noise):
    w0 = 2*np.pi/(N)
    frequencies = np.fft.rfft(noise)[:cutoff]
    l = len(frequencies)
    tInt= np.r_[0,integrate.cumtrapz (signal, dx = dt)]    
    phase = np.add(w0*np.tensordot(range(int(N)), range(l), axes = 0), alphaFM*reshape(repeat(tInt, l), (-1,l)))
    re, im  = np.real(frequencies), np.imag(frequencies)
    ret = np.dot(np.cos(phase),re)-np.dot(np.sin(phase), im)
    #ret = np.dot(np.r_[np.cos(phase), np.sin(phase)],np.r_[re,-im])
    return 2./N*ret
    

#function that creates mean modulation (given signal and noise)
def MM (signal, noise):
    return np.array(signal)+np.array(noise)

#function that generates AM
def AM (signal,noise):
    return (1+alphaAM*np.array(signal))*np.array(noise)
    #return [(1+alphaAM*x)*y for x,y in zip (signal,noise)]

# actual variance modulation
def AM_var(signal,noise):
    #return np.sqrt(np.abs(1+alphaAM*np.array(signal)))*np.array(noise)
    #return np.real(np.sqrt(1+alphaAM*np.array(signal)))*np.array(noise)
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

if __name__ == '__main__':

#FM test in gerneral; test multiprocessing FM_n
#plot for simple modulations
    if True:
        alphaAM = 1.
        alphaFM = 25.
        N = 1e3
        #a = OU(tauS,sigmaS)
        #b = OU(tauN,sigmaN)
        b = [1.*np.cos(37*np.pi*i/(1.3*N)+0*1.5)+ 0.*np.cos(20*np.pi*i/(N*1.3)+4) for i in np.arange(N)]
        a = [1.*np.cos (4.5*np.pi*i/(1.3*N))+ 0*np.sin (8*np.pi*i/(N*1.3)) for i in np.arange(N)]
        #d = [1.*np.cos (50*np.pi*i/(N)+(alphaFM)/(6*np.pi/(N*dt))*np.sin(6*np.pi*i/(N))) for i in np.arange(N)]
        A = [x+1 for x in a]
        AA = [-x-1 for x in a]
        cpu = cpu_count()
        mm, am , fm = MM(a,b), AM(a,b), FM_n(a,b)
        # print effectiveTau(a), effectiveSigma(a)
        # print effectiveTau(b), effectiveSigma(b)
        # print effectiveTau(c), effectiveSigma(c)
        # print np.subtract(c,d)
        #sys.exit()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(511)
        ax1.set_ylabel('signal')
        ax1.plot(np.arange(N),a,'black')
        ax2 = fig1.add_subplot(512)
        ax2.plot(np.arange(N),b,'green')
        ax2.set_ylabel('noise')
        ax4 = fig1.add_subplot(513)
        ax4.plot(np.arange(N),mm,'red',np.arange(N),a,'black')
        ax4.set_ylabel('MM')
        ax3= fig1.add_subplot(514)
        ax3.set_ylabel('AM')
        ax3.plot(np.arange(N),am,'blue',np.arange(N),A,'black',np.arange(N),AA,'black')
        ax5= fig1.add_subplot(515)
        ax5.set_ylabel('FM')
        #ax5.plot(np.arange(N),fm,'grey',np.arange(N),a,'black')
        [x.axes.get_xaxis().set_ticklabels([]) for x in [ax1,ax2,ax3,ax4,ax5]]
        [x.axes.get_yaxis().set_ticklabels([]) for x in [ax1,ax2,ax3,ax4,ax5]]
        plt.show ()
        sys.exit()

#test variance of OU
    if False:
        """
        end1 = np.zeros(1000)
        end2 = np.zeros(1000)
        beg = np.zeros(1000)
        for i in np.arange(1000):
            a = OU(tauS,sigmaS)
            beg[i]= a[1]
            end1[i] = a[5e3-5]
            end2[i] = a[5e3-2]
        print 'beginning: ', np.var(beg)
        print 'end1:       ', np.var(end1)
        print 'end2:       ', np.var(end2)
        print 'target:    ', sigmaS**2
        #sys.exit()
        """
        a = OU(tauS,sigmaS)
        b = OU(tauN,sigmaN)
        c = FM_n(a,b,4)
        
        corr = np.correlate(a,a,mode='full')
        size_ = corr[corr.size / 2:]
        result = size_
        result  /= result[result.argmax()]
        corr2 = np.correlate(b,b,mode='full')
        result2 = corr2[corr2.size/2:]
        result2  /= result2[result2.argmax()]
        corr3 = np.correlate(c,c,mode='full')
        result3 = corr3[corr3.size/2:]
        result3  /= result3[result3.argmax()]

        print result
        
        time = np.linspace(0,N*dt,N)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(time,result,'g',time,result2,'r',time,result3,'b')
        plt.show ()

#effective tau
    if False:
        a = OU(tauS,sigmaS)
        b = OU(tauN,sigmaN)
        start = time.time()
        c = FM_n(a,b,4)
        print params()
        print effectiveTau(a), effectiveSigma(a)
        print effectiveTau(b), effectiveSigma(b)
        print effectiveTau(c), effectiveSigma(c)
        start2 = time.time() - start
        print 'time: ', start2
        sys.exit()
        c = FM_n(a,b,1)
        print params()
        print effectiveTau(a), effectiveSigma(a)
        print effectiveTau(b), effectiveSigma(b)
        print effectiveTau(c), effectiveSigma(c)
        print 'time: ', time.time()-start2


    if False:
        p = []
        v = []
        l = logspace(-3,0.,5)
        for i in l:
            dt = i
            print i
            b = []
            for i in range(20):
                a = OU(tauS, sigmaS)
                b =np.r_[b,effectiveTau(a)[1]]
            p  += [np.mean(b)]
            v  += [np.std(b)] 
