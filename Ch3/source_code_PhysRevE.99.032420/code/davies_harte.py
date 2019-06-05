from pylab import *
#from joblib import Parallel, delayed  
#import multiprocessing

def process(spectrum): #expects analytic spectrum and dt
    spec_ana = spectrum[1]
    dt = spectrum[2]
    acf_ana = irfft(spec_ana)/dt  
    if min(spec_ana) < -1e-5:
        print "negative coefficient"
        return "negative coefficient!"
    else:
        #acf = r_[acf, acf[1::-1]]
        n = len(spec_ana)
        a1 = np.random.normal(0,1.,n)
        a2 = np.random.normal(0,1.,n)
        b = sqrt(2*(n-1)/dt*spec_ana/2.)*(a1+1j*a2) #as in Percival paper; length in time space 2(n-1)
        p = irfft(b)
        spec_sim = dt/(2*n-2)*abs(b)**2 #according to spectral density estimation: periodogram
        acf_sim = irfft(spec_sim)/dt
        return p , spec_ana, spec_sim, acf_ana, acf_sim, var(p)
        
#NOT CHECKED
def sqrt_exp(sig,tau, w0, N):
    x = arange(N/2+1)
    spec = exp(-(2*pi*x-w0)**2/(2*tau)**2) + exp(-(2*pi*x+w0)**2/(2*tau)**2)
    return N*sig**2*sqrt(pi)/(2*tau)*spec

def sqrt_exp_t(sig, tau, w0, N, dt=1.):
    x = arange(-N/2,N/2)
    a = exp(-(x*dt)**2/(tau**2))*sig**2*cos(w0*x*dt)
    return a, dt*rfft(fftshift(a)), dt

## fast implementation of OUP
def OUP(tau, sig, N, dt= .1):
    np.random.seed()
    x = arange(-N/2,N/2)
    a = exp(-abs(x)*dt/tau)*sig**2
    ss = dt*rfft(fftshift(a))
    n = len(ss)
    a1 = np.random.normal(0,1.,n)
    a2 = np.random.normal(0,1.,n)
    b = sqrt(2*(n-1)/dt*ss/2.)*(a1+1j*a2)
    p = irfft(b)
    ##FOR CONTROL PURPOSES; COMMENT OUT WHEN FAST CALCULATION NEEDED
    #spec = dt/(2*n-2)*abs(b)**2
    #auto = irfft(spec)
    return p#, spec, ss, auto, var(p)   

## mono expontetial (abs) * cos correlation function

def mono_exp(tau, sig, w0, N, dt= .1):
    np.random.seed()
    x = arange(-N/2,N/2)
    a = exp(-abs(x*dt/tau))*sig**2*cos(w0*x/tau*dt) #note: w0 in units of 2*pi/tau
    ss = dt*rfft(fftshift(a))
    n = len(ss)
    a1 = np.random.normal(0,1.,n)
    a2 = np.random.normal(0,1.,n)
    b = sqrt(2*(n-1)/dt*ss/2.)*(a1+1j*a2)
    p = irfft(b)
    ##FOR CONTROL PURPOSES; COMMENT OUT WHEN FAST CALCULATION NEEDED
    #spec = dt/(2*n-2)*abs(b)**2
    #auto = irfft(spec)/dt
    return p#, spec, ss, auto, var(p)


## squared expontetial * cos correlation function
def sq_exp(tau, sig, w0, N, dt= .1):
    np.random.seed()
    x = arange(-N/2,N/2)
    a = exp(-(x*dt/tau)**2)*sig**2*cos(w0*x*dt) #only difference to OUP (?)
    ss = dt*rfft(fftshift(a))
    n = len(ss)
    a1 = np.random.normal(0,1.,n)
    a2 = np.random.normal(0,1.,n)
    b = sqrt(2*(n-1)/dt*ss/2.)*(a1+1j*a2)
    p = irfft(b)
    ##FOR CONTROL PURPOSES; COMMENT OUT WHEN FAST CALCULATION NEEDED
    #spec = dt/(2*n-2)*abs(b)**2
    #auto = irfft(spec)/dt
    return p#, spec, ss, auto, var(p)

'''
#x = ifftshift(arange(-N,N))
#y = exp(-(x+N/300)**2/20)
#y_fft = fft(y)

#print min(y_fft)
x = arange(N/2)
d = sqrt(N/(2*cosh(.001*(x-2000))))
d = sqrt(N*exp(-.0002*(x-2000)**2)/2)
e = r_[d[::-1],d]

a1 = np.random.normal(0,1.,N)
a2 = np.random.normal(0,1.,N)

#b = e*(a1+1j*a2)
b = e*(a1+1j*a2)
#b = [sqrt(N*y_fft[x]/2.)*(a1[x]+a2[x]*1j) for x in range(int(N))]
#b =  [sqrt(N/cosh(.01*x-400))*(a1[x]+a2[x]*1j) for x in range(int(N))]
#b = [sqrt(N*exp(-.001*(x-N-200)**2)/2)*(a1[x]+a2[x]*1j) for x in range(int(N))]

print min(b)
b = fftshift(b)
c = ifft(b)
'''
