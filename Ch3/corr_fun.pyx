#### run with:
#### python setup.py build_ext --inplace
####

# coding: utf-8
import numpy as np
import fm
cimport numpy as np

def corr(np.ndarray a, c=1, d=1):
    le = xrange(len(a))
    #new
    #z = zip(le, np.roll(le,1))
    z = [(x,y) for x in le for y in le if x < y]
    cdef np.ndarray[complex] co = np.zeros(int(fm.N2/4.), dtype = complex)
    cdef np.ndarray[complex, ndim = 2] ft = np.fft.rfft(a)
    ft = np.array([x[:int(fm.N2/4.)] for x in ft])
    cdef np.ndarray[complex, ndim = 2] ftC = np.conjugate(ft)
    for i,k in z:
        co = co + ftC[i]*ft[k]
    cdef np.ndarray co2 = np.float32(np.real(co)/len(z))
    co2 = np.r_[co2, np.zeros(int(fm.N2/4.+1))]
    #return co2, np.roll(np.fft.irfft(co2), int(fm.N2/2))
    return np.float32(co2), np.array([])
'''
def corr(np.ndarray a, c=1, d=1):
    le = xrange(len(a))
    #new
    #z = zip(le, np.roll(le,1))
    z = [(x,y) for x in le for y in le if x < y]
    cdef np.ndarray[np.complex64_t] co = np.zeros(int(fm.N2/4.), dtype = np.complex64)
    cdef np.ndarray[np.complex64_t, ndim = 2] ft = np.complex64(np.fft.rfft(a))
    ft = np.array([x[:int(fm.N2/4.)] for x in ft])
    cdef np.ndarray[np.complex64_t, ndim = 2] ftC = np.complex64(np.conjugate(ft))
    for i,k in z:
        co = co + ftC[i]*ft[k]
    cdef np.ndarray co2 = np.float32(np.real(co)/len(z))
    co2 = np.r_[co2, np.zeros(int(fm.N2/4.+1))]
    #return co2, np.roll(np.fft.irfft(co2), int(fm.N2/2))
    return np.float32(co2), np.array([])
'''

'''
cdef class SimpleIF(object):
# Simple Integrated-and-fire model of the form:
# tau*dV/dt =-(V-El) + R*I
# tau is the time constant
# R is the leak resistance
# El is the resting membrane potential
# I is the applied current
#tref is the refractory time
#ures is the reste potential
    cdef public float _peak, _tau, _R, _El, _Vinit, _tref, _ures, _Vthr,
    #cdef int _counts, _N2
    cdef np.ndarray spiketrain2, spiketrain_isi

    def __init__(self, tau, R, El, V_thr = 20., Vinit=-0., tref=0, ures = -5., dt2 = fm.dt2):
        fm.dt2 = dt2
        fm.N2 = int(fm.dt*fm.N/dt2)
        self._peak = 40.
        self._tau = float(tau) # in ms
        self._R = float(R) # in GOhm
        self._El = float(El) # in mV
        self._Vinit = float(Vinit) # in mV
        self._tref = float(tref) # in ms
        self._ures = float(ures)
        self._Vthr = V_thr # in mV
        #print LIF parameters
  
    def print_LIFparams(self):
        return  'tau_m = %.2f, R = %.2f \n El = %.2f, V_th = %.2f ' % (self._tau, self._R, self._El, self._Vthr)

    def spiketrain2(self, np.ndarray curr):
        cdef np.ndarray[double] current =  curr
        cdef np.ndarray[np.int_t] stimes = np.array([], np.int)
        cdef np.ndarray[np.int_t] st = np.zeros(fm.N2, dtype = np.int)
        cdef double voltage =  self._Vinit  # initial condition
        cdef float dt = fm.dt #fm.dt # step size integration
        cdef int tfire = -10000
        cdef double diff = 0
# Eulers solver
        cdef int i
        cdef double dVdt
        cdef double r = fm.dt/fm.dt2
        cdef int n = int(len(current))
        #print sum(voltage)
        for i in xrange(n):
            dVdt =  1./self._tau*(current[i-1]*self._R - voltage + self._El)
            voltage = voltage + dt*dVdt
            if voltage >= self._Vthr:
                diff = voltage-self._Vthr
                #if i == tfire + 1: print "HERE"
                tfire = i
                voltage = self._ures + diff
                stimes = np.append(stimes, tfire)
        for x in stimes:
            st[int(x*r)] = 1
        return st

    def spiketrain_isi(self, np.ndarray curr):
        cdef np.ndarray[double] current =  curr
        cdef np.ndarray[np.int_t] stimes = np.array([], np.int)
        cdef float voltage =  self._Vinit  # initial condition
        cdef float dt = fm.dt #fm.dt # step size integration
        cdef int tfire = -10000
        cdef double diff = 0
# Eulers solver
        cdef int i
        cdef double dVdt
        cdef int n = int(len(current))
        #print sum(voltage)
        for i in xrange(n):
            dVdt =  1./self._tau*(current[i-1]*self._R - voltage + self._El)
            voltage = voltage + dt*dVdt
            if voltage >= self._Vthr:
                diff = voltage-self._Vthr
                #if i == tfire + 1: print "HERE"
                tfire = i
                voltage = self._ures + diff
                stimes = np.append(stimes, tfire)
        stimes = np.diff(stimes)
        return stimes*fm.dt
'''
'''
    def timecourse(self, np.ndarray current):
        cdef np.ndarray voltage = np.empty(len(current))
# initial condition
        voltage[0] = self._Vinit
        cdef double dt = fm.dt #fm.dt # step size integration
        cdef int tfire = -1000
        cdef double diff = 0
# Eulers solver
        cdef int i
        cdef double dVdt
        for i in range(1, len(current)):
            if i==tfire+1:
                voltage[i] = self._ures + diff
            else:
                dVdt =  (1/self._tau*(current[i-1]*self._R - voltage[i-1] + self._El))
                voltage[i] = voltage[i-1] + dt*dVdt
    # action potential threshold 
                if voltage[i] >= self._Vthr:
                    diff = voltage[i]-self._Vthr
                    tfire = i
                    voltage[i] = self._peak #spike
        self._Vmem = voltage
        return voltage    

    def spiketrain(self, np.ndarray time_course):        
        cdef np.ndarray st = np.array([1 if x == self._peak else 0 for x in time_course])
        self._spiketrain = st
        return st
'''
'''     
    def spiketrain0(self, time_course):
        st = np.array(time_course)//self._peak
        self._spiketrain = st
        return st
'''

'''
# coding: utf-8
import numpy as np

def timecourse(tau, R, Vthr, El, ures, current, peak):
    voltage = np.empty(len(current))
    # initial condition
    voltage[0] = El
    dt = fm.dt # step size integration
    tfire = -e9
    diff = 0
    # Eulers solver
    for i in range(1, len(current)):
        if i==tfire+1:
            voltage[i] = ures + diff
        else:
            voltage[i] = voltage[i-1] + dt*((1./tau*(current[i-1]*R - voltage[i-1] + El)))
            # action potential threshold 
            if voltage[i] >= Vthr:
                diff = voltage[i]-Vthr
                tfire = i
                voltage[i] = peak #spike
    return voltage
'''
