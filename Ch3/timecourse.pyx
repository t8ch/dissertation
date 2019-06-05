#### run with:
#### python setup.py build_ext --inplace
####


###
### computing voltage evolution and spike times for LIF and EIf neurons (implementation of solving differential equation for given input current via Euler forward)
### implemented in cython for better performance
###

# cython: profile=True

# coding: utf-8
import numpy as np
import fm

cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

#import multiprocessing
#import os

#make multicore work available?
#os.system("taskset -p 0xff %d" % os.getpid())

cdef class SimpleIF(object):
# Simple Integrated-and-fire model of the form:
# tau*dV/dt =-(V-El) + R*I
# tau is the time constant
# R is the leak resistance
# El is the resting membrane potential
# I is the applied current
#tref is the refractory time
#ures is the reste potential
    cdef public float _peak, _tau, _R, _El, _Vinit, _tref, _ures, _Vthr, _vol
    #cdef int _counts, _N2
    cdef np.ndarray spiketrain2, spiketrain_isi

    def __init__(self, tau, R, El, V_thr = 20., Vinit=-0., tref=0, ures = -5.,):# dt2 = fm.dt2):
        dt2 = fm.dt2
        fm.N2 = int(fm.dt*fm.N/dt2)
        self._peak = 40.
        self._tau = float(tau) # in ms
        self._R = float(R) # in GOhm
        self._El = float(El) # in mV
        self._Vinit = float(Vinit) # in mV
        self._tref = float(tref) # in ms
        self._ures = float(ures)
        self._Vthr = V_thr # in mV
        self._vol = 0
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
            #self._vol = voltage
            if voltage >= self._Vthr:
                diff = voltage-self._Vthr
                #if i == tfire + 1: print "HERE"
                tfire = i
                voltage = self._ures + diff
                #stimes = np.append(stimes, tfire)
                st[int(tfire*r)] = 1
        #for x in stimes:
            #st[int(x*r)] = 1
        return st
    
    def voltagecourse(self, np.ndarray curr):
        cdef np.ndarray[double] current =  curr
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
        self._vol = voltage

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
#########
#     ************* EIF ************
###########

cdef class EIF(object):
# tau is the time constant
# R is the leak resistance
# El is the resting membrane potential
# I is the applied current
#tref is the refractory time
#ures is the reste potential
    cdef public float _peak, _tau, _R, _El, _Vinit, _tref, _ures, _Vthr, DeltaT, _vol
    #cdef int _counts, _N2
    cdef np.ndarray spiketrain2, spiketrain_isi

    def __init__(self, tau, R, El, V_thr = 15., Vinit=-0., tref= 5., ures = 0., DeltaT= 1.5):# dt2 = fm.dt2):
        dt2 = fm.dt2
        fm.N2 = int(fm.dt*fm.N/dt2)
        self._peak = 40.
        self._tau = float(tau) # in ms
        self._R = float(R) # in GOhm
        self._El = float(El) # in mV
        self._Vinit = float(Vinit) # in mV
        self._tref = float(tref) # in ms
        self._ures = float(ures)
        self._Vthr = V_thr # in mV
        self.DeltaT = float(DeltaT)
        self._vol = 0
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
        cdef float ref = self._tref/fm.dt
        cdef float thr_num = self._Vthr + 50
        #print sum(voltage)
        for i in xrange(n):
            if i-tfire < ref:
                voltage = self._ures
            else:
                dVdt =  1./self._tau*(current[i-1]*self._R - voltage + self._El + self.DeltaT*exp((voltage-self._Vthr)/self.DeltaT))
                voltage = voltage + dt*dVdt
                if voltage >= thr_num:
                    #diff = voltage-self._Vthr-50
                    #if i == tfire + 1: print "HERE"
                    tfire = i
                    voltage = self._ures# + diff
                    #stimes = np.append(stimes, tfire)
                    st[int(tfire*r)] = 1
        #for x in stimes:
        #st[int(x*r)] = 1
        return st

    def voltagecourse(self, np.ndarray curr):
        cdef np.ndarray[double] current =  curr
        cdef double voltage =  self._Vinit  # initial condition
        cdef float dt = fm.dt #fm.dt # step size integration
        cdef int tfire = -10000
        cdef double diff = 0
# Eulers solver
        cdef int i
        cdef double dVdt
        cdef double r = fm.dt/fm.dt2
        cdef int n = int(len(current))
        cdef float ref = self._tref/fm.dt
        cdef float thr_num = self._Vthr + 50
        #print sum(voltage)
        for i in xrange(n):
            if i-tfire < ref:
                voltage = self._ures
            else:
                dVdt =  1./self._tau*(current[i-1]*self._R - voltage + self._El + self.DeltaT*exp((voltage-self._Vthr)/self.DeltaT))
                voltage = voltage + dt*dVdt
                if voltage >= thr_num:
                    #diff = voltage-self._Vthr-50
                    #if i == tfire + 1: print "HERE"
                    tfire = i
                    voltage = self._ures# + diff
        self._vol = voltage
        

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
        cdef float ref = self._tref/fm.dt
        cdef float thr_num = self._Vthr + 50
        #print sum(voltage)
        for i in xrange(n):
            if i-tfire < ref:
                voltage = self._ures
            else:
                dVdt = 1./self._tau*(current[i-1]*self._R - voltage + self._El + self.DeltaT*exp((voltage-self._Vthr)/self.DeltaT))
                voltage = voltage + dt*dVdt
                if voltage >= thr_num:
                    #diff = 0 # voltage-self._Vthr-50 
                    #if i == tfire + 1: print "HERE"
                    tfire = i
                    voltage = self._ures# + diff
                    stimes = np.append(stimes, tfire)
        stimes = np.diff(stimes)
        return stimes*fm.dt

