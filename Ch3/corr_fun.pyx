#### run with:
#### python setup.py build_ext --inplace
####

####
#### computes cross-correlations based on convolution theorem (in cython for better performance)

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

                tfire = i
                voltage[i] = peak #spike
    return voltage
