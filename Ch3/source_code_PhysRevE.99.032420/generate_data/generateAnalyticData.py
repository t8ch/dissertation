###
## This script generates the numerical evaluation of the analytic expressions of the
## linear response theory as described in the manuscript
###

import sys
sys.path.append("../code")
import time
from pylab import *
from MM_AM_LRT_mono import analytic_all
import multiprocessing

from mpmath import mp
mp.dps = 190 #numerical precision of hypergeometric functions

mu = 300.

snr = array([.15, .05, .25, .5, .75, 1., 2.])
sigN_a = arange(175., 325., 25)
tau_a = arange(5., 35., 5)
w0_a = append(linspace(0, 8, 64), logspace(-3, 1, 96))

par_a = {'snr': snr, 'sigN': sigN_a, 'tau': tau_a, 'w0' : w0_a}

##
# MM
##f
b = [[[[analytic_all(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s*mu, tS = [t_x, w0_x*t_x], tauN = 0., n = 800, N = 1e6, dt = .01, method = 0, omega_end = 1.5*w0_x+9) for w0_x in w0_a] for t_x in tau_a] for sigN_x in sigN_a] for s in snr]

b = array(b)
name = 'ana-mm'
savez_compressed(name, ana = b)
print '{} done'.format(name)

##
# AM
##
b = [[[[analytic_all(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = 0., n = 800, N = 1e6, dt = .01, method = 1, omega_end = 1.5*w0_x+9) for w0_x in w0_a] for t_x in tau_a] for sigN_x in sigN_a] for s in snr]

b = array(b)
name = 'ana-vm'
savez_compressed(name, ana = b)
print '{} done'.format(name)


