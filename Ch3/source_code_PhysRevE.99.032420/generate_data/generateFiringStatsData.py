#### test firing statistics for LIF and EIF at signal turned off (MM and VM are the same)

import sys
sys.path.append("../code")
import time
from pylab import *
from MM_AM_LRT_mono import simulate_auto

#from tqdm import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

snr = array([0.])
tau_s = [10.]
w0_s = [0.]
########
### LIF
########
### ----------------------------------------------------------------------------------
#### tau_n = 0
### -----------------------------------------------------------------------------------
mu = 300.
tau_n = 0.
sigN_s = [200., 250., 300.]

name = 'firing-stats-LIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02, method = 1, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 2.5
### -----------------------------------------------------------------------------------
mu = 330.
tau_n = 2.5
sigN_s = [95, 130, 165]

name = 'firing-stats-LIF-tauN{:.1f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 5
### -----------------------------------------------------------------------------------
mu = 350.
tau_n = 5.
sigN_s = [40., 70., 100.]

name = 'firing-stats-LIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 10
### -----------------------------------------------------------------------------------
mu = 365.
tau_n = 10.
sigN_s =  [20, 45, 70]

name = 'firing-stats-LIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)

####################
###### EIF
####################
### ----------------------------------------------------------------------------------
#### tau_n = 0
### -----------------------------------------------------------------------------------
mu = 300.
tau_n = 0.
sigN_s = [350, 500, 650]

name = 'firing-stats-EIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 2.5
### -----------------------------------------------------------------------------------
mu = 300.
tau_n = 2.5
sigN_s = [175, 300, 425]

name = 'firing-stats-EIF-tauN{:.1f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 5
### -----------------------------------------------------------------------------------
mu = 310.
tau_n = 5.
sigN_s = [100, 200, 300]

name = 'firing-stats-EIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)


### ----------------------------------------------------------------------------------
#### tau_n = 10
### -----------------------------------------------------------------------------------
mu = 330.
tau_n = 10.
sigN_s = [75, 175, 275]

name = 'firing-stats-EIF-tauN{:.0f}'.format(tau_n)
print '{} started'.format(name)
a = [[[[simulate_auto(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 5000, N = 2e5, dt = .02,  method = 1, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]
a = array(a)
par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)
