#####
### This script generates the correlation functions and information for the parameters as used
### in the manuscript - for both LIF and EIF models.
#####
import sys
sys.path.append("../code")
import time
from pylab import *
from MM_AM_LRT_mono import combined_par

#from tqdm import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

snr = array([.15, .05, .25, .5, .75, 1., 2.])
tau_s = [10., 20., 30.]
w0_s = [0, 0.25, .508, 1., 2.54, 7.111]

##
### ***LIF***
##
### ----------------------------------------------------------------------------------
#### set values for tau_n and corresponding mu,sigN to be sampled
### -----------------------------------------------------------------------------------
mu = [300., 330., 350., 365.]
tau_n = [0., 2.5, 5., 10.]
sigN_s = [[200., 250., 300.], [95., 130., 165.], [40., 70., 100.], [20., 45., 70.]]

for mu, tau_n, sigN_s in zip(mu, tau_n, sigN_s):
    ##
    # MM
    ##
    name = 'sim-mm-LIF-tauN{:.0f}'.format(tau_n)
    print '{} started'.format(name)
    a = [[[[combined_par(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = mu*s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 4000, N = 2e5, dt = .02, ns = 2, method = 0, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

    a = array(a)
    par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
    savez_compressed(name, sim = a, par = par_s)
    print '{} done'.format(name)

    ##
    # VM
    ##
    name = 'sim-vm-LIF-tauN{:.0f}'.format(tau_n)
    print '{} started'.format(name)
    a = [[[[combined_par(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 4000, N = 2e5, dt = .02, ns = 2, method = 1, model = 'LIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

    a = array(a)
    par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
    savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)

##
### ***EIF***
##
### ----------------------------------------------------------------------------------
#### set values for tau_n and corresponding mu,sigN to be sampled
### -----------------------------------------------------------------------------------

mu = [300., 300., 310., 330.]
tau_n = [0., 2.5, 5., 10.]
sigN_s = [[350., 500., 650.], [175., 300., 425.], [100., 200., 300.], [75., 175., 275.]]

for mu, tau_n, sigN_s in zip(mu, tau_n, sigN_s):
    ##
    # MM
    ##
    name = 'sim-mm-EIF-tauN{:.0f}'.format(tau_n)
    print '{} started'.format(name)
    a = [[[[combined_par(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = mu*s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 4000, N = 2e5, dt = .02, ns = 2, method = 0, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

    a = array(a)
    par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
    savez_compressed(name, sim = a, par = par_s)
    print '{} done'.format(name)

    ##
    # VM
    ##
    name = 'sim-vm-EIF-tauN{:.0f}'.format(tau_n)
    print '{} started'.format(name)
    a = [[[[combined_par(R=.04, theta = 15., mu = mu, sigN = sigN_x, tM = 10., Vres = 0, sigS = s, tS = [t_x, w0_x*t_x], tauN = tau_n, n = 4000, N = 2e5, dt = .02, ns = 2, method = 1, model = 'EIF') for w0_x in w0_s] for t_x in tau_s] for sigN_x in sigN_s] for s in snr]

    a = array(a)
    par_s = {'snr': snr, 'sigN': sigN_s, 'tau': tau_s, 'w0' : w0_s}
    savez_compressed(name, sim = a, par = par_s)
print '{} done'.format(name)
