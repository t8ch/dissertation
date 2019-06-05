####
## This script generates the data that have been used to demonstrate
## the Gaussianity of the Fourier coefficients in signal and spike trains
####
import sys
sys.path.append("../code")
import time
from pylab import *
from gaussianity_check import spikes_check_complete, signal_check_complete
#from tqdm import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

snr = .25
tau_s = 20.
w0_s = 0.508
### ----------------------------------------------------------------------------------
#### tau_n = 0
### -----------------------------------------------------------------------------------
mu = 300.
tau_n = 0.
sigN_s = 250.

name = 'test-gaussianity-LIF-tauN{:.0f}-1'.format(tau_n)
print '{} started'.format(name)
a = signal_check_complete(mu = mu, sigN = sigN_s, sigS = snr, tS = [tau_s, tau_s*w0_s], tauN = tau_n, n = 500, N = 1e5, dt = .02, lim1 = 5000, lim2 = 300)

b = spikes_check_complete(mu = mu, sigN = sigN_s, sigS = snr, tS = [tau_s, tau_s*w0_s], tauN = tau_n, n = 500, N = 1e6, dt = .02, lim1 = 5000, lim2 = 300)

savez_compressed(name, sig = a, spi = b)
print '{} done'.format(name)

