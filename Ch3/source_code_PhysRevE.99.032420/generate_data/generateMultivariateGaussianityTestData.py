####
## This script generates the data that have been used to demonstrate
## the multivariate Gaussianity (and independence) of the Fourier coefficients in spike trains
####
import sys
sys.path.append("../code")
import time
from pylab import *
from gaussianity_check import normality_tests, merge_test_results
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


######
#### NINE FREQUENCIES AROUND OMEGA_0
#####
name = './multivariate-normality-test-around-Om0'
print ('{} started'.format(name))

b = array([normality_tests(mu = mu, sigN = sigN_s, sigS = snr, tS = [tau_s, tau_s*w0_s], tauN = tau_n, n = 500, N = 1e6, dt = .02, lim1 = range(1617-4,1617+5), lim2 = 9, index = k+1) for k in range(250)])

data,fake = merge_test_results(b[:,0]), merge_test_results(b[:,1])

savez_compressed(name, data = data, fake = fake)
print ('{} done'.format(name))
######
#### NINE RANDOM FREQUENCIES
######
name = './multivariate-normality-test-random-frequencies'
print ('{} started'.format(name))

b = array([normality_tests(mu = mu, sigN = sigN_s, sigS = snr, tS = [tau_s, tau_s*w0_s], tauN = tau_n, n = 500, N = 1e6, dt = .02, lim1 = 6000, lim2 = 9, verbose = 1, index = k+1) for k in range(250)])

data,fake = merge_test_results(b[:,0]), merge_test_results(b[:,1])

savez_compressed(name, data = data, fake = fake)
print ('{} done'.format(name))

