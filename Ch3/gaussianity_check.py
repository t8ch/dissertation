###
### checking normality, correlatedness, and multivariate Gaussianity (of spike trains)
###



from __future__ import division
from pylab import *
import fm
import multiprocessing
from joblib import Parallel, delayed
#import seaborn as sns
#import pandas
from scipy.stats import normaltest, norm, shapiro
from MM_AM_LRT_mono import check_Gauss_repeats, check_Gauss_uniques
import random

#####
##### FUNDAMENTAL FUNCTIONS
#####

### CHECKS FOR SIGNAL (SPIKE TRAINS IMPORTED)
def check_Gauss_AC_s(mu = 300., sigS = .25, sigN = 300, tS = [20, 10.], tauN = 0., n = 100, method = 1):
    if method == 0:
        g = fm.MM
    if method ==1:
        g = fm.AM_var
    N= int(fm.N)
    sAM = np.zeros([n, N])
    for x in range(n):
        sig = fm.mono_spec(tS[0], sigS, tS[1])
        sigAM = g(sig, fm.OU(tauN, sigN)) + mu
        sAM[x] = sigAM
    CC = rfft(sAM)
    CC = (transpose(real(CC)), transpose(imag(CC)))
    return CC

def check_Gauss_CC_s(sig, mu = 300., sigS = .25, sigN = 300, tS = [20, 10.], tauN = 0., n = 100, method = 1):
    if method == 0:
        g = fm.MM
    if method ==1:
        g = fm.AM_var
    N= int(fm.N)
    sAM = np.zeros([n, N])
    for x in range(n):
        sigAM = g(sig, fm.OU(tauN, sigN)) + mu
        sAM[x] = sigAM
    CC = rfft(sAM)
    CC = (transpose(real(CC)), transpose(imag(CC)))
    return CC


def check_Gauss_uniques_s(mu = 300., sigS = .25, sigN = 300, tS = [20, 10.], tauN = 0., 
                        n = 100, method = 1,  N = 5e5, dt = .02):
    fm.N = N
    fm.dt = dt
    fm.N2 = fm.N
    num_cores = multiprocessing.cpu_count()
    ac = Parallel(n_jobs=num_cores)(delayed(check_Gauss_AC_s)(mu, sigS, sigN, tS, tauN, n, method) for x in range(num_cores))
    return dstack(ac)

def check_Gauss_repeats_s(mu = 300., sigS = .25, sigN = 300, tS = [20, 10.], tauN = 0., 
                        n = 100, method = 1,  N = 5e5, dt = .02):
    fm.N = N
    fm.dt = dt
    fm.N2 = fm.N
    num_cores = multiprocessing.cpu_count()
    signal = fm.mono_spec(tS[0], sigS, tS[1])
    ac = Parallel(n_jobs=num_cores)(delayed(check_Gauss_CC_s)(signal, mu, sigS, sigN, tS, tauN, n, method) for x in range(num_cores))
    #return hstack(ac[0]), hstack(ac[1])
    return dstack(ac)

#### extract means of distributions
def get_means(a, lim):
    a = array(a)[:, 1:lim+1]
    return mean(a[0], 1), mean(a[1],1)
    #return mean(a[0], 1)/std(a[0], 1), mean(a[1],1)/std(a[1],1)

#### extract vars of distributions
def get_std(a, lim):
    a = array(a)[:, 1:lim+1]
    return std(a[0], 1), std(a[1],1)
    #return mean(a[0], 1)/std(a[0], 1), mean(a[1],1)/std(a[1],)

#### control distribution
def create_normal_from_dist(a, lim):
    N = shape(a)[-1]
    a = array(a)[:, 1:lim+1]
    ref1 = [np.random.normal(mean(x), std(x), N) for x in a[0]]
    ref2 = [np.random.normal(mean(x), std(x), N) for x in a[1]]
    return ref1, ref2

### compute corrcoef of data and control
def collect_corrcoef(a, lim1, lim2, ref):
    N = shape(a)[-1]
    lim2_ind = choice(int(lim1), int(lim2), replace = False)
    a1 = array(a)[:, 1:lim1+1]
    a2 = array(a1)[:, lim2_ind]
    l1, l2 = range(int(lim1)), range(int(lim2))
    ref01 = array(ref[0]) #array([np.random.normal(mean(x), std(x), N) for x in a1[0]])
    ref02 = array(ref[1]) #array([np.random.normal(mean(x), std(x), N) for x in a1[1]])
    ref1, ref2 = ref01[lim2_ind], ref02[lim2_ind]
    c_equal = []
    for k in l1: #@ equal frequencies
        c_equal= append(c_equal, corrcoef(array([a1[0][k], a1[1][k]]))[0,1])
    ref_equal = []
    for k in l1: #@ equal frequencies
        ref_equal= append(ref_equal, corrcoef(array([ref01[k], ref02[k]]))[0,1])

    c_mixed = []
    for k1 in l2: #all comb. @ non-equal frequencies
        for k2 in l2:
            if k1 > k2:
                c_mixed = append(c_mixed, corrcoef(array([a2[0][k1], a2[1][k2]]))[0,1])
                c_mixed = append(c_mixed, corrcoef(array([a2[0][k1], a2[0][k2]]))[0,1])
                c_mixed = append(c_mixed, corrcoef(array([a2[1][k1], a2[1][k2]]))[0,1])
    ref_mixed = []
    for k1 in l2: #all comb. @ non-equal frequencies
        for k2 in l2:
            if k1 > k2:
                ref_mixed = append(ref_mixed, corrcoef(array([ref1[k1], ref2[k2]]))[0,1])
                ref_mixed = append(ref_mixed, corrcoef(array([ref1[k1], ref1[k2]]))[0,1])
                ref_mixed = append(ref_mixed, corrcoef(array([ref2[k1], ref2[k2]]))[0,1])
    return append(c_equal, c_mixed), append(ref_equal, ref_mixed) #just put them together
    #return [c_equal, ref_equal], [c_mixed, ref_mixed]

### compute normaltest of data and control
def gauss_test(a, lim, ref):
    N = shape(a)[-1]
    a = array(a)[:, 1:lim+1]
    ref1 = ref[0]#[np.random.normal(mean(x), std(x), N) for x in a[0]]
    ref2 = ref[1]#[np.random.normal(mean(x), std(x), N) for x in a[1]]
    test_a = []
    test_ref = []
    for x in a[0]:
        test_a = append(test_a, normaltest(x)[0])
    print('1')
    for x in a[1]:
        test_a = append(test_a, normaltest(x)[0])
        #test_a = append(test_a, shapiro(x,)[0])
    print('2')
    for x in ref1:
        test_ref = append(test_ref, normaltest(x)[0])
    print('3')
    for x in ref2:
        test_ref = append(test_ref, normaltest(x)[0])
    print('4')
    return test_a, test_ref

def spikes_check_complete(R=.04, theta = 15., mu = 300., sigN = 300, tM = 10, Vres = 0, sigS = .25, tS = [20., 15.], tauN = 0, n = 100, N = 1e5, dt = .02, method = 1, ns = 1, model = 'LIF', DeltaT = 1.5, tref = 5., lim1 = 5000., lim2 = 300):
    num_cores = multiprocessing.cpu_count()

    un_raw = check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref), check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)
    
    re_raw = check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref), check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)

    #means_am = Parallel(n_jobs=num_cores)(delayed(get_means)(x, lim1) for x in [un_raw[1], re_raw[1]])
    #means_mm = Parallel(n_jobs=num_cores)(delayed(get_means)(x, lim1) for x in [un_raw[0], re_raw[0]])

    #std_am = Parallel(n_jobs=num_cores)(delayed(get_std)(x, lim1) for x in [un_raw[1], re_raw[1]])
    #std_mm = Parallel(n_jobs=num_cores)(delayed(get_std)(x, lim1) for x in [un_raw[0], re_raw[0]])

    ref_am = [create_normal_from_dist(x, lim1) for x in [un_raw[1], re_raw[1]]]
    ref_mm = [create_normal_from_dist(x, lim1) for x in [un_raw[0], re_raw[0]]]
    
    #print shape(zip([un_raw[1][:, 1:lim1+1], re_raw[1][:, 1:lim1+1]], ref_am))
    corr_am = [collect_corrcoef(x, lim1, lim2, y) for x, y  in zip([un_raw[1], re_raw[1]], ref_am)]
    corr_mm = [collect_corrcoef(x, lim1, lim2, y) for x, y  in zip([un_raw[0], re_raw[0]], ref_mm)]
    print ('done')

    test_am = [gauss_test(x, lim1, y) for x, y  in zip([un_raw[1], re_raw[1]], ref_am)]
    test_mm = [gauss_test(x, lim1, y) for x, y  in zip([un_raw[0], re_raw[0]], ref_mm)]
    
    #return {'means': [means_mm, means_am], 'std': [std_mm, std_am], 'control':[ref_am, ref_mm], 'corrcoeff':[corr_am, corr_mm], 'normtest':[test_am, test_mm]}# 'raw':[un_raw, re_raw],}
    return {'corrcoeff':[corr_am, corr_mm], 'normtest':[test_am, test_mm]}# 'raw':[un_raw, re_raw],}

#######
####### COMMENTED OUT SOME OUTPUTS (ABOVE AND BELOW) TO REDUCE FILE SIZE FOR LARGER LIM1
######

def signal_check_complete(mu = 300., sigS = .25, sigN = 300, tS = [20, 10.], tauN = 0., n = 100, method = 1,  N = 1e5, dt = .02, lim1 = 5000., lim2 = 300):
    num_cores = multiprocessing.cpu_count()
    
    un_raw = check_Gauss_uniques_s(mu, sigS*mu, sigN, tS, tauN,n, 0, N, dt), check_Gauss_uniques_s(mu, sigS, sigN, tS, tauN,n, 1, N, dt)
    re_raw = check_Gauss_repeats_s(mu, sigS*mu, sigN, tS, tauN,n, 0, N, dt), check_Gauss_repeats_s(mu, sigS, sigN, tS, tauN,n, 1, N, dt)

    #means_am = Parallel(n_jobs=num_cores)(delayed(get_means)(x, lim1) for x in [un_raw[1], re_raw[1]])
    #means_mm = Parallel(n_jobs=num_cores)(delayed(get_means)(x, lim1) for x in [un_raw[0], re_raw[0]])

    #std_am = Parallel(n_jobs=num_cores)(delayed(get_std)(x, lim1) for x in [un_raw[1], re_raw[1]])
    #std_mm = Parallel(n_jobs=num_cores)(delayed(get_std)(x, lim1) for x in [un_raw[0], re_raw[0]])

    ref_am = [create_normal_from_dist(x, lim1) for x in [un_raw[1], re_raw[1]]]
    ref_mm = [create_normal_from_dist(x, lim1) for x in [un_raw[0], re_raw[0]]]

    #print shape(zip([un_raw[1][:, 1:lim1+1], re_raw[1][:, 1:lim1+1]], ref_am))
    corr_am = [collect_corrcoef(x, lim1, lim2, y) for x, y  in zip([un_raw[1], re_raw[1]], ref_am)]
    corr_mm = [collect_corrcoef(x, lim1, lim2, y) for x, y  in zip([un_raw[0], re_raw[0]], ref_mm)]
    print ('done')

    test_am = [gauss_test(x, lim1, y) for x, y  in zip([un_raw[1], re_raw[1]], ref_am)]
    test_mm = [gauss_test(x, lim1, y) for x, y  in zip([un_raw[0], re_raw[0]], ref_mm)]
    

    #return {'means': [means_mm, means_am], 'std': [std_mm, std_am], 'control':[ref_am, ref_mm], 'corrcoeff':[corr_am, corr_mm], 'normtest':[test_am, test_mm]}#'raw':[un_raw, re_raw], }
    return {'corrcoeff':[corr_am, corr_mm], 'normtest':[test_am, test_mm]}#'raw':[un_raw, re_raw], }

#######
####### ARE SUPERPOISTIONS OF TWO FC GAUSSIAN (INDEPENDENCE CHECK)?
#######
def spikes_FC(R=.04, theta = 15., mu = 300., sigN = 300, tM = 10, Vres = 0, sigS = .25, tS = [20., 15.], tauN = 0, n = 100, N = 1e5, dt = .02, method = 1, ns = 1, model = 'LIF', DeltaT = 1.5, tref = 5., freqs = [3,4,5,6,7]):

    '''
    only generates Fourier coefficients
    '''
    un_raw = check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref)[:,freqs,:], check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)[:,freqs,:]
    
    re_raw = check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref)[:,freqs,:], check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)[:,freqs,:]

    fc = {}
    
    for k_mod,mod in enumerate(['mm','vm']):
        fc[mod] = {}
        for k_uni_rep,uni_rep in enumerate(['uni', 'rep']):
            fc[mod][uni_rep] = {}
            for k_part,part in enumerate(['real','imag']):
                fc[mod][uni_rep][part] = {}
                fc[mod][uni_rep][part] = array([[un_raw[0], re_raw[0]],[un_raw[1], re_raw[1]]])[k_mod,k_uni_rep,k_part]

    return fc#{'coeff': [[un_raw[0], re_raw[0]],[un_raw[1], re_raw[1]]]} #MM/VM, uni/rep,real/imag,freq, trial

def normality_tests(R=.04, theta = 15., mu = 300., sigN = 300, tM = 10, Vres = 0, sigS = .25, tS = [20., 15.], tauN = 0, n = 100, N = 1e5, dt = .02, method = 1, ns = 1, model = 'LIF', DeltaT = 1.5, tref = 5., lim1 = 10000, lim2 = 300, rnd_seed = 444, verbose = 1, index = 0):
    '''
    multivariate normality tests
    '''
    from copy import deepcopy

    r = random.Random(rnd_seed) # seed number is arbitrary 
    if size(lim1) > 1:
        freqs = r.sample(map(int,lim1), int(lim2) )
    else:
        freqs = r.sample(range(int(lim1)), int(lim2))
    print(freqs)
    
    # # select lim2 random frequencies from the first lim1
    # if size(lim1) > 1:
    #     freqs = r.choice(map(int,lim1), int(lim2), replace = False, )
    # else:
    #     freqs = r.choice(int(lim1), int(lim2), replace = False)
    # print(freqs)

    un_raw = check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref)[:,freqs,:], check_Gauss_uniques(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)[:,freqs,:]

    
    re_raw = check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref)[:,freqs,:], check_Gauss_repeats(R, theta, mu, sigN, tM, Vres, sigS, tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)[:,freqs,:]

    def normality_test_on_array(data):
        henze_score, henze_pvalue = multivariate_normality(data.T)[1:]
        mardias1, mardias2,  = Mardias_test(data)[:2]
        return henze_score, mardias1, mardias2, henze_pvalue #weird order but preserves previous order of first three

    def generate_surrogate(data):
        mu = data.mean(1, keepdims = 1)
        sigma = data.std(1, keepdims = 1)
        return np.random.normal(size = data.shape, loc = mu, scale = sigma)

    fc = {}
    for k_mod,mod in enumerate(['mm','vm']):
        fc[mod] = {}
        for k_uni_rep,uni_rep in enumerate(['uni', 'rep']):
            fc[mod][uni_rep] = {}
            for k_part,part in enumerate(['real','imag']):
                fc[mod][uni_rep][part] = {}
                
                fc[mod][uni_rep][part] = normality_test_on_array(array([[un_raw[0], re_raw[0]],[un_raw[1], re_raw[1]]])[k_mod,k_uni_rep,k_part])

    #same for surrogate data
    fc_sur  = deepcopy(fc)
    for k_mod,mod in enumerate(['mm','vm']):
        for k_uni_rep,uni_rep in enumerate(['uni', 'rep']):
            for k_part,part in enumerate(['real','imag']):                
                fc_sur[mod][uni_rep][part] = normality_test_on_array(generate_surrogate(array([[un_raw[0], re_raw[0]],[un_raw[1], re_raw[1]]])[k_mod,k_uni_rep,k_part]))

    if verbose:
        print('{} runs completed.'.format(index))
    
    return fc, fc_sur#{'coeff': [[un_raw[0], re_raw[0]],[un_raw[1], re_raw[1]]]} #MM/VM, uni/rep,real/imag,freq, trial

def multivariate_normality(X, alpha=.05):
    """Henze-Zirkler multivariate normality test.
    Parameters
    ----------
    X : np.array
        Data matrix of shape (n, p) where n are the observations and p the
        variables.
    alpha : float
        Significance level.
    Returns
    -------
    normal : boolean
        True if X comes from a multivariate normal distribution.
    p : float
        P-value.
    See Also
    --------
    normality : Test the univariate normality of one or more variables.
    homoscedasticity : Test equality of variance.
    sphericity : Mauchly's test for sphericity.
    Notes
    -----
    The Henze-Zirkler test has a good overall power against alternatives
    to normality and is feasable for any dimension and any sample size.
    Translated to Python from a Matlab code by Antonio Trujillo-Ortiz.
    Tested against the R package MVN.
    References
    ----------
    .. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
       tests for multivariate normality. Communications in Statistics-Theory
       and Methods, 19(10), 3595-3617.
    .. [2] Trujillo-Ortiz, A., R. Hernandez-Walls, K. Barba-Rojo and L.
       Cupul-Magana. (2007). HZmvntest: Henze-Zirkler's Multivariate
       Normality Test. A MATLAB file.
    Examples
    --------
    1. Test for multivariate normality of 2 variables
        >>> import numpy as np
        >>> from pingouin import multivariate_normality
        >>> np.random.seed(123)
        >>> mean, cov, n = [4, 6], [[1, .5], [.5, 1]], 30
        >>> X = np.random.multivariate_normal(mean, cov, n)
        >>> normal, p = multivariate_normality(X, alpha=.05)
        >>> print(normal, p)
            True 0.7523511059223078
    2. Test for multivariate normality of 3 variables
        >>> import numpy as np
        >>> from pingouin import multivariate_normality
        >>> np.random.seed(123)
        >>> mean, cov = [4, 6, 5], [[1, .5, .2], [.5, 1, .1], [.2, .1, 1]]
        >>> X = np.random.multivariate_normal(mean, cov, 50)
        >>> normal, p = multivariate_normality(X, alpha=.05)
        >>> print(normal, p)
            True 0.46074660317578175
    """
    from scipy.stats import lognorm
  
    # Check input
    X = np.asarray(X)
    assert X.ndim == 2
    n, p = X.shape
    assert p >= 2

    # Covariance matrix
    S = np.cov(X, rowvar=False, bias=True)
    S_inv = np.linalg.inv(S)
    difT = X - X.mean(0)
    # Squared-Mahalanobis distances
    Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
    Y = np.linalg.multi_dot([X, S_inv, X.T])
    Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + np.tile(np.diag(Y.T), (n, 1))

    # Smoothing parameter
    b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * (n**(1 / (p + 4)))
 
    if np.linalg.matrix_rank(S) == p:
        hz = n * (1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk))) - 2
                  * ((1 + (b**2))**(-p / 2)) * (1 / n)
                  * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
                  + ((1 + (2 * (b**2)))**(-p / 2)))
    else:
        hz = n * 4

    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2
    # Mean and variance
    mu = 1 - a**(-p / 2) * (1 + p * b**2 / a + (p * (p + 2)
                                                * (b**4)) / (2 * a**2))
    si2 = 2 * (1 + 4 * b**2)**(-p / 2) + 2 * a**(-p) *  (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4)) - 4 * wb**(-p / 2) * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))

    # Lognormal mean and variance
    pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
    psi = np.sqrt(np.log((si2 + mu**2) / mu**2))

    #print((si2 + mu**2) / mu**2)
    #print(hz, mu, si2, pmu, psi)

    # P-value
    pval = lognorm.sf(hz, psi, scale=np.exp(pmu))
    normal = True if pval > alpha else False

    # i think HZ can be used as score (pval can still be evaluated later)
    return normal, hz, pval

def Mardias_test(data):
    from scipy.stats import chi2
    #following wiki: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Multivariate_normality_tests
    n,k = data.shape[1], data.shape[0]
    sigma = cov(data)
    sigma_inverse = linalg.inv(sigma)
    # substract mean
    data = data - data.mean(-1, keepdims = 1)
#    A_matrix = array([[matmul(xj.T,matmul(sigma_inverse,xi))**3 for xj in data.T] for xi in data.T])
    A_matrix = np.linalg.multi_dot([data.T,sigma_inverse, data])**3
    A = 1./(6*n)*sum(A_matrix)
    B_matrix = array([matmul(xi.T,matmul(sigma_inverse,xi))**2 for xi in data.T])
    denom_B = sqrt(8*k*(k+2)*(n-3)*(n-k-1)*(n-k+1)/((n+3)*(n+5)))
#   B = sqrt(n/(8.*k*(k+2)))*(1./n*sum(B_matrix)-k*(k+2))
    B = (sum(B_matrix)-k*(k+2)*(n-1))/denom_B
    
    #p-values from scores
    df = 1./6*k*(k+1)*(k+2)
    A_pvalue = chi2.pdf(A, df)
    B_pvalue = 1/(sqrt(2*pi))*exp(-1./2.*B**2)

    #those p-values are BS
    return A,B,A_pvalue,B_pvalue

def merge_test_results(results):
    # to be applied for each result dictionry separately
    fc = {}
    for k_mod,mod in enumerate(['mm','vm']):
        fc[mod] = {}
        for k_uni_rep,uni_rep in enumerate(['uni', 'rep']):
            fc[mod][uni_rep] = {}
            for k_part,part in enumerate(['real','imag']):
                fc[mod][uni_rep][part] = {}
                fc[mod][uni_rep][part] = array([x[mod][uni_rep][part] for x in results])
    return fc
