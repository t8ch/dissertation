###
### calculating correlation functions and information for combined encoding (simulation and analytic)
###


from pylab import *
#from lif import *
from hypergeometric import nu0, U, dU, d2U, C0, chiM, chiA, Sst, S_mono, MI
from MM_AM_LRT_mono import CcrossAM, CcrossMM, CautoAM, simulate_all
import fm
from AmAutoAll import auto_corr #, isi, get_isi2_parallel
#from AmCrossAll import corr
from corr_fun import corr
import multiprocessing
from joblib import Parallel, delayed
from scipy.integrate import cumtrapz
from scipy.stats import gmean

#import pyximport
#pyximport.install()
import timecourse as tc

#import numeric MI integration from cython module
import MI_integrator as mi
MI_ana = mi.MI_int_ex
MI_ana_w = mi.MI_w_par

####
#ANALYTIC PART
########
def CcrossComb(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, mod1 = 0, mod2 = 1):
    '''return cross correlation for combined input
    
    mod1, mod2 -- used modulation (either 0 or 1 for MM/VM)    
    '''
    # sigN is total noise, must therefore be divided by sqrt(2) for each channel
    S1 = Sst(sigS[0], tauS[0], w, tauS[1])
    S2 = Sst(sigS[1], tauS[0], w, tauS[1])
    if mod1 == 0:
        chi1 = mu*chiM(R, theta, mu, sigN, tauM, Vr, w)
    if mod1 == 1:
        chi1 = sigN**2/2.*chiA(R, theta, mu, sigN, tauM, Vr, w)
    if mod2 == mod1:
        chi2 = chi1
    else:
        if mod2 == 0:
            chi2 = mu*chiM(R, theta, mu, sigN, tauM, Vr, w)
        if mod2 ==1:
            chi2 = sigN**2/2.*chiA(R, theta, mu, sigN, tauM, Vr, w)
    c1 = abs(chi1)**2
    c2 = abs(chi2)**2
    cMix = 2*(conjugate(chi1)*chi2).real
    return c1*S1, c2*S2, c1*S1+c2*S2+cMix*sqrt(S1*S2)

def CautoComb(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, mod1 = 0, mod2 = 1):
    c = CcrossComb(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, mod1, mod2)
    S0 = C0(R, theta, mu, sigN, tauM, Vr, w)
    return c+S0

#cross and auto for combined case
def combined_all(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, mod1 = 0, mod2 = 1):
    c = CcrossComb(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, mod1, mod2)
    S0 = C0(R, theta, mu, sigN, tauM, Vr, w)
    a = [S0 + x for x in c]
    return a, c

def analyticCombAndSingle(R=.04, theta = 15., mu = 300., sigN = 300, tM = 10, Vres = -5., sigS = [.1, .1], tS = [20., 0.], tauN = 0., n = 180, N = 5e5, dt = .02, mod1 = 0, mod2 = 1, ns = 1, omega_start = 0, omega_end = 4.):
    if mod1 != 0 or mod2 != 1: raise ValueError('mod1 is not MM or mod2 is not VM')
    fm.N = N
    fm.dt = dt
    tauM = tM
    ## omega_e = omega_end*fm.N*fm.dt/(2*pi) #choose Omega0+sqrt(prec)/sqrt(tauS)
    ## omega_s = omega_start*fm.N*fm.dt/(2*pi)+1
    ## t = [2*pi*x/(fm.N*fm.dt) for x in linspace(omega_s, omega_e, n)]
    t = linspace(omega_start+2*pi/(fm.N*fm.dt), omega_end, n)
    num_cores = multiprocessing.cpu_count()
    print "analytic ACF/CCF"
    ana =  Parallel(n_jobs=num_cores)(delayed(combined_all)(R, theta, mu, sigN, tauM, Vres, sigS, tS, x, mod1, mod2) for x in t)
    ana = swapaxes(array(ana), 0, 2)
    res = {}
    #only valid if mod1 = 0 and mod2 = 1
    for ind, name in enumerate(['mm','vm', 'comb']):
        ac_ana, cc_ana = ana[ind]
        mi = cumtrapz(-.5*log2(double(1 - array(cc_ana)/array(ac_ana))), dx = t[1]-t[0])
        res[name] =  double(ac_ana), double(cc_ana), double(t), mi
    print "analytic ACF/CCF done"
    print "analytic MI (cumtrapz)"
    return res

####
#SIMULATION PART
########
def CC_sim_comb(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300., tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, mod1 = 0, mod2 = 1, model = 'LIF', DeltaT = 1.5, tref = 5.):
    if model == 'LIF': cell = tc.SimpleIF(tM, R, 0., ures = Vres, V_thr = theta)
    if model == 'EIF': cell = tc.EIF(tM, R, 0., ures = Vres, V_thr = theta, DeltaT= DeltaT, tref=tref)
    #generate mods, voltages and spike trains for cross corr analysis
    sig = fm.mono_spec(tS[0], 1., tS[1])
    #add factor mu for signal in MM such that mu(1+s) is implemented
    pre = [mu, 1]
    sig1, sig2 = sig*pre[mod1]*sigS[0], sig*pre[mod2]*sigS[1]
    g1 = [fm.MM, fm.AM_var][mod1]
    g2 = [fm.MM, fm.AM_var][mod2]
    N2 = int(fm.dt*fm.N/fm.dt2)
    sAM = np.zeros([n, N2])
    for x in range(n):    
        # total noise is sigN
        sigAM = g1(sig1, fm.OU(tauN, sigN/sqrt(2.))) + g2(sig2, fm.OU(tauN, sigN/sqrt(2.))) + mu
        cell._Vinit = R*mu
        cell.voltagecourse(sigAM[::-1][:int(3*tS[0]/fm.dt)])
        cell._Vinit = cell._vol
        sAM[x] = cell.spiketrain2(sigAM)
    #print "calculate CCF"
    CC = array(corr(sAM, 1, 1))/(fm.N2*fm.dt2**2)
    #print 'single CCF done'
    return CC

def AC_sim_comb(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300., tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, mod1 = 0, mod2 = 1, model = 'LIF', DeltaT = 1.5, tref = 5.):
    if model == 'LIF': cell = tc.SimpleIF(tM, R, 0., ures = Vres, V_thr = theta)
    if model == 'EIF': cell = tc.EIF(tM, R, 0., ures = Vres, V_thr = theta, DeltaT= DeltaT, tref=tref)
    #generate mods, voltages and spike trains for cross corr analysis
    g1 = [fm.MM, fm.AM_var][mod1]
    g2 = [fm.MM, fm.AM_var][mod2]
    pre = [mu, 1]
    N2 = int(fm.dt*fm.N/fm.dt2)
    sAM = np.zeros([n, N2])
    for x in range(n):    
        sig = fm.mono_spec(tS[0], 1., tS[1])
        #add factor mu for signal in MM such that mu(1+s) is implemented
        sig1, sig2 = sig*pre[mod1]*sigS[0], sig*pre[mod2]*sigS[1]
        # total noise is sigN
        sigAM = g1(sig1, fm.OU(tauN, sigN/sqrt(2.))) + g2(sig2, fm.OU(tauN, sigN/sqrt(2.))) + mu
        cell._Vinit = R*mu
        cell.voltagecourse(sigAM[::-1][:int(3*tS[0]/fm.dt)])
        cell._Vinit = cell._vol
        sAM[x] = cell.spiketrain2(sigAM)
    #print "calculate ACF"
    AC = array(auto_corr(sAM, 1, 1))/(fm.N2*fm.dt2**2)
    #print 'single ACF done'
    return AC

def simulate_all_comb(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300, tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, ns = 1, mod1 = 0, mod2 = 1, N = 5e5, dt = .02, model = 'LIF', DeltaT = 1.5, tref = 5.):
    fm.N = N
    fm.dt = dt
    fm.N2 = int(fm.dt*fm.N/fm.dt2)
    num_cores = multiprocessing.cpu_count()
    ac = Parallel(n_jobs=num_cores)(delayed(AC_sim_comb)(R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref) for x in range(num_cores))
    ac_sim_f = mean([x[0] for x in ac], 0)
    #print 'AC sim. done'
    cc =  Parallel(n_jobs=num_cores)(delayed(CC_sim_comb)(R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref) for x in range(ns*num_cores))
    cc_sim_f = mean([x[0] for x in cc], 0)
    #print 'CC sim. done'
    t = [2*pi*x/(fm.N*fm.dt) for x in range(len(cc_sim_f))]
    #mutual information added
    mi = cumtrapz(-.5*nan_to_num(log2(1 - array(cc_sim_f[1:])/array(ac_sim_f[1:]), dtype = double)), dx = 2*pi/(fm.N*fm.dt))
    ####used to calculate the properly averaged MI through 1/2*log(auto/sigma_Rs)
    #sigma_Rs = gmean([x[0] for x in array(ac)-array(cc)], 0)
    return ac_sim_f*fm.dt2, cc_sim_f*fm.dt2, float16(t), float16(mi), #sigma_Rs

#parallel computing and correct normalization + freq axis
def simulate_cross_comb(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300, tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, mod1 = 0, mod2 = 1, N = 5e5, dt = .02, model = 'LIF', DeltaT = 1.5, tref = 5.):
    fm.N = N
    fm.dt = dt
    fm.N2 = int(fm.dt*fm.N/fm.dt2)
    num_cores = multiprocessing.cpu_count()
    cc =  Parallel(n_jobs=num_cores)(delayed(CC_sim_comb)(R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref) for x in range(num_cores))
    cc_sim_f = mean([x[0] for x in cc], 0)
    print 'CC sim. done'
    t = [2*pi*x/(fm.N*fm.dt) for x in range(len(cc_sim_f))]
    return cc_sim_f*fm.dt2, t

def simulate_single_and_comb(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300, tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, ns = 1, mod1 = 0, mod2 = 1, N = 5e5, dt = .02, model = 'LIF', DeltaT = 1.5, tref = 5.):
    res = {}
    res['comb'] = simulate_all_comb(R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, ns, mod1, mod2, N, dt, model, DeltaT, tref)
    print("comb sim. done")
    res['vm'] = simulate_all_comb(R, theta, mu, [0, sigS[1]], sigN, tS, tM, Vres, tauN, n, ns, mod1, mod2, N, dt, model, DeltaT, tref)
    print("vm sim. done")
    res['mm'] = simulate_all_comb(R, theta, mu, [sigS[0], 0], sigN, tS, tM, Vres, tauN, n, ns, mod1, mod2, N, dt, model, DeltaT, tref)
    print("mm sim. done")
    #use 'old' implementations here (factor mu neccessary for MM)
    ## res['mm'] = simulate_all(R, theta, mu, sigN, tM, Vres, sigS[0]*mu, tS, tauN, n, N, dt, 0, ns, model, DeltaT, tref)
    ## res['vm'] = simulate_all(R, theta, mu, sigN, tM, Vres, sigS[1], tS, tauN, n, N, dt, 1, ns, model, DeltaT, tref)
    return res


##############################
### FUNCTIONS THAT EVALUATE RATE AND FF FOR A GIVEN SIGNAL
##############################


def rate_and_var(sig, R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300., tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, mod1 = 0, mod2 = 1, model = 'LIF', DeltaT = 1.5, tref = 5., bin_size = 1):
    if model == 'LIF': cell = tc.SimpleIF(tM, R, 0., ures = Vres, V_thr = theta)
    if model == 'EIF': cell = tc.EIF(tM, R, 0., ures = Vres, V_thr = theta, DeltaT= DeltaT, tref=tref)
    #generate mods, voltages and spike trains for cross corr analysis
    #sig = fm.mono_spec(tS[0], 1., tS[1])
    #add factor mu for signal in MM such that mu(1+s) is implemented
    pre = [mu, 1]
    sig1, sig2 = sig*pre[mod1]*sigS[0], sig*pre[mod2]*sigS[1]
    g1 = [fm.MM, fm.AM_var][mod1]
    g2 = [fm.MM, fm.AM_var][mod2]
    N2 = int(fm.dt*fm.N/fm.dt2)
    sAM = np.zeros([n, N2])
    for x in range(n):    
        # total noise is sigN
        #sigAM = g1(sig1, fm.OU(tauN, sigN/sqrt(2.))) + g2(sig2, fm.mono_spec(100., sigN/sqrt(2.), 100.*2*pi*.07)) + mu
        sigAM = g1(sig1, fm.OU(tauN, sigN/sqrt(2.))) + g2(sig2, fm.OU(tauN, sigN/sqrt(2.))) + mu
        cell._Vinit = R*mu
        cell.voltagecourse(sigAM[::-1][:int(3*tS[0]/fm.dt)])
        cell._Vinit = cell._vol
        sAM[x] = cell.spiketrain2(sigAM)
    if bin_size>0:
        bin_width = int(bin_size/fm.dt2)
        sAM = array([[mean(x[i:i+bin_width]) for i in xrange(0, len(sAM[0]), bin_width)] for x in sAM])
    return sAM

def rate_and_var_par(R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300, tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, ns = 1, mod1 = 0, mod2 = 1, N = 5e5, dt = .02, model = 'LIF', DeltaT = 1.5, tref = 5., bin_size = 10.):
    fm.N = N
    fm.dt = dt
    fm.N2 = int(fm.dt*fm.N/fm.dt2)
    sig = fm.mono_spec(tS[0], 1., tS[1])
    num_cores = multiprocessing.cpu_count()
    rv = Parallel(n_jobs=num_cores)(delayed(rate_and_var)(sig, R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref) for x in range(num_cores))
    rv = vstack(asarray(rv, uint8))
    bin_width = int(bin_size/fm.dt2)
    binned = array([[mean(x[i:i+bin_width]) for i in xrange(0, len(rv[0]), bin_width)] for x in rv])
    sig_binned =  array([mean(sig[i:i+bin_width]) for i in xrange(0, len(sig), int(bin_size/fm.dt))])
    rate, vari = mean(binned, 0), var(binned, 0)
    print 'rate_and_var done'
    return rv, sig, sig_binned, binned, rate, vari

###### used to generate spike trains and binned signal for decoding ######
##### MM, VM and comb ####
def spike_trains_all_comb(sig = 'None', R=.04, theta = 15., mu = 300., sigS = [.1, .1], sigN = 300, tS = [20, 0], tM = 10, Vres = -5, tauN = 0., n = 100, ns = 1, mod1 = 0, mod2 = 1, N = 5e5, dt = .02, model = 'LIF', DeltaT = 1.5, tref = 5., bin_size = 5):
    fm.N = N
    fm.dt = dt
    fm.N2 = int(fm.dt*fm.N/fm.dt2)
    if sig == 'None': sig = fm.mono_spec(tS[0], 1., tS[1])
    num_cores = multiprocessing.cpu_count()
    res = {}
    rv = Parallel(n_jobs=num_cores)(delayed(rate_and_var)(sig, R, theta, mu, sigS, sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref, bin_size) for x in range(num_cores))
    res['comb'] = vstack(asarray(rv, float16))
    print("comb sim. done")
    rv = Parallel(n_jobs=num_cores)(delayed(rate_and_var)(sig, R, theta, mu, [0, sigS[1]], sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref, bin_size) for x in range(num_cores))
    res['vm'] = vstack(asarray(rv, float16))
    print("vm sim. done")
    rv = Parallel(n_jobs=num_cores)(delayed(rate_and_var)(sig, R, theta, mu, [sigS[0], 0], sigN, tS, tM, Vres, tauN, n, mod1, mod2, model, DeltaT, tref, bin_size) for x in range(num_cores))
    res['mm'] = vstack(asarray(rv, float16))
    print("mm sim. done")
    bin_width = int(bin_size/fm.dt)
    sig_binned =  array([mean(sig[i:i+bin_width]) for i in xrange(0, len(sig), bin_width)])
    print 'rate_and_var done'
    return res, sig_binned#,sig, binned, rate, vari


## smaller files to be saved
def binary_to_times(spikes, dt):
    """
    maps spike trains represented with 0s and 1s to spike times
    Params
    --------------

    """
    return array([argwhere(spikes[:, x]).ravel()*dt for x in range(shape(spikes)[1])])
