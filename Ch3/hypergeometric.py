###
### helper functions for analytic calculations in linear response theory
###


#from lif import *
from pylab import *
from mpmath import *
#import fm
import sys
from joblib import Parallel, delayed

mp.dps = 170
mp.pretty = True

def gFunctionLIF(l, R, Vth, mu, sigma, tauM, Vres):
    return exp(((Vres/R - mu)**2 - (Vth/R - mu)**2)/(2*sigma**2/tauM))*pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vres/R))/pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vth/R))

def nu0(R_, theta_, mu_, sigN_, tauMe_, Vres_):
    return -1./diff(lambda l, R, Vth, mu, sigma, tauM, Vres:  exp(((Vres/R - mu)**2 - (Vth/R - mu)**2)/(2*sigma**2/tauM))*pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vres/R))/pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vth/R)), (0, R_,theta_, mu_, sigN_, tauMe_, Vres_), (1,0,0,0,0,0,0))

def deltaT(R, theta, mu, sigN, tauMe, Vres):
    upper = (theta/R-mu)*sqrt(tauMe)/(sigN)
    lower = (Vres/R-mu)*sqrt(tauMe)/(sigN)
    f = lambda z: quad(lambda y: exp(y**2)*(1+erf(y))**2, [z, inf], maxdegree = 4)
    integral = quad(f, [lower, upper], maxdegree = 4)#[0]
    T = 2*pi*integral
    return T

## for white noise only

def VarT(R_, theta_, mu_, sigN_, tauMe_, Vres_):
    return diff(lambda l, R, Vth, mu, sigma, tauM, Vres:  exp(((Vres/R - mu)**2 - (Vth/R - mu)**2)/(2*sigma**2/tauM))*pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vres/R))/pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vth/R)), (0, R_,theta_, mu_, sigN_, tauMe_, Vres_), (2,0,0,0,0,0,0))- diff(lambda l, R, Vth, mu, sigma, tauM, Vres:  exp(((Vres/R - mu)**2 - (Vth/R - mu)**2)/(2*sigma**2/tauM))*pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vres/R))/pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vth/R)), (0, R_,theta_, mu_, sigN_, tauMe_, Vres_), (1,0,0,0,0,0,0))**2

def CV(R_, theta_, mu_, sigN_, tauMe_, Vres_):
    nu = nu0(R_, theta_, mu_, sigN_, tauMe_, Vres_)
    c = diff(lambda l, R, Vth, mu, sigma, tauM, Vres:  exp(((Vres/R - mu)**2 - (Vth/R - mu)**2)/(2*sigma**2/tauM))*pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vres/R))/pcfd(-l*tauM, sqrt(2)/sigma*sqrt(tauM)*(mu - Vth/R)), (0, R_,theta_, mu_, sigN_, tauMe_, Vres_), (2,0,0,0,0,0,0)) - 1/nu**2
    cv = sqrt(c)*nu
    return nu, cv

def S_oup(sigS, tauS, w):
    return (2*sigS**2*tauS)/(1+w**2*tauS**2)

def S_sqrt(sig, tau, w, w0):
    return sig**2*tau*sqrt(pi)/2.*(exp(-1/4.*(w0+w)**2*tau**2)+exp(-1/4.*(w0-w)**2*tau**2))

def S_mono(sig, tau, w, w0):
    return sig**2*tau*(1/(1+(w0-tau*w)**2)+1/(1+(w0+tau*w)**2))

def Sst(*args):
    if len(args) == 3: return S_oup(args[0], args[1], args[2])
    #if len(args) == 4: return S_sqrt(args[0], args[1], args[2], args[3])
    if len(args) == 4: return S_mono(args[0], args[1], args[2], args[3])

def U(a,b,y):
    return (hyp1f1(a, b, y**2, eps=e-5)*fp.gamma(1-b)/fp.gamma(a-b+1) - y**(2-2*b)*fp.hyp1f1(a-b+1, 2-b, y**2)*fp.gamma(b-1)/fp.gamma(a))/sqrt(pi)

def dU(a,b,y):
    return -2*y*a*U(a+1, b+1, y)

def d2U(a,b,y):
    return -2*a*U(a+1, b+1, y) + 4*y**2*a*(a+1)*U(a+2, b+2,y)

def C0(R, theta, mu, sig, tauM, Vr, w):
    delta = (Vr**2 - theta**2 + 2*R*mu*(theta - Vr))/(2*R**2*sig**2/tauM)
    dR = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - Vr/R)/sig)
    dT = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - theta/R)/sig)
    return nu0(R, theta, mu, sig, tauM, Vr)*(abs(dT)**2-exp(2*delta)*abs(dR)**2)/(abs(dT-exp(delta)*dR)**2)

#USING U INSTEAD OF PARABOLIC CYLINDER

def chiM(R, theta, mu, sig, t, Vr, w):
    delta = (Vr**2 - theta**2 + 2*R*mu*(theta - Vr))/(2*R**2*sig**2/t)
    dR = pcfd(j*w*t, sqrt(2*t)*(mu - Vr/R)/sig)
    dT = pcfd(j*w*t, sqrt(2*t)*(mu - theta/R)/sig)
    dR1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - Vr/R)/sig)
    dT1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - theta/R)/sig)
    return nu0(R, theta, mu, sig, t, Vr)*j*w*t*sqrt(2*t)/(sig*(j*w*t-1))*((dT1 - exp(delta)*dR1)/(dT-exp(delta)*dR))

def chiA(R, theta, mu, sig, t, Vr, w):
    delta = (Vr**2 - theta**2 + 2*R*mu*(theta - Vr))/(2*R**2*sig**2/t)
    dR = pcfd(j*w*t, sqrt(2*t)*(mu - Vr/R)/sig)
    dT = pcfd(j*w*t, sqrt(2*t)*(mu - theta/R)/sig)
    dR2 = pcfd(j*w*t-2, sqrt(2*t)*(mu - Vr/R)/sig)
    dT2 = pcfd(j*w*t-2, sqrt(2*t)*(mu - theta/R)/sig)
    return nu0(R, theta, mu, sig, t, Vr)*j*w*t*(j*w*t-1)/(sig**2*(2-j*w*t))*((dT2 - exp(delta)*dR2)/(dT-exp(delta)*dR))

def corr_combined(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, method = 0):
    if method == 0:
        g = CcrossMM
    if method == 1:
        g = CcrossAM
    cross = g(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w)
    c0 = C0(R, theta, mu, sigN, tauM, Vr, w)
    auto = cross + c0
    return auto, cross


def MI(R, theta, mu, sig, tauM, Vr, sigS, tauS, w, method = 0):
    t = tauM
    delta = (Vr**2 - theta**2 + 2*R*mu*(theta - Vr))/(2*R**2*sig**2/tauM)
    dR = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - Vr/R)/sig)
    dT = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - theta/R)/sig)
    nu = nu0(R, theta, mu, sig, t, Vr)
    if method == 0:
        dR1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - Vr/R)/sig)
        dT1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - theta/R)/sig)
        g = j*w*t*sqrt(2*t)/(sig*(j*w*t-1))*((dT1 - exp(delta)*dR1))#/(dT-exp(delta)*dR))
        ss = Sst(sigS, tauS[0], w, tauS[1])
    if method == 1:
        dR2 = pcfd(j*w*t-2, sqrt(2*t)*(mu - Vr/R)/sig)
        dT2 = pcfd(j*w*t-2, sqrt(2*t)*(mu - theta/R)/sig)
        g = j*w*t*(j*w*t-1)/(sig**2*(2-j*w*t))*((dT2 - exp(delta)*dR2))#/(dT-exp(delta)*dR))
        ss = Sst(sigS*sig**2, tauS[0], w, tauS[1])
    return .5*log2(1 + abs(g)**2*ss*nu/(abs(dT)**2-exp(2*delta)*abs(dR)**2), dtype = double)#/(abs(dT-exp(delta)*dR)**2)

'''
def MI2(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, method = 0):
    delta = (Vr**2 - theta**2 + 2*R*mu*(theta - Vr))/(2*R**2*sig**2/tauM)
    dR = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - Vr/R)/sig)
    dT = pcfd(j*w*tauM, sqrt(2*tauM)*(mu - theta/R)/sig)
    if method == 0:
        dR1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - Vr/R)/sig)
        dT1 = pcfd(j*w*t-1, sqrt(2*t)*(mu - theta/R)/sig)
        g = j*w*t*sqrt(2*t)/(sig*(j*w*t-1))*((dT1 - exp(delta)*dR1))#/(dT-exp(delta)*dR))
    if method == 1:
        dR1 = pcfd(j*w*t-2, sqrt(2*t)*(mu - Vr/R)/sig)
        dT1 = pcfd(j*w*t-2, sqrt(2*t)*(mu - theta/R)/sig)
        g = j*w*t*(j*w*t-1)/(sig**2*(2-j*w*t))*((dT2 - exp(delta)*dR2))#/(dT-exp(delta)*dR))    
    return .5*log2(1 + abs(g)**2/(abs(dT)**2-exp(2*delta)*abs(dR)**2), dtype = double)#/(abs(dT-exp(delta)*dR)**2)
    

def MI(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w, method = 0):
    if method == 0:
        g = CcrossMM
    if method == 1:
        g = CcrossAM
    cross = g(R, theta, mu, sigN, tauM, Vr, sigS, tauS, w)
    c0 = C0(R, theta, mu, sigN, tauM, Vr, w)
    return .5*log2(1 + cross/c0, dtype = double)
'''
def MI_int(R, theta, mu, sigN, tauM, Vr, sigS, tauS, method, w_end, w_start = 0):
    return quad(lambda x: MI(R, theta, mu, sigN, tauM, Vr, sigS, tauS, x, method,), [w_start, w_end], maxdegree = 6, verbose = True )
