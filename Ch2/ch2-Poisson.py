# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python [conda env:py27]
#     language: python
#     name: conda-env-py27-py
# ---

# # import relevant packages 

# +
# %pylab
import scipy.integrate as integrate
import seaborn as sns
sns.set_style('ticks')

import mpmath as mp
mp.dps = 200
# -

# # code and figures

# ## spectra and correlation functions for Ornstein-Uhlenbeck, uniform and telegraph rate processes

# +
def ou_spec(sigma, tau, w, process):
    if process == 'ou' or process == 'tele' or process == 'uni':
        return 1/(2*pi)*2*sigma**2*tau/(1+w**2*tau**2)
        #return sqrt(2/pi)*sigma**2*tau/(1+w**2*tau**2)
        #tau = tau/4.
        #return 2/(2*pi)*2*sigma**2*tau/(1+w**2*tau**2)

def cross(sigma, tau, w, process):
    return ou_spec(sigma, tau, w, process)

def auto(nu, sigma, tau, w, process):
    return ou_spec(sigma, tau, w, process)+nu

def gauss(x, nu, sigma):
    return 1./sqrt(2*pi*sigma**2)*exp(-(x-nu)**2/(2*sigma**2))

# mpmath for higher precision (I think obsolete due to closed form solutions)
def log2(x):
    return mp.log(x,2)
log2 = vectorize(log2)
# -

# ## class that calculates mutual information

# includes correlation method, info for ind. spikes and analytic limit; for both processes

class info():
    
    def __init__(self, nu, sigma, tau, process = 'ou'):
        self.nu = nu
        self.sigma = sigma
        self.tau = tau
        self.p = process
        
    def Poisson_info(self):
#        r, step = linspace(0, r_max, N, restep = 1)
        if self.p == 'ou':
            #return integrate.quad(lambda x: x/self.nu*log2(x/self.nu)*gauss(x, self.nu, self.sigma), 0, 4*self.nu)[0]
            return integrate.quad(lambda x: x/self.nu*log2(x/self.nu)*gauss(x, self.nu, self.sigma), 0, 100*self.sigma)[0]        
        if self.p == 'tele':
            x = array([self.nu-self.sigma, self.nu+self.sigma])
            return mean(x/self.nu*log2(x/self.nu)) 
            #        gauss = 1/sqrt(2*pi*sigma**2)*exp(-(r-nu)**2/(2*sigma**2))
        if self.p == 'uni':
            return (-2*sqrt(3)*self.nu*self.sigma - (self.nu**2 - 2*sqrt(3)*self.nu*self.sigma + 3*self.sigma**2)*log(1 - (sqrt(3)*self.sigma)/self.nu) + (self.nu**2 + 2*sqrt(3)*self.nu*self.sigma + 3*self.sigma**2)*log(1 + (sqrt(3)*self.sigma)/self.nu))/(sqrt(3)*self.nu*self.sigma*log(16))
    # old, obsolete
#    def Poisson_info2(self):
#        if self.p == 'ou':
#            return -.5*log2(2*pi*e*self.sigma**2)/self.nu - log2(self.nu)

    def corr_info(self, f_max = 0):
        if f_max == 0: 
            f_max = 1000./self.tau
            #print(f_max)
        return mp.quad(lambda x: -1./(1.*self.nu)*mp.log(1-cross(self.sigma, self.tau, x, self.p)/auto(self.nu, self.sigma, self.tau, x, self.p), 2), [0, f_max])#[0]
   
# analytic solution for exp decaying ACF
    def corr_info_exp(self):
        return (sqrt(pi)*self.sigma**2*self.tau + pi**1.5*self.nu - pi*sqrt(self.nu*(self.sigma**2*self.tau + pi*self.nu)))/(self.tau*sqrt(self.nu**3*(self.sigma**2*self.tau + pi*self.nu))*log(2))
    
# I_0
    def limit(self, s):
        return s**2/(2*log(2)*self.nu**2)
    #test for implementing DeWeese toy case
    def square_info(self):
        return 1.05-.28*(self.tau*self.nu)-6*(self.tau*self.nu)**2

# function that samples information values for different $\sigma_r$ and $\tau$

def scan_snr(nu, snr, tau, process = 'tele'):
    jo = info(nu, 1., 10., process)
    info_train = zeros((len(tau), len(snr)))
    info_spike = zeros((len(tau), len(snr)))
    for kx, x in enumerate(snr):
        for kt, t in enumerate(tau):
            jo.sigma = x
            jo.tau = t
            info_train[kt, kx] = jo.corr_info_exp()#1st version jo.corr_info()
            info_spike[kt, kx] = jo.Poisson_info()
    return array(info_train), array(info_spike)

# ## Fig 3: info as function of $\sigma_r$

nu = 2. #1st version: 1
snr = linspace(0.05, .75, 50)*nu
tau = array([.5, 5, 30])/nu #1st version [5., 10., 50.]

#1st version: ou in first line, tele in 2nd
info_train, info_spike = scan_snr(nu, snr, tau, 'tele')
#info_train_tele, info_spike_tele = scan_snr(nu, snr, tau, 'uni')

figure(figsize = (5.5,4.5))
for x in range(len(tau)):
    if x==0: semilogy(snr/nu, info_spike[x], label = r'$\mathcal{{I}}_\mathrm{{ind}}$'.format(tau[x]*nu))
    semilogy(snr/nu, info_train[x], label = r'$\mathcal{{I}}_\mathrm{{corr}}$, $\tau={}/\nu$'.format(tau[x]*nu), c = cm.Reds(.7*float(x)/len(tau)+.3))
semilogy(snr/nu, [x**2/(2*log(2)*nu**2) for x in snr], label = r'$\mathcal{{I}}_0$', ls = ':', c = 'grey')
legend()
ylabel("information [bits/spike]")
xlabel(r"rate modulation depth $\sigma_r/\nu$")
gca().set_yscale('linear')
tight_layout()
xlim([snr[0]/nu, snr[-1]/nu])
show()

# ## Fig 2: info as function of $\tau$

nu = 1.
tau2 = logspace(-2, 2, 40, endpoint=1)/nu
snr2 = [.5]#for 1st version: [.75]

info_trainTau, info_spikeTau = scan_snr(nu, snr2, tau2, 'uni') #1st version: OU
info_train_teleTau, info_spike_teleTau = scan_snr(nu, snr2, tau2, 'tele')
info_trainOUtau, info_spikeOUtau = scan_snr(nu, snr2, tau2, 'ou') #1st version: OU

# +
figure(figsize = (5.5,4.5))
snr_index = 0 #1st version 3
print(linspace(0.05, .75, 50)[snr_index])

plot(tau2, info_trainTau[:, snr_index], label = r'$\mathcal{{I}}^\mathrm{{uni}}_\mathrm{{corr}}$', c = sns.color_palette()[3])
plot(tau2, info_train_teleTau[:, snr_index], 'o', label = r'$\mathcal{{I}}^\mathrm{{tele}}_\mathrm{{corr}}$', c = sns.color_palette()[3],  linestyle='None', ms = 6)
plot(tau2, info_spikeTau[:, snr_index], label = r'$\mathcal{{I}}^\mathrm{{uni}}_\mathrm{{ind}}$', c = sns.color_palette()[0])
plot(tau2, info_spike_teleTau[:, snr_index], 'o', label = r'$\mathcal{{I}}^\mathrm{{tele}}_\mathrm{{ind}}$', c = sns.color_palette()[0],  linestyle='None', ms = 6)
# OUP
plot(tau2, info_trainOUtau[:, snr_index], 'x', label = r'$\mathcal{{I}}^\mathrm{{OU}}_\mathrm{{corr}}$', c = sns.color_palette()[3],  linestyle='None', ms = 5)
plot(tau2, info_spikeOUtau[:, snr_index], 'x', label = r'$\mathcal{{I}}^\mathrm{{OU}}_\mathrm{{ind}}$', c = sns.color_palette()[0],  linestyle='None', ms = 5)

axhline(snr2[snr_index]**2/(2*log(2)*nu**2), label = r'$I_0$', ls = ':', c = 'grey')
gca().set_xscale('log')
xlim([tau2[0], tau2[-1]])
xlabel(r'rate auto-correlation time $\tau$')
ylabel('information [bits/spike]')

legend(loc = 6, ncol = 3, )

tight_layout()
show()
# -

# ## $\nu$ as first axis

def infos(nu, sigma, tau, process = 'uni', f_max = 0):
    jo = info(nu, sigma, tau, process)
    return jo.corr_info_exp(), jo.Poisson_info(), jo.limit(sigma)

# **I removed the firing rate normalization of information (compared to dissertation figure)**

# +
nu_range = linspace(.01, 15, 50)
k_sigma = .5
infos_uni = array([infos(nu, k_sigma*nu, 10., 'uni') for nu in nu_range])
infos_tele = array([infos(nu, k_sigma*nu, 10., 'tele') for nu in nu_range])
figure()
plot(nu_range, infos_uni[:, 0]*nu_range, label = r'$\mathcal{{I}}^\mathcal{{uni}}_\mathcal{{corr}}$', c = sns.color_palette()[3])
plot(nu_range, infos_tele[:, 0]*nu_range, 'o', ls='-',label = r'$\mathcal{{I}}^\mathcal{{tele}}_\mathcal{{corr}}$', c = sns.color_palette()[3],  linestyle='None', ms = 6)
plot(nu_range, infos_uni[:, 1]*nu_range, label = r'$\mathcal{{I}}^\mathcal{{uni}}_\mathcal{{ind}}$', c = sns.color_palette()[0])
plot(nu_range, infos_tele[:, 1]*nu_range, 'o', label = r'$\mathcal{{I}}^\mathcal{{tele}}_\mathcal{{ind}}$', c = sns.color_palette()[0], ms = 6)

xlim([nu_range[0], nu_range[-1]])
xlabel(r'mean firing rate $\nu$ [a.u.] (with $\sigma_r/\nu={}$ fixed)'.format(k_sigma))
ylabel('information [bits/spike]')

ax1 = gca()
ax2 = ax1.twiny()
ax2.plot(k_sigma*nu_range, .18*np.ones(50), alpha = 0) # Create a dummy plot
ax2.set_xlabel('firing rate standard deviation $\sigma_r$ [a.u]')
#ax2.cla()

ax1.legend(loc = 'best')
tight_layout()
show()
# -

close('all')

# ## code for figure 1 (schematic)

def raster(event_times_list, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.25, **kwargs)
    plt.ylim(.5, len(event_times_list) + .5)
    return ax

def poisson_const(rate, dt, N):
    a = zeros(N)
    for x,X in enumerate(rate):
        if random.uniform()<X*dt:
            a[x] = 1
    return array([x*dt for x in nonzero(a)]) #spike times in ms

def regular_spikes(rate, total_time):
    spike_times = [x/(rate)+random.normal(0, .05/rate) for x in range(5)]
    return array(spike_times)

def bell_correlation(times,rate,width):
    return sum(array([exp(-(times-x/rate)**2/(2*(width+(x/1.6))**2)) for x in range(1,6)]), axis = 0)*exp(-times/rate*.00001)
#    return exp(-(times-1./rate)**2/(2*width**2))+exp(-(times-2./rate)**2/(2*width**2))+exp(-(times-3./rate)**2/(2*width**2))++exp(-(times-4./rate)**2/(2*width**2))+exp(-(times-5./rate)**2/(2*width**2))        

mpl.rcParams['axes.linewidth'] = 0.5 #set the value globally

# +
fig, axs =  subplots(3,3, figsize = (6, 3))

for ix in axs:
    for ax in ix: 
        ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.autoscale(enable=True, axis='x', tight=True)


ylabels = ['stimulus', 'spikes', 'spike\n auto-corr.']
titles = ['intrinsically induced\n correlations', 'independent spikes\n (correlation-free)', 'signal induced\n correlations']
for i in range(3):
    axs[2,i].set_xlabel('time')
    axs[i,0].set_ylabel(ylabels[i])
    axs[0,i].set_title(titles[i])
#    axs[0,i].autoscale(enable=True, axis='x', tight=True)
#    axs[2,i].autoscale(enable=True, axis='x', tight=True)

# plot rates/stimulus
axs[0,0].plot([1,1])
axs[0,1].plot([1,1])
axs[0,2].plot(cos(arange(0, 5.*2*pi, .1)))
axs[0,2].annotate("", xy=(62, 0.75), xycoords='data', xytext=(128, 0.75), textcoords='data',
                         arrowprops=dict(arrowstyle= '|-|, widthA=.4, widthB=.4',lw= 1., ls='-', connectionstyle= 'angle'))
axs[0,2].annotate("$\Delta$", xy=(0.1, 0.55), xycoords='data',
            xytext=((128+62)/2., 0.8), textcoords='data', horizontalalignment = 'center', size = 10.)

# spike rasters
rate = 10./1000.
dt = .1
T = .5*1000.
N = int(T/dt)
sca(axs[1,0])
raster([regular_spikes(rate, N*dt) for _ in range(6)], lw = 1.5)
axs[1,0].annotate("", xy=(100, 1.5), xycoords='data', xytext=(210, 1.5), textcoords='data',
                         arrowprops=dict(arrowstyle= '|-|, widthA=.4, widthB=.4', lw= 1., ls='-', connectionstyle= 'angle'), )
axs[1,0].annotate("$\Delta+\epsilon$", xy=(0.1, 0.55), xycoords='data',
            xytext=(155, 1.59), textcoords='data', horizontalalignment = 'center', fontsize = 8.,)
sca(axs[1,1])
raster([poisson_const(rate*ones(int(N)), dt, N) for _ in range(6)], lw = 1.5)
sca(axs[1,2])
raster([regular_spikes(rate, N*dt) for _ in range(6)], lw = 1.5)

axs[2,0].plot(bell_correlation(arange(0., 520., .01), .01, .05/.01), c = sns.color_palette()[2])
axs[2,0].annotate("", xy=(10000, 0.5), xycoords='data', xytext=(20000, 0.5), textcoords='data',
                         arrowprops=dict(arrowstyle= '|-|, widthA=.4, widthB=.4', lw= 1., ls='-', connectionstyle= 'angle'), )
axs[2,0].annotate("$\Delta$", xy=(0.1, 0.55), xycoords='data',
            xytext=(15000, 0.55), textcoords='data', horizontalalignment = 'center', size = 10.,)

axs[2,1].plot([1,1],c = sns.color_palette()[2])
axs[2,1].annotate(r"$\nu^2$", xy=(0.7, 1.02), xycoords='data',
            xytext=(0.92, 1.01), textcoords='data', horizontalalignment = 'center', size = 10.)


axs[2,2].plot(bell_correlation(arange(0., 520., .01), .01, .05/.01), c = sns.color_palette()[2])
axs[2,2].annotate("", xy=(10000, 0.5), xycoords='data', xytext=(20000, 0.5), textcoords='data',
                         arrowprops=dict(arrowstyle= '|-|, widthA=.4, widthB=.4', lw= 1., ls='-'), )
axs[2,2].annotate("$\Delta$", xy=(0.1, 0.55), xycoords='data', xytext=(15000, 0.55), textcoords='data', horizontalalignment = 'center', size = 10.,)
#sca(axs[2,2])
#axvline(x=0,c = sns.color_palette()[2])

tight_layout()
show()
# -

# # schematic for uni and tele process distributions

# +
fig,axs = subplots(2,2,)
axs[0,0].axvline(x = .35, ymax = .5, lw = 5)
axs[0,0].axvline(x = .65, ymax = .5, lw = 5)
axs[0,0].axvline(x=.5, ymax = 1, lw = 1, ls = 'dashed', c ='k')
axs[0,0].set_xlabel('firing rate')
axs[0,0].set_ylabel('probability')
axs[0,0].set_title('binary distribution')

axs[0,1].axhline(y=1/(3.*sqrt(3)), xmin=.5-sqrt(3)*.15, xmax=.5+sqrt(3)*.15, lw = 2)
axs[0,1].axvline(x=.5-sqrt(3)*.15, ymax = 1/(3.*sqrt(3)), lw = 2)
axs[0,1].axvline(x=.5+sqrt(3)*.15, ymax = 1/(3.*sqrt(3)), lw = 2)
axs[0,1].set_xlim(0,1)
axs[0,1].axvline(x=.5, ymax = 1, lw = 1, ls = 'dashed', c ='k')
axs[0,1].set_xlabel('firing rate')
axs[0,1].set_ylabel('probability')
axs[0,1].set_title('uniform distribution')

axs[1,0].plot(exp(-linspace(0,3.9,100)))
axs[1,0].set_xlim(0,None)
axs[1,0].tick_params(labelleft=False)
axs[1,0].set_yticks([])
axs[1,0].set_xlabel('time lag')
axs[1,0].set_ylabel('auto-correlation')
axs[1,0].set_title('exp. decaying correlation')


for ax in axs.flatten():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.set_xticks([])

tight_layout()
show()
