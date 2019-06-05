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

from __future__ import division
from scipy.integrate import cumtrapz
import sys
sys.path.append('../code')
sys.path.append('../generate_data/')
from pylab import *
from signalsmooth import smooth
import seaborn as sns
sns.set_style('white')
sns.set(style = "ticks", color_codes = True)
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# global plotting settings

from style_sheet import custom_style, set_inset_params
custom_style(paper = True)
#sns.set_context("paper")

# %matplotlib
#nbagg

# +
def find_nearest(array,value):
    return (np.abs(array-value)).argmin()

def get_coh(x):
    return 1-4**(-x)

def logplot(x, y, *args, **kwargs):
    return plot(x, log10(y), *args, **kwargs)

sig = [200, 250, 300] #numbers of no relevance
uppers = map(chr, range(65, 91))
# -

# %cd ../generate_data/

# # load data (LIF)

sigN = [200., 250., 300.]
sigNc= [[200., 250., 300.], [95, 130, 165], [40., 70., 100.], [10., 34., 58.],  [20., 45., 70.]] 
muc = [300., 330., 350., 375., 365.]
tauN = [0., 2.5, 5., 10., 15.]
tauS = [10., 20., 30.]
snr = array([.15, .05, .25, .5, .75, 1., 2.])
snr2 = snr
w0_s = [0, 0.25, .508, 1., 2.54, 7.111]
order = argsort(w0_s)
w0_s = sort(w0_s)

# ## simulations (exact)

#w0_s = [0, .508, 2.54, 7.111]
par_s = {'snr': snr, 'sigN': sigN, 'tau': tauS, 'w0' : w0_s}
#freq. resolved MI from exact
def MI(x, s = 25.):
    return -.5*nan_to_num(log2(real(1 - array(smooth(x[1][1:], s))/array(smooth(x[0][1:], s))), dtype = float32))
def MI_cum(x, s = 25):
    d = MI(x, s)/(2*pi)
    res = cumtrapz(d, dx = x[2][1])
    res2 = res[:int(argmax(d)*3+3//x[2][1]+1)]
    res = append(res2, [res2[-1]]*(len(res)-len(res2)))
    return x[0], x[1], x[2], res, x[3], x[4] 

# +
As = [load("sim-vm-LIF-tauN0.npz"), load("sim-vm-LIF-tauN2.npz"), load("sim-vm-LIF-tauN5.npz"), load("sim-vm-LIF-tauN10.npz")]
Bs = [load("sim-mm-LIF-tauN0.npz"), load("sim-mm-LIF-tauN2.npz"), load("sim-mm-LIF-tauN5.npz"), load("sim-mm-LIF-tauN10.npz")]
sim_am, sim_mm = [x.f.sim for x in As], [x.f.sim for x in Bs]

sim_am = array([[[[[MI_cum(x, 21) for x in y[order]] for y in z] for z in jo] for jo in ja] for ja in sim_am])
sim_mm = array([[[[[MI_cum(x, 21) for x in y[order]] for y in z] for z in jo] for jo in ja] for ja in sim_mm])
# -

# ### lower bound estimation

# +
coh_mm = sim_mm
coh_am = sim_am

par_coh = [{'sigN': sigNc[x], 'mu': muc[x], 'tauN':tauN, 'tauS': tauS, 'snr': snr, 'w0': w0_s} for x in range(len(sim_am))]
par_s = par_coh[0]
par_s['tau'] = par_s['tauS']
w0_coh = w0_s#[0, .508, 2.54, 7.111, 12.]
# -

rates_am = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float16)
rates_mm = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float16)
cv_am = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float16)
cv_mm = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float16)

def I_LB(a, smoothing = 30):
    b, c, d = smooth(a[0][1:], smoothing), smooth(a[5][1:], smoothing), smooth(a[4][1:], smoothing)
#    D = max(d)
#    d[d<D/230.] = 0
    coherence = abs(d)**2/(b*c)
    #coherence[argmax(coherence)*2+3//(2*pi/(2e5*.02)):] = 0
    # substract mean of info at very large values
    coherence -= mean(coherence[4*len(b)//5:])
    return -.5*log2(1-coherence)

# +
#w = linspace(2*pi/(2e5*.02), 2*pi/(.02)/(2.*5), int(2e5/(2*5))) #only first 1/5
N, dt = 2e5, .02
w = 2*pi*linspace(1/(N*dt), 1./dt/(5*2), int(N/(2*5))) #consider only first 5th of frequencies which accords to dt = .1

li_am = array([[[[[cumtrapz(I_LB(x1, 1), dx = w[0])/(2*pi) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float32)
li_mm = array([[[[[cumtrapz(I_LB(x1, 1), dx = w[0])/(2*pi) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float32)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## analytic LRT (only $\tau_n = 0$)

# + {"hidden": true}
sigN_a = arange(175., 325., 25)
tau_a = arange(5., 35., 5)
w0_a = linspace(0, 8, 64)
w0_a2 = sort(append(w0_a, logspace(-3, 1, 96)))
lis = argsort(append(w0_a, logspace(-3, 1, 96)))

par_a = {'snr': snr, 'sigN': sigN_a, 'tau': tau_a, 'w0' : w0_a2}
#print par_a

# + {"hidden": true}
A = np.load('ana-vm.npz')
mi_am_ana = A.f.ana
mi_am_ana = array([[[x[lis] for x in y] for y in z] for z in mi_am_ana])
mi_am_ana[:,:,:,:, -1] /= 2*pi #normalize omega integration for total information

A = np.load('ana-mm.npz')
mi_mm_ana = A.f.ana
mi_mm_ana = array([[[x[lis] for x in y] for y in z] for z in mi_mm_ana])
mi_mm_ana[:,:,:,:, -1] /= 2*pi #normalize omega integration for total information
# -

# # Figures 3,4,5,6 and S1,S11

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. 4 (top row)

# + {"code_folding": [], "hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot, logplot][0]

fig = figure(figsize=(5., 4.8))
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
sec = fig.add_subplot(gs[0,1])
others0 = fig.add_subplot(gs[1,0], sharex = first)
others = fig.add_subplot(gs[1,1], sharex = sec)
axs = [first, sec, others0, others]

tit = ['LIF', 'LIF']
for i, ax in enumerate(hstack(axs)):
    #ax.tick_params(axis='both', which='major', pad= 4)
    #ax.xaxis.labelpad = 4
    #ax.yaxis.labelpad = 4
    ax.set_xlabel('central freq. $\Omega_0$ [2$\pi$ kHz]', usetex = False)
    if i == 2 or i == 3: 
        ax.axis('off')
    else:
        if i%2==0:
           # setp(ax.xaxis.get_ticklabels(), visible = False)
            ax.set_ylabel(r'information $\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]', usetex = False)
        ax.set_title('{}'.format(tit[i]), y = 1.15, fontsize = 12, usetex = False)
        if i%2==1:
            ax.set_ylabel(r'info. ratio $\beta^{\mathrm{tot}} = \mathcal{I}^{\mathrm{tot}}_{\mathrm{MM}}/\mathcal{I}^{\mathrm{tot}}_{\mathrm{VM}}$', usetex = False)
            ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
for i, ax in enumerate(hstack(array(axs).reshape(2,2).T)):
    if i%2==0: ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False)
    
#suptitle('LIF vs EIF: $\Omega_0$ dependence of MI')

###--------------------------------------w0---------------
ind = 0 #snr
s_ind = ind
t = 1 #sigN
y = 1
for tau in [0, 2]:
    linst = '-'
    #if tau==0:
    #    linst = 'None'
    subplot(axs[0])
    if AM:
        p(par_s['w0'], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                            for x in range(6)], marker= markers[t], linestyle = linst, linewidth = .35, 
                            color = c(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]) )
    if MM:
        p(par_s['w0'], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                            for x in range(6)], marker= markers[t], linestyle = linst, linewidth = .35, 
                            color = c2(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
    #ana for tauN = 0
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    if tau == 0:
        if AM: p(w0_a2, [mi_am_ana[s_ind][t*2+1][Y][x][3][-1]/mi_am_ana[s_ind][t*2+1][Y][x][0][-1] for x in range(160)],
                 color = c(.15*tau+.1), lw = 1.5, linestyle = 'dashed')
        if MM: p(w0_a2, [mi_mm_ana[s_ind][t*2+1][Y][x][3][-1]/mi_mm_ana[s_ind][t*2+1][Y][x][0][-1] for x in range(160)],
                 color = c2(.15*tau+.1), lw = 1.5, linestyle = 'dashed')
    xlim([0, 7.5])
handles, labels = array(axs[0].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (.18, .8), frameon = False)
##### INSET ############
inset_axis =  inset_axes(gca(),
                width= '35%', height= "35%",  loc=3,
                 bbox_to_anchor=(0.55, 0.37, 1, 1,),
                 bbox_transform=axs[0].transAxes)
s_ind = 3
inset_axis.text(2.6, .2, r'$\sigma_{s}=$'+'{:.1f}'.format(snr2[s_ind]), fontsize= 7, usetex = False)
for tau in [0, 2]:
    linst = '-'
    if tau==0:
        linst = 'None'
    if AM:
        p(par_s['w0'], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                            for x in range(6)], marker= markers[t], linestyle = linst, lw = .25, color = c(.15*tau+.1),
                            label ='VM: {}ms'.format(tauN[tau]), markersize = 4)
    if MM:
        p(par_s['w0'], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                            for x in range(6)], marker= markers[t], linestyle = linst, lw = .25, color = c2(.15*tau+.1),
                            label ='MM: {}ms'.format(tauN[tau]), markersize = 4)

    #ana for tauN = 0
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    if tau == 0:
        if AM: p(w0_a2, [mi_am_ana[s_ind][t*2+1][Y][x][3][-1]/mi_am_ana[s_ind][t*2+1][Y][x][0][-1] for x in range(160)],
                         color = c(.15*tau+.1), linestyle = 'dashed', lw = .65)
        if MM: p(w0_a2, [mi_mm_ana[s_ind][t*2+1][Y][x][3][-1]/mi_mm_ana[s_ind][t*2+1][Y][x][0][-1] for x in range(160)],
                         color = c2(.15*tau+.1), linestyle = 'dashed', lw = .65)
    inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize=6)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    xlim([0,7.5])

#####
####### RATIOS  ####################################################
####
c2 = cm.BuGn_r#plt.cm.winter
c = cm.Greys_r#plt.cm.copper

p = [semilogy, plot, logplot][0]

###--------------------------------------w0---------------
ind = 0 #snr
s_ind = ind
t = 1 #sigN
y = 1
for tau in [0, 2]:
    linst = '-'
    if tau==0:
        linst = 'None'
    subplot(axs[1])
    p(par_s['w0'], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                            for x in range(6)], marker= markers[t], linestyle = linst, linewidth = .35, 
                            color = c(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))

    #ana for tauN = 0
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    if tau == 0:
        p(w0_a2, [mi_mm_ana[s_ind][t*2+1][Y][x][3][-1]/mi_am_ana[s_ind][t*2+1][Y][x][3][-1] for x in range(160)],
                 color = c(.15*tau+.1), lw = 1.5, linestyle = 'dashed')
    xlim([0, 7.5])
handles, labels = array(axs[1].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 3, ncol = 1, 
       markerscale = .8, columnspacing = .2, bbox_to_anchor = (.25, .8), frameon = False)
##### INSET ############
inset_axis = inset_axes(gca(),
                width= '35%', height= "35%",  loc=3,
                 bbox_to_anchor=(0.55, 0.35, 1, 1,),
                 bbox_transform= axs[1].transAxes)
s_ind = 3
inset_axis.text(2.6, 2, r'$\sigma_{s} = $'+'{:.1f}'.format(snr2[s_ind]), fontsize= 7, usetex = False)
for tau in [0, 2]:   
    linst = '-'
    if tau==0:
        linst = 'None'
    p(par_s['w0'], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for x in range(6)], marker= markers[t], linestyle = linst, lw = .25, color = c(.15*tau+.1),
                    label ='VM: {}ms'.format(tauN[tau]), markersize = 4)


    #ana for tauN = 0
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    if tau == 0:
        p(w0_a2, [mi_mm_ana[s_ind][t*2+1][Y][x][3][-1]/mi_am_ana[s_ind][t*2+1][Y][x][3][-1] for x in range(160)],
                         color = c(.15*tau+.1), linestyle = 'dashed', lw = .65)

    inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize=6)
    inset_axis.axhline(y=1, color = 'k', linestyle = ':', lw = .25)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    xlim([0,7.5])
tight_layout()   
show()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. 3

# + {"code_folding": [], "hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot, logplot][0]

fig = figure(figsize = (4.5, 6.))
gs = GridSpec(3, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
sec = fig.add_subplot(gs[1,0], )
third = fig.add_subplot(gs[2,0],)
others0 = fig.add_subplot(gs[0,1], sharex = first)
others = [fig.add_subplot(gs[i+1, 1], sharex = k, sharey = None) for i, k in enumerate([sec, third])]
axs = [first,sec,third, others0] + others

#axs = [subplot(x) for x in gs]
#fig, axs = plt.subplots(nrows = 2, ncols=3, sharey=True, figsize = (9., 4.8), gridspec_kw={'width_ratios': [1., 1., 1.]})
xlab = ['signal strength $\sigma_{s}$', 'noise strength $\hat \sigma_n$', 'noise time const. $\\tau_n$ [ms]']*2
tit = [r'$\sigma_{s}$',  '$\hat \sigma_n$', r'$\tau_n$']
for i, ax in enumerate(hstack(axs)):
    #ax.tick_params(axis='both', which='major', pad= 2)
    #ax.xaxis.labelpad = 2
    #ax.yaxis.labelpad = 2
    ax.set_xlabel(xlab[i])
    if i<3:
       # setp(ax.xaxis.get_ticklabels(), visible = False)
        ax.set_ylabel(r'information $\mathcal{I}$ [bits/sp.]')
        #ax.set_title('influence of {}'.format(tit[i]), y = 1.15)

    if i >2:
        ax.set_ylabel(r'info. ratio $\beta = \mathcal{I}_{\mathrm{\mathrm{MM}}}/\mathcal{I}_{\mathrm{\mathrm{VM}}}$')
        ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
#bc figures were transposed in a brute force way
for i, ax in enumerate(hstack(array(axs).reshape(2,3))): 
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex= False)

for ax in [axs[1], axs[4]]:
        ax.set_xticks(sig)
        ax.set_xticklabels([r'$\hat \sigma_n^{{({0})}}{{\mathrel{{\widehat=}}}}\nu_{0}$'.format(x) for x in sig])
#axs[1].annotate('', xy=(.45, .5), xytext=(.05, .5), xycoords = 'axes fraction', size = 5.,arrowprops=dict(facecolor=cm.Greys_r(.5),  arrowstyle = 'simple'),)
#axs[1].annotate('increasing noise', xy=(.55, .52), xytext=(.05, .52), xycoords = 'axes fraction', size = 8., color = cm.Greys_r(.5) )
#suptitle('LIF: overview mutual information per spike')

###--------------------------------------SNR---------------
snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = 0
t = 1 #sigN
y = 1 #tauS
for tau in [0, 2]:#range(4)[:3]:
    subplot(axs[0])
    ts = par_s['tau'][y]
    if AM:
        p(snr2, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]))

    if MM:
        p(snr2, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c2(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]))

    #ana for tauN = 0
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    X = find_nearest(par_s['w0'][x], par_a['w0'])
    if tau == 0:
        if AM: p(snr2, [mi_am_ana[s_ind][t*2+1][Y][X][3][-1]/mi_am_ana[s_ind][t*2+1][Y][X][0][-1] for s_ind in ran],
                 color = c(.15*tau+.1), linestyle = 'dashed', lw =1.5)
        if MM: p(snr2, [mi_mm_ana[s_ind][t*2+1][Y][X][3][-1]/mi_mm_ana[s_ind][t*2+1][Y][X][0][-1] for s_ind in ran],
                 color = c2(.15*tau+.1), linestyle = 'dashed', lw = 1.5)
    #ylim([1e-4, None])
    xlim([0, 2.2])

handles, labels = array(axs[0].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 2, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (.0, 1.15), frameon = False)

##### INSET
x = 1
inset_axis = inset_axes(gca(),
                width="40%", # width = 30% of parent_bbox
                height= "40%",  #height : 1 inch)
                bbox_to_anchor=(.1, 0, 1, 1,),
                bbox_transform=axs[0].transAxes,
                loc=4, borderpad = 2
                       )

inset_axis.text(.3, .0005, '$\Omega_0 = $'+"\n"+' {:.2f} $2\pi\cdot$ kHz'.format(1*par_s['w0'][x]), fontsize= 6, usetex  = True)

for tau in [0, 2]:#range(4)[:3]:
    if AM:
        p(snr2, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c(.15*tau+.1),
                    label ='VM: {}ms'.format(tauN[tau]), markersize = 4)
    if MM:
        p(snr2, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c2(.15*tau+.1),
                    label ='MM: {}ms'.format(tauN[tau]), markersize = 4)
    #ana for tauN = 0
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    X = find_nearest(par_s['w0'][x], par_a['w0'])
    if tau == 0:
        if AM: p(snr2, [mi_am_ana[s_ind][t*2+1][Y][X][3][-1]/mi_am_ana[s_ind][t*2+1][Y][X][0][-1] for s_ind in ran],
                 color = c(.15*tau+.1), linestyle = 'dashed', lw = 0.65)
        if MM: p(snr2, [mi_mm_ana[s_ind][t*2+1][Y][X][3][-1]/mi_mm_ana[s_ind][t*2+1][Y][X][0][-1] for s_ind in ran],
                 color = c2(.15*tau+.1), linestyle = 'dashed', lw = 0.65)
    inset_axis.tick_params(axis='both', which= 'both', pad= 1, labelsize = 6)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    xlim([0., 2.2])
    
###--------------------------------------sigN---------------    
s_ind = 0
ind = 0
x = 0
y = 1 #tauS
X = find_nearest(par_s['w0'][x], par_a['w0'])
for tau in [0,2]:#range(4):
    subplot(axs[1])    
    axs[1].set_yscale("log")
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    if AM:
        p(sig, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c(.15*tau+.1), label ='VM: {}ms'.format(tauN[tau]) )    
        if tau == 0:
            p(sig_a, [(mi_am_ana[s_ind][t][Y][X][3][-1]/mi_am_ana[s_ind][t][Y][X][0][-1]) for t in range(6)],
                           marker= 'None', linestyle = '--', linewidth = 1.5, color = c(.15*tau+.1))
    

    if MM:
        p(sig, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c2(.15*tau+.1), label ='MM: {}ms'.format(tauN[tau]) )
        if tau == 0:
                p(sig_a, [(mi_mm_ana[s_ind][t][Y][X][3][-1]/mi_mm_ana[s_ind][t][Y][X][0][-1]) for t in range(6)],
                               marker= 'None', linestyle = '--', linewidth = 1.5, color = c2(.15*tau+.1))
    handles, labels = array(axs[0].get_legend_handles_labels()) #legend from SNR axis
    leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 5, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (1, .5), frameon = False)
    #xlim([0,12])
    
###--------------------------------------tauN/tauS---------------    
s_ind = 0
ind = 0
x = 0 #w0
y = 1 #tauS
t = 1 #sigN
ts = par_s['tau'][y]
Y = argwhere(par_a['tau']==ts).flatten()[0]
for t in [0, 2]:#range(3):
    subplot(axs[2])
    if AM:
        p(tauN[:4], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                        for tau in range(4)], marker = ['X', 'o', '>'][1], linestyle = '-', linewidth = .35, color = c(.2*t+.1), 
          label = '$\hat \sigma_n$ = '+ '$ \hat \sigma_n^{{({})}}$'.format(t+1))
#        scatter(tauN[:4], [(sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
#                        for tau in range(4)], marker = ['X', 'o', '>'][t], linestyle = '-', linewidth = .35, 
#                        color = [c(.15*C+.1) for C in range(4)], zorder = 10)
        
    if MM:
        p(tauN[:4], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                        for tau in range(4)], marker= ['X', 'o', '>'][1], linestyle = '-', linewidth = .35, color = c2(.2*t+.1), 
          label = r'$\hat \sigma_n^{{({0})}}\mathrel{{\widehat=}}\nu_{0}$'.format(t+1))
#        scatter(tauN[:4], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])
#                        for tau in range(4)], marker = ['X', 'o', '>'][t], linestyle = '-', linewidth = .35, 
#                        color = [c2(.15*C+.1) for C in range(4)], zorder = 10)
        
handles, labels = array(axs[2].get_legend_handles_labels()) #legend from SNR axis
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[1], labels[3]), fontsize = 'x-small', loc = 5, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (1, .5), frameon = False)
#####
####### RATIOS  ####################################################
####
c2 = cm.BuGn_r#plt.cm.winter
c = cm.Greys_r#plt.cm.copper

p = [semilogy, plot, logplot][0]

snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = 0
t = 1 #sigN
y = 1 #tauS
for tau in [0, 2]:#range(4)[:3]:
    subplot(axs[3])
    p(snr2, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.15*tau+.1),
                    label =r'$\tau_n$ = {} ms'.format(tauN[tau]))
        #ana for tauN = 0
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    X = find_nearest(par_s['w0'][x], par_a['w0'])
    if tau == 0:
        p(snr2, [mi_mm_ana[s_ind][t*2+1][Y][X][3][-1]/mi_am_ana[s_ind][t*2+1][Y][X][3][-1] for s_ind in ran],
                 color = c(.15*tau+.1), linestyle = 'dashed', lw =1.5)
    #ylim([1e-4, None])
xlim([0, 2.2])
handles, labels = array(axs[3].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 3, ncol = 1, 
       markerscale = .8, columnspacing = .2,  bbox_to_anchor = (.05, .002), frameon = False)
##### INSET
x = 1
inset_axis = inset_axes(gca(),
                    width= '40%', height= "40%",  loc=3,
                    bbox_to_anchor=(0.45, 0.55, 1, 1,),
                    bbox_transform= gca().transAxes)
inset_axis.text(.5, 45.5, r'$\Omega_0 = $'+"\n"+'{:.2f} $2\pi\cdot$kHz'.format(1*par_s['w0'][x]), fontsize= 6, usetex = True)

for tau in [0, 2]:#range(4)[:3]:
    p(snr2, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
             (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c(.15*tau+.1),
                    label ='VM: {}ms'.format(tauN[tau]), markersize = 4)
    
    #ana for tauN = 0
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    X = find_nearest(par_s['w0'][x], par_a['w0'])
    if tau == 0:
        p(snr2, [mi_mm_ana[s_ind][t*2+1][Y][X][3][-1]/mi_am_ana[s_ind][t*2+1][Y][X][3][-1] for s_ind in ran],
                 color = c(.15*tau+.1), linestyle = 'dashed', lw = 0.65)
    inset_axis.tick_params(axis='both', which='both', pad= 1, labelsize = 6)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    xlim([0., 2.2])
    
###--------------------------------------sigN---------------    
p = [semilogy, plot, logplot][1]

s_ind = 0
ind = 0
x = 0
y = 1 #tauS
X = find_nearest(par_s['w0'][x], par_a['w0'])
for tau in [0,2]:#range(4):
    subplot(axs[4])    
    ts = par_s['tau'][y]
    Y = argwhere(par_a['tau']==ts).flatten()[0]
    p(sig, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
    #for t in range(3):
        #scatter(sig[t], (sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
         #           (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]), marker = markers[t], color = c(.15*tau+.1))
    
    if tau == 0:
        p(sig_a, [mi_mm_ana[s_ind][t][Y][X][3][-1]/(mi_am_ana[s_ind][t][Y][X][3][-1]) for t in range(6)],
                           marker= 'None', linestyle = '--', linewidth = 1.5, color = c(.15*tau+.1))

    #xlim([0,12])
    legend(fontsize = 'x-small', loc = 10, ncol = 1,  markerscale = .8)
    
###--------------------------------------tauN/tauS---------------    
s_ind = 0
ind = 0
x = 0 #w0
y = 1 #tauS
t = 1 #sigN
for t in [0, 2]:#range(3):
    subplot(axs[5])
    #ts = par_s['tau'][y]
    p(tauN[:4], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                        for tau in range(4)], marker= ['None', 'o', 'None'][1], linestyle = '-', linewidth = .35, color = c(.15*t+.1), 
          label = r'$\hat \sigma_n^{{({0})}}\mathrel{{\widehat=}}\nu_{0}$'.format(t+1))
    legend(fontsize = 'x-small', loc = 'best', ncol = 1,  markerscale = .8)
tight_layout()
show()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. S1

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot, logplot][1]

fig = figure()
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
second = fig.add_subplot(gs[0,1], sharex = first)
others = [fig.add_subplot(gs[1, i], sharex = k, sharey = None) for i, k in enumerate([first, second])]
axs = array([first,second] + others).reshape(2,2)

tit = [r'$\sigma_{s}$',  '$\hat \sigma_n$', r'$\tau_n$ and $\tau_s$']
for i, ax in enumerate(hstack(axs)):
    ax.set_xlabel(r'$\tau_s$ [ms]')
    ax.set_ylabel(r'information $\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]')
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = 0) 

y = 1 #tauS
t = 1 #sigN
for i,s_ind in enumerate([0, 3]):
    for j,x in enumerate([0, 2]):
        subplot(axs[i,j])
        axs[i,j].set_title(r'$\sigma_{s} =\ $'+'{}, '.format(snr2[s_ind])+  r'$\Omega_0 =\ $'+'{:.2f} $2\pi\cdot$kHz'.format(1*par_s['w0'][x]))
        X = find_nearest(par_s['w0'][x], par_a['w0'])
        for tau in [0,2]:#range(4):
            #title(r'invariance in $\tau_s$', size = 8)
            if AM:
                p(tauS, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                                for y in range(3)], marker= markers[t], linestyle = '-', linewidth = .3, color = c(.15*tau+.1), label = r'MM, $\tau_n=${} ms'.format(tauN[tau]))
            if MM:
                p(tauS, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                                for y in range(3)], marker= markers[t], linestyle = '-', linewidth = .3, color = c2(.15*tau+.1),label = r'VM, $\tau_n=${} ms'.format(tauN[tau]))
axs[0,1].legend()
tight_layout()
show()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. 5 (left side) and S3 (left side)

# + {"hidden": true}
x_am = real(array([[[[[sim_am[tau][s_ind][t][y][x][3][-1] for x in range(0,6,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))
y_am = real(array([[[[[1/sim_am[tau][s_ind][t][y][x][3][-1]*li_am[tau][s_ind][t][y][x][-1] 
                                                           for x in range(0,6,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))

x_mm = real(array([[[[[sim_mm[tau][s_ind][t][y][x][3][-1] for x in range(0,6,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))
y_mm = real(array([[[[[1/sim_mm[tau][s_ind][t][y][x][3][-1]*li_mm[tau][s_ind][t][y][x][-1] 
                                                           for x in range(0,6,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot][1]

figure(figsize = (5, 4.5))
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1.,1]
                       )

xlab = [r'signal strength $\sigma_{s}$', r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]",r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]", r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]"]
tit = ['LIF', 'EIF']
axs = [subplot(x) for x in gs]
for i, ax in enumerate(hstack(axs)):
    if i > 0: 
        ax.set_xscale('log')
        ax.set_yscale('linear')
    ax.set_xlabel(xlab[i], usetex = False)
    ax.set_ylabel(r'lin. index $\lambda^{\mathrm{ld}}=\mathcal{I}^{\mathrm{ld}}/\mathcal{I}^{\mathrm{tot}}$', usetex = False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.text(-.28, 1.05, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False) #seemingly, transform=... uses relative coord.
    ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
    if i==0 or i==1:   ax.set_title('{}'.format(tit[i]), y = 1., fontsize = 12, usetex = False)

    
###--------------------------------------SNR---------------
snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = -1
t = 1 #sigN
y = 1 #tauS
tau = 3

#for tau in [0, 2]:#range(4)[:3]:
for x in [0,4]:
    subplot(gs[0])
    if AM:
        p(snr2[:], [1./sim_am[tau][s_ind][t][y][x][3][-1]*li_am[tau][s_ind][t][y][x][-1]
                    for s_ind in ran[:]], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.1*x+.1),
                    label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))
    if MM:
        p(snr2[:], [1./sim_mm[tau][s_ind][t][y][x][3][-1]*li_mm[tau][s_ind][t][y][x][-1]
                    for s_ind in ran[:]], marker= markers[t], linestyle = '-', linewidth = .35, color = c2(.1*x+.1),
                    label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))
    #ylim([.7, 5.5])
    #xlim([0.1, None])
handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
       markerscale = .8, columnspacing = -.7, bbox_to_anchor = (.38, .8), frameon = False)
    

###-----------------------------lambda vs I (sigN)---------------
snr2 = sort(snr)
ran = argsort(snr)

subplot(gs[1])


[[scatter(x_mm[:,s_ind,t], y_mm[:,s_ind,t], facecolor = c2(.2*t+.1), edgecolor = 'none',
             alpha = .6, marker = markers[1], s = 9,label = '$\hat \sigma_n =$' +' $ \hat \sigma_n^{{({})}}$'.format(t+1)),
scatter(x_am[:,s_ind,t], y_am[:,s_ind,t], facecolor = c(.2*t+.1), edgecolor = 'none',
             alpha = .6, marker = markers[1], s = 9,label = '$\hat \sigma_n =$' +' $ \hat \sigma_n^{{({})}}$'.format(t+1))] for s_ind in range(6) for t in [0,2]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (.0, .3), frameon = False)


###---------------------------lambda vs I (w0)---------------
subplot(gs[2])
#scatter([x_am, x_mm], [y_am,y_mm], alpha = .5, c= 'lightgrey', s = 7)
[[scatter(x_mm[:,s_ind,:,:,x], y_mm[:,s_ind,:,:,x], c = c2(.1*x+.1), 
             alpha = .6, marker = markers[1], s = 9,label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x])),
scatter(x_am[:,s_ind,:,:,x], y_am[:,s_ind,:,:,x], c = c(.1*x+.1), 
             alpha = .6, marker = markers[1], s = 9, label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))] for s_ind in range(6) for x in [0,4]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (-.05, .2), frameon = False)

    
###-------------------------lambda vs I (tauN)---------------    
subplot(gs[3])
#scatter([x_am, x_mm], [y_am,y_mm], alpha = .5, c= 'lightgrey', s = 7)
[[scatter(x_am[tau,s_ind], y_am[tau][s_ind], c = c(.2*tau+.1), 
             alpha = .6, marker = markers[1], s = 9, label = r'$\tau_n$ = {} ms'.format(tauN[tau])),
scatter(x_mm[tau][s_ind], y_mm[tau][s_ind], c = c2(.2*tau+.1), 
             alpha = .6, marker = markers[1], s = 9,label = r'$\tau_n$ = {} ms'.format(tauN[tau]))] for s_ind in range(6) for tau in [0,2]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (-.05, .2), frameon = False)

tight_layout()
show()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. 6

# + {"hidden": true}
sns.set_context('paper')

# + {"hidden": true}
def relative_difference(a,b):
    c = 2*(a-b)/(abs(a)+abs(b))*(abs(a)-max(a)/10000. > 0) #best so far
    #c = 2*(a-b)/maximum(abs(a),abs(a))#*(abs(a)-max(a)/10000. > 0)
    #c = (a-b)/a#*(abs(a)-max(a)/10000. > 0)
    #c = 2*(a-b)/maximum(a,b)
    return c

# + {"hidden": true}
sig = [200, 250, 300]
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
tauN = [0, 2.5, 5., 10., 15.]
W0 = linspace(0, 8, 64)
t = 1 #sig = [150, 200, 250]
y = 1 #tauS
x = 2 #Omega0
s_ind = 0 #snr
tau = 0 #tauN

#
low_snr = 1,1,2,0,2
high_snr = 1,1,1,3,0
t,y,x, s_ind, tau = [low_snr, high_snr][0]

### ADD A FUNCTIONALITY THAT PLOTS EITHER LB OR LRT OR BOTH
which_one = 0 #0,1,2

ts = par_s['tau'][y]
Y = argwhere(par_a['tau']==ts).flatten()[0]
X2 = find_nearest(par_s['w0'][x], par_a['w0'])
T = argwhere(par_a['sigN'] == sig[t]).flatten()[0]

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(5, 4.5))

for i, ax in enumerate(hstack(axs)): 
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText= False))
    #ax.ticklabel_format(scilimits=(-2, 1))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.28, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False) 
    
#suptitle(r' $\Omega_0 =  {:.2f}, \tau_s$={}, $\sigma_n=${}, snr = {}, $\tau_n$ = {}'.format(w0_s[x], tauS[y], t, snr[s_ind], tauN[tau]), size =16)

### INSETS showing exact-LB
inset_axis = inset_axes(axs[0,0], width= '40%', height= "40%",  loc=3, bbox_to_anchor=(0.5, 0.25, 1, 1), bbox_transform=axs[0,0].transAxes)
inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize=6)
inset_axis.spines['bottom'].set_linewidth(0.5)
inset_axis.spines['left'].set_linewidth(0.5)
ylabel(r'rel. diff. $(\mathcal{I}_{\mathrm{tot}}, \mathcal{I}_{\mathrm{ld}})$', fontsize = 7, usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
b = coh_mm[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(relative_difference(MI(a, 2),I_LB(b, 2)), 25), label=r'rel. diff. $(\mathcal{I}_{\mathrm{tot}}, \mathcal{I}_{\mathrm{ld}})$' , color = 'k', lw = 1.) #exact
#plot(w, I_LB(b, 25), label=r'$\mathcal{I}_{\mathrm{ld}}$ (sim.)',color = c2(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    ana_freqs = [find_nearest(mi_mm_ana[s_ind][T][Y][X2][2][1:], F) for F in w]
    mi = mi_mm_ana[s_ind][T][Y][X2]
    plot(w, (MI(a, 25)+.5*log2(1-mi[1][ana_freqs])/mi[0][ana_freqs])/(-.5*log2(1-mi[1][ana_freqs]/mi[0][ana_freqs])),
         color = 'gray', label=r'$\mathcal{I}_{\mathrm{tot}}-\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.) #theoryxlim([0, 4])
xlim([0, 2.7])
ylim([-0, 2.5])
#legend(fontsize = 'x-small', loc = 3, bbox_to_anchor = (.15, .5), bbox_transform=inset_axis.transAxes)

inset_axis = inset_axes(axs[0,1], width= '40%', height= "40%",  loc=3, bbox_to_anchor=(0.5, 0.2, 1, 1), bbox_transform=axs[0,1].transAxes)
inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize=6)
inset_axis.spines['bottom'].set_linewidth(0.5)
inset_axis.spines['left'].set_linewidth(0.5)
ylabel(r'rel. diff. $(\mathcal{I}_{\mathrm{tot}}, \mathcal{I}_{\mathrm{ld}})$', fontsize = 7, usetex = False)
a = sim_am[tau][s_ind][t][y][x]
b = coh_am[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(relative_difference(MI(a, 2),I_LB(b, 2)), 25), label=r'$\mathcal{I}_{\mathrm{tot}}-\mathcal{I}_{\mathrm{ld}}$' , color = 'k', lw = 1.) #exact
#plot(w, I_LB(b, 25), label=r'$\mathcal{I}_{\mathrm{ld}}$ (sim.)',color = c2(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    mi = mi_am_ana[s_ind][T][Y][X2]
    plot(w, (MI(a, 25)+.5*log2(1-mi[1][ana_freqs])/mi[0][ana_freqs])/(-.5*log2(1-mi[1][ana_freqs]/mi[0][ana_freqs])),
       color = 'k', label=r'$\mathcal{I}_{\mathrm{tot}}-\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.2) #theoryxlim([0, 4])
ylim([-2, 2.5])
xlim([0, 2.7])
#legend(fontsize = 'x-small', loc = 3, bbox_to_anchor = (.15, .6), bbox_transform=inset_axis.transAxes)


### INSETS showing ACF
inset_axis = inset_axes(axs[1,0], width= '40%', height= "40%",  loc=3,bbox_to_anchor=(0.5, 0.25, 1, 1), bbox_transform=axs[1,0].transAxes)
inset_axis.tick_params(axis='both', which='major', pad= 1.4, labelsize=6)
inset_axis.spines['bottom'].set_linewidth(0.5)
inset_axis.spines['left'].set_linewidth(0.5)
ylabel(r'auto-corr. [kHz]', fontsize = 7, usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], a[0][1:], label=r'$C_{\mathrm{auto}}$', color = c2(.05), lw = 1.) #exact
b = coh_mm[tau][s_ind][t][y][x]
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2], mi_mm_ana[s_ind][T][Y][X2][0] , label=r'$C^{\mathrm{lin}}_{\mathrm{auto}}$', 
         linestyle = ':', color = c2(.65), lw = .8) #theory
xlim([0, 3])
legend(fontsize = 'x-small', loc = 4, bbox_to_anchor = (1.1, .2), bbox_transform=inset_axis.transAxes)

inset_axis = inset_axes(axs[1,1], width= '40%', height= "40%",  loc=3,bbox_to_anchor=(0.52, 0.25, 1, 1), bbox_transform=axs[1,1].transAxes)
inset_axis.tick_params(axis='both', which='major', pad= 1.4, labelsize=6)
inset_axis.spines['bottom'].set_linewidth(0.5)
inset_axis.spines['left'].set_linewidth(0.5)
ylabel(r'auto-corr. [kHz]', fontsize = 7, usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], a[0][1:], label=r'$C_{\mathrm{auto}}$', color = c(.05), lw = 1.) #exact
b = coh_am[tau][s_ind][t][y][x]
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2], mi_am_ana[s_ind][T][Y][X2][0], color = c(.55), 
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{auto}}$', lw = .8) #theory
xlim([0, 3])
legend(fontsize = 'x-small', loc = 4, bbox_to_anchor = (1.1, .2), bbox_transform=inset_axis.transAxes)

### CCF
sca(axs[1,0])
ylabel(r'cross-correlation. [kHz]', usetex = False)
xlabel(r'angular frequency $\omega$ [2$\pi$ kHz]', usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(a[1][1:], 25), label=r'$C_{\mathrm{cross}}$', color = c2(.05), lw = 1.5) #exact
b = coh_mm[tau][s_ind][t][y][x]
plot(w, get_coh(I_LB(b, 25))*b[0][1:], label='$|S_{sr}|^2/S_{ss}$', color = c2(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2][1:], mi_mm_ana[s_ind][T][Y][X2][1][1:], color = c2(.55),
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{cross}}$', lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 1, ncol = 1)

sca(axs[1,1])
ylabel(r'cross-correlation. [kHz]', usetex = False)
xlabel(r'angular frequency $\omega$ [2$\pi$ kHz]', usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(a[1][1:], 25), label=r'$C_{\mathrm{cross}}$',color = c(.05), lw = 1.5) #exact
b = coh_am[tau][s_ind][t][y][x]
plot(w, get_coh(I_LB(b, 20))*b[0][1:], label='$|S_{sr}|^2/S_{ss}$', color = c(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2][1:], mi_am_ana[s_ind][T][Y][X2][1][1:], color = c(.55),
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{cross}}$', lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 1, ncol = 1)

### INFORMATION
sca(axs[0,0])
ylabel(r'information [bits]', usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], MI(a, 25), label=r'$\mathcal{I}_{\mathrm{tot}}$' , color = c2(.05), lw = 1.5) #exact
b = coh_mm[tau][s_ind][t][y][x]
plot(w, I_LB(b, 25), label=r'$\mathcal{I}_{\mathrm{ld}}$',color = c2(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2][1:], -.5*log2(1-mi_mm_ana[s_ind][T][Y][X2][1][1:]/mi_mm_ana[s_ind][T][Y][X2][0][1:]),
         color = c2(.65), label=r'$\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.2) #theoryxlim([0, 4])

#ax2 = gca().twinx()
#plot(w, MI(a,25)/I_LB(b, 25), label=r'$\mathcal{I}_{\mathrm{ld}}$ (sim.)',color =  'k', linestyle = '-', lw = 1.2)
legend(fontsize = 'small', loc = 'best', ncol = 1)

sca(axs[0,1])
ylabel(r'information [bits]', usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], MI(a, 25), label= r'$\mathcal{I}_{\mathrm{tot}}$', color = c(.05), lw = 1.5) #exact
b = coh_am[tau][s_ind][t][y][x]
plot(w, I_LB(b, 25), label= r'$\mathcal{I}_{\mathrm{ld}}$', color = c(.45), linestyle = '-', lw = 1.2) #coherence
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2][1:], -.5*log2(1-mi_am_ana[s_ind][T][Y][X2][1][1:]/mi_am_ana[s_ind][T][Y][X2][0][1:]),
         color = c(.65), label=r'$\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.2) #theory
xlim([0, 2.7])
legend(fontsize = 'small', loc = 'best', ncol = 1)


tight_layout()
show()

# + {"hidden": true}
custom_style(paper = True)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. S11

# + {"hidden": true}
sig = [200, 250, 300]
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
tauN = [0, 2.5, 5., 10., 15.]
W0 = linspace(0, 8, 64)
t = 1 #sig = [150, 200, 250]
y = 1 #tauS
x = 1 #Omega0
s_ind = 4 #snr
tau = 0 #tauN

low_snr = 1,1,2,0,2
high_snr = 1,1,2,3,0

t,y,x,s_ind,tau = [low_snr, high_snr][1]

ts = par_s['tau'][y]
Y = argwhere(par_a['tau']==ts).flatten()[0]
X2 = find_nearest(par_s['w0'][x], par_a['w0'])
T = argwhere(par_a['sigN'] == sig[t]).flatten()[0]

fig, axs = plt.subplots(3, 2, sharex=True)

for i, ax in enumerate(hstack(axs)): 
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText= False))
    #ax.ticklabel_format(scilimits=(-2, 1))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False) 
    
#suptitle(r' $\Omega_0 =  {:.2f}, \tau_s$={}, $\sigma_n=${}, snr = {}, $\tau_n$ = {}'.format(w0_s[x], tauS[y], t, snr[s_ind], tauN[tau]), size = 8)

sca(axs[2,0])
ylabel(r'auto-corr. [kHz]', usetex = False)
xlabel(r'angular frequency $\omega$ [2$\pi$ kHz]', usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], a[0][1:], label=r'$C_{\mathrm{auto}}$ (sim.)', color = c2(.05), lw = 1.5) #exact
b = coh_mm[tau][s_ind][t][y][x]
#plot(w, b[0][1:], label='$S_{xs}$', color = c2(.45), linestyle = ':') #coherence
#plot(w_S0, a_mm(tau, s_ind, t, y, X), label='LRT') #theory
#plot(w, auto_coh_mm(tau, s_ind, t, y, x), label='LRT-coh')
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2], mi_mm_ana[s_ind][T][Y][X2][0] , label=r'$C^{\mathrm{lin}}_{\mathrm{auto}}$', 
         linestyle = ':', color = c2(.55), lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 'best')

sca(axs[2,1])
ylabel(r'auto-corr. [kHz]', usetex = False)
xlabel(r'angular frequency $\omega$ [2$\pi$ kHz]', usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], a[0][1:], label=r'$C_{\mathrm{auto}}$ (sim.)', color = c(.05), lw = 1.5) #exact
b = coh_am[tau][s_ind][t][y][x]
#plot(w, b[0][1:], label='$S_{ss}$', color = c(.45), linestyle = ':') #coherence
#plot(w_S0, a_am(tau, s_ind, t, y, X), label='LRT') #theory
#plot(w, auto_coh_am(tau, s_ind, t, y, x), label='LRT-coh')
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2], mi_am_ana[s_ind][T][Y][X2][0], color = c(.55), 
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{auto}}$', lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 'best')

sca(axs[1,0])
ylabel(r'cross-corr. [kHz]', usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(a[1][1:], 25), label=r'$C_{\mathrm{cross}}$ (sim.)', color = c2(.05), lw = 1.5) #exact
b = coh_mm[tau][s_ind][t][y][x]
plot(w, get_coh(I_LB(b, 25))*b[0][1:], label='$|S_{sr}|^2/S_{ss}$ (sim.)', color = c2(.45), linestyle = '-', lw = 1.2) #coherence
#plot(w, b[1][1:]**2/b[2][:1], label='$S_{sr}/S_{ss}$') #coherence
#plot(w_S0, c_mm(tau, s_ind, t, y, X), label='LRT') #theory
#plot(w, cross_coh_mm(tau, s_ind, t, y, x), label='LRT2') #theory
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2][1:], mi_mm_ana[s_ind][T][Y][X2][1][1:], color = c2(.55),
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{cross}}$', lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 'best')

sca(axs[1,1])
ylabel(r'cross-corr. [kHz]', usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], smooth(a[1][1:], 25), label=r'$C_{\mathrm{cross}}$ (sim.)',color = c(.05), lw = 1.5) #exact
b = coh_am[tau][s_ind][t][y][x]
plot(w, get_coh(I_LB(b, 20))*b[0][1:], label='$|S_{sr}|^2/S_{ss}$ (sim.)', color = c(.45), linestyle = '-', lw = 1.2) #coherence
#plot(w, b[1][1:]**2/b[2][:1], label='$S_{sr}/S_{ss}$') #coherence
#plot(w_S0, c_mm(tau, s_ind, t, y, X), label='LRT') #theory
#plot(w, cross_coh_am(tau, s_ind, t, y, x), label='LRT2') #theory
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2][1:], mi_am_ana[s_ind][T][Y][X2][1][1:], color = c(.55),
         linestyle = ':', label=r'$C^{\mathrm{lin}}_{\mathrm{cross}}$', lw = 1.2) #theory
xlim([0, 4])
legend(fontsize = 'small', loc = 'best')

sca(axs[0,0])
title('MM')
ylabel(r'information [bits]', usetex = False)
a = sim_mm[tau][s_ind][t][y][x]
plot(a[2][1:], MI(a, 25), label=r'$\mathcal{I}_{\mathrm{tot}}$ (sim.)' , color = c2(.05), lw = 1.5) #exact
b = coh_mm[tau][s_ind][t][y][x]
plot(w, I_LB(b, 25), label=r'$\mathcal{I}_{\mathrm{ld}}$ (sim.)',color = c2(.45), linestyle = '-', lw = 1.2) #coherence
#plot(w_S0, mi_mm(tau, s_ind, t, y, X), label='LRT') #theory
if tau == 0:
    plot(mi_mm_ana[s_ind][T][Y][X2][2][1:], -.5*log2(1-mi_mm_ana[s_ind][T][Y][X2][1][1:]/mi_mm_ana[s_ind][T][Y][X2][0][1:]),
         color = c2(.55), label=r'$\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.2) #theoryxlim([0, 4])
legend(fontsize = 'small', loc = 'best')

sca(axs[0,1])
title('VM')
ylabel(r'information [bits]', usetex = False)
a = sim_am[tau][s_ind][t][y][x]
plot(a[2][1:], MI(a, 25), label= r'$\mathcal{I}_{\mathrm{tot}}$ (sim.)', color = c(.05), lw = 1.5) #exact
b = coh_am[tau][s_ind][t][y][x]
plot(w, I_LB(b, 25), label= r'$\mathcal{I}_{\mathrm{ld}}$ (sim.)', color = c(.45), linestyle = '-', lw = 1.2) #coherence
#plot(w_S0, mi_am(tau, s_ind, t, y, X), label='LRT') #theory
if tau == 0:
    plot(mi_am_ana[s_ind][T][Y][X2][2][1:], -.5*log2(1-mi_am_ana[s_ind][T][Y][X2][1][1:]/mi_am_ana[s_ind][T][Y][X2][0][1:]),
         color = c(.55), label=r'$\mathcal{I}_{\mathrm{lin}}$', linestyle = ':', lw = 1.2) #theory
xlim([0, 2.7])
legend(fontsize = 'small', loc = 'best')


tight_layout()
show()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # load data (EIF)

# + {"hidden": true}
As = [load("sim-vm-EIF-tauN0.npz"), load("sim-vm-EIF-tauN2.npz"), load("sim-vm-EIF-tauN5.npz"), load("sim-vm-EIF-tauN10.npz")]
Bs = [load("sim-mm-EIF-tauN0.npz"), load("sim-mm-EIF-tauN2.npz"), load("sim-mm-EIF-tauN5.npz"), load("sim-mm-EIF-tauN10.npz")]
sim_am, sim_mm = [x.f.sim for x in As], [x.f.sim for x in Bs]

sim_am = array([[[[[MI_cum(x, 21) for x in y[order]] for y in z] for z in jo] for jo in ja] for ja in sim_am])
sim_mm = array([[[[[MI_cum(x, 21) for x in y[order]] for y in z] for z in jo] for jo in ja] for ja in sim_mm])

# + {"hidden": true}
coh_mm = sim_mm
coh_am = sim_am

par_coh = [{'sigN': sigNc[x], 'mu': muc[x], 'tauN':tauN, 'tauS': tauS, 'snr': snr, 'w0': w0_s} for x in range(len(sim_am))]
par_s = par_coh[0]
par_s['tau'] = par_s['tauS']
w0_coh = w0_s#[0, .508, 2.54, 7.111, 12.]

# + {"hidden": true}
rates_am = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float16)
rates_mm = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float16)
cv_am = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float16)
cv_mm = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float16)

# + {"hidden": true}
#w = linspace(2*pi/(2e5*.02), 2*pi/(.02)/(2.*5), int(2e5/(2*5))) #only first 1/5
N, dt = 2e5, .02
w = 2*pi*linspace(1/(N*dt), 1./dt/(5*2), int(N/(2*5))) #consider only first 5th of frequencies which accords to dt = .1

li_am = array([[[[[cumtrapz(I_LB(x1, 1), dx = w[0])/(2*pi) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_am], dtype = float32)
li_mm = array([[[[[cumtrapz(I_LB(x1, 1), dx = w[0])/(2*pi) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_mm], dtype = float32)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # Figures 4,9 and S2,S3

# + {"heading_collapsed": true, "hidden": true, "cell_type": "markdown"}
# ## Fig. 4 (bottom)

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']
snr2 = snr

AM, MM = 1, 1

p = [semilogy, plot, logplot][0]

fig = figure(figsize=(5., 4.8))
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
sec = fig.add_subplot(gs[0,1])
others0 = fig.add_subplot(gs[1,0], sharex = first)
others = fig.add_subplot(gs[1,1], sharex = sec)
axs = [first, sec, others0, others]

tit = ['LIF', 'LIF', 'EIF', 'EIF']
for i, ax in enumerate(hstack(axs)):
    #ax.tick_params(axis='both', which='major', pad= 4)
    #ax.xaxis.labelpad = 4
    #ax.yaxis.labelpad = 4
    ax.set_xlabel('central freq. $\Omega_0$ [2$\pi$ kHz]', usetex=False)
    if i == 0 or i == 1: 
        ax.axis('off')
    else:
        if i%2==0:
           # setp(ax.xaxis.get_ticklabels(), visible = False)
            ax.set_ylabel(r'information $\mathcal{I}^{tot}$ [bits/sp.]', usetex = False)
        ax.set_title('{}'.format(tit[i]), y = 1.15, fontsize =12)
        if i%2==1:
            ax.set_ylabel(r'info. ratio $\beta^{tot} = \mathcal{I}^{tot}_{\mathrm{MM}}/\mathcal{I}^{tot}_{\mathrm{VM}}$',usetex = False)
            ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
for i, ax in enumerate(hstack(array(axs).reshape(2,2).T)):
        if i%2==1: ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False)
    
#suptitle('LIF vs EIF: $\Omega_0$ dependence of MI')

###--------------------------------------w0---------------
ind = 0 #snr
s_ind = ind
t = 1 #sigN
y = 1
for tau in [0, 2]:
    subplot(axs[2])
    if AM:
        p(par_s['w0'][:5], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                            for x in range(5)], marker= markers[t], linestyle = '-', linewidth = .35, 
                            color = c(.15*tau+.1), label ='VM: {}ms'.format(tauN[tau]) )
    if MM:
        p(par_s['w0'][:5], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                            for x in range(5)], marker= markers[t], linestyle = '-', linewidth = .35, 
                            color = c2(.15*tau+.1), label ='MM: {}ms'.format(tauN[tau]) )
    #xlim([0, 7.5])
handles, labels = array(axs[2].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (.0, .0), frameon = False)
##### INSET ############
inset_axis = inset_axes(gca(),
                    width= '40%', height= "40%",  loc=3,
                    bbox_to_anchor=(0.55, 0.58, 1, 1,),
                    bbox_transform= gca().transAxes)

s_ind = 3
inset_axis.text(1, 10**(-.8), r'$\sigma_{s} = $'+'{:.1f}'.format(snr2[s_ind]), fontsize= 7, usetex = False)

for tau in [0, 2]:   
    if AM:
        p(par_s['w0'][:5], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                            for x in range(5)], marker= markers[t], linestyle = '-', lw = .25, color = c(.15*tau+.1),
                            label ='VM: {}ms'.format(tauN[tau]), markersize = 4)
    if MM:
        p(par_s['w0'][:5], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                            for x in range(5)], marker= markers[t], linestyle = '-', lw = .25, color = c2(.15*tau+.1),
                            label ='MM: {}ms'.format(tauN[tau]), markersize = 4)

    inset_axis.tick_params(axis='both', which='both', pad= 1, labelsize=6)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    #xlim([0,7.5])

#####
####### RATIOS  ####################################################
####
c2 = cm.BuGn_r#plt.cm.winter
c = cm.Greys_r#plt.cm.copper

p = [semilogy, plot, logplot][0]

###--------------------------------------w0---------------
ind = 0 #snr
s_ind = ind
t = 1 #sigN
y = 1
for tau in [0, 2]:
    subplot(axs[3])
    p(par_s['w0'][:5], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                            for x in range(5)], marker= markers[t], linestyle = '-', linewidth = .35, 
                            color = c(.15*tau+.1), label ='VM: {}ms'.format(tauN[tau]) )
    #xlim([0, 7.5])
    ylim([None, 50])
handles, labels = array(axs[3].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 1, ncol = 1, 
       markerscale = .8, columnspacing = .2, bbox_to_anchor = (1., .4), frameon = False)
##### INSET ############
inset_axis = inset_axes(gca(),
                    width= '40%', height= "40%",  loc=3,
                    bbox_to_anchor=(0.55, 0.58, 1, 1,),
                    bbox_transform= gca().transAxes)
s_ind = 3
inset_axis.text(1, 9, r'$\sigma_{s} = $'+'{:.1f}'.format(snr2[s_ind]), fontsize= 7, usetex = False)

for tau in [0, 2]:   
    p(par_s['w0'][:5], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for x in range(5)], marker= markers[t], linestyle = '-', lw = .25, color = c(.15*tau+.1),
                    label ='VM: {}ms'.format(tauN[tau]), markersize = 4)

    inset_axis.tick_params(axis='both', which='both', pad= 1, labelsize=6)
    inset_axis.axhline(y=1, color = 'k', linestyle = ':', lw = .25)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
    #ylim([0,7.5])
tight_layout()    
show()

# + {"heading_collapsed": true, "hidden": true, "cell_type": "markdown"}
# ## Fig. S2

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot, logplot][1]

fig = figure()
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
second = fig.add_subplot(gs[0,1], sharex = first)
others = [fig.add_subplot(gs[1, i], sharex = k, sharey = None) for i, k in enumerate([first, second])]
axs = array([first,second] + others).reshape(2,2)

tit = [r'$\sigma_{s}$',  '$\hat \sigma_n$', r'$\tau_n$ and $\tau_s$']
for i, ax in enumerate(hstack(axs)):
    ax.set_xlabel(r'$\tau_s$ [ms]')
    ax.set_ylabel(r'information $\mathcal{I}^{tot}$ [bits/sp.]')
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = 0) 

y = 1 #tauS
t = 1 #sigN
for i,s_ind in enumerate([0, 3]):
    for j,x in enumerate([0, 2]):
        subplot(axs[i,j])
        axs[i,j].set_title(r'$\sigma_{s} =\ $'+'{}, '.format(snr2[s_ind])+  r'$\Omega_0 =\ $'+'{:.2f} $2\pi\cdot$kHz'.format(1*par_s['w0'][x]))
        for tau in [0,2]:#range(4):
            #title(r'invariance in $\tau_s$', size = 8)
            if AM:
                p(tauS, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                                for y in range(3)], marker= markers[t], linestyle = '-', linewidth = .3, color = c(.15*tau+.1), label = r'MM, $\tau_n=${} ms'.format(tauN[tau]))
            if MM:
                p(tauS, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                                for y in range(3)], marker= markers[t], linestyle = '-', linewidth = .3, color = c2(.15*tau+.1), label = r'VM, $\tau_n=${} ms'.format(tauN[tau]))
axs[0,0].legend()
tight_layout()
show()

# + {"heading_collapsed": true, "hidden": true, "cell_type": "markdown"}
# ## Fig. S3 (right side)

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot, logplot][0]

fig = figure(figsize = (9, 4.8))
gs = GridSpec(2, 3,
                       width_ratios=[1,1,1],
                       height_ratios=[1,1],
                       #hspace= .15)
             )
first = fig.add_subplot(gs[0,0])
sec = fig.add_subplot(gs[0,1], )
third = fig.add_subplot(gs[0,2],)
others0 = fig.add_subplot(gs[1,0], sharex = first)
others = [fig.add_subplot(gs[1, i+1], sharex = k, sharey = None) for i, k in enumerate([sec, third])]
axs = [first,sec,third, others0] + others

#axs = [subplot(x) for x in gs]
#fig, axs = plt.subplots(nrows = 2, ncols=3, sharey=True, figsize = (9., 4.8), gridspec_kw={'width_ratios': [1., 1., 1.]})
xlab = [r'signal strength $\sigma_{s}$', 'noise strength $\hat \sigma_n$', r'noise time const. $\tau_n$ [ms]']*2
tit = [r'$\sigma_{s}$',  '$\hat \sigma_n$', r'$\tau_n$']
for i, ax in enumerate(hstack(axs)):
    #ax.tick_params(axis='both', which='both', pad= 2)
    #ax.xaxis.labelpad = 2
    #ax.yaxis.labelpad = 2
    ax.set_xlabel(xlab[i])
    if i<3:
       # setp(ax.xaxis.get_ticklabels(), visible = False)
        ax.set_ylabel(r'information $\mathcal{I}$ [bits/sp.]', usetex = False)
        #ax.set_title('influence of {}'.format(tit[i]), y = 1.15)
    if i >2:
        ax.set_ylabel(r'info. ratio $\beta^{tot} = \mathcal{I}^{tot}_{\mathrm{MM}}/\mathcal{I}^{tot}_{\mathrm{VM}}$', usetex = False)
        #ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False)
for ax in [axs[1], axs[4]]:
    ax.set_xticks(sig)
    ax.set_xticklabels([r'$\hat \sigma_n^{{({0})}}{{\mathrel{{\widehat=}}}}\nu_{0}$'.format(x) for x in sig])

#suptitle('EIF: overview mutual information per spike')

###--------------------------------------SNR---------------
snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = 0
t = 1 #sigN
y = 1 #tauS
for tau in [0, 2]:#range(4)[:3]:
    subplot(axs[0])
    if AM:
        p(snr2, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]))

    if MM:
        p(snr2, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c2(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
handles, labels = array(axs[0].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (.07, .007), frameon = False)    
##### INSET
x = 1
inset_axis = inset_axes(gca(),
                width="40%", # width = 30% of parent_bbox
                height= "40%",  #height : 1 inch)
                bbox_to_anchor=(0.09, 0.1, 1, 1),
                bbox_transform= axs[0].transAxes,
                loc=4, borderpad = 2.
                       )
inset_axis.text(.3, 0.002, r'$\Omega_0 =${:.1f} $2\pi\cdot$kHz'.format(1*par_s['w0'][x]), fontsize= 7, usetex = False)

for tau in [0, 2]:#range(4)[:3]:
    if AM:
        p(snr2, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c(.15*tau+.1),
                    label ='VM: {}ms'.format(tauN[tau]), markersize = 4)
    if MM:
        p(snr2, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c2(.15*tau+.1),
                    label ='MM: {}ms'.format(tauN[tau]), markersize = 4)
    tick_params(axis='both', which='both', labelsize= 7)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)
###--------------------------------------sigN---------------    
s_ind = 0
ind = 0
x = 0
y = 1 #tauS
for tau in [0,2]:#range(4):
    subplot(axs[1])
    axs[1].set_yscale('log')
    if AM:
        p(sig, [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
        #for t in range(3):
        #    scatter(sig[t], log10(sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]), marker = markers[t], color = c(.15*tau+.1))
    if MM:
        p(sig, [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c2(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
        #for t in range(3):
        #    scatter(sig[t], log10(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]), marker = markers[t], color = c2(.15*tau+.1))
handles, labels = array(axs[1].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 1, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (1, 1), frameon = False)
###--------------------------------------tauN/tauS---------------    
s_ind = 0
ind = 0
x = 0 #w0
y = 1 #tauS
t = 1 #sigN
for t in [0, 2]:#range(3):
    subplot(axs[2])
    if AM:
        p(tauN[:4], [sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]
                        for tau in range(4)], marker = ['None', 'o', 'None'][1], linestyle = '-', linewidth = .35, color = c(.2*t+.1), 
                        label = '$\hat \sigma_n =$' +'$ \hat \sigma_n^{{({})}}$'.format(t+1))
#        scatter(tauN[:4], [(sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
#                        for tau in range(4)], marker= ['X', 'o', '>'][t], linestyle = '-', linewidth = .35, 
#                        color = [c(.15*C+.1) for C in range(4)], zorder = 10)
        
    if MM:
        p(tauN[:4], [sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1]
                        for tau in range(4)], marker= ['None', 'o', 'None'][1], linestyle = '-', linewidth = .35, color = c2(.2*t+.1),
                        label = r'$\hat \sigma_n^{{({0})}}\mathrel{{\widehat=}}\nu_{0}$'.format(t+1))
#        scatter(tauN[:4], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])
#                        for tau in range(4)], marker= ['X', 'o', '>'][t], linestyle = '-', linewidth = .35, 
 #                       color = [c2(.15*C+.1) for C in range(4)], zorder = 10)
handles, labels = array(axs[2].get_legend_handles_labels())
leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
       markerscale = .8, columnspacing = -.5, bbox_to_anchor = (.4, .6), frameon = False)

#####
####### RATIOS  ####################################################
####
c2 = cm.BuGn_r#plt.cm.winter
c = cm.Greys_r#plt.cm.copper

p = [semilogy, plot, logplot][0]

snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = 0
t = 1 #sigN
y = 1 #tauS
for tau in [0, 2]:#range(4)[:3]:
    subplot(axs[3])
    p(snr2, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
handles, labels = array(axs[3].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 3, ncol = 1, 
       markerscale = .8, columnspacing = .2, bbox_to_anchor = (.52, .22), frameon = False)    
##### INSET
x = 1
inset_axis = inset_axes(axs[3],
                    width= '40%', height= "40%",  loc=3,
                    bbox_to_anchor=(0.5, 0.5, 1, 1,),
                    bbox_transform= gca().transAxes)
inset_axis.text(.25, 2, r'$\Omega_0 =${:.1f} $2\pi\cdot$kHz'.format(1*par_s['w0'][x]), fontsize= 7, usetex = False)
inset_axis.axhline(y=1, color = 'k', linestyle = ':', lw = .25)

for tau in [0, 2]:#range(4)[:3]:
    p(snr2, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
             (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                    for s_ind in ran], marker= markers[t], linestyle = '-', linewidth = .25, color = c(.15*tau+.1),
                    label = r'$\tau_n$ = {} ms'.format(tauN[tau]), markersize = 4)
    tick_params(axis='both', which='both', labelsize= 7)
    inset_axis.spines['bottom'].set_linewidth(0.5)
    inset_axis.spines['left'].set_linewidth(0.5)

###--------------------------------------sigN---------------    
p = [semilogy, plot, logplot][1]
s_ind = 0
ind = 0
x = 1
y = 1 #tauS
for tau in [0,2]:#range(4):
    subplot(axs[4])    
    p(sig, [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                        for t in range(3)], marker= 'o', linestyle = '-', linewidth = .35, color = c(.15*tau+.1), label = r'$\tau_n$ = {} ms'.format(tauN[tau]))
    #for t in range(3):
        #scatter(sig[t], (sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/(sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]), marker = markers[t], color = c(.15*tau+.1))
handles, labels = array(axs[3].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 1, ncol = 1, 
       markerscale = .8, columnspacing = .2, bbox_to_anchor = (1, 1), frameon = False)    
###--------------------------------------tauN/tauS---------------    
s_ind = 0
ind = 0
x = 0 #w0
y = 1 #tauS
t = 1 #sigN
for t in [0, 2]:#range(3):
    subplot(axs[5])
    #ts = par_s['tau'][y]
    p(tauN[:4], [(sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1])
                        for tau in range(4)], marker= ['None', 'o', 'None'][1], linestyle = '-', linewidth = .35, color = c(.2*t+.1),
                        label = r'$\hat \sigma_n^{{({0})}}\mathrel{{\widehat=}}\nu_{0}$'.format(t+1))
#    scatter(tauN[:4], [((sim_mm[tau][s_ind][t][y][x][3][-1]/sim_mm[tau][s_ind][t][y][x][0][-1])/
#                    (sim_am[tau][s_ind][t][y][x][3][-1]/sim_am[tau][s_ind][t][y][x][0][-1]))
#                        for tau in range(4)], marker= ['X', 'o', '>'][t], linestyle = '-', linewidth = .35, 
#                        color = [c(.15*C+.1) for C in range(4)], zorder = 10)
handles, labels = array(axs[3].get_legend_handles_labels())
legend(handles[[0,1]], (labels[0], labels[1]), fontsize = 'x-small', loc = 2, ncol = 1, 
       markerscale = .8, columnspacing = .2, bbox_to_anchor = (0,1), frameon = False)

tight_layout()
show()

# + {"heading_collapsed": true, "hidden": true, "cell_type": "markdown"}
# ## Fig. S6

# + {"hidden": true}
x_am = real(array([[[[[sim_am[tau][s_ind][t][y][x][3][-1] for x in range(0,5,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))
y_am = real(array([[[[[1./sim_am[tau][s_ind][t][y][x][3][-1]*li_am[tau][s_ind][t][y][x][-1] 
                                                           for x in range(0,5,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))

x_mm = real(array([[[[[sim_mm[tau][s_ind][t][y][x][3][-1] for x in range(0,5,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))
y_mm = real(array([[[[[1./sim_mm[tau][s_ind][t][y][x][3][-1]*li_mm[tau][s_ind][t][y][x][-1] 
                                                           for x in range(0,5,1)]  for y in range(3)] for t in range(3)] for s_ind in [0,2,3,4,5,6]] for tau in range(4)]))

# + {"hidden": true}
c = cm.Blues_r#plt.cm.winter
c2 = cm.Reds_r#plt.cm.copper
sig = [1, 2, 3]
sig_a = (sigN_a-200)/(50.)+1
tauN = [0, 2.5, 5., 10., 15.]
markers = ['X', 'o', '>']

snr = par_coh[0]['snr']

AM, MM = 1, 1

p = [semilogy, plot][1]

figure(figsize = (5, 4.5))
gs = GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1.,1]
                       )

xlab = [ r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]",r'signal strength $\sigma_{s}$',r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]", r"$\mathcal{I}^{\mathrm{tot}}$ [bits/sp.]"]
tit = ['LIF', 'EIF']
axs = [subplot(x) for x in gs]
for i, ax in enumerate(hstack(axs)):
    if i != 1: 
        ax.set_xscale('log')
        ax.set_yscale('linear')
        #ax.set_xlim([1e-5,.7])
        #ax.set_ylim(0,1.1)
    ax.set_xlabel(xlab[i], usetex = False)
    ax.set_ylabel(r'lin. index $\lambda^{\mathrm{ld}}=\mathcal{I}^{\mathrm{ld}}/\mathcal{I}^{\mathrm{tot}}$', usetex = False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.text(-.28, 1.05, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = False) #seemingly, transform=... uses relative coord.
    ax.axhline(y=1, color = 'k', linestyle = ':', lw = 0.4)
    if i==0 or i==1:   ax.set_title('{}'.format(tit[i]), y = 1., fontsize = 12, usetex = False)

    
###--------------------------------------SNR---------------
snr2 = sort(snr)
ran = argsort(snr)

s_ind = 0
ind = 0
x = -1
t = 1 #sigN
y = 1 #tauS
tau = 3

#for tau in [0, 2]:#range(4)[:3]:
for x in [0,4]:
    subplot(gs[1])
    if AM:
        p(snr2[1:], [1./sim_am[tau][s_ind][t][y][x][3][-1]*li_am[tau][s_ind][t][y][x][-1]
                    for s_ind in ran[1:]], marker= markers[t], linestyle = '-', linewidth = .35, color = c(.1*x+.1),
                    label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))
    if MM:
        p(snr2[1:], [1./sim_mm[tau][s_ind][t][y][x][3][-1]*li_mm[tau][s_ind][t][y][x][-1]
                    for s_ind in ran[1:]], marker= markers[t], linestyle = '-', linewidth = .35, color = c2(.1*x+.1),
                    label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))
    #ylim([.7, 5.5])
    #xlim([0.1, None])
handles, labels = array(gca().get_legend_handles_labels())
#leg = legend(handles[[1,3,0,2]], (r'', r'', labels[0], labels[2]), fontsize = 'x-small', loc = 3, ncol = 2, 
#       markerscale = .8, columnspacing = -.7, bbox_to_anchor = (.38, .8), frameon = False)
    

###-----------------------------lambda vs I (sigN)---------------
snr2 = sort(snr)
ran = argsort(snr)

subplot(gs[0])


[[scatter(x_mm[:,s_ind,t], y_mm[:,s_ind,t], facecolor = c2(.2*t+.1), edgecolor = 'none',
             alpha = .6, marker = markers[1], s = 9,label = '$\hat \sigma_n =$' +' $ \hat \sigma_n^{{({})}}$'.format(t+1)),
scatter(x_am[:,s_ind,t], y_am[:,s_ind,t], facecolor = c(.2*t+.1), edgecolor = 'none',
             alpha = .6, marker = markers[1], s = 9,label = '$\hat \sigma_n =$' +' $ \hat \sigma_n^{{({})}}$'.format(t+1))] for s_ind in range(6) for t in [0,2]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (.12, .27), frameon = False)


###---------------------------lambda vs I (w0_s)---------------
subplot(gs[3])
#scatter([x_am, x_mm], [y_am,y_mm], alpha = .5, c= 'lightgrey', s = 7)
[[scatter(x_mm[:,s_ind,:,:,x], y_mm[:,s_ind,:,:,x], c = c2(.1*x+.1), 
             alpha = .6, marker = markers[1], s = 9,label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x])),
scatter(x_am[:,s_ind,:,:,x], y_am[:,s_ind,:,:,x], c = c(.1*x+.1), 
             alpha = .6, marker = markers[1], s = 9, label = r'$\Omega_0 = {:.2f} \cdot2\pi \mathrm{{kHz}}$'.format(w0_s[x]))] for s_ind in range(6) for x in [0,4]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (-.05, .2), frameon = False)

    
###-------------------------lambda vs I (tauN)---------------    
subplot(gs[2])
#scatter([x_am, x_mm], [y_am,y_mm], alpha = .5, c= 'lightgrey', s = 7)
[[scatter(x_am[tau,s_ind], y_am[tau][s_ind], c = c(.2*tau+.1), 
             alpha = .6, marker = markers[1], s = 9, label = r'$\tau_n$ = {} ms'.format(tauN[tau])),
scatter(x_mm[tau][s_ind], y_mm[tau][s_ind], c = c2(.2*tau+.1), 
             alpha = .6, marker = markers[1], s = 9,label = r'$\tau_n$ = {} ms'.format(tauN[tau]))] for s_ind in range(6) for tau in [0,2]]

handles, labels = array(gca().get_legend_handles_labels())
leg = legend(handles[[1,3,4,2]], (r'', r'', labels[0], labels[7]), fontsize = 'x-small', loc = 'center left', ncol = 2, 
       markerscale = 1.5,  columnspacing = -.5, bbox_to_anchor = (.12, .27), frameon = False)

tight_layout()
show()
