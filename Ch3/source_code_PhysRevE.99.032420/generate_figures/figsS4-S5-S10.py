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
sys.path.append('../generate_data/')
sys.path.append('../code/')
from pylab import *
from signalsmooth import smooth
import seaborn as sns
sns.set_style('white')
sns.set(style = "ticks", color_codes = True)
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# %matplotlib 
#nbagg

uppers = map(chr, range(65, 91))

# #### rc params

rc('figure', titlesize= 'x-large')
rc('axes', labelpad = 4);
rc('xtick.major', pad = 4); rc('xtick.minor', pad = 4); rc('ytick.major', pad = 4); rc('ytick.minor', pad = 4);

from style_sheet import custom_style
custom_style()
mpl.rcParams['text.usetex'] = False

# # load data

# %cd ../generate_data/

data = load('test-gaussianity-LIF-tauN0-1.npz')
signal, spikes = data.f.sig[()], data.f.spi[()] #weird, but recovers dict

signal.keys()

# # Figures

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Fig. S10

# + {"hidden": true}
spikes_or_sig = 0
data = [spikes, signal][spikes_or_sig]

w = array([2*pi/(1e5*.02)*(x+1) for x in xrange(len((data['means'][0][0][0])))])
T = .02*1e5/1000. #in seconds
normC = 1/T #because of FT and C = 1/T *(r r)


means = array([[data['means'][0][0], data['means'][1][0]], [data['means'][0][1], data['means'][1][1]]])
stds = array([[data['std'][0][0], data['std'][1][0]], [data['std'][0][1], data['std'][1][1]]])

col = [cm.Reds_r(.1) , cm.Blues_r(.1)]  
col2 = [cm.Reds_r(.35) ,cm.Blues_r(.35)]
C = col, col2

text1 = ['varying sig.', 'repeating sig.']
text2 = ['MM', 'VM']

fig, axs = plt.subplots(2, 2, sharex= False, sharey = False)

tit = [r'$\epsilon_{snr}$',  '$\hat \sigma_n$', r'$\tau_n$ and $\tau_s$', '$\Omega_0$']

#suptitle("LIF: mean values of FC of spike trains")
suptitle('Mean of Fourier coefficients obtained from {} in LIF neurons'.format(['spike trains', 'signal'][spikes_or_sig]), size =15)
for i, ax in enumerate(hstack(axs)):
    #ax.set_title('influence of '+tit[i], y = 1.15)
    ax.set_xlabel(r'$\omega$ [$2\pi\cdot$ kHz]')
    ax.set_ylabel(r'mean $\mu_c(\omega)$ of FC [Hz$^{1/2}$]')
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,1), labelsize=7)
    ax.text(-.15, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold') #seemingly, transform=... uses relative coord.
    #ax.set_ylim(min(means.flatten())*normC, max(means.flatten())*normC)
    ax.set_xlim([0, 7.5])

for mod in range(2):
    for ur in range(2):
        subplot(axs[ur, mod])
        axs[ur, mod].text(.8, 1.02, text2[mod], ha='right', va ='bottom',  weight = 'bold', color = [cm.Reds_r(.0), cm.Blues_r(.0)][mod], fontsize = 12, transform=axs[ur, mod].transAxes)
        axs[ur, mod].text(.2, 1.02, text1[ur], ha='left', va ='bottom',  color = C[ur][mod], fontsize = 12, transform=axs[ur, mod].transAxes)
        plot(w, means[ur][mod][0]*normC, c = C[ur][mod], label = 'real'),# plot(means[ur][mod][1], c = C[ur][mod])
        #axs[ur, mod].fill_between(w, means[ur][mod][0]+stds[ur][mod][0], means[ur][mod][0]-stds[ur][mod][0], facecolor=C[ur][mod], alpha=0.5)
        axs[ur,mod].set_ylim(min(means[:, mod].flatten())*normC, max(means[:, mod].flatten())*normC)
        #legend(loc = 2)

        inset_axis = inset_axes(gca(),
                width="35%", # width = 30% of parent_bbox
                height= "35%",  #height : 1 inch)
                loc = 1, borderpad = .92)
        inset_axis.text(0.5, 1.05, r'imag(FC)'.format(), fontsize= 7, transform= inset_axis.transAxes, horizontalalignment = 'center')
        #inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize = 7)
        plot(w, means[ur][mod][1]*normC, c = C[ur][mod], linestyle = 'dashed', label = 'imag.')
        inset_axis.set_ylim(min(means[:, mod].flatten())*normC, max(means[:, mod].flatten())*normC)
        inset_axis.set_xticklabels([]), inset_axis.set_yticklabels([])
        inset_axis.set_xlim([0, 12.5])
        #legend(fontsize = 'x-small')

tight_layout()
show()
# -

# ## Fig. S4 and S5

# **set the variable spikes_or_sig to 1 for Fig. S4 and to 0 for Fig. S5**

# +
binN = 75

spikes_or_sig = 1
data = [spikes, signal][spikes_or_sig]

c_un, c_re = [data['corrcoeff'][0][0], data['corrcoeff'][1][0]], [data['corrcoeff'][0][1], data['corrcoeff'][1][1]]
test_un, test_re = [data['normtest'][0][0], data['normtest'][1][0]], [data['normtest'][0][1], data['normtest'][1][1]]
      
col = [cm.Reds_r(.1) , cm.Blues_r(.1)]  
col2 = [cm.Reds_r(.35) ,cm.Blues_r(.35)]

gs1 = GridSpec(2, 2)
gs1.update(left=0.14, right=0.52, hspace = 0.24, wspace=0.0, bottom = .07, top = .86)
gs2 = GridSpec(2, 2)
gs2.update(left=0.6, right=0.97, hspace=0.24, wspace = 0.0, bottom = .07, top =.86)

fig = figure()

topax_left = fig.add_subplot(gs1[0,0])
others = fig.add_subplot(gs1[0,1], sharey = topax_left)
bottomax_left = fig.add_subplot(gs1[1,0])
others2 = fig.add_subplot(gs1[1,1], sharey = bottomax_left)
axs = array([[topax_left, others],[bottomax_left, others2]])

topax_left2 = fig.add_subplot(gs2[0,0], sharey = topax_left)
others2 = fig.add_subplot(gs2[0,1], sharey = topax_left2)
bottomax_left2 = fig.add_subplot(gs2[1,0], sharey = bottomax_left)
others22 = fig.add_subplot(gs2[1,1], sharey = bottomax_left)
axs2 = array([[topax_left2, others2],[bottomax_left2, others22]])

for i, ax in enumerate([axs[0, 0], axs2[0,0], axs[1,0] , axs2[1,0]]):
    #ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(scilimits=(-2, 1))
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,1), labelsize=7)
    ax.text(-.16, 1.06, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold')
    #ax.set_ylim([-.3, .3])
    #ax.set_xlim([0, 15])

for x, ax in enumerate([axs, axs2]):
#    if x ==1: 
#        setp(ax[0,0].get_yticklabels(), visible=False)
#        setp(ax[0,0].get_yticklines(), visible=False)
#    if x == 1: 
#        ax[0,1].yaxis.tick_right()
#        setp(ax[0,1].get_yticklabels(), visible = False)

    setp(ax[0,1].get_yticklabels(), visible=False)
    setp(ax[0,1].get_yticklines(), visible=False)
    setp(ax[1,1].get_yticklabels(), visible=False)
    setp(ax[1,1].get_yticklines(), visible=False)
    #ax[0, 0].set_xticks([])
    #ax[0, 1].set_xticks([])
    setp(ax[0,0].get_xticklabels(), visible = False)
    setp(ax[0,1].get_xticklabels(), visible = False)
    setp(ax[1,0].get_xticklabels(), visible = False)
    setp(ax[1,1].get_xticklabels(), visible = False)

    ax[0,0].text(.7, -.15, 'norm. count', fontweight = 'normal', fontsize = 12, transform=ax[0,0].transAxes)
    ax[1,0].text(.7, -.15, 'norm. count', fontweight = 'normal', fontsize = 12, transform=ax[1,0].transAxes)

suptitle('FC of {} in LIF neurons are independent and Gaussian'.format(['spike trains', 'signal'][spikes_or_sig]), size =15)

for x, ax in enumerate([axs, axs2]):
    
    subplot(ax[0, 0])
    sns.distplot(c_un[x][0], bins = binN, hist = True, kde=False, norm_hist = True, 
                 vertical = True, color = col[x], hist_kws = {'alpha':.8})
    sns.distplot(c_un[x][1], bins = binN, hist = True, kde=False, norm_hist = True, 
                 hist_kws={"histtype": "step", "linewidth": 1.3, "alpha": .75},
                         vertical = True, color = 'k')
    if x ==0: 
        ylabel(r'corr. coef. $\rho_{c,c}$', rotation = 90, fontweight = 'normal', fontsize = 12, labelpad = 10), #title('coefficient of correlation', loc = 'left'), xlabel('counts')
    
    subplot(ax[0, 1])
    sns.distplot(c_re[x][0], bins = binN, hist = True, kde=False, norm_hist = True, 
                 vertical = True, color = col2[x], hist_kws = {'alpha':.8})
    sns.distplot(c_re[x][1], bins = binN, hist = True, kde=False, norm_hist = True, 
                 hist_kws={"histtype": "step", "linewidth": 1.3, "alpha": .75},
                         vertical = True, color = 'k')
    print min(c_re[x][0]), min(c_re[x][1]), min(c_un[x][0]), min(c_un[x][1])
    print max(c_re[x][0]), max(c_re[x][1]), max(c_un[x][0]), max(c_un[x][1])

    #xlabel('norm. count', fontweight = 'normal', fontsize = 13, labelpad = 15)
    
    limits = [min(ax[0,0].get_xlim()[0], ax[0,1].get_xlim()[0]), max(ax[0,0].get_xlim()[1], ax[0,1].get_xlim()[1])]
    ax[0,1].set_xlim(limits)
    ax[0,0].set_xlim(limits)
    ax[0,0].invert_xaxis()
    #ax[0,0].set_ylim(-.4, .4)
    
    ax[0,0].set_title('varying sig.', color = col[x]), ax[0,1].set_title('repeating sig.', color = col2[x])
    ax[0,1].text(.65, .05, ['MM', 'VM'][x], fontweight = 'bold', color = [cm.Reds_r(.0), cm.Blues_r(.0)][x],
                 fontsize = 15, transform=ax[0,1].transAxes)# horizontalalignment='center',verticalalignment='center',
      
###    
### ------------------------------------------------------
###
    subplot(ax[1, 0])
    sns.distplot(log(test_un[x][0]), bins = binN, hist = True, kde=False, norm_hist = True, 
                 vertical = True, color = col[x], hist_kws = {'alpha':.8})
    sns.distplot(log(test_un[x][1]), bins = binN, hist = True, kde=False, norm_hist = True, 
                 hist_kws={"histtype": "step", "linewidth": 1.3,"alpha": .75},
                 vertical = True, color = 'k')
    gca().set(xscale="linear", yscale="linear")

    #xlabel('norm. count', fontweight = 'normal', fontsize = 13, labelpad = 15)
    if x ==0: ylabel('log($z^2$)', rotation = 90, fontweight = 'normal', fontsize = 13, labelpad = 10), #title('normtest (log(z-score))', loc = 'left')

    subplot(ax[1, 1])
    sns.distplot(log(test_re[x][0]), bins = binN, hist = True, kde=False, norm_hist = True, rug = False,
                 vertical = True, color = col2[x], hist_kws = {'alpha':.8})
    sns.distplot(log(test_re[x][1]), bins = binN, hist = True, kde=False, norm_hist = True,
                          hist_kws={"histtype": "step", "linewidth": 1.3, "alpha": .75},
                         vertical = True, color = 'k')
    gca().set(xscale="linear", yscale="linear")
    #gca().set_yticks([])
    limits = [min(ax[1,0].get_xlim()[0], ax[1,1].get_xlim()[0]), max(ax[1,0].get_xlim()[1], ax[1,1].get_xlim()[1])]
    ax[1,1].set_xlim(limits)
    ax[1,0].set_xlim(limits)
    ax[1,0].invert_xaxis()

    ax[1,1].text(.65, .05, ['MM', 'VM'][x], fontweight = 'bold', color = [cm.Reds_r(.0), cm.Blues_r(.0)][x],fontsize = 15, transform=ax[1,1].transAxes)
   
#tight_layout()
    #title('coefficient of correlation', loc = 'left'), xlabel('counts')    
    subplot(ax[0, 1])
    sns.distplot(c_re[x][0], bins = binN, hist = True, kde=False, norm_hist = True, 
                 vertical = True, color = col2[x], hist_kws = {'alpha':.8})
    sns.distplot(c_re[x][1], bins = binN, hist = True, kde=False, norm_hist = True, 
                 hist_kws={"histtype": "step", "linewidth": 1.3, "alpha": .75},
                         vertical = True, color = 'k')
    
    #ax[0,1].set_yticks([])

    limits = [min(ax[0,0].get_xlim()[0], ax[0,1].get_xlim()[0]), max(ax[0,0].get_xlim()[1], ax[0,1].get_xlim()[1])]
    ax[0,1].set_xlim(limits)
    ax[0,0].set_xlim(limits)
    ax[0,0].invert_xaxis()
    #ax[0,0].set_ylim(-.4, .4)
    
    ax[0,0].set_title('varying sig.', color = col[x]), ax[0,1].set_title('repeating sig.', color = col2[x])
    ax[0,1].text(.65, .05, ['MM', 'VM'][x], fontweight = 'bold', color = [cm.Reds_r(.0), cm.Blues_r(.0)][x],
                 fontsize = 15, transform=ax[0,1].transAxes)# horizontalalignment='center',verticalalignment='center',
      
###    
### ------------------------------------------------------
###
    subplot(ax[1, 0])
    sns.distplot(log(test_un[x][0]), bins = binN, hist = True, kde=False, norm_hist = True, 
                 vertical = True, color = col[x], hist_kws = {'alpha':.8})
    sns.distplot(log(test_un[x][1]), bins = binN, hist = True, kde=False, norm_hist = True, 
                 hist_kws={"histtype": "step", "linewidth": 1.3,"alpha": .75},
                 vertical = True, color = 'k')
    gca().set(xscale="linear", yscale="linear")

    #xlabel('norm. count', fontweight = 'normal', fontsize = 13, labelpad = 15)
    if x ==0: ylabel('log($z^2$)', rotation = 90, fontweight = 'normal', fontsize = 13, labelpad = 10), #title('normtest (log(z-score))', loc = 'left')

    subplot(ax[1, 1])
    sns.distplot(log(test_re[x][0]), bins = binN, hist = True, kde=False, norm_hist = True, rug = False,
                 vertical = True, color = col2[x], hist_kws = {'alpha':.8})
    sns.distplot(log(test_re[x][1]), bins = binN, hist = True, kde=False, norm_hist = True,
                          hist_kws={"histtype": "step", "linewidth": 1.3, "alpha": .75},
                         vertical = True, color = 'k')
    gca().set(xscale="linear", yscale="linear")
    #gca().set_yticks([])
    limits = [min(ax[1,0].get_xlim()[0], ax[1,1].get_xlim()[0]), max(ax[1,0].get_xlim()[1], ax[1,1].get_xlim()[1])]
    ax[1,1].set_xlim(limits)
    ax[1,0].set_xlim(limits)
    ax[1,0].invert_xaxis()

    ax[1,1].text(.65, .05, ['MM', 'VM'][x], fontweight = 'bold', color = [cm.Reds_r(.0), cm.Blues_r(.0)][x],
                 fontsize = 15, transform=ax[1,1].transAxes)
show()
