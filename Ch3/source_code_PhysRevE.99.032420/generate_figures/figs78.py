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

# #### global plotting settings

from style_sheet import custom_style, set_inset_params
custom_style(paper = True)
#sns.set_context("paper")

# %matplotlib
#nbagg

# # load data

sig = [200, 250, 300]
uppers = map(chr, range(65, 91))
sigN = [200., 250., 300.]
tauN = [0., 2.5, 5., 10., 15.]
tauS = [10., 20., 30.]
snr = array([.15, .05, .25, .5, .75, 1., 2.])
snr2 = snr
w0_s = [0, 0.25, .508, 1., 2.54, 7.111]
order = argsort(w0_s)
w0_s = sort(w0_s)

# normalized noise strengths

# +
sigNc= [[200., 250., 300.], [95, 130, 165], [40., 70., 100.], [20., 45., 70.], [10., 34., 58.]]
sigN_lif = append([sigNc[0]], [[sigNc[y][x]*sqrt(tauN[y]*2) for x in range(3)] for y in range(1,4)], axis = 0)
mu_lif = [300., 330., 350., 365., 375.]

mu_eif = [300., 300., 310., 330.]
sigN_eif = [[350., 500., 650.], [175., 300., 425.], [100., 200., 300.], [75., 175., 275.]]
sigN_eif = append([sigN_eif[0]], [[sigN_eif[y][x]*sqrt(tauN[y]*2) for x in range(3)] for y in range(1,4)], axis = 0)
# -

sigN_lif, sigN_eif

# %cd ../generate_data/

# +
As = [load("firing-stats-LIF-tauN0.npz"), load("firing-stats-LIF-tauN2.5.npz"),load("firing-stats-LIF-tauN5.npz"), load("firing-stats-LIF-tauN10.npz")]
Bs = [load("firing-stats-EIF-tauN0.npz"), load("firing-stats-EIF-tauN2.5.npz"),load("firing-stats-EIF-tauN5.npz"), load("firing-stats-EIF-tauN10.npz")]

sim_lif, sim_eif = [x.f.sim for x in As], [x.f.sim for x in Bs]
# -

rates_lif = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_lif], dtype = float16)
rates_eif = array([[[[[x1[0][-1] for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_eif], dtype = float16)
cv_lif = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_lif], dtype = float16)
cv_eif = array([[[[[sqrt(x1[0][1]/x1[0][-1]) for x1 in x2] for x2 in x3] for x3 in x4] for x4 in x5] for x5 in sim_eif], dtype = float16)

# # Figures 7 and 8

# is_eif is a boolean that determine whether EIF or LIF data are to be plotted. **is_eif = 0 corresponds to Fig 7, is_eif = 1 to Fig 8**

# +
markers = ['X', 'o', '>']
is_rates = 1

is_eif = 0
rates_or_cv = [[cv_lif, cv_eif],[rates_lif, rates_eif]][is_rates][is_eif] # choose rates vs CV and LIF vs EIF
sigNc = [sigN_lif, sigN_eif][is_eif]
muc = [mu_lif, mu_eif][is_eif]

rates = array([rates_or_cv[:,0,x,0,0]*1000 for x in range(3)])
rates_err = array([std(rates_or_cv[:,0,x,0,0])*1000 for x in range(3)])
rates_mean = array([mean(rates_or_cv[:,0,x,0,0])*1000 for x in range(3)])

print('rates : ' ,rates_mean, rates_err)

fig, axs = plt.subplots(nrows=3, ncols=1, sharex= 'col', sharey=False, 
                            gridspec_kw={'height_ratios': [1.25, 1., 1.]},
                            figsize=(3.5, 6.)
                       )
for i, ax in enumerate(hstack(axs)):
    ax.text(-.25, 1.1, uppers[i], size = 14, transform=ax.transAxes, weight  = 'bold', usetex = 0) 
### tau vs sigma/mu plot
ax1 = axs[0]
[ax1.plot(tauN[:4], array(sigNc)[:4,x], ls = '-', c= 'k', marker = markers[x], linewidth = .35, label = r'$\hat\sigma_n^{{({})}}\sim \nu_{{{}}}$'.format(x+1, x+1)) for x in range(3)]
ax2 = ax1.twinx()
ax2.plot(tauN[:4], muc[:4], 's', c = 'darkkhaki', ls = '-', lw = .35)
ax2.tick_params('y', colors='darkkhaki')
ax2.spines['right'].set_visible(1)
ax2.spines['right'].set_color('darkkhaki')

#ax1.set_xlabel(r"noise time constant $\tau_n$ [ms]")
ax1.set_ylabel(r"$\sigma_n$ [pA  $\cdot \mathrm{ms}^{{1/2}}$]")
ax2.set_ylabel(r"$\mu$ [pA]", color = 'darkkhaki')

#### rate and CV plot
for is_rates in [True, False]:
    rates_or_cv = [[cv_lif, cv_eif],[rates_lif, rates_eif]][is_rates][is_eif] # choose rates vs CV and LIF vs EIF

    rates = array([rates_or_cv[:,0,x,0,0]*1000 for x in range(3)])
    rates_err = array([std(rates_or_cv[:,0,x,0,0])*1000 for x in range(3)])
    rates_mean = array([mean(rates_or_cv[:,0,x,0,0])*1000 for x in range(3)])
    
    sca(axs[2-is_rates])
    if is_rates == 0:
        rates /= 1000.
        ylim([.5, 1.])
    [plot(tauN[:4], rates[t], ls = '-', c = 'k', marker = markers[t], linewidth = .35, label = r'$\hat\sigma_n^{{({})}}\sim \nu_{{{}}}$'.format(t+1, t+1)) for t in range(3)]
    if is_rates == 1:
        [hlines(rates_mean[x], tauN[0], tauN[3], color = 'lightgrey') for x in range(3)]
        [fill_between(tauN[:4], rates_mean[x]+rates_err[x], rates_mean[x]-rates_err[x], color = 'lightgrey', alpha = .3) for x in range(3)]
        [text(tauN[3]-.5, rates_mean[x]+1, r'$\nu_{{{}}}$'.format(x+1), horizontalalignment = 'right') for x in range(3)]
    ylabel([r'coeff of var. $CV_{\mathrm{ISI}}$', r'firing rate $\nu$ [Hz]'][is_rates])

xlabel(r'noise time constant $\tau_n$ [ms]')
if is_eif == 0: ax1.legend(fontsize = 'small', labelspacing = .3, columnspacing = 1.2, loc = 'upper left', bbox_to_anchor = (.05, .350))
else: ax1.legend(fontsize = 'small', labelspacing = .3, columnspacing = 1.2, loc = 'upper left', bbox_to_anchor = (.02, 1.05))
tight_layout()
show()
# -


