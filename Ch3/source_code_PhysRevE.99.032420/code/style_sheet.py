from pylab import *
import seaborn as sns

def custom_style(paper = False):
	sns.set_style('white')
	sns.set(style = "ticks", color_codes = True)
	
	plt.rc('figure', titlesize= 'large')

	if paper:
		print 'paper style set'
		sns.set_context('paper')
	#turned out to be very helpful for alignment of tex text in legends	
	mpl.rcParams['text.usetex'] = True
	mpl.rcParams['text.latex.preview'] = True

	# label and tick padding
	rcParams['axes.labelpad'] = 3;	
	plt.rcParams['xtick.major.pad']= 2
	plt.rcParams['ytick.major.pad']= 2
	plt.rcParams['xtick.minor.pad']= 2
	plt.rcParams['ytick.minor.pad']= 2
	#remove spines	
	plt.rc('axes.spines', right = False, top = False);
	plt.rc('xtick', top = False)
	plt.rc('ytick', right = False)

	plt.rcParams['legend.frameon'] = False

	plt.rcParams['lines.markersize'] = 5.5
	'''
	plt.rcParams['figure.figsize'] = (8, 3)
	plt.rcParams['font.size'] = 10
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
	plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
	plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
	plt.rcParams['xtick.major.size'] = 3
	plt.rcParams['xtick.minor.size'] = 3
	plt.rcParams['xtick.major.width'] = 1
	plt.rcParams['xtick.minor.width'] = 1
	plt.rcParams['ytick.major.size'] = 3
	plt.rcParams['ytick.minor.size'] = 3
	plt.rcParams['ytick.major.width'] = 1
	plt.rcParams['ytick.minor.width'] = 1

	plt.rcParams['legend.loc'] = 'center left'
	plt.rcParams['axes.linewidth'] = 1
	'''

def set_inset_params(inset_axis):
	inset_axis.spines['bottom'].set_linewidth(0.5)
	inset_axis.spines['left'].set_linewidth(0.5)
	inset_axis.tick_params(axis='both', which='major', pad= 1, labelsize = 7)
