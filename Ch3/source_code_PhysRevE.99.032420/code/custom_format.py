from pylab import *
from mpl_toolkits.mplot3d import Axes3D


def f(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%.0f}$' % x

def f2(x, pos):
    return r'$%g$' % 10**x

def f3(x, pos):
    x1 = int(x)
    x2 = 10**(x-x1)
    return r'${:.3f}\cdot 10^{{{:d}}}$'.format(x2,x1)

formatter = FuncFormatter(f)
formatter2 = FuncFormatter(f2)
formatter3 = FuncFormatter(f3)
