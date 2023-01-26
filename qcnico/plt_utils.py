import numpy as np
from matplotlib import rcParams, cm
import matplotlib.pyplot as plt


def get_cm(vals, cmap_str, max_val=0.7):
    '''Creates a list of colours from an array of numbers. Can be used to colour-code curves corresponding to
        different values of a given paramter.'''
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    sorted_vals = np.sort(vals)
    delta = sorted_vals[-1] - sorted_vals[0]
    x = max_val * (vals - sorted_vals[0]) / delta
    if isinstance(cmap_str, str):
        if cmap_str[:3] == 'cm.':
            cmap = eval(cmap_str)
        else:
            cmap = eval('cm.' + cmap_str)
    else:
        print('[get_cm] ERROR: The colour map must be specified as a string (e.g. "plasma" --> cm.plasma).\nSetting the colour map to viridis.')
        cmap = cm.viridis
    return cmap(x)


def setup_tex(preamble_str=None):
    rcParams['text.usetex'] = True
    if preamble_str:
        rcParams['text.latex.preamble'] = preamble_str
    else:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}  \usepackage{bm}'


def histogram(values,nbins=100,normalised=False,xlabel=None,ylabel=None,show=True,plt_kwargs=None):
    hist, bins = np.histogram(values,nbins)
    dx = bins[1:] - bins[:-1]
    centers = (bins[1:] + bins[:-1])/2
    hist = hist.astype(np.float64)

    if normalised:
        hist /= values.size #sum of the bin counts will be equal to 1
    
    if plt_kwargs: # Note: plt_kwargs is a dictionary of keyword arguments
        if 'color' in plt_kwargs:
            plt.bar(centers, hist,align='center',width=dx,**plt_kwargs)
        else:
            plt.bar(centers, hist,align='center',width=dx,color='r',**plt_kwargs)
    else:
        plt.bar(centers, hist,align='center',width=dx,color='r')
    if xlabel:
        plt.xlabel(xlabel)
    
    if ylabel:
        plt.ylabel(ylabel)
    elif ylabel == None and normalised:
        plt.ylabel('Normalised counts')
    else:
        plt.ylabel('Counts')
    if show:
        plt.show()
