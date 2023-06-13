import numpy as np
from matplotlib import rcParams, cm
import matplotlib.pyplot as plt


def get_cm(vals, cmap_str, max_val=0.7, min_val=0.0):
    '''Creates a list of colours from an array of numbers. Can be used to colour-code curves corresponding to
        different values of a given paramter.'''
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    sorted_vals = np.sort(vals)
    delta = sorted_vals[-1] - sorted_vals[0]
    x = min_val + ( (max_val-min_val) * (vals - sorted_vals[0]) / delta )
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


def histogram(values,nbins=100,normalised=False,xlabel=None,ylabel=None,log_counts=False,plt_objs=None,show=True,plt_kwargs=None,print_dx=True):
    hist, bins = np.histogram(values,nbins)
    dx = bins[1:] - bins[:-1]
    centers = (bins[1:] + bins[:-1])/2
    hist = hist.astype(np.float64)

    if print_dx:
        print(f"[plt_utils.histogram] dx = {np.mean(dx)}")

    if normalised:
        hist /= values.size #sum of the bin counts will be equal to 1

    # If integrating histogram w/in a pre-exisiting figure (as created by `plt.subplots()`),
    # unpack plt_objs. Else, create figure.
    if plt_objs:
        fig, ax = plt_objs
    else:
        fig, ax = plt.subplots()
    
    if plt_kwargs: # Note: plt_kwargs is a dictionary of keyword arguments
        if 'color' in plt_kwargs:
            ax.bar(centers, hist,align='center',width=dx,**plt_kwargs)
        else:
            ax.bar(centers, hist,align='center',width=dx,color='r',**plt_kwargs)
    else:
        ax.bar(centers, hist,align='center',width=dx,color='r')
    if xlabel:
        ax.set_xlabel(xlabel)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    elif ylabel == None and normalised:
        ax.set_ylabel('Normalised counts')
    else:
        ax.set_ylabel('Counts')

    if log_counts:
        ax.set_yscale('log')

    if show:
        plt.show()
