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


def setup_tex(fontsize=18,preamble_str=None):
    rcParams['text.usetex'] = True
    rcParams['font.size'] = fontsize
    if preamble_str:
        rcParams['text.latex.preamble'] = preamble_str
    else:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}  \usepackage{bm}'


def histogram(values,nbins=100,normalised=False,density=False,xlabel=None,ylabel=None,log_counts=False,
              plt_objs=None,show=True,usetex=True,plt_kwargs=None,print_dx=True,return_data=False,return_data_w_dx=False):
    hist, bins = np.histogram(values,nbins,density=density)
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
    
    if usetex:
        setup_tex()

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
    
    else:
        if return_data:
            return fig, ax, centers, hist
        
        elif return_data_w_dx:
            return fig, ax, centers, hist, dx
        
        else:
            return fig, ax

     
    if return_data:
        return centers, hist
    
    elif return_data_w_dx:
        return centers, hist, dx
    


def multiple_histograms(vals_arr, labels, nbins=100, colors=None, alpha=0.6, normalised=False,density=False,xlabel=None,ylabel=None,log_counts=False,
              plt_objs=None,show=True,plt_kwargs=None,print_dx=True,return_data=False,usetex=True,title=None):
    
    ndatasets = len(vals_arr)
    
    if ndatasets == 2 and colors is None:
        colors = ['r', 'b']
    elif colors is None:
        cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
        colors = [d['color'] for d in list(cyc[0:ndatasets])]
    
    if usetex:
        setup_tex()

    if plt_objs is None:
        plt_objs = plt.subplots()
        
    fig, ax = plt_objs
    
    
    if title is not None:
        ax.set_title(title)

    for k, vals in enumerate(vals_arr):
        if plt_kwargs is None:
            plt_kwargs2 = {'color': colors[k],'label':labels[k], 'alpha': alpha}
        else:
            plt_kwargs2 = plt_kwargs | {'color': colors[k],'label':labels[k], 'alpha': alpha}
       
        if (k == ndatasets - 1):
            if show:
                histogram(vals, nbins=nbins, normalised=normalised, density=density, xlabel=xlabel, ylabel=ylabel,
                          log_counts=log_counts, plt_objs=plt_objs, plt_kwargs=plt_kwargs2, print_dx=print_dx, show=False)
                plt.legend()
                plt.show()
            else: 
                histogram(vals, nbins=nbins, normalised=normalised, density=density, xlabel=xlabel, ylabel=ylabel,
                          log_counts=log_counts, plt_objs=plt_objs, plt_kwargs=plt_kwargs2, print_dx=print_dx, show=False)
                return fig, ax
        
        else:
                histogram(vals, nbins=nbins, normalised=normalised, density=density, xlabel=xlabel, ylabel=ylabel,
                          log_counts=log_counts, plt_objs=plt_objs, plt_kwargs=plt_kwargs2, print_dx=print_dx, show=False)


def MAC_ensemble_colors():
    """A handy function to enforce a consistent color scheme when plotting quantities obtained from 
    MAC structures of the three following ensembles:
        * ensemble generated by Michael's PixelCNN model
        * ensemble generated by Ata's model with T = 0.6
        * ensemble generated by Ata's model with T = 0.5
    
    The color attributed to each of these ensembles is derived from that ensembles degrees of crystallinity (as
    measured by its fraction of crystalline hexagons.
    """

    dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

    ring_data_tempdot5 = np.load(dd_rings + 'avg_ring_counts_tempdot5_new_model_relaxed.npy')
    ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')
    ring_data_tempdot6 = np.load(dd_rings + 'avg_ring_counts_tempdot6_new_model_relaxed.npy')

    p6c_tempdot6 = ring_data_tempdot6[4] / ring_data_tempdot6.sum()
    p6c_tempdot5 = ring_data_tempdot5[4] / ring_data_tempdot5.sum()
    p6c_pCNN = ring_data_pCNN[4]
    # p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1,p6c_tempdot6])
    p6c = np.array([p6c_pCNN,p6c_tempdot6,p6c_tempdot5])

    clrs = get_cm(p6c, 'inferno',min_val=0.25,max_val=0.7)

    return clrs



    
    
