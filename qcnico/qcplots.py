
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .plt_utils import setup_tex
from .qchemMAC import MO_rgyr, MCO_com, MCO_rgyr


def plot_atoms(pos,dotsize=45.0,colour='k',show_cbar=False, usetex=True,show=True, plt_objs=None, return_plt_objs=False):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    rcParams['font.size'] = 16

    if usetex:
        setup_tex()
        
    if plt_objs == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    ye = ax.scatter(*pos.T, c=colour, s=dotsize)

    #uncomment below to remove whitespace around plot
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #ax.axis('tight')

    ax.set_xlabel('$x$ [\AA]')
    ax.set_ylabel('$y$ [\AA]')
    ax.set_aspect('equal')

    if show_cbar:
        cbar = fig.colorbar(ye,ax=ax,orientation='vertical')

    if show:
        plt.show()
        #plt.close()

    if return_plt_objs: # should not be True if `show=True`
        return fig, ax



def plot_MO(pos,MO_matrix,n,dotsize=45.0,show_COM=False,show_rgyr=False,usetex=True,show=True, plt_objs=None):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    psi = np.abs(MO_matrix[:,n])**2

    rcParams['font.size'] = 16

    if usetex:
        setup_tex()

    #if plot_type == 'nanoribbon':
    #    #rcParams['figure.figsize'] = [30.259946/2,7/2]
    #    figsize = [12,11/2]
    #elif plot_type == 'square':
    #    figsize = [4,4]  
    #else:
    #    print('Invalid plot type. Using default square plot type.')
    #    figsize = [4,4]

    if plt_objs is None:
        fig, ax1 = plt.subplots()
    else:
        fig, ax1 = plt_objs
    #fig.set_size_inches(figsize,forward=True)

    ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=dotsize,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1,orientation='vertical')
    plt.suptitle('$|\langle\\varphi_n|\psi_{%d}\\rangle|^2$'%n)
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    if show_COM or show_rgyr:
        com = psi @ pos
        ax1.scatter(*com, s=dotsize*10,marker='*',c='r')
    if show_rgyr:
        rgyr = MO_rgyr(pos,MO_matrix,n,center_of_mass=com)
        loc_circle = plt.Circle(com, rgyr, fc='none', ec='r', ls='--', lw=1.0)
        ax1.add_patch(loc_circle)

    #line below turns off x and y ticks 
    #ax1.tick_params(axis='both',which='both',bottom=False,top=False,right=False, left=False)

    if show:
        plt.show()


def plot_MCO(pos,P,Pbar,n,dotsize=45.0,show_COM=False,show_rgyr=False,plot_dual=False,usetex=True,show=True):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    if plot_dual:
        psi = np.abs(Pbar[:,n])**2
        plot_title = '$|\langle\\varphi_n|\\bar{\psi}_{%d}\\rangle|^2$'%n
    else:
        psi = np.abs(P[:,n])**2
        plot_title = '$|\langle\\varphi_n|\psi_{%d}\\rangle|^2$'%n

    rcParams['font.size'] = 16

    if usetex:
        setup_tex()

    fig, ax1 = plt.subplots()
    #fig.set_size_inches(figsize,forward=True)

    ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=dotsize,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1,orientation='vertical')
    plt.suptitle(plot_title)
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    if show_COM or show_rgyr:
        com = MCO_com(pos, P, Pbar, n)
        print(com)
        ax1.scatter(*com, s=dotsize+1,marker='*',c='r')
    if show_rgyr:
        rgyr = MCO_rgyr(pos,P,Pbar,n,center_of_mass=com)
        loc_circle = plt.Circle(com, rgyr, fc='none', ec='r', ls='--', lw=1.0)
        ax1.add_patch(loc_circle)

    if show:
        plt.show()


def plot_loc_discrep(iprs, rgyrs, energies, dotsize=10, cmap='viridis' ,usetex=True, show=True, plt_objs=None, show_cbar=True):

    iprs = 1/np.sqrt(iprs)
    
    if plt_objs is None:
        fig, ax1 = plt.subplots()
    else:
        fig, ax1 = plt_objs

    rcParams['font.size'] = 16

    if usetex:
        setup_tex()

    ye = ax1.scatter(iprs,rgyrs,marker='o',c=energies,s=dotsize, cmap=cmap)
    if show_cbar:
        cbar = fig.colorbar(ye, ax=ax1)
    ax1.set_ylabel('$\sqrt{\langle R^2\\rangle - \langle R\\rangle^2}$')
    ax1.set_xlabel('1/$\sqrt{IPR}$')
    if show:
        plt.show()

def size_to_clr(n):
    """Associates a colour string to each ring size"""

    if n == 3:
        return 'fuchsia'
    elif n == 4:
        return 'aqua'
    elif n == 5:
        return 'red'
    elif n == 6:
        return 'darkgreen'
    elif n == -6: #crystallite hexagons
        return "limegreen"
    elif n == 7 or n == 8:
        return 'blue'
    else:
        return 'lightsteelblue'


def plot_rings_MAC(pos,M,ring_sizes,ring_centers,dotsize_atoms=45.0,dotsize_centers=300.0,plt_objs=None,show=True,return_plt_objs=False):
    pos = pos[:,:2] # assume all z coords are 0 (project everything to xy plane)
    ring_centers = ring_centers[:,:2]
    
    if plt_objs is None:
        fig, ax = plot_atoms(pos,dotsize=dotsize_atoms,show=False,return_plt_objs=True)
    else:
        fig, ax = plot_atoms(pos,dotsize=dotsize_atoms,show=False,plt_objs=plt_objs,return_plt_objs=True)

    pairs = np.vstack(M.nonzero()).T
    
    for i,j in pairs:
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'k-', lw=0.5)
    
    ring_colours = list(map(size_to_clr,ring_sizes))
    ax.scatter(*ring_centers.T,c=ring_colours,s=dotsize_centers)

    if return_plt_objs:
        return fig, ax

    if show:
        plt.show()