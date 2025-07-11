
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from .plt_utils import setup_tex
from .qchemMAC import MO_rgyr, MCO_com, MCO_rgyr, MO_rgyr_hyperlocal
from .graph_tools import adjacency_matrix_sparse


def plot_atoms(pos,dotsize=45.0,colour='k',show_cbar=False, usetex=True,show=True, plt_objs=None,zorder=3):

    if pos.shape[1] == 3:
        pos = pos[:,:2]

    rcParams['font.size'] = 16

    if usetex:
        setup_tex()
        
    if plt_objs == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    # if all atoms have the same size/colour, `plot` is faster than `scatter`
    if hasattr(dotsize, '__iter__') or hasattr(colour,'__iter__'): # checks if `dotsize` or `colour` are iterable
        ye = ax.scatter(*pos.T, c=colour, s=dotsize, zorder=zorder,alpha=1.0)
    else:
        ye = ax.plot(*pos.T, c=colour, ms=dotsize, zorder=zorder,alpha=1.0)

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
    else: # should not be True if `show=True`
        return fig, ax

def plot_atoms_w_bonds(pos,A,dotsize=45.0,colour='k', bond_colour='k', bond_lw=0.5,usetex=True,show=True, plt_objs=None,zorder_atoms=3,zorder_bonds=3):

    if usetex:
        setup_tex()

    if plt_objs is None:
        fig, ax = plot_atoms(pos,dotsize=dotsize,colour=colour,show=False,zorder=zorder_atoms)
    else:
        fig, ax = plot_atoms(pos,dotsize=dotsize,colour=colour,show=False,plt_objs=plt_objs,zorder=zorder_atoms)

    pairs = np.vstack(A.nonzero()).T
    
    for i,j in pairs:
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], c=bond_colour, ls='-', lw=bond_lw,zorder=zorder_bonds)
    
    if show: 
        plt.show()
    
    else: # should not be True if `show=True`
        return fig, ax
    



def plot_MO(pos,MO_matrix, n, dotsize=45.0, cmap='plasma', cnorm=None,show_COM=False, show_rgyr=False, show_bonds=False , plot_amplitude=False, 
            scale_up=1.0, com_clr = 'r', title=None, show_title =True, usetex=True, show=True, plt_objs=None, zorder=1,scale_up_threshold=0.001,
            show_cbar=True,loc_centers=None, loc_radii=None, c_clrs='r',c_markers='h',c_labels=None,c_rel_size=5,c_lw=3.0,bond_size=0.5):

    if pos.shape[1] == 3:
        pos = pos[:,:2]
    if isinstance(n, int) or isinstance(n, np.int64):
        psi = MO_matrix[:,n]
        density = np.abs(psi)**2
    else:
        print(type(n))
        n = np.array(n)
        psi = MO_matrix[:,n].sum(axis=1)
        density = (np.abs(MO_matrix[:,n])**2).sum(axis=1)


    # rcParams['font.size'] = 16

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

    sizes = dotsize * np.ones(pos.shape[0])

    sizes[density > scale_up_threshold] *= scale_up #increase size of high-probability sites
    
    if plot_amplitude:
        ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=sizes,cmap=cmap,norm=colors.CenteredNorm(),zorder=zorder) #CenteredNorm() sets center of cbar to 0
    else:
        ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=density,s=sizes,cmap=cmap,zorder=zorder,norm=cnorm) #CenteredNorm() sets center of cbar to 0

    if show_cbar:
        cbar = fig.colorbar(ye,ax=ax1,orientation='vertical',label='Density $|\langle\\varphi_i|\psi\\rangle|^2$',ticks=[])

    if show_title:
        if title is None: 
            if isinstance(n,int):
                if plot_amplitude:
                    plt.suptitle('$\langle\\varphi_n|\psi_{%d}\\rangle$'%n)
                else:
                    plt.suptitle('$|\langle\\varphi_n|\psi_{%d}\\rangle|^2$'%n)
        else:
            plt.suptitle(title)

    if show_bonds:
        add_bonds_MO(pos, psi, ax1, zorder_bonds = zorder-1,cmap=cmap, bond_size=bond_size)


    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    
    if show_COM or show_rgyr:
        com = density @ pos
        label = '$\langle\psi|\\bm{r}|\psi\\rangle$'
        if not show_rgyr:
            ax1 = add_MO_centers(com[None,:],ax1,marker='*',clr=com_clr,dotsize=dotsize*c_rel_size,zorder=zorder+1,labels=label,lw=c_lw)
    if show_rgyr:
        rgyr = MO_rgyr(pos,MO_matrix,n,center_of_mass=com)
        print('RGYR = ', rgyr)
        ax1 = add_MO_centers(com[None,:],ax1,[rgyr],marker='*',clr=com_clr,dotsize=dotsize*c_rel_size,zorder=zorder+1,labels=label,lw=c_lw)
        # ax1 = add_MO_centers(com[None,:],ax1,[rgyr],clr=com_clr,zorder=zorder+1)

    if loc_centers is not None:
        ax1 = add_MO_centers(loc_centers,ax1,radii=loc_radii,clr=c_clrs,marker=c_markers,labels=c_labels,dotsize=dotsize*c_rel_size,zorder=zorder+1,lw=c_lw)    

    #line below turns off x and y ticks 
    #ax1.tick_params(axis='both',which='both',bottom=False,top=False,right=False, left=False)

    if c_labels is not None or show_COM:
        ax1.legend()

    if show:
        plt.show()
    else:
        return fig, ax1


def add_bonds_MO(pos, psi, ax, bond_size=0.5,zorder_bonds=0,cmap='plasma', rCC = 1.8, plot_amplitude=False, cnorm=None): 
    A = adjacency_matrix_sparse(pos, rCC)
    pairs = np.vstack(A.nonzero()).T
    t = np.linspace(0,1,100)

    if not plot_amplitude:
        psi = psi**2

    if cnorm is None:
        cnorm = colors.Normalize(vmin=np.min(psi),vmax=np.max(psi))

    for ij in pairs:
        # Define (i,j) ordering as psi[i] <= psi[j]
        psis = psi[ij]
        isorted = np.argsort(psis)
        i = ij[isorted[0]]
        j = ij[isorted[1]]
        edge_pts = pos[i,:] + t[:,None] * (pos[j,:] - pos[i,:])
        edge_psis =  psi[i] + t * (psi[j] - psi[i])
        ax.scatter(edge_pts[:,0], edge_pts[:,1], c=edge_psis, cmap=cmap, norm=cnorm,zorder=zorder_bonds,s=bond_size,edgecolor='none')
    # return ax
        

def add_MO_centers(centers, ax, radii=None,  clr='r', marker='*',labels=None , dotsize=10,zorder=2,ls='--',lw=3.0):
    """This function adds localisation centers (either COM or hopping sites) to a pre-existing MO figure.
    
    Parameters
    ----------
    centers: `np.ndarray`, shape=(N,2)
        Array containing Cartesian coords of the points
    ax: `matplotlib.axes.Axes`
        Axes object of the MO plot
    clr: `str` or `list` of `str`
        String(s) describing the colour(s) of the centers on the plot
    marker: `str`
        String(s) describing the marker style(s) of the centers on the plot
    labels:  `str` or `list` of `str`
        Labels of the centers
    dotsize: `float`
        Dot size of the centers on the plot
    ls: `str` or `list` of `str`
        String(s) describing the linestyle(s) of the radii on the plot
    lw: `float` or iterable of `float`
       Linewidth(s) of the radii on the plot
    
    Returns
    -------
    ax: Same as input arg, but of the modified figure
    """

    if not isinstance(centers,np.ndarray):
        centers = np.array(centers)
    
    if centers.ndim == 1:
        centers = centers.reshape(1,-1)

    if labels is not None:
        if isinstance(labels,list):
            ax.scatter(*centers[-1].T, s=dotsize,marker=marker,c=clr,zorder=zorder,label=labels)
        else: #if only a single label is specicfied, apply it only to the last center (avoids having many times the same legend)
            ax.scatter(*centers[:-1].T, s=dotsize,marker=marker,c=clr,zorder=zorder,edgecolors='k',lw=1.0)
            ax.scatter(*centers[-1].T, s=dotsize,marker=marker,c=clr,zorder=zorder,label=labels,edgecolors='k',lw=1.0)
    else:
        ax.scatter(*centers.T, s=dotsize,marker=marker,c=clr,zorder=zorder+1,edgecolors='k',lw=0.5)
    
    if radii is not None:
        assert len(radii) == centers.shape[0], f'Number of radii ({len(radii)}) does not match number of centers ({centers.shape[0]})!'
        if not isinstance(clr, list):
            clr = [clr] * len(radii) 
        if not isinstance(ls, list):
            ls = [ls] * len(radii) 
        if not isinstance(lw, list) and not isinstance(lw,np.ndarray):
            lw = [lw] * len(radii) 
        for c, r, cl, w, s in zip(centers,radii,clr,lw,ls):
            loc_circle = plt.Circle(c, r, fc='none', ec=cl, ls=s, lw=w,zorder=zorder)
            ax.add_patch(loc_circle)
    return ax


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
    elif n == -6:
        return 'darkgreen'
    elif n == 6: #crystallite hexagons
        return "limegreen"
    elif n == 7 or n == 8:
        return 'blue'
    else:
        return 'lightsteelblue'


def plot_rings_MAC(pos,M,ring_sizes,ring_centers,atom_labels=None,dotsize_atoms=45.0,dotsize_centers=300.0,plt_objs=None,show=True):
    pos = pos[:,:2] # assume all z coords are 0 (project everything to xy plane)
    ring_centers = ring_centers[:,:2]
    
    if atom_labels is not None:
        if np.unique(atom_labels).shape[0] > 2:
            atom_colours = list(map(size_to_clr,atom_labels))
        else: # if only 6c atoms are labelled (binary labelling)
            atom_colours = ['limegreen' if l else 'k' for l in atom_labels]
    else:
        atom_colours = ['k'] * pos.shape[0]
    
    if plt_objs is None:
        fig, ax = plot_atoms(pos,colour=atom_colours,dotsize=dotsize_atoms,show=False,zorder=10)
    else:
        fig, ax = plot_atoms(pos,colour=atom_colours,dotsize=dotsize_atoms,show=False,plt_objs=plt_objs,zorder=10)

    pairs = np.vstack(M.nonzero()).T
    
    for i,j in pairs:
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'k-', lw=0.8,zorder=10)
    
    ring_colours = list(map(size_to_clr,ring_sizes))
        
    ax.scatter(*ring_centers.T,c=ring_colours,s=dotsize_centers)


    if show:
        plt.show()
    else:
        return fig, ax
