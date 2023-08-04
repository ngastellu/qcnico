#!/usr/bin/env python

import numpy as np
from scipy.fftpack import fft
from scipy.integrate import simps


# Set of functions used to evaluate the vibrational density of states (VDOS) of a molecule given its atomic
# velocities obtained from a MD run (e.g. from a LAMMPS dump file). Pretty much all of this code is shamelessly
# copied/pasted from Steve Schemerler's PWtools package (https://github.com/elcorto/pwtools/tree/master/pwtools).
# The function `vdos` is the main function of this module. It is a version of PWtools' `direct_pdos` function.



def welch(M, sym=1):
    """Welch window. Function skeleton shamelessly stolen from
    scipy.signal.bartlett() and others."""
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M+1
    n = np.arange(0,M)
    w = 1.0-((n-0.5*(M-1))/(0.5*(M-1)))**2.0
    if not sym and not odd:
        w = w[:-1]
    return w

def slicetake(a, sl, axis=None, copy=False):
    """The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.

    Parameters
    ----------
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None
            `sl` is a list or tuple of slice objects, one for each axis.
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view

    Returns
    -------
    A view into `a` or copy of a slice of `a`."""

    if axis is None:
        slices = sl
    else:
        # Note that these are equivalent:
        #   a[:]
        #   a[s_[:]]
        #   a[slice(None)]
        #   a[slice(None, None, None)]
        #   a[slice(0, None, None)]
        slices = [slice(None)] * a.ndim
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
    slices = tuple(slices)
    if copy:
        return a[slices].copy()
    else:
        return a[slices]

def norm_int(y, x, area=1.0, scale=True, func=simps):
    """Normalize integral area of y(x) to `area`.

    Parameters
    ----------
    x,y : numpy 1d arrays
    area : float
    scale : bool, optional
        Scale x and y to the same order of magnitude before integration.
        This may be necessary to avoid numerical trouble if x and y have very
        different scales.
    func : callable
        Function to do integration (like scipy.integrate.{simps,trapz,...}
        Called as ``func(y,x)``. Default: simps

    Returns
    -------
    scaled y

    Notes
    -----
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to the order used in the scipy.integrate routines.
    """
    if scale:
        fx = np.abs(x).max()
        fy = np.abs(y).max()
        sx = x / fx
        sy = y / fy
    else:
        fx = fy = 1.0
        sx, sy = x, y
    # Area under unscaled y(x).
    _area = func(sy, sx) * fx * fy
    return y * area / _area

def pad_zeros(arr, axis=0, where='end', nadd=None, upto=None, tonext=None,
              tonext_min=None):
    """Pad an nd-array with zeros. Default is to append an array of zeros of
    the same shape as `arr` to arr's end along `axis`.

    Parameters
    ----------
    arr :  nd array
    axis : the axis along which to pad
    where : string {'end', 'start'}, pad at the end ("append to array") or
        start ("prepend to array") of `axis`
    nadd : number of items to padd (i.e. nadd=3 means padd w/ 3 zeros in case
        of an 1d array)
    upto : pad until arr.shape[axis] == upto
    tonext : bool, pad up to the next power of two (pad so that the padded
        array has a length of power of two)
    tonext_min : int, when using `tonext`, pad the array to the next possible
        power of two for which the resulting array length along `axis` is at
        least `tonext_min`; the default is tonext_min = arr.shape[axis]

    Use only one of nadd, upto, tonext.

    Returns
    -------
    padded array
    """
    if tonext == False:
        tonext = None
    lst = [nadd, upto, tonext]
    assert lst.count(None) in [2,3], "`nadd`, `upto` and `tonext` must be " +\
           "all None or only one of them not None"
    if nadd is None:
        if upto is None:
            if (tonext is None) or (not tonext):
                # default
                nadd = arr.shape[axis]
            else:
                tonext_min = arr.shape[axis] if (tonext_min is None) \
                             else tonext_min
                # beware of int overflows starting w/ 2**arange(64), but we
                # will never have such long arrays anyway
                two_powers = 2**np.arange(30)
                assert tonext_min <= two_powers[-1], ("tonext_min exceeds "
                    "max power of 2")
                power = two_powers[np.searchsorted(two_powers,
                                                  tonext_min)]
                nadd = power - arr.shape[axis]
        else:
            nadd = upto - arr.shape[axis]
    if nadd == 0:
        return arr
    add_shape = list(arr.shape)
    add_shape[axis] = nadd
    add_shape = tuple(add_shape)
    if where == 'end':
        return np.concatenate((arr, np.zeros(add_shape, dtype=arr.dtype)), axis=axis)
    elif where == 'start':
        return np.concatenate((np.zeros(add_shape, dtype=arr.dtype), arr), axis=axis)
    else:
        raise Exception("illegal `where` arg: %s" %where)
    

def vdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
         npad=1, tonext=False, mirr=False):
    """Phonon DOS by FFT of the VACF or direct FFT of atomic velocities.

    Integral area is normalized to `area`. It is possible (and recommended) to
    zero-padd the velocities (see `npad`).

    Parameters
    ----------
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,),
        atomic mass array, if None then mass=1.0 for all atoms is used
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    npad : {None, int}
        method='direct' only: Length of zero padding along `axis`. `npad=None`
        = no padding, `npad > 0` = pad by a length of ``(nstep-1)*npad``. `npad
        > 5` usually results in sufficient interpolation.
    tonext : bool
        method='direct' only: Pad `vel` with zeros along `axis` up to the next
        power of two after the array length determined by `npad`. This gives
        you speed, but variable (better) frequency resolution.
    mirr : bool
        method='vacf' only: mirror one-sided VACF at t=0 before fft

    Returns
    -------
    if full_out = False
        | ``(faxis, pdos)``
        | faxis : 1d array [1/unit(dt)]
        | pdos : 1d array, the phonon DOS, normalized to `area`
    if full_out = True
        | if method == 'direct':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx))``
        | if method == 'vavcf':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))``
        |     fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        |     vacf : 1d array, the VACF
    """
    mass = m
    # assume vel.shape = (nstep,natoms,3)
    axis = 0
    assert vel.shape[-1] == 3
    if mass is not None:
        assert len(mass) == vel.shape[1], "len(mass) != vel.shape[1]"
        # define here b/c may be used twice below
        mass_bc = mass[None,:,None]
    if window:
        sl = [None]*vel.ndim
        sl[axis] = slice(None)  # ':'
        vel2 = vel*(welch(vel.shape[axis])[tuple(sl)])
    else:
        vel2 = vel

    # padding
    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, tonext=True,
                             tonext_min=vel2.shape[axis] + nadd,
                             axis=axis)
        else:
            vel2 = pad_zeros(vel2, tonext=False, nadd=nadd, axis=axis)
    
    # Do the FFT
    full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
    full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
    split_idx = len(full_faxis)//2
    faxis = full_faxis[:split_idx]

    # First split the array, then multiply by `mass` and average. 
    arr = full_fft_vel
    fft_vel = slicetake(arr, slice(0, split_idx), axis=axis, copy=False)
    if mass is not None:
        fft_vel *= mass_bc

    # average remaining axes, summing is enough b/c normalization is done below
    # sums: (nstep, natoms, 3) -> (nstep, natoms) -> (nstep,)
    while fft_vel.ndims > 1:
        fft_vel = fft_vel.sum(-1)
    vdos = norm_int(fft_vel,faxis,area=area)
    out = (faxis, vdos)
    return out