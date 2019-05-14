"""circstats.py -- circular statistics module

This software is licensed under the terms of the MIT License as
follows:

Copyright (c) 2013 Jessica B. Hamrick, Peter W. Battaglia

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

__all__ = [
    'difference',
    'wrapdiff',
    'normalize',
    'csr',
    'to_complex',
    'to_real',
    'mean',
    'nanmean',
    'resvec',
    'nanresvec',
    'var',
    'nanvar',
    'std',
    'nanstd',
    'confmean',
    'nanconfmean',
    'corrcc',
    'nancorrcc',
    'vmpar',
    'nanvmpar',
    'vmlogpdf',
    'vmpdf',
    'kappa',
    'nankappa'
]

import numpy as np
from scipy.stats import chi2
from scipy.special import iv


def difference(a1, a2, deg=False):
    """Compute the smallest difference between two angle arrays.

    Parameters
    ----------
    a1, a2 : np.ndarray
        The angle arrays to subtract
    deg : bool (default=False)
        Whether to compute the difference in degrees or radians

    Returns
    -------
    out : np.ndarray
        The difference between a1 and a2

    """

    diff = a1 - a2
    return wrapdiff(diff, deg=deg)


def wrapdiff(diff, deg=False):
    """Given an array of angle differences, make sure that they lie
    between -pi and pi.

    Parameters
    ----------
    diff : np.ndarray
        The angle difference array
    deg : bool (default=False)
        Whether the angles are in degrees or radians

    Returns
    -------
    out : np.ndarray
        The updated angle differences

    """

    if deg:
        base = 360
    else:
        base = np.pi * 2

    i = np.abs(diff) > (base / 2.0)
    out = diff.copy()
    out[i] -= np.sign(diff[i]) * base
    return out


def normalize(data, deg=False):
    """Make sure that all values in `data` lie between 0 and 2*pi (or
    0 and 360, for degrees).

    Parameters
    ----------
    data : number or np.array
        The value(s) to normalize
    deg : bool (default=False)
        Whether the values are in degrees or radians

    Returns
    -------
    out : number np.ndarray
        The normalized value(s)

    """

    if deg:
        base = 360
    else:
        base = np.pi * 2

    data = data % base
    return data


def csr(data, axis=None, nanrobust=False, deg=False):
    """Finds the mean cosines, sines, and radii of an array of angles
    along the given axis.

    Parameters
    ----------
    data : np.ndarray
        The angle array
    axis : int (default=None)
        The axis along which to take the mean computations
    nanrobust : bool (default=False)
        Ignore nans
    deg : bool (default=False)
        Whether the values are in degrees or radians

    Returns
    -------
    out : tuple
        (mean cosine, mean sine, mean radius)

    """

    if deg:
        data = np.radians(data)

    if nanrobust:
        n = np.sum(~np.isnan(data), axis=axis)
        C = np.nansum(np.cos(data), axis=axis) / n
        S = np.nansum(np.sin(data), axis=axis) / n
    else:
        C = np.mean(np.cos(data), axis=axis)
        S = np.mean(np.sin(data), axis=axis)

    R = (C ** 2 + S ** 2) ** 0.5

    return (C, S, R)


def to_complex(data, deg=False):
    """Convert an array of angles to complex values, e ** (data * 1j).

    Parameters
    ----------
    data : np.ndarray
        The angle values
    deg : bool (default=False)
        Whether the values are in degrees or radians

    Returns
    -------
    out : np.ndarray
        An array of complex-valued angles

    """

    if deg:
        d = np.radians(data)
    else:
        d = data.copy()
    return np.exp(d * 1j)


def to_real(data, deg=False):
    """Convert an array of complex values to angles.

    Parameters
    ----------
    data : np.ndarray
        The complex values
    deg : bool (default=False)
        Whether the values are in degrees or radians

    Returns
    -------
    out : np.ndarray
        An array of angles

    """

    return np.angle(data, deg=deg)


def _mean(alpha, nanrobust, axis=None, w=None, fconf=False):
    if w is None:
        w = np.ones(alpha.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha)
    if nanrobust:
        r = np.nansum(t, axis=axis)
        w[np.isnan(t)] = 0
    else:
        r = np.sum(t, axis=axis)

    # obtain mean by
    #mu = normalize(np.angle(r))
    mu = np.angle(r)

    if fconf:
        # confidence limits if desired
        if nanrobust:
            t = nanconfmean(alpha, w=w, axis=axis)
        else:
            t = confmean(alpha, w=w, axis=axis)
        ul = mu + t
        ll = mu - t

        return (mu, ul, ll)
    else:
        return mu


def mean(alpha, axis=None, w=None, fconf=False, nancheck=True):
    """Calculate the mean of an array of angles using circular
    statistics.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    fconf : bool (default=False)
        Whether or not to return confidence intervals along with the
        means
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : np.ndarray OR (np.ndarray, np.ndarray, np.ndarray)
        The resulting numpy array, or a tuple of the array, the upper
        confidence limit, and the lower confidence limit

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  "
              "If you want to ignore this message, "
              "please set nancheck to False.  If you "
              "want to make the mean robust to NaNs, "
              "please use the nanmean function instead.")

    return _mean(alpha, False, axis=axis, w=w, fconf=fconf)


def nanmean(alpha, axis=None, w=None, fconf=False):
    """Calculate the mean of an array of angles using circular
    statistics.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    fconf : bool (default=False)
        Whether or not to return confidence intervals along with the
        means

    Returns
    -------
    out : np.ndarray OR (np.ndarray, np.ndarray, np.ndarray)
        The resulting numpy array, or a tuple of the array, the upper
        confidence limit, and the lower confidence limit

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _mean(alpha, True, axis=axis, w=w, fconf=fconf)


def _resvec(alpha, nanrobust, axis=None, w=None, d=0):
    if w is None:
        w = np.ones(alpha.shape)

    # weight sum of cos & sin angles
    t = w * np.exp(1j * alpha)
    if nanrobust:
        r = np.nansum(t, axis=axis)
        w[np.isnan(t)] = 0
    else:
        r = np.sum(t, axis=axis)

    # obtain length
    r = np.abs(r) / np.sum(w, axis=axis)

    # for data with known spacing, apply correction factor to correct
    # for bias in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d != 0:
        r *= d / 2 / np.sin(d / 2)

    return r


def resvec(alpha, axis=None, w=None, d=0, nancheck=True):
    """Computes mean resultant vector length for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : np.ndarray
        The resulting numpy array of resultant vector lengths

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  If you "
              "want to ignore this message, please set nancheck "
              "to False.  If you want to make the mean robust to "
              "NaNs, please use the nanresvec function instead.")

    return _resvec(alpha, False, axis=axis, w=w, d=d)


def nanresvec(alpha, axis=None, w=None, d=0):
    """Computes mean resultant vector length for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned

    Returns
    -------
    out : np.ndarray
        The resulting numpy array of resultant vector lengths

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _resvec(alpha, True, axis=axis, w=w, d=d)


def _var(alpha, nanrobust, axis=None, w=None, d=0, angvar=False):
    if w is None:
        w = np.ones(alpha.shape)

    # compute mean resultant vector length
    if nanrobust:
        r = nanresvec(alpha, axis=axis, w=w, d=d)
    else:
        r = resvec(alpha, axis=axis, w=w, d=d)

    # apply transformation to var
    S = 1 - r  # circular variance
    if angvar:
        var = 2 * S  # angular variance
    else:
        var = S

    return var


def var(alpha, axis=None, w=None, d=0, angvar=False, nancheck=True):
    """Computes circular variance for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned
    angvar : bool (default=False):
        If True, calculate angular variance, otherwise calculate
        circular variance
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The calculated circular variance and angular variance

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  If you want "
              "to ignore this message, please set nancheck to "
              "False.  If you want to make the mean robust to NaNs, "
              "please use the nanvar function instead.")

    return _var(alpha, False, axis=axis, w=w, d=d, angvar=angvar)


def nanvar(alpha, axis=None, w=None, d=0, angvar=False):
    """Computes circular variance for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned
    angvar : bool (default=False):
        If True, calculate angular variance, otherwise calculate
        circular variance

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The calculated circular variance and angular variance

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _var(alpha, True, axis=axis, w=w, d=d, angvar=angvar)


def _std(data, nanrobust, axis=None, deg=False, varroot=False):
    # convert to radians
    if deg:
        data = np.radians(data)

    # compute mean resultant vector length
    if nanrobust:
        R = nanresvec(data, axis=axis)
    else:
        R = resvec(data, axis=axis)

    # if just compute the square root, compute the variance and then
    # take the square root
    if varroot:
        V = 1 - R
        sd = np.sqrt(V)
    # otherwise calculate true circular standard deviation
    else:
        sd = np.sqrt(-2 * np.log(R))

    # convert back to degrees, if necessary
    if deg:
        sd = np.degrees(sd)

    return sd


def std(data, axis=None, deg=False, varroot=False, nancheck=True):
    """Computes circular standard deviation for circular data.

    Parameters
    ----------
    data : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    deg : bool (default=False)
        Whether the data is in degrees or radians
    varroot : bool (default=False)
        If True, compute the square root of the variance.  If False,
        compute the circlar standard deviation.
    nancheck : bool (default=True):
        Whether to check for NaNs in the input array

    Returns
    -------
    out : np.ndarray
        The calculated circular standard deviation

    References
    ----------
    http://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread

    """

    if nancheck and np.isnan(data).any():
        raise (ValueError,
               "NaN(s) detected in the input array.  If you want "
               "to ignore this message, please set nancheck to "
               "False.  If you want to make the stddev robust to "
               "NaNs, please use the nanstd function instead.")
    return _std(data, False, axis=axis, deg=deg, varroot=varroot)


def nanstd(data, axis=None, deg=False, varroot=False):
    """Computes circular standard deviation for circular data and is
    robust to invalid entries in the array.

    Parameters
    ----------
    data : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    deg : bool (default=False)
        Whether the data is in degrees or radians
    varroot : bool (default=False)
        If True, compute the square root of the variance.  If False,
        compute the circlar standard deviation.

    Returns
    -------
    out : np.ndarray
        The calculated circular standard deviation

    References
    ----------
    http://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread

    """

    return _std(data, True, axis=axis, deg=deg, varroot=varroot)


def _confmean(alpha, nanrobust, axis=None, ci=0.95, w=None, d=0):
    if w is None:
        w = np.ones(alpha.shape)

    # compute ingredients for conf. lim.
    if nanrobust:
        r = nanresvec(alpha, axis=axis, w=w, d=d)
    else:
        r = resvec(alpha, axis=axis, w=w, d=d)
    n = np.sum(w, axis=axis)
    R = n * r
    c2 = chi2.isf(1-ci, 1)

    # check for resultant vector length and select appropriate formula
    t = np.zeros(r.shape)

    tscalar = np.isscalar(r)
    if tscalar:
        r = np.array([r])
        n = np.array([n])
        R = np.array([R])
        t = np.array([t])

    # fill in values
    i = (r < 0.9) & (r > np.sqrt(c2 / 2. / n))
    t[i] = np.sqrt((2 * n[i] * (2 * R[i]**2 - n[i] * c2)) / (4 * n[i] - c2))

    j = r >= 0.9
    t[j] = np.sqrt(n[j] ** 2 - (n[j] ** 2 - R[j] ** 2) * np.exp(c2 / n[j]))

    t[~(i | j)] = np.nan

    # apply final transform
    t = np.arccos(t / R)
    if tscalar:
        t = t[0]

    return t


def confmean(alpha, axis=None, ci=0.95, w=None, d=0, nancheck=True):
    """Calculate the confidence limit of the mean of circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    ci : number (default=0.95)
        The confidence interval, between 0 and 1
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : np.ndarray
        The confidence limit(s) of the input array

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  If you want "
              "to ignore this message, please set nancheck to False.  "
              "If you want to make the mean robust to NaNs, please use "
              "the nanconfmean function instead.")

    return _confmean(alpha, False, axis=axis, ci=ci, w=w, d=d)


def nanconfmean(alpha, axis=None, ci=0.95, w=None, d=0):
    """Calculate the confidence limit of the mean of circular data.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    ci : number (default=0.95)
        The confidence interval, between 0 and 1
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned

    Returns
    -------
    out : np.ndarray
        The confidence limit(s) of the input array

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _confmean(alpha, True, axis=axis, ci=ci, w=w, d=d)


def _corrcc(alpha1, alpha2, nanrobust, axis=None):
    if axis is not None and alpha1.shape[axis] != alpha2.shape[axis]:
        raise(ValueError, "shape mismatch")

    # compute mean directions
    if axis is None:
        n = alpha1.size
    else:
        n = alpha1.shape[axis]

    #################################################################
    c1 = np.cos(alpha1)
    c1_2 = np.cos(2*alpha1)
    c2 = np.cos(alpha2)
    c2_2 = np.cos(2*alpha2)
    s1 = np.sin(alpha1)
    s1_2 = np.sin(2*alpha1)
    s2 = np.sin(alpha2)
    s2_2 = np.sin(2*alpha2)

    if nanrobust:
        sumfunc = lambda x: np.nansum(x, axis=axis)
    else:
        sumfunc = lambda x: np.sum(x, axis=axis)

    num = 4 * (sumfunc(c1*c2) * sumfunc(s1*s2) -
               sumfunc(c1*s2) * sumfunc(s1*c2))
    den = np.sqrt((n**2 - sumfunc(c1_2)**2 - sumfunc(s1_2)**2) *
                  (n**2 - sumfunc(c2_2)**2 - sumfunc(s2_2)**2))

    rho = num / den

    return rho


def corrcc(alpha1, alpha2, axis=None, nancheck=True):
    """Calculate the correlation coefficient between two sets of
    circular data.

    Parameters
    ----------
    alpha1 : np.ndarray
        The first set of circular data
    alpha2 : np.ndarray
        The second set of circular data
    axis : int (default=None)
        The axis along which to take the mean
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The correlation coefficient(s) and the p-value(s)

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    Fisher, N. & Lee, A. (1983). A correlation coefficient for circular
      data. Biometrika. 70 (2), 327-332.

    """

    if nancheck and (np.isnan(alpha1).any() or np.isnan(alpha2).any()):
        raise ValueError("NaN(s) detected in the input array. "
                         "If you want to ignore this message, please set "
                         "nancheck to False. If you want to make the mean "
                         "robust to NaNs, please use the nancorrcc function "
                         "instead.")

    return _corrcc(alpha1, alpha2, False, axis=axis)


def nancorrcc(alpha1, alpha2, axis=None):
    """Calculate the correlation coefficient between two sets of
    circular data.

    Parameters
    ----------
    alpha1 : np.ndarray
        The first set of circular data
    alpha2 : np.ndarray
        The second set of circular data
    axis : int (default=None)
        The axis along which to take the mean

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The correlation coefficient(s) and the p-value(s)

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10


    Fisher, N. & Lee, A. (1983). A correlation coefficient for circular
      data. Biometrika. 70 (2), 327-332.

    """

    return _corrcc(alpha1, alpha2, True, axis=axis)


def _vmpar(alpha, nanrobust, axis=None, w=None, d=0):
    if w is None:
        w = np.ones(alpha.shape)

    if nanrobust:
        _kappa = nankappa(alpha, axis=axis, w=w, d=d)
        _thetahat = nanmean(alpha, axis=axis, w=w)
    else:
        _kappa = kappa(alpha, axis=axis, w=w, d=d)
        _thetahat = mean(alpha, axis=axis, w=w)

    return (_thetahat, _kappa)


def vmpar(alpha, axis=None, w=None, d=0, nancheck=True):
    """Estimate the parameters of a Von Mises distribution.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The theta and kappa parameters, respectively

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  "
              "If you want to ignore this message, please "
              "set nancheck to False.  If you want to make "
              "the mean robust to NaNs, please use the nanvmpar "
              "function instead.")

    return _vmpar(alpha, False, axis=axis, w=w, d=d)


def nanvmpar(alpha, axis=None, w=None, d=0):
    """Estimate the parameters of a Von Mises distribution.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    d : number (default=0)
        The bin spacing, if the data is binned

    Returns
    -------
    out : tuple (np.ndarray, np.ndarray)
        The theta and kappa parameters, respectively

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _vmpar(alpha, True, axis=axis, w=w, d=d)


def vmlogpdf(alpha, thetahat=0.0, kappa=1.0):
    """Computes the circular Von Mises log-PDF with preferred
    direction thetahat and concentration kappa at each of the angles
    in alpha.

    The vmpdf is given by f(phi) =
    (1/(2pi*I0(kappa))*exp(kappa*cos(phi-thetahat)

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    thetahat : number
        The preferred direction
    kappa : number
        Concentration at each of the angles in alpha

    Returns
    -------
    out : np.ndarray
        The Von Mises log-PDF for alpha

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    C = -np.log(2 * np.pi * iv(0, kappa))
    p = C + (kappa * np.cos(alpha - thetahat))
    return p


def vmpdf(alpha, thetahat=0.0, kappa=1.0):
    """Computes the circular Von Mises PDF with preferred direction
    thetahat and concentration kappa at each of the angles in alpha.

    The vmpdf is given by f(phi) =
    (1/(2pi*I0(kappa))*exp(kappa*cos(phi-thetahat)

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    thetahat : number
        The preferred direction
    kappa : number
        Concentration at each of the angles in alpha

    Returns
    -------
    out : np.ndarray
        The Von Mises PDF for alpha

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    p = np.exp(vmlogpdf(alpha, thetahat=thetahat, kappa=kappa))
    return p


def _kappa(alpha, nanrobust, axis=None, w=None, d=0):
    if w is None:
        w = np.ones(alpha.shape)

    if axis is None:
        n = alpha.size
    else:
        n = alpha.shape[axis]

    if nanrobust:
        R = nanresvec(alpha, axis=axis, w=w, d=d)
    else:
        R = resvec(alpha, axis=axis, w=w, d=d)

    if type(R) is not np.ndarray or len(R.shape) == 0:
        R = np.array([R])

    i1 = R < 0.53
    i2 = (R >= 0.53) & (R < 0.85)
    i3 = ~(i1 | i2)

    kappa = np.empty(R.shape)
    kappa[i1] = 2*R[i1] + R[i1]**3 + 5*R[i1]**5/6
    kappa[i2] = -.4 + 1.39*R[i2] + 0.43/(1-R[i2])
    kappa[i3] = 1/(R[i3]**3 - 4*R[i3]**2 + 3*R[i3])

    if (n < 15) & (n > 1):
        i1 = kappa < 2
        i2 = ~i1
        kappa[i1] = np.max(np.concatenate((kappa[i1]-2*(n*kappa[i1])**-1,
                                           np.zeros(i1.shape)),
                                          axis=i1.ndim-1),
                           axis=i1.ndim-1)
        kappa[i2] = (n-1)**3 * kappa[i2] / (n**3 + n)

    return kappa


def kappa(alpha, axis=None, w=None, d=0, nancheck=True):
    """Computes an approximation to the ML estimate of the
    concentration parameter kappa of the von Mises distribution.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired
    nancheck : bool (default=True)
        Whether or not to check for NaNs in the array

    Returns
    -------
    out : np.ndarray
        Kappa parameter estimations

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    if nancheck and np.isnan(alpha).any():
        raise(ValueError,
              "NaN(s) detected in the input array.  If "
              "you want to ignore this message, please set "
              "nancheck to False.  If you want to make the "
              "mean robust to NaNs, please use the nankappa "
              "function instead.")

    return _kappa(alpha, False, axis=axis, w=w, d=d)


def nankappa(alpha, axis=None, w=None, d=0):
    """Computes an approximation to the ML estimate of the
    concentration parameter kappa of the von Mises distribution.

    Parameters
    ----------
    alpha : np.ndarray
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    w : np.ndarray (default=None)
        An array of weights, if a weighted mean is desired

    Returns
    -------
    out : np.ndarray
        Kappa parameter estimations

    References
    ----------
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular
      Statistics. Journal of Statistical Software, 31(10). Available
      from http://www.jstatsoft.org/v31/i10

    """

    return _kappa(alpha, True, axis=axis, w=w, d=d)
