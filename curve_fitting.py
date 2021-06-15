#!/usr/bin/env python3

"""
Assorted methods for fitting non-linear functions.
"""

import itertools
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, minimize


#### Function definitions ####

def sigmoid(x, x0, k):
    """
    Sigmoid (logistic) function.

    Arguments
    ---------
    x : array like, required
        Values to plot over.
    x0 : float, required
        Mid-point of function (y=0.5 when x=x0).
    k : float, required
        Slope parameter of function.

    Returns
    -------
    y : numpy array
        y-values of function.
    """
    return 1 / ( 1 + np.exp(-k*(x-x0)) )

def weibullCDF(x, gamma, k):
    """
    Weibull cumulative density function.

    Arguments
    ---------
    x : array like, required
        Values to plot over.
    gamma : float, required
        Scale parameter.  Parameter is often referred to as lambda, but that
        name has a special meaning in Python so we use gamma instead here.
    k : float, required
        Shape parameter.

    Returns
    -------
    y : numpy array
        y-values of function.
    """
    y = np.zeros_like(x, dtype=float)
    y[x>=0] = 1 - np.exp(-(x[x>=0]/gamma)**k)
    return y


##### Class definitions #####

class _BaseFitFunction(object):
    """
    Base class arguments
    --------------------
    func : function instance, optional
        Objective function to fit.  Must accept an array of x-values as its
        first argument, and then any further arguments should be function
        parameters that are to be estimated via the fitting process.  Default
        is to use a Gaussian CDF.
    invfunc : function instance or None, optional
        Inverse of main function. Only necessary if wanting to use the
        .getXForY method.
    fit_method : 'mle' or 'lsq', optional
        Fitting method to use.  Pass 'mle' to use maximum-likelihood estimation
        (default), or 'lsq' to use non-linear least squares.
    ymin : float, optional
        Expected minimum value of y (e.g. use to adjust for chance level). Can
        also specify as 'optim' to add this as an additional optimisation
        parameter - note that in this case the ymin parameter must be appended
        as the final (if lapse != 'optim') or penultimate (if lapse == 'optim')
        value in any starting parameters, bounds, etc.
    lapse : float | 'optim', optional
        Lapse parameter (expected ymax = 1 - lapse). Can also specify as str
        'optim' to add this as an additional optimisation parameter - note
        that in this case the lapse parameter must then be appended as the
        final value in any starting parameters, bounds, etc.

    Base class methods
    ------------------
    .doFit
        Performs function fitting.
    .doInterp
        Returns interpolated values for x and y variables, e.g. for plotting.
    .getFittedParams
        Return fitted parameters, assuming .doFit has already been run.
    .getXForY
        Use inverse function to get x-value for given y-value.
    """

    """
    Separate docstring so it's not inherited by child classes. Base class
    provides methods for performing function fits. May pass to child classes.

    If wanting to do MLE, child class must implement a method .negLogLik,
    taking an array of parameters as its only argument, and returning
    a single-value giving the negative log-likelihood.

    Child class must assign self.x attribute - an array containing values for
    predictor variable. If wanting to do least square, child class must
    additionally assign self.y attribute - an array containing values for
    outcome variable.
    """
    def __init__(self, func=stats.norm.cdf, invfunc=stats.norm.ppf,
                 fit_method='mle', ymin=0, lapse=0):

        # Error check
        if fit_method not in ['lsq','mle']:
            raise ValueError("fit_method must be one of 'lsq' or 'mle', " \
                             f"but received: {fit_method}")

        if not ( (isinstance(ymin, str) and ymin == 'optim') \
                 or isinstance(ymin, (int, float)) ):
            raise ValueError("ymin must be numeric or 'optim', " \
                             f"but received: {ymin}")

        if not ( (isinstance(lapse, str) and lapse == 'optim') \
                 or isinstance(lapse, (int, float)) ):
            raise ValueError("lapse must be numeric or 'optim', " \
                             f"but received: {lapse}")

        # Assign args to class
        self.fit_method = fit_method
        self.ymin = ymin
        self.lapse = lapse
        self.fit = None  # place holder for fit results
        self.selected_x0 = None  # place holder for grid search results

        # Assign funcs last as setter methods need access to other attributes
        self.func = func
        self.invfunc = invfunc

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, fun):
        """Setter adjusts forward func for ymin and lapse params"""
        def _func(x, *params):
            params = list(params)
            lapse = params.pop(-1) if self.lapse == 'optim' else self.lapse
            ymin = params.pop(-1) if self.ymin == 'optim' else self.ymin
            return ymin + (1 - ymin - lapse) * np.asarray(fun(x, *params))
        self._func = _func

    @property
    def invfunc(self):
        return self._invfunc

    @invfunc.setter
    def invfunc(self, fun):
        """Setter adjusts inverse func for ymin and lapse params"""
        if fun is None:
            self._invfunc = None
            return

        def _invfunc(y, *params):
            params = list(params)
            lapse = params.pop(-1) if self.lapse == 'optim' else self.lapse
            ymin = params.pop(-1) if self.ymin == 'optim' else self.ymin
            return fun((np.asarray(y) - ymin) / (1 - ymin - lapse), *params)
        self._invfunc = _invfunc

    def doFit(self, *args, **kwargs):
        """
        Performs the function fit.  May include further positional and / or
        keyword arguments to be passed to the relevant optimization function.
        These vary depending on the fitting method chosen:
        * If using MLE, these should be additional arguments for the
          scipy.optimize.minimize function. Note that this requires a first
          positional argument or keyword argument 'x0' containing initial
          guesses for the non-linear function parameters.
        * If using non-linear least squares, should be additional arguments for
          the scipy.optimize.curve_fit function. Although not required, it is
          recommended that you supply a first positional argument or keyword
          argument 'p0' containing initial guesses for the non-linear
          function parameters.
        * See the documentation for the minimize / curve_fit functions for
          more details on other available arguments.
        * If optimising ymin parameter (self.ymin == 'optim') then this
          parameter must be included as the final (if self.lapse != 'optim')
          or penultimate (if self.lapse == 'optim') value in any relevant args.
        * If optimising lapse parameter (self.lapse == 'optim') then this
          parameter must be included as the final value in any relevant args.
        * Starting parameters can also be specified as a list of lists, where
          each inner list contains a range of values for a given parameter
          (i.e. each inner list represents a function parameter in turn).
          In this case a grid search is performed over all parameter
          combinations, and the best performing set is selected for the
          optimisation procedure. The selected values will be stored in the
          .selected_x0 attribute.

        Results are stored within the .fit attribute of this class, and can
        also be accessed with the .getFittedParams method.
        """
        # Pop out starting params (so we can check for grid search)
        if self.fit_method == 'mle' and 'x0' in kwargs:
            x0 = kwargs.pop('x0')
        elif self.fit_method == 'lsq' and 'p0'in kwargs:
            x0 = kwargs.pop('p0')
        else:
            args = list(args)
            x0 = args.pop(0)

        # Grid search?
        if all(hasattr(_x0, '__iter__') for _x0 in x0):
            x0grid = list(itertools.product(*x0))
            errs = []
            for _x0 in x0grid:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if self.fit_method == 'mle':
                        err = self.negLogLik(_x0)
                    elif self.fit_method == 'lsq':
                        err = np.sum((self.func(self.x, *_x0) - self.y)**2)
                errs.append(err)
            self.selected_x0 = x0 = x0grid[np.nanargmin(errs)]

        # Main fit
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.fit_method == 'mle':
                self.fit = minimize(self.negLogLik, x0, *args, **kwargs)
            elif self.fit_method == 'lsq':
                self.fit = curve_fit(self.func, self.x, self.y, x0,
                                     *args, **kwargs)

    def getFittedParams(self):
        """
        Return fitted parameters if .doFit has been run.
        """
        if self.fit is None:
            raise RuntimeError('Must call .doFit method first')

        if self.fit_method == 'mle':
            return self.fit.x
        elif self.fit_method == 'lsq':
            return self.fit[0]

    def getXForY(self, y):
        """
        Use inverse function to get corresponding x-value for given y-value.
        Requires that an inverse function has been supplied and .doFit method
        has been run.
        """
        params = self.getFittedParams()
        if self.invfunc is None:
            raise RuntimeError('Inverse function was not supplied')
        return self.invfunc(y, *params)

    def doInterp(self, npoints=100, interpX=None):
        """
        Returns an (x,y) tuple of values interpolated along x-dimension(s),
        which can be used for plotting.

        Arguments
        ---------
        npoints : int or list of ints, optional
            Number of points to interpolate. Ignored if interpX is not None.
        interpX : array-like or None, optional
            Interpolated x-values to calculate y-values over. If None, will
            create a default range (see npoints).

        Returns
        -------
        interpX, interpY
            Interpolated x- and y-values respectively.
        """
        params = self.getFittedParams()
        if interpX is None:
            interpX = np.linspace(self.x.min(), self.x.max(), npoints)
        interpY = self.func(interpX, *params)
        return (interpX, interpY)

    def negLogLik(self):
        """
        Place holder for negLogLik function to be implemented by child class.
        """
        raise NotImplementedError()


class FitFunction(_BaseFitFunction):
    """
    Class provides functions for fitting a non-linear function via maximum
    likelihood (using a normal PDF) or non-linear least squares.

    Arguments
    ---------
    x, y : array-like, required
        Predictor and outcome variables, respectively, which the function is to
        be fit to.  Each should be an (nsamples,) 1D array.
    mle_costfunc : function instance
        Cost function for evaluating log-likelihood of residuals. Ignored if
        fit_method is 'lsq'. Should accept array of y-axis residual values
        as its only argument. Default is a Gaussian PDF.
    *args, **kwargs :
        Futher arguments passed to base class (see below)

    Methods
    -------
    .negLogLik
        Return negative log-likelihood for a given set of params

    See also
    --------
    * MLEBinomFitFunction: Class for fitting a non-linear function via MLE
      using a Binomial PMF - use for data derived from Bernoulli trials.

    Example usage
    -------------
    Generate some (slightly) noisy data from a Gaussian CDF.

    >>> true_mu, true_sigma = -3, 1.5
    >>> x = np.linspace(-10, 10, 30)
    >>> y = scipy.stats.norm.cdf(x, true_mu, true_sigma)
    >>> ynoise = y + (np.random.rand(len(y)) - 0.5) * 0.3

    Fit the function using MLE.  For both the initial guess (x0) and the bound
    values, the parameters refer to the Gaussian CDF parameters (mu, sigma).
    Bounds are specified as (min, max) tuples for each of parameter in turn.

    >>> mle_fit = FitFunction(x, ynoise, fit_method='mle')
    >>> mle_x0 = (0, 1)
    >>> mle_bounds = ( (min(x), max(x)), (1e-5, None))
    >>> mle_fit.doFit(x0=mle_x0, bounds=mle_bounds, method='L-BFGS-B')
    >>> mle_params = mle_fit.getFittedParams()

    Fit the function using non-linear least squares. For both the initial
    guess (p0) and the bound values, the parameters refer to the Gaussian CDF
    parameters (mu, sigma).  Bounds are specified as two tuples giving the
    minimum and maximum values respectively, each containing values for each
    of the function parameters in turn.

    >>> lsq_fit = FitFunction(x, ynoise, fit_method='lsq')
    >>> lsq_p0 = (0, 1)
    >>> lsq_bounds = ( (min(x), 1e-5), (max(x), np.inf) )
    >>> lsq_fit.doFit(p0=lsq_p0, bounds=lsq_bounds, method='trf')
    >>> lsq_params = lsq_fit.getFittedParams()

    Inspect function fits.

    >>> print(f'True parameters: mu = {true_mu}, sigma = {true_sigma}')
    >>> print(f'MLE parameter estimates: mu = {mle_params[0]}, sigma = {mle_params[1]}')
    >>> print(f'LSQ parameter estimates: mu = {lsq_params[0]}, sigma = {lsq_params[1]}')

    Make a plot.

    >>> mle_interpX, mle_interpY = mle_fit.doInterp()
    >>> lsq_interpX, lsq_interpY = lsq_fit.doInterp()
    >>> plt.figure()
    >>> plt.plot(x, y, 'k-', label='True fit')
    >>> plt.plot(x, ynoise, 'go', label='Noisy data')
    >>> plt.plot(mle_interpX, mle_interpY, 'r--', label='MLE fit')
    >>> plt.plot(lsq_interpX, lsq_interpY, 'b:', label='LSQ fit')
    >>> plt.legend()
    >>> plt.show()
    """
    __doc__ += _BaseFitFunction.__doc__

    def __init__(self, x, y, mle_costfunc=stats.norm.pdf, *args, **kwargs):
        self.x = np.asarray(list(x))
        self.y = np.asarray(list(y))
        self.mle_costfunc = mle_costfunc
        super(FitFunction, self).__init__(*args, **kwargs)

    def negLogLik(self, params):
        """
        Computes negative log likelihood for a given set of function params.
        """
        ypred = self.func(self.x, *params)
        resid = self.y - ypred
        return -np.log(self.mle_costfunc(resid).clip(min=1e-10)).sum()


class MLEBinomFitFunction(_BaseFitFunction):
    """
    Class contains functions for fitting a non-linear function in the special
    case where the data are derived from Bernoulli trials, i.e. where an event
    either happens or doesn't.  Fit is done via MLE using a binomial PMF cost
    function.

    Arguments
    ---------
    x : array-like, required
        The values along which the predictor variable varies. May be bin values
        in the case where x is discrete, or just a list of the individual
        x-values where x is continuous.  Should be an (nsamples,) 1D array.
    counts : array-like of ints, required
        The counts (sums) of the number of occurences of the measured event
        within each level of x. These can be the 0 and 1 values from the
        Bernoulli trials themselves if x is continous.
    n : int or array-like of ints, required
        The total number of trials.  Can be a single integer in which case the
        same number is assumed for all levels of x, or a list of separate
        integers for each level of x. This can be set to 1 if x is continuous.
    *args, **kwargs :
        Further arguments passed to base class (see below). Note that the
        fit_method argument will be overidden and set to 'mle'.

    Methods
    -------
    .negLogLik
        Return negative log-likelihood for a given set of params

    Example usage
    -------------
    Generate some random data.  We will have x-values in 11 discrete bins
    ranging from -10 to +10.  For each bin we will simulate a random number of
    trials (between 5 and 8), generating a random set of binary responses
    (i.e. as per a Bernoulli trial) with the probability of responding 0 or 1
    at the given x-value determined by a Gaussian CDF.

    >>> true_mu, true_sigma = -2, 3
    >>> x = np.linspace(-10, 10, 11)
    >>> responses = []
    >>> for this_x in x:
    ...     this_n = np.random.choice((5,6,7,8))
    ...     p = scipy.stats.norm.cdf(this_x, true_mu, true_sigma)
    ...     these_resps = np.random.choice((0,1), size=this_n, p=(1-p,p))
    ...     responses.append(these_resps)
    >>> counts = [sum(r) for r in responses]
    >>> n = [len(r) for r in responses]

    Fit the function.  For both the inital guess (x0) and the bound values,
    parameters refer to those of the Gaussian CDF parameters (mu, sigma).
    Bounds are specified as (min, max) tuples for each of the parameters in
    turn.

    >>> fit = MLEBinomFitFunction(x, counts, n)
    >>> x0 = (0, 1)
    >>> bounds = ( (min(x), max(x)), (1e-5, None) )
    >>> fit.doFit(x0=x0, bounds=bounds, method='L-BFGS-B')
    >>> mle_params = fit.getFittedParams()

    Inspect function fits.

    >>> print(f'True parameters: mu = {true_mu}, sigma = {true_sigma}')
    >>> print(f'MLE parameter estimates: mu = {mle_params[0]}, sigma = {mle_params[1]}')

    Make a plot.

    >>> interpX, interpY = fit.doInterp()
    >>> plt.figure()
    >>> plt.plot(interpX, stats.norm.cdf(interpX, true_mu, true_sigma),
    ...          'k-', label='True fit')
    >>> plt.plot(x, list(map(np.mean, responses)), 'bo', label='Mean response')
    >>> plt.plot(interpX, interpY, 'r--', label='MLE fit')
    >>> plt.legend()
    >>> plt.show()
    """
    __doc__ += _BaseFitFunction.__doc__

    def __init__(self, x, counts, n, *args, **kwargs):
        self.x = np.asarray(list(x))
        self.counts = np.asarray(list(counts))
        if hasattr(n, '__iter__'):
            self.n = np.asarray(list(n))
        else:
            self.n = n
        kwargs['fit_method'] = 'mle'  # force MLE
        super(MLEBinomFitFunction, self).__init__(*args, **kwargs)

    def negLogLik(self, params):
        """
        Computes negative log likelihood via a Binomial PMF cost function,
        using probability values generated by the given objective function.
        Params should be for the objective function.
        """
        p = self.func(self.x, *params)
        return -np.log(stats.binom.pmf(self.counts, self.n, p).clip(min=1e-10)).sum()
