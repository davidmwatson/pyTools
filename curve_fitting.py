#!/usr/bin/env python3
"""
Provides class for fitting non-linear functions to a binary outcome variable
via MLE.

Also provides classes giving forward and inverse functions for some common
psychometric function types (actually the functions are already in scipy,
these classes just wrap those into a more convenient format):
 * Gaussian - gives Gaussian CDF (forward) and quantile (inverse) functions.
 * Sigmoid - gives Sigmoid logistic (forward) and logit (inverse) functions.
 * Weibull - gives Weibull CDF (forward) and quantile (inverse) functions.
"""

import itertools
import numpy as np
import scipy.stats
from scipy.optimize import minimize


class Gaussian(object):
    """
    Gaussian cumulative density and quantile functions.

    Wraps scipy.stats.norm.

    Parameters
    ----------
    x,y : array like
        1D array of x/y values to plot over

    mu : float
        Mean parameter; ``y = 0.5`` when ``x = mu``

    sigma : float
        Standard deviation parameter

    Equations
    ---------
    **CDF**

    .. math::

        f(x) = \\frac{1}{2} \\left[ 1 + \\text{erf} \\left(
                   \\frac{x - \\mu}{\\sigma \\sqrt{2} }
                   \\right) \\right]

    **Quantile**

    .. math::

        f(y) = \\mu + \\sigma \\sqrt{2} \\ \\text{erf}^{-1} (2y - 1)
    """
    @staticmethod
    def cdf(x, mu, sigma):
        return scipy.stats.norm.cdf(x, loc=mu, scale=sigma)

    @staticmethod
    def quantile(y, mu, sigma):
        return scipy.stats.norm.ppf(y, loc=mu, scale=sigma)


class Sigmoid(object):
    """
    Sigmoid logistic and logit (inverse of logistic) functions.

    Wraps scipy.stats.logistic.

    Parameters
    ----------
    x,y : array like
        1D array of x/y values to plot over

    x0 : float
        Mid-point parameter; ``y = 0.5`` when ``x = x0``

    k : float
        Slope parameter

    Equations
    ---------
    **Logistic**

    .. math::

        f(x) = \\frac{1}{1 + \\exp(-k(x - x_0))}

    **Logit**

    .. math::

        f(y) = x_0 + \\frac{ \\ln \\left( \\frac{y}{1-y} \\right) } {k}
    """
    @staticmethod
    def logistic(x, x0, k):
        return scipy.stats.logistic.cdf(x, loc=x0, scale=1/k)

    @staticmethod
    def logit(y, x0, k):
        return scipy.stats.logistic.ppf(y, loc=x0, scale=1/k)


class Weibull(object):
    """
    Weibull cumulative density and quantile functions.

    Wraps scipy.stats.weibull_min.

    Parameters
    ----------
    x,y : array like
        1D array of x/y values to plot over. Note that the function is
        undefined for ``x < 0``.

    gamma : float, required
        Scale parameter.  Parameter is often referred to as lambda, but that
        name has a special meaning in Python so we use gamma instead here.

    k : float, required
        Shape parameter.

    Equations
    ---------
    **CDF**

    .. math::

        f(x) = 1 - \\exp( -(x / \\gamma)^k )

    **Quantile**

    .. math::

        f(y) = \\gamma (-\\ln(1-y))^{1/k}
    """
    @staticmethod
    def cdf(x, gamma, k):
        return scipy.stats.weibull_min.cdf(x, c=k, scale=gamma)

    @staticmethod
    def quantile(y, gamma, k):
        return scipy.stats.weibull_min.ppf(y, c=k, scale=gamma)


class MLEBinomFitFunction(object):
    """
    Class contains functions for fitting a non-linear function in the case
    where the data are derived from Bernoulli trials. Fit is done via MLE using
    a binomial PMF cost function.

    Arguments
    ---------
    x : 1D array-like, required
        The values along which the predictor variable varies. May be bin values
        in the case where x is discrete, or just a list of the individual
        x-values where x is continuous.

    counts : 1D array-like of ints, required
        The counts (sums) of the number of occurences of the measured event
        within each level of x. These can be the 0 and 1 values from the
        Bernoulli trials themselves if x is continous.

    n : int or 1D array-like of ints, required
        The total number of trials.  Can be a single integer in which case the
        same number is assumed for all levels of x, or a list of separate
        integers for each level of x. This can be set to 1 if x is continuous.

    func : function instance, optional
        Objective function to fit.  Must accept a 1D array of x-values as its
        first argument, and then any further arguments should be function
        parameters that are to be estimated via the fitting process.  Default
        is to use a Gaussian CDF.

    invfunc : function instance or None, optional
        Inverse of main function. Must accept a 1D array of y-values as its
        first argument, then any further arguments should be function
        parameters that have been estimated via the fitting process. Default
        is to use a Gaussian quantile function. Only necessary if wanting to
        use the .getXForY method.

    ymin : float or  'optim', optional
        Expected minimum value of y (e.g. use to adjust for chance level). Can
        also specify as string 'optim' to instead optimise the parameter - note
        that in this case the ymin parameter must be appended as the final
        (if ``lapse != 'optim``') or penultimate (if ``lapse == 'optim'``)
        value in any starting parameters, bounds, etc.

    lapse : float or 'optim', optional
        Lapse parameter (expected ymax = 1 - lapse). Can also specify as string
        'optim' to instead optimise the parameter - note that in this case the
        lapse parameter must be appended as the final value in any starting
        parameters, bounds, etc.

    Class methods
    -------------
    .doFit
        Performs function fitting.

    .doInterp
        Returns interpolated values for x and y variables, e.g. for plotting.

    .getFittedParams
        Return fitted parameters, assuming .doFit has already been run.

    .getXForY
        Use inverse function to get x-value for given y-value.

    .negLogLik
        Return negative log-likelihood for a given set of params

    Example usage
    -------------
    Generate some random data.  We will have x-values in 11 discrete bins
    ranging from -10 to +10.  For each bin we will simulate a random number of
    trials (between 10 and 15), generating a random set of binary responses
    (i.e. as per a Bernoulli trial) with the probability of responding 0 or 1
    at the given x-value determined by a Gaussian CDF.

    >>> import numpy as np
    >>> RNG = np.random.default_rng(42)
    >>>
    >>> true_mu, true_sigma = -2, 3
    >>> x = np.linspace(-10, 10, 11)
    >>> responses = []
    >>> for this_x in x:
    ...     this_n = RNG.integers(low=10, high=15)
    ...     p = Gaussian.cdf(this_x, true_mu, true_sigma)
    ...     these_resps = RNG.choice([0,1], size=this_n, p=[1-p, p])
    ...     responses.append(these_resps)
    >>> counts = [sum(r) for r in responses]
    >>> n = [len(r) for r in responses]

    Fit the function.  For both the inital guess (x0) and the bound values,
    parameters refer to those of the Gaussian CDF parameters (mu, sigma).

    * For the initial guess, we specify a list of tuples to perform an initial
      gridsearch over all combinations of ``mu = (-5, -2.5, 0, +2.5, +5)`` and
      ``sigma = (1, 3, 5)``.
    * Bounds are specified as (min, max) tuples for each of the parameters in
      turn. We'll bound ``mu`` between the x-value limits. For ``sigma`` we'll
      place a lower bound slightly above zero, but no upper bound.

    >>> fit = MLEBinomFitFunction(x, counts, n)
    >>> x0 = [(-5, -2.5, 0, 2.5, 5), (1, 3, 5)]
    >>> bounds = [(min(x), max(x)), (1e-5, None)]
    >>> fit.doFit(x0=x0, bounds=bounds, method='L-BFGS-B')
    >>> fit_mu, fit_sigma = fit.getFittedParams()

    Check fitted parameters.

    >>> print(f'True parameters: mu = {true_mu}, sigma = {true_sigma}')
    >>> print(f'Parameter estimates: mu = {fit_mu:.2f}, sigma = {fit_sigma:.2f}')
    >>> print(f'Grid search selected starting params: {fit.selected_x0}')

    ::

        True parameters: mu = -2, sigma = 3
        Parameter estimates: mu = -1.81, sigma = 2.54
        Grid search selected starting params: (-2.5, 3)

    Make a plot.

    >>> import matplotlib.pyplot as plt
    >>> interpX, interpY = fit.doInterp()
    >>> plt.figure()
    >>> plt.plot(interpX, Gaussian.cdf(interpX, true_mu, true_sigma),
    ...          'k-', label='True function')
    >>> plt.plot(x, list(map(np.mean, responses)), 'bo', label='Mean response')
    >>> plt.plot(interpX, interpY, 'r--', label='MLE fit')
    >>> plt.legend()
    >>> plt.show()

    """

    def __init__(self, x, counts, n, func=Gaussian.cdf,
                 invfunc=Gaussian.quantile, ymin=0, lapse=0):

        # Error check
        if not ( (isinstance(ymin, str) and ymin == 'optim') \
                 or isinstance(ymin, (int, float)) ):
            raise ValueError("ymin must be numeric or 'optim', " \
                             f"but received: {ymin}")

        if not ( (isinstance(lapse, str) and lapse == 'optim') \
                 or isinstance(lapse, (int, float)) ):
            raise ValueError("lapse must be numeric or 'optim', " \
                             f"but received: {lapse}")

        # Assign args to class
        self.x = np.asarray(x)
        self.counts = np.asarray(counts)
        self.n = np.asarray(n)
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
        def f(x, *params):
            params = list(params)
            lapse = params.pop(-1) if self.lapse == 'optim' else self.lapse
            ymin = params.pop(-1) if self.ymin == 'optim' else self.ymin
            return ymin + (1 - ymin - lapse) * np.asarray(fun(x, *params))

        self._func = f

    @property
    def invfunc(self):
        return self._invfunc

    @invfunc.setter
    def invfunc(self, fun):
        """Setter adjusts inverse func for ymin and lapse params"""
        if fun is None:
            self._invfunc = None
            return

        def f(y, *params):
            params = list(params)
            lapse = params.pop(-1) if self.lapse == 'optim' else self.lapse
            ymin = params.pop(-1) if self.ymin == 'optim' else self.ymin
            return fun((np.asarray(y) - ymin) / (1 - ymin - lapse), *params)

        self._invfunc = f

    def doFit(self, x0=None, *args, **kwargs):
        """
        Performs the function fit.

        Arguments
        ---------
        x0 : array-like, optional
            Optional (but recommended) list of starting parameter values for
            the optimisation. Can also be specified as a list of lists, where
            each inner list contains a range of values for a given parameter
            (i.e. each inner list represents a function parameter in turn). In
            this case an initial grid search is performed over all parameter
            combinations, and the best performing set is selected for the
            optimisation. The selected values will be stored in the
            ``.selected_x0`` attribute.

        *args, **kwargs
            Additional arguments passed to ``scipy.optimize.minimize``.

        Notes
        -----
        * If optimising ymin parameter (``ymin == 'optim'``) then this
          parameter must be included as the final (if ``lapse != 'optim'``) or
          penultimate (if ``lapse == 'optim'``) value in any relevant args.

        * If optimising lapse parameter (``lapse == 'optim'``) then this
          parameter must be included as the final value in any relevant args.

        * If specifying parameter bounds via the ``bounds`` keyword argument,
          the bounds are specified as ``(lower, upper)`` array-likes for each
          paramter in turn::

              bounds = [(p1_lower, p1_upper), (p2_lower, p2_upper), etc]

          To leave a parameter unbounded, specify the value as ``None``.

        * Results are stored within the .fit attribute of this class, and can
          also be accessed with the ``.getFittedParams`` method.
        """
        # Grid search?
        if x0 is not None and all(hasattr(p, '__iter__') for p in x0):
            x0grid = list(itertools.product(*x0))
            errs = []
            for p in x0grid:
                with np.errstate(divide='ignore', invalid='ignore'):
                    err = self.negLogLik(p)
                errs.append(err)
            x0 = x0grid[np.nanargmin(errs)]

        self.selected_x0 = x0

        # Main fit
        with np.errstate(divide='ignore', invalid='ignore'):
            self.fit = minimize(self.negLogLik, x0, *args, **kwargs)

    def getFittedParams(self):
        """
        Return fitted parameters if .doFit has been run.
        """
        if self.fit is None:
            raise RuntimeError('Must call .doFit method first')
        return self.fit.x

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

    def doInterp(self, interpX=100):
        """
        Returns an (x,y) tuple of values interpolated along x-dimension(s),
        which can be used for plotting.

        Arguments
        ---------
        interpX : int or array-like, optional
            Interpolated x-values to calculate y-values over. If an integer,
            will create a default range between min(x) and max(x) with the
            specified number of samples. If an array, will take these as the
            x-values directly. The default is 100.

        Returns
        -------
        interpX, interpY
            Interpolated x- and y-values respectively.
        """
        params = self.getFittedParams()
        if isinstance(interpX, int):
            interpX = np.linspace(self.x.min(), self.x.max(), interpX)
        else:
            interpX = np.asarray(interpX)
        interpY = self.func(interpX, *params)
        return (interpX, interpY)

    def negLogLik(self, params):
        """
        Computes negative log likelihood via a Binomial PMF cost function,
        using probability values generated by the given objective function.
        Params should be for the objective function.
        """
        p = self.func(self.x, *params)
        L = scipy.stats.binom.pmf(self.counts, self.n, p)
        return -np.log(L.clip(min=1e-16).sum())
