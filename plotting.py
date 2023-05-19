#!/usr/bin/env python3
"""
Assorted plotting tools
"""

from __future__ import division
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap


def _parse_cmap(cmap):
    """
    Convenience function. If cmap is str or None, converts to matplotlib
    colormap instance. Otherwise, returns as is. FSL colormaps supported
    if str starts with 'fsl-'
    """
    if isinstance(cmap, str) and cmap.startswith('fsl-'):
        cmap = get_fsl_cmap(cmap.split('fsl-')[1])
    elif cmap is None or isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    return cmap


def get_fsl_cmap(cmap=None):
    """
    Defines various fslview colormaps

    Arguments
    ---------
    cmap : str
        Name of colormap. Enter None or other invalid name to see list of
        available options. Colormaps can be reversed by appending '_r' to
        end of name.

    Returns
    -------
    colormap - LinearSegmentedColormap instance
    """
    # Lookup of cdicts for all possible colormaps
    x = 155/255 # value which comes up in a lot of colormaps
    cdicts = {
        'red' : {'red':[(0,x,x), (1,1,1)],
                 'green':[(0,0,0),(1,0,0)],
                 'blue':[(0,0,0), (1,0,0)]},
        'green' : {'red':[(0,0,0), (1,0,0)],
                   'green':[(0,x,x), (1,1,1)],
                   'blue':[(0,0,0), (1,0,0)]},
        'blue' : {'red':[(0,0,0), (1,0,0)],
                  'green':[(0,0,0),(1,0,0)],
                  'blue':[(0,x,x), (1,1,1)]},
        'yellow' : {'red':[(0,x,x), (1,1,1)],
                    'green':[(0,x,x),(1,1,1)],
                    'blue':[(0,0,0), (1,0,0)]},
        'pink' : {'red':[(0,1,1), (1,1,1)],
                  'green':[(0,x,x), (1,1,1)],
                  'blue':[(0,x,x), (1,1,1)]},
        'redyellow' : {'red':[(0,1,1), (1,1,1)],
                       'green':[(0,0,0), (1,1,1)],
                       'blue':[(0,0,0), (1,0,0)]},
        'bluelightblue' : {'red':[(0,0,0), (1,0,0)],
                           'green':[(0,0,0), (1,1,1)],
                           'blue':[(0,1,1), (1,1,1)]},
        'grey' : {'red':[(0,0,0), (1,1,1)],
                  'green':[(0,0,0), (1,1,1)],
                  'blue':[(0,0,0), (1,1,1)]},
        'cool' : {'red':[(0,0,0),(1,1,1)],
                  'green':[(0,1,1),(1,0,0)],
                  'blue':[(0,1,1),(1,1,1)]},
        'copper' : {'red':[(0,0,0), (1/1.2,1,1), (1,1,1)],
                    'green':[(0,0,0), (1,0.8,0.8)],
                    'blue':[(0,0,0), (1,0.5,0.5)]},
        'hot' : {'red':[(0,x,x), (1/3,1,1), (1,1,1)],
                 'green':[(0,0,0), (1/3,x,x), (2/3,1,1), (1,1,1)],
                 'blue':[(0,0,0), (2/3,x,x), (1,1,1)]}
        }
    cdicts['gray'] = cdicts['grey']

    if cmap is not None:
        cmap = cmap.lower()

        # Handle reverse colourmap names
        if cmap.endswith('_r'):
            reverse = True
            cmap = cmap[:-2]
        else:
            reverse = False

        # Handle fsl- prefix
        if cmap.startswith('fsl-'):
                cmap = cmap[4:]

    # Create colormap, return
    try:
        cmap_obj = LinearSegmentedColormap(cmap, cdicts[cmap])
        if reverse:
            cmap_obj = cmap_obj.reversed()
        return cmap_obj
    except KeyError:
        raise KeyError('FSL cmap must be one of: fsl-' \
                       + ', fsl-'.join(sorted(cdicts.keys())))


def plot_cbar(vmin=0, vmax=1, dp=2, cmap=None, label=None, labelsize=32,
              nticks=6, ticks=None, ticksize=24, tickpos=None, ticklabels=None,
              font='sans-serif', fontcolor='black', ori='vertical',
              figsize=None, segmented=False, segment_interval=1):
    """
    Plots single colorbar.

    Arguments
    ---------
    vmin, vmax : float, optional
        Minimum and maximum values for colormap limits
    dp : int, optional
        Number of decimal places for colorbar axis ticks.  Ignored if
        ticklabels is not None.
    cmap : str or matplotlib cmap instance), optional
        May be any valid matplotlib colormap (string or matplotlib cmap
        instance) or fsl colormap string (see get_fsl_cmap function).
    label : str, optional
        Axis label for colorbar
    labelsize : int, optional
        Fontsize for axis label
    nticks : int, optional
        Number of ticks to have on colorbar axis.
    ticks : list of floats, optional
        Exact values of ticks (overrides nticks)
    ticksize : int, optional
        Fontsize for ticks
    tickpos : str, optional
        Where to place ticks.  May be 'right' (default) or 'left' for a
        vertical bar, or 'bottom' (default) or 'top' for a horizontal bar
    ticklabels : list of strings, optional
        May manually specify tick label strings to plot against colorbar.
        If specified, number of elements must match specified <nticks> value.
        If omitted, labels will be <nticks> evenly spaced values between
        <vmin> and <vmax>, formatted to specified number of dp.
    fontcolor : str, rgb / rgba tuple, or hex value, optional
        Font color of all text
    ori : str, optional
        Orientation of colorbar. Can be 'vertical' (default) or 'horizontal'
    figsize  : (width,length) tuple, optional
        Tuple giving figure size (in inches).  Defaults to (3,8) for vertical
        or (8,3) for horizontal bar.
    segmented : bool, optional
        Set to True to use a segmented colormap
    segment_interval : float, optional
        Interval between segments of colorbar (ignored if segmented = False)

    Returns
    -------
    fig, ax : figure and axis handles
    """
    # Check orientation
    if ori == 'v':
        ori = 'vertical'
    elif ori == 'h':
        ori = 'horizontal'
    if ori not in ['vertical', 'horizontal']:
        raise ValueError("ori must be 'vertical' or 'horizontal'")

    # Handle cmap
    cmap = _parse_cmap(cmap)

    # Work out our tick position setting defaults as necessary
    if tickpos is None:
        if ori == 'vertical':
            tickpos = 'right'
        else: # ori == 'horizontal'
            tickpos = 'bottom'

    # Sanity check
    if (ori == 'vertical' and tickpos not in ['right', 'left']) or \
       (ori == 'horizontal' and tickpos not in ['top', 'bottom']):
        raise ValueError("tickpos '{}' not valid for {} colorbar"\
                         .format(tickpos, ori))

    # Handle fig size from orientation
    if figsize is None:
        if ori == 'vertical':
            figsize = (3,8)
        else:
            figsize = (8,3)

    # Handle axes size from orientation
    if ori == 'vertical':
        if tickpos == 'right':
            rect = [0.1, 0.05, 0.2, 0.9] # [l,b,w,h]
        else: # tickpos == 'left'
            rect = [0.5, 0.05, 0.2, 0.9]
    else: # ori == 'horizontal'
        if tickpos == 'bottom':
            rect = [0.1, 0.6, 0.8, 0.3]
        else: # tickpos == 'top'
            rect = [0.1, 0.4, 0.8, 0.3]

    # Define figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    # Create a normalised color range - this needs handling differently
    # depending on whether the colormap should be segmented or not
    if segmented:
        # If doing a segmented map, we use a boundary norm.  First, need to get
        # bounds (values at which boundaries between segments occur). This needs
        # to be extended by 2 units to include both the vmax value and the
        # final boundary at the top of the colorbar
        bounds = np.arange(vmin, vmax+(2*segment_interval), segment_interval)
        # Create norm
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        # If not doing a segmented map, just define a continuous norm from vmin
        # to vmax
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Define colorbar
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)

    # Calculate & set tick intervals
    if ticks is None:
        ticks = np.linspace(vmin, vmax, nticks)
        # For a segmented colormap, the ticks will occur at the boundaries by
        # default - we want them next to the segments themselves so we need to
        # place half a segment_interval up
        if segmented:
            ticks += segment_interval/2
    else:
        ticks = np.asarray(ticks)

    if ori == 'vertical':
        cb.ax.yaxis.set_ticks_position(tickpos)
    else:
        cb.ax.xaxis.set_ticks_position(tickpos)
    cb.set_ticks(ticks)

    # Assign requested ticklabels if provided, else format some default ones
    if ticklabels is not None:
        if not len(ticklabels) == nticks:
            raise ValueError('Number of ticklabels must match value of nticks')
        cb.set_ticklabels(ticklabels)
    else:
        cb.set_ticklabels(['{:.{}f}'.format(x, dp) for x in ticks])

    # Define label from args
    if label is not None:
        cb.set_label(label, size=labelsize, color=fontcolor, family=font)

    # Loop through ticks and adjust font size & color
    if ori == 'vertical':
        labs = cb.ax.get_yticklabels()
    else:
        labs = cb.ax.get_xticklabels()
    for t in labs:
        t.set_family(font)
        t.set_fontsize(ticksize)
        t.set_color(fontcolor)

    # Return figure handle
    return fig, ax


def plot_matrix(array, cbar=True, annotate_vals=True, avdiag=False,
                nodiag=False, grid=True, xlabel='', ylabel='', ticklabels='',
                xticklabels=None, yticklabels=None, tickrotation=0,
                xtickrotation=None, ytickrotation=None,
                xtickalignment='center', ytickalignment='center', title='',
                fontsize=18, titlesize=None, labelsize=None, ticksize=None,
                lims=None, dp=2, cbarlabel='', cbar_nticks=5,
                cbar_ticklabels=None, font='sans-serif', cmap=None,
                fontcolor='black', segmented=False, segment_interval=1,
                ax=None):
    """
    Function for plotting confusion / similarity matrix.

    Arguments
    ---------
    array : numpy.ndarray, required
        array of numbers you wish to plot (note - NaN or None values will
        appear blank in the plot)
    cbar : bool, optional
        Set to True to plot a colorbar
    annotate_vals : bool or numpy.ndarray, optional
        Set to True to overlay text values on matrix cells. If a numpy array
        the same size as <array> is supplied, will annotate the values
        contained on this array instead.
    avdiag : bool or str ('done'), optional
        Set to True to average values across diagonal. Set as 'done' if this
        has already been done. Will be forced to False if array is not square.
    nodiag : bool, optional
        If True, will omit diagonal from plot (note - avdiag will be forced to
        True). Will be forced to False if array is not square.
    grid : bool, optional
        Set to True to overlay gridlines
    xlabel, ylabel : str, optional
        x- and y-axis labels
    ticklabels, xticklabels, yticklabels : list, optional
        x- and y-axis tick-labels.  If xticklabels or yticklabels are None,
        they will default to the value of ticklabels
    tickrotation, xtickrotation, ytickrotation : float, optional
        Rotation of x- and y-axis ticklabels in degrees.  If xtickrotation or
        ytickrotation is None, they will default to the value of tickrotation
    xtickalignment, ytickalignment : str, optional
        Alignment of x- and y-axis ticklabels.  xtickalignment can be 'left',
        'center' (default), or 'right'.  ytickalignment can be 'top',
        'center' (default), 'bottom', or 'baseline'
    title : str, optional
        Plot title
    fontsize : int, optional
        Global fontsize for plot (see also titlesize, labelsize, ticksize)
    titlesize : int, optional
        Title fontsize (default = fontsize + 2)
    labelsize : int, optional
        Axis and colorbar label fontsize (default = fontsize)
    ticksize : int, optional
        Fontsize for axis ticklabels and value labels (default = fontsize-2)
    lims : (min, max) tuple or None, optional
        Colormap limits. If None (default) will use data range
    dp : int, optional
        Decimal places for colorbar ticks and matrix value labels
    cbarlabel : str, optional
        colorbar label
    cbar_nticks : int, optional
        Number of ticks to put on colorbar
    cbar_ticklabels : list of strings, optional
        Tick label strings to plot against colorbar. Number of elements must
        match specified <cbar_nticks> value. If omitted, labels will be
        <cbar_nticks> evenly spaced values between <lims>, formatted to
        specified number of dp.
    font : str, optional
        Font style for text
    cmap : str or matplotlib cmap instance, optional
        May be any valid matplotlib colormap (string or matplotlib cmap
        instance) or fsl colormap string (see get_fsl_cmap function).
    fontcolor : str, rgb / rgba tuple, or hex value), optional
        Font color of all text
    segmented : bool, optional
        Set to True to use a segmented colormap
    segment_interval : float, optional
        Interval between segments of colorbar (ignored if segmented = False)
    ax : axis handle or None, optional
        If provided, attach plot to existing axis.

    Returns
    -------
    fig, ax : figure and axis handles
    """
    # Handle cmap
    cmap = _parse_cmap(cmap)

    # Apply values of x- / y-ticklabels and rotation if necessary
    if xticklabels is None:
        xticklabels = ticklabels
    if yticklabels is None:
        yticklabels = ticklabels
    if xtickrotation is None:
        xtickrotation = tickrotation
    if ytickrotation is None:
        ytickrotation = tickrotation

    # Apply default fontsizes
    if titlesize is None:
        titlesize = fontsize + 2
    if labelsize is None:
        labelsize = fontsize
    if ticksize is None:
        ticksize = fontsize - 2

    # If we are not showing diagonal, force avdiag to True
    if nodiag:
        avdiag = True

    # Prep array
    array = np.asarray(array, dtype='float').copy() # force numpy array
    # Get array shape
    nYConds, nXConds = array.shape
    if nXConds != nYConds:
        avdiag = False
        nodiag = False

    # Error check annotate_vals
    if not isinstance(annotate_vals, (bool, np.ndarray)):
        raise TypeError('annotate_vals must be bool or numpy.ndarray')
    elif (isinstance(annotate_vals, np.ndarray) and
          annotate_vals.shape != array.shape):
        raise ValueError('annotate_vals must be same shape as array')

    # Average across diagonal if requested
    if avdiag:
        # Iterate through all unique x,y index pairs (this block should only
        # exectute if nXConds == nYConds, so we just arbitrarily use nXConds)
        for x,y in itertools.combinations(range(nXConds),2):
            array[y,x] = np.mean([array[x,y],array[y,x]]) # set lower left to av value
            array[x,y] = np.nan # set upper right to nan

    # If not plotting diagonal, remove diagonal, 1st row, and last col
    if nodiag:
        # This block should only execute if nXConds == nYConds, so we just
        # arbitrarily make mask from nXConds
        array[np.eye(nXConds).astype(bool)] = np.nan
        array = array[1:, :-1]
        if isinstance(annotate_vals, np.ndarray):
            annotate_vals = annotate_vals[1:, :-1]
        nXConds -= 1
        nYConds -= 1

    # Determine min/max from lims (if specified) or array otherwise
    if lims is None:
        mArr = np.ma.masked_array(array, np.isnan(array))
        vmin, vmax = mArr.min(), mArr.max()
    else:
        vmin, vmax = lims

    # Set bounds and norm depending on if continuous or segmented colourmap
    if segmented:
        # Define the bounds, i.e. values at which boundaries between color
        # segments should occur.  We extend this range by 2 units - the
        # first to make the range include the upper limit value, and the
        # second to include the boundary at the top of the colormap.
        bounds = np.arange(vmin, vmax+(2*segment_interval), segment_interval)
        norm = BoundaryNorm(bounds, cmap.N)
    else:
        bounds = None
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Create plot and axes
    if ax is None:
        fig, _ax = plt.subplots()
    else:
        fig = ax.figure
        _ax = ax

    # Set title, x and y labels
    _ax.set_title(title, size=titlesize, family=font, color=fontcolor)
    _ax.set_xlabel(xlabel, size=labelsize, color=fontcolor, family=font)
    _ax.set_ylabel(ylabel, size=labelsize, color=fontcolor, family=font)

    # Plot matrix
    im = _ax.imshow(array, cmap=cmap, norm=norm, interpolation='nearest',
                    aspect='equal')

    # Add gridlines if requested. Draw lines across full plot if not
    # averaging over diagonal, or just up to to diagonal if averaging
    if grid:
        # Loop x coords and plot vertical gridlines
        for i, x in enumerate(np.arange(0.5, nXConds-0.5)):
            if avdiag == False:
                _ax.axvline(x, color='k', ls='-', lw=1)
            elif avdiag in [True, 'done']:
                _ax.axvline(x, ymax=1-(i/nXConds), color='k', ls='-', lw=1)
        # Loop y coords and plot horizontal gridlines
        for j, y in enumerate(np.arange(0.5, nYConds-0.5)):
            if avdiag == False:
                _ax.axhline(y, color='k', ls='-', lw=1)
            elif avdiag in [True, 'done']:
                _ax.axhline(y, xmax=(j+2)/nYConds, color='k', ls='-', lw=1)

    # Add matrix tick labels
    if nodiag:
        xticklabels = xticklabels[:-1]
        yticklabels = yticklabels[1:]
    _ax.set_xticks(np.arange(nXConds))
    _ax.set_yticks(np.arange(nYConds))
    _ax.set_xticklabels(xticklabels, size=ticksize, family=font, color=fontcolor,
                        ha=xtickalignment, rotation=xtickrotation)
    _ax.set_yticklabels(yticklabels, size=ticksize, family=font, color=fontcolor,
                        va=ytickalignment, rotation=ytickrotation)

    # Set axis limits (can get messed up by preceding plot tweaks)
    _ax.set_xlim(-0.5, nXConds-0.5)
    _ax.set_ylim(nYConds-0.5, -0.5)

    # Iterate through matrix overlaying values if requested
    if annotate_vals is not False:
        for x,y in itertools.product(range(nXConds), range(nYConds)):
            if isinstance(annotate_vals, bool):
                val = array[y,x]
                if np.isnan(val):
                    continue  # ignore nans
                else:
                    txt = '{0:.{1}f}'.format(val, dp)
            elif isinstance(annotate_vals, np.ndarray):
                txt = str(annotate_vals[y,x])

            _ax.text(x, y, txt, ha='center',
                     va='center', fontsize=ticksize, family=font,
                     color=fontcolor)

    # Plot colorbar if requested
    if cbar:
        # Set up tick range and labels
        cbrng = np.linspace(vmin, vmax, cbar_nticks)
        if cbar_ticklabels is not None:
            if len(cbar_ticklabels) != cbar_nticks:
                raise ValueError('Number of colorbar tick labels must match '
                                 'number of colorbar ticks')
        else:
            cbar_ticklabels = ['{:.{}f}'.format(x, dp) for x in cbrng]

        if segmented:
            # By default ticks will be at boundaries - we want them next to
            # the segments themselves, so increment up half a unit
            cbrng += segment_interval / 2

        # Do colorbar
        cb = fig.colorbar(im, boundaries=bounds)
        cb.set_ticks(cbrng)
        cb.set_ticklabels(cbar_ticklabels)
        cb.set_label(cbarlabel, size=labelsize, family=font, color=fontcolor)
        cb.ax.tick_params(labelsize=ticksize, labelcolor=fontcolor)
        for t in cb.ax.get_yticklabels():
            t.set_family(font)

    # Return figure and axes
    return fig, _ax


def polar_pcolormesh(C, r=None, theta=None, hemi='both', theta_units='rad',
                     theta_direction='clockwise', cmap=None, vmin=None,
                     vmax=None, shading='gouraud', xticks=None, yticks=None,
                     xticklabels=None, yticklabels=None, font='sans-serif',
                     fontcolor='black', fontsize=10, grid=False, cbar=False,
                     dp=2, cbarlabel='', cbar_nticks=5, cbar_ticklabels=None,
                     ax=None):
    """
    Makes polar pcolormesh plot, e.g. for making polar angle or eccentricity
    colourmaps.

    Arguments
    ---------
    C : str ('angle' | eccentricity') or 2D numpy array, required
        Plot data. Selected strings 'angle' or 'eccentricity' can be
        used to make polar angle or eccentricity maps. Abbreviations 'pol' and
        'ecc' also accepted. Alternatively, can supply own data as a numpy
        array with columns corresponding to values of <theta>, and rows
        corresponding to values of <r>.
    r : 1D numpy array, optional
        Radial values to plot over. Will use a default range between 0 and 1 if
        not specified.
    theta : 1D numpy array, optional
        Angular values to plot over. Will use a default range (in radians)
        dependent on <hemi> if not specified. Note that 0 is at 12 o'clock.
    hemi : str ('both' | 'left' | 'right' | 'top' | 'bottom'), optional
        Which hemifield to plot. Ignored if <theta> is specified.
    theta_units : str ('rad' | 'deg'), optional
        Are angular units specified in radians or degrees?
    theta_direction : str or int, optional
        Should theta increment clockwise ('clockwise', 'cw', or -1) or
        counter-clockwise ('counterclockwise', 'ccw', or 1)?
    cmap : str or matplotlib cmap instance, optional
        May be any valid matplotlib colormap (string or matplotlib cmap
        instance) or fsl colormap string (see get_fsl_cmap function).
    vmin, vmax : float, optional
        Limits for colourmap. If not specified, will default to data limits.
    shading : str ('flat' | 'nearest' | 'gouraud' | 'auto'), optional
        Fill style for the qudrilatera's (see pcolormesh)
    xticks, yticks : list, optional
        Tick positions along angular and radial axes respectively.
        <xticks> should be specified in units of <theta_units>.
    xticklabels, yticklabels : list, optional
        Tick labels for angular and radial axes respectively.
    font : str, optional
        Text font style
    fontcolor : str, optional
        Colour for text
    fontsize : int, optional
        Text size
    grid : bool, optional
        Sets whether to display grid overlay on plot.
    cbar : bool, optional
        Sets whether to display colorbar next to plot.
    dp : int, optional
        Decimal places for colorbar tick labels. Ignored if <cbar> is False.
    cbarlabel : str, optional
        Colorbar axis label. Ignored if <cbar> is False.
    cbar_nticks : int, opttional
        Number of ticks to put on colorbar. Ignored if <cbar> is False.
    cbar_ticklabels : list of strings, optional
        Tick label strings to plot against colorbar. Number of elements must
        match specified <cbar_nticks> value. If omitted, labels will be
        <cbar_nticks> evenly spaced values, formatted to specified <dp>.
        Ignored if <cbar> is False.
    ax : axis handle or None, optional
        If provided, attach plot to existing axis. Note that existing axis
        MUST be created with polar projection.

    Returns
    -------
    fig, ax : figure and axis handles
    """

    # Error check args
    if not isinstance(C, (np.ndarray, str)):
        raise TypeError('C must be numpy array or str')

    if hemi and hemi not in ['both', 'left', 'right', 'top', 'bottom']:
        raise ValueError('Invalid hemi')

    if theta_direction not in ['clockwise','counterclockwise','cw','ccw',1,-1]:
        raise ValueError('Invalid theta_direction')

    if not (theta_units.startswith('rad') or theta_units.startswith('deg')):
        raise ValueError('Invalid theta_units')

    # Assign further defaults where necessary
    cmap = _parse_cmap(cmap)

    if theta_direction in ['clockwise', 'cw']:
        theta_direction = -1
    elif theta_direction in ['counterclockwise', 'ccw']:
        theta_direction = 1

    if r is None:
        r = np.linspace(0, 1, 100)

    if theta is None:
        if hemi == 'both':
            theta_lims = (0, 360)
        elif hemi == 'left':
            if theta_direction == -1:  # cw
                theta_lims = (-180, 0)
            else:  # ccw
                theta_lims = (0, 180)
        elif hemi == 'right':
            if theta_direction == -1:
                theta_lims = (0, 180)
            else:  # ccw
                theta_lims = (-180, 0)
        elif hemi == 'top':
            theta_lims = (-90, 90)
        else:
            theta_lims = (90, 270)
        theta = np.radians(np.arange(theta_lims[0], theta_lims[1] + 1, 1))
    else:
        if theta_units.startswith('deg'):
            theta = np.radians(theta)
        theta_lims = np.degrees((theta.min(), theta.max()))

    # Meshgrid
    [fr, ftheta] = np.meshgrid(r, theta)

    # Assign C if necessary
    if isinstance(C, str):
        if C.startswith('ecc'):
            C = fr
        elif C in ['pol', 'angle']:
            C = ftheta
        else:
            raise ValueError(f'Invalid string argument for C: {C}')

    # Check color limits
    if vmin is None:
        vmin = C.min()
    if vmax is None:
        vmax = C.max()

    # Create figure
    if ax is None:
        fig, _ax = plt.subplots(subplot_kw={'polar':True})
    else:
        _ax = ax
        fig = _ax.figure

    # Plot
    im = _ax.pcolormesh(ftheta, fr, C, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading=shading)

    # Set axis details
    _ax.set_theta_direction(theta_direction)
    _ax.set_theta_zero_location('N', 0)
    _ax.set_thetamin(theta_lims[0])
    _ax.set_thetamax(theta_lims[1])

    if xticks is not None:
        if theta_units.startswith('deg'):
            xticks = np.radians(xticks)
        _ax.set_xticks(xticks)
    if yticks is not None:
        _ax.set_yticks(yticks)
    if xticklabels is not None:
        _ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        _ax.set_yticklabels(yticklabels)

    _ax.tick_params(labelsize=fontsize, labelcolor=fontcolor)
    for t in _ax.get_xticklabels():
        t.set_family(font)
    for t in _ax.get_yticklabels():
        t.set_family(font)

    # Plot colorbar if requested
    if cbar:
        cbrng = np.linspace(vmin, vmax, cbar_nticks)
        if cbar_ticklabels is not None:
            if len(cbar_ticklabels) != cbar_nticks:
                raise ValueError('Number of colorbar tick labels must match '
                                 'number of colorbar ticks')
        else:
            cbar_ticklabels = ['{:.{}f}'.format(x, dp) for x in cbrng]

        cb = fig.colorbar(im, boundaries=None, norm=None)
        cb.set_ticks(cbrng)
        cb.set_ticklabels(cbar_ticklabels)
        cb.set_label(cbarlabel, size=fontsize, family=font, color=fontcolor)
        cb.ax.tick_params(labelsize=fontsize, labelcolor=fontcolor)
        for t in cb.ax.get_yticklabels():
            t.set_family(font)

    # Return
    return fig, _ax
