#!/usr/bin/env python3
"""
Assorted plotting tools
"""

import os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform

FSL_COLOURMAP_DIR = 'fsleyes_colourmaps'


def get_colourmap(cmap):
    """
    Retrieve colourmap.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap instance or str
        Input colourmap. Can be a colourmap object (e.g., plt.cm.viridis) or a
        string giving the name of the colourmap (e.g., 'viridis').

        * If the string starts with 'fsl-' you can use one of the classic
          FSLeyes colourmaps (e.g., 'fsl-red-yellow')

        * If the string starts with 'fsl-brain-' you can use one of the newer
          FSLeyes brain colourmaps (e.g., 'fsl-brain-hot_iso')

        * See the ``fsleyes_colourmaps`` directory for a list of available
          FSLeyes colourmaps.

        * If the string ends with '_r', the colourmap will be reversed.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        Colourmap instance
    """
    # Handle FSL colourmaps
    if isinstance(cmap, str) and cmap.startswith('fsl-'):
        # Determine colourmap name and directory
        if cmap.startswith('fsl-brain-'):
            cmap_name = cmap[10:]
            cmap_dir = os.path.join(FSL_COLOURMAP_DIR, 'brain_colours')
        else:
            cmap_name = cmap[4:]
            cmap_dir = FSL_COLOURMAP_DIR

        # Check if colourmap should be reversed, adjust name accordingly
        reverse = cmap.endswith('_r')
        if reverse:
            cmap_name = cmap_name[:-2]

        # Load RGB values, create colourmap
        RGB = np.loadtxt(os.path.join(cmap_dir, f'{cmap_name}.cmap'))
        cmap = ListedColormap(RGB, name=cmap_name)
        if reverse:
            cmap = cmap.reversed()

    # Handle matplotlib colourmaps
    else:
        cmap = plt.get_cmap(cmap)

    # Return
    return cmap


def plot_cbar(vmin=0, vmax=1, dp=2, cmap=None, label=None, labelsize=20,
              nticks=6, ticks=None, ticksize=18, tickpos=None, ticklabels=None,
              font='sans-serif', fontcolor='black', ori='vertical',
              figsize=None, segmented=False, segment_interval=1):
    """
    Plots single colorbar.

    Arguments
    ---------
    vmin, vmax : float
        Minimum and maximum values for colormap limits.

    dp : int
        Number of decimal places for colorbar axis ticks.  Ignored if
        ticklabels is not None.

    cmap : str or matplotlib.colors.Colormap instance
        Any valid input to ``get_colourmap`` function.

    label : str
        Axis label for colorbar.

    labelsize : int
        Fontsize for axis label.

    nticks : int
        Number of ticks to have on colorbar axis.

    ticks : list of floats
        Exact values of ticks (overrides nticks).

    ticksize : int
        Fontsize for ticks.

    tickpos : str
        Where to place ticks.  May be 'right' (default) or 'left' for a
        vertical bar, or 'bottom' (default) or 'top' for a horizontal bar.

    ticklabels : list of strings
        May manually specify tick label strings to plot against colorbar.
        If specified, number of elements must match specified <nticks> value.
        If omitted, labels will be <nticks> evenly spaced values between
        <vmin> and <vmax>, formatted to specified number of dp.

    fontcolor : str, rgb / rgba tuple, or hex value
        Font color of all text.

    ori : str
        Orientation of colorbar. Can be 'vertical' (default) or 'horizontal'.
        Also accepts 'v' and 'h'.

    figsize  : (width,length) tuple
        Tuple giving figure size (in inches).  Defaults to (1.5, 6) for
        vertical or (8, 1) for horizontal bar.

    segmented : bool
        Set to True to use a segmented colormap.

    segment_interval : float
        Interval between segments of colorbar (ignored if segmented = False).

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
    cmap = get_colourmap(cmap)

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
            figsize = (1.5, 6)
        else:
            figsize = (8, 1)

    # Handle axes size from orientation
    if ori == 'vertical':
        if tickpos == 'right':
            rect = [0.1, 0.05, 0.2, 0.9] # [l,b,w,h]
        else: # tickpos == 'left'
            rect = [0.7, 0.05, 0.2, 0.9]
    else: # ori == 'horizontal'
        if tickpos == 'bottom':
            rect = [0.1, 0.6, 0.8, 0.3]
        else: # tickpos == 'top'
            rect = [0.1, 0.1, 0.8, 0.3]

    # Define figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.addax_es(rect)

    # Create a normalised color range - this needs handling differently
    # depending on whether the colormap should be segmented or not
    if segmented:
        # If doing a segmented map, we use a boundary norm.  First, need to get
        # bounds (values at which boundaries between segments occur). This
        # needs to be extended by 2 units to include both the vmax value and
        # the final boundary at the top of the colorbar
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
        nticks = len(ticks)

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
        if ori == 'vertical':
            ax.yaxis.set_label_position(tickpos)
        else:
            ax.xaxis.set_label_position(tickpos)

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
                mask_diag=False, mask_tril=False, mask_triu=False,
                grid=True, grid_color='dimgrey', xlabel='', ylabel='',
                ticklabels=None, xticklabels=None, yticklabels=None,
                tickrotation=0, xtickrotation=None, ytickrotation=None,
                xtickalignment='center', ytickalignment='center', title='',
                fontsize=12, titlesize=None, labelsize=None, ticksize=None,
                annotsize=None, lims=None, dp=2, cbarlabel='', cbar_nticks=5,
                cbar_ticklabels=None, font='sans-serif', cmap=None,
                fontcolor='black', segmented=False, segment_interval=1,
                ax=None):
    """
    Function for plotting confusion / similarity matrix.

    Arguments
    ---------
    array : numpy.ndarray
        Array to plot. Can be a 2D rectangular array, or a 1D vector of values
        from upper triangle. (Note - NaN or None values will appear blank).

    cbar : bool
        Set to True to plot a colorbar.

    annotate_vals : bool or numpy.ndarray
        Set to True to overlay text values on matrix cells. If a numpy array
        the same size as <array> is supplied, will annotate the values
        contained on this array instead.

    avdiag : bool
        Set to True to average values across diagonal. Will be forced to False
        if matrix is not square.

    mask_diag, mask_tril, mask_triu : bool
        Setting to True will mask out the diagonal, upper-triangle, and/or
        lower-triangle elements. Cannot have both mask_tril and mask_triu set
        to True. Will be forced to False is matrix is not square.

    grid : bool or float
        Set to True to overlay grid lines with line width = 1. Set to a float
        value to overlay grid lines with this value as the line width.

    grid_color : str
        Any valid matplotlib colour for grid lines.

    xlabel, ylabel : str
        x- and y-axis labels.

    ticklabels, xticklabels, yticklabels : list
        x- and y-axis tick-labels.  If xticklabels or yticklabels are None,
        they will default to the value of ticklabels.

    tickrotation, xtickrotation, ytickrotation : float
        Rotation of x- and y-axis ticklabels in degrees.  If xtickrotation or
        ytickrotation is None, they will default to the value of tickrotation.

    xtickalignment, ytickalignment : str
        Alignment of x- and y-axis ticklabels.  xtickalignment can be 'left',
        'center' (default), or 'right'.  ytickalignment can be 'top',
        'center' (default), 'bottom', or 'baseline'.

    title : str
        Plot title.

    fontsize : float
        Global fontsize for plot (see also titlesize, labelsize, ticksize,
        and annotsize).

    titlesize : float
        Title fontsize (default = fontsize + 2).

    labelsize : float
        Axis and colorbar label fontsize (default = fontsize).

    ticksize : int
        Fontsize for axis ticklabels (default = fontsize - 2).

    annotsize : float
        Fontsize for cell annotations (default = fontsize - 2).

    lims : (min, max) tuple or None
        Colormap limits. If None (default) will use data range.

    dp : int
        Decimal places for colorbar ticks and matrix value labels.

    cbarlabel : str
        colorbar label.

    cbar_nticks : int
        Number of ticks to put on colorbar.

    cbar_ticklabels : list of strings
        Tick label strings to plot against colorbar. Number of elements must
        match specified <cbar_nticks> value. If omitted, labels will be
        <cbar_nticks> evenly spaced values between <lims>, formatted to
        specified number of dp.

    font : str
        Font style for text.

    cmap : str or matplotlib cmap instance
        May be any valid matplotlib colormap (string or matplotlib cmap
        instance) or fsl colormap string (see get_fsl_cmap function).

    fontcolor : str, rgb / rgba tuple, or hex value).
        Font color of all text

    segmented : bool
        Set to True to use a segmented colormap.

    segment_interval : float
        Interval between segments of colorbar (ignored if segmented = False).

    ax : axis handle or None
        If provided, attach plot to existing axis.

    Returns
    -------
    fig, ax : figure and axis handles
    """
    # Handle cmap
    cmap = get_colourmap(cmap)

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
    if annotsize is None:
        annotsize = fontsize - 2

    # Prep array
    array = np.array(array, dtype='float', copy=True)

    # Convert to square form?
    if np.ndim(array) == 1:
        array = squareform(array)
    if isinstance(annotate_vals, np.ndarray) and np.ndim(annotate_vals) == 1:
        annotate_vals = squareform(annotate_vals)

    # Get array shape
    nYConds, nXConds = array.shape
    if nXConds != nYConds:
        avdiag = mask_diag = mask_tril = mask_triu = False

    # Error check annotate_vals
    if not isinstance(annotate_vals, (bool, np.ndarray)):
        raise TypeError('annotate_vals must be bool or numpy.ndarray')
    elif (isinstance(annotate_vals, np.ndarray) and
          annotate_vals.shape != array.shape):
        raise ValueError('annotate_vals must be same shape as array')

    # Average across diagonal if requested
    if avdiag:
        array += array.T
        array /= 2

    # Mask elements if requested
    if mask_tril and mask_triu:
        raise Exception('Cannot set both mask_tril and mask_triu to True')
    if mask_diag:
        np.fill_diagonal(array, np.nan)
    if mask_tril:
        array[np.tril_indices_from(array, k = -1)] = np.nan
    if mask_triu:
        array[np.triu_indices_from(array, k = 1)] = np.nan

    # Determine min/max from lims (if specified) or array otherwise
    if lims is None:
        vmin = np.nanmin(array)
        vmax = np.nanmax(array)
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
        fig, ax_ = plt.subplots()
    else:
        fig = ax.figure
        ax_ = ax

    # Set title, x and y labels
    ax_.set_title(title, size=titlesize, family=font, color=fontcolor)
    ax_.set_xlabel(xlabel, size=labelsize, color=fontcolor, family=font)
    ax_.set_ylabel(ylabel, size=labelsize, color=fontcolor, family=font)

    # Plot matrix
    im = ax_.imshow(array, cmap=cmap, norm=norm, interpolation='nearest',
                    aspect='equal')

    # Add gridlines if requested. Draw lines across full plot if not
    # averaging over diagonal, or just up to to diagonal if averaging
    if grid:
        lw = float(grid)

        # Loop x coords and plot vertical gridlines
        for x in np.arange(0.5, nXConds - 0.5):
            if mask_tril:
                ymin = -0.5
                ymax = x if mask_diag else x + 1
            elif mask_triu:
                ymin = x if mask_diag else x - 1
                ymax = nXConds - 0.5
            else:
                ymin = -0.5
                ymax = nXConds - 0.5
            ax_.plot([x, x], [ymin, ymax], color=grid_color, ls='-', lw=lw)

        # Loop y coords and plot horizontal gridlines
        for y in np.arange(0.5, nYConds - 0.5):
            if mask_tril:
                xmin = y if mask_diag else y - 1
                xmax = nYConds - 0.5
            elif mask_triu:
                xmin = -0.5
                xmax = y if mask_diag else y + 1
            else:
                xmin = -0.5
                xmax = nYConds - 0.5
            ax_.plot([xmin, xmax], [y, y], color=grid_color, ls='-', lw=lw)

    # Add matrix tick labels
    ax_.set_xticks(np.arange(nXConds))
    ax_.set_yticks(np.arange(nYConds))
    if xticklabels is not None:
        ax_.set_xticklabels(
            xticklabels, size=ticksize, family=font, color=fontcolor,
            ha=xtickalignment, rotation=xtickrotation, rotation_mode='anchor'
            )
    if yticklabels is not None:
        ax_.set_yticklabels(
            yticklabels, size=ticksize, family=font, color=fontcolor,
            va=ytickalignment, rotation=ytickrotation, rotation_mode='default'
            )

    # Set axis limits (can get messed up by preceding plot tweaks)
    ax_.set_xlim(-0.5, nXConds-0.5)
    ax_.set_ylim(nYConds-0.5, -0.5)

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
            ax_.text(x, y, txt, ha='center', va='center', fontsize=annotsize,
                     family=font, color=fontcolor)

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

        # Add colorbar
        cb = fig.colorbar(im, boundaries=bounds, pad=0.03)
        cb.set_ticks(cbrng)
        cb.set_ticklabels(cbar_ticklabels)
        cb.set_label(cbarlabel, size=labelsize, family=font, color=fontcolor)
        cb.ax.tick_params(labelsize=ticksize, labelcolor=fontcolor)
        for t in cb.ax.get_yticklabels():
            t.set_family(font)

    # Return figure and axes
    return fig, ax_


def polar_pcolormesh(C, r=None, theta=None, hemi='both', theta_units='rad',
                     theta_direction='clockwise', cmap=None, vmin=None,
                     vmax=None, shading='gouraud', xticks=None, yticks=None,
                     xticklabels=None, yticklabels=None, font='sans-serif',
                     fontcolor='black', fontsize=10, grid=False,
                     grid_color='dimgrey', cbar=False, dp=2, cbarlabel='',
                     cbar_nticks=5, cbar_ticklabels=None, ax=None):
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

    r : 1D numpy array
        Radial values to plot over. Will use a default range between 0 and 1 if
        not specified.

    theta : 1D numpy array
        Angular values to plot over. Will use a default range (in radians)
        dependent on <hemi> if not specified. Note that 0 is at 12 o'clock.

    hemi : str ('both' | 'left' | 'right' | 'top' | 'bottom')
        Which hemifield to plot. Ignored if <theta> is specified.

    theta_units : str ('rad' | 'deg')
        Are angular units specified in radians or degrees?

    theta_direction : str or int
        Should theta increment clockwise ('clockwise', 'cw', or -1) or
        counter-clockwise ('counterclockwise', 'ccw', or 1)?

    cmap : str or matplotlib cmap instance
        May be any valid matplotlib colormap (string or matplotlib cmap
        instance) or fsl colormap string (see get_fsl_cmap function).

    vmin, vmax : float
        Limits for colourmap. If not specified, will default to data limits.

    shading : str ('flat' | 'nearest' | 'gouraud' | 'auto')
        Fill style for the qudrilateral's (see pcolormesh).

    xticks, yticks : list
        Tick positions along angular and radial axes respectively.
        <xticks> should be specified in units of <theta_units>.

    xticklabels, yticklabels : list
        Tick labels for angular and radial axes respectively.

    font : str
        Text font style.

    fontcolor : str
        Colour for text.

    fontsize : int
        Text size.

    grid : bool or float
        Set to True to overlay grid lines with line width = 1. Set to a float
        value to overlay grid lines with this value as the line width.

    grid_color : str
        Any valid matplotlib colour for grid lines.

    cbar : bool
        Sets whether to display colorbar next to plot.

    dp : int
        Decimal places for colorbar tick labels. Ignored if <cbar> is False.

    cbarlabel : str
        Colorbar axis label. Ignored if <cbar> is False.

    cbar_nticks : int, opttional
        Number of ticks to put on colorbar. Ignored if <cbar> is False.

    cbar_ticklabels : list of strings
        Tick label strings to plot against colorbar. Number of elements must
        match specified <cbar_nticks> value. If omitted, labels will be
        <cbar_nticks> evenly spaced values, formatted to specified <dp>.
        Ignored if <cbar> is False.

    ax : axis handle or None
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
    cmap = get_colourmap(cmap)

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
        fig, ax_ = plt.subplots(subplot_kw={'polar':True})
    else:
        ax_ = ax
        fig = ax_.figure

    # Plot
    im = ax_.pcolormesh(ftheta, fr, C, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading=shading)

    # Set axis details
    ax_.set_theta_direction(theta_direction)
    ax_.set_theta_zero_location('N', 0)
    ax_.set_thetamin(theta_lims[0])
    ax_.set_thetamax(theta_lims[1])

    if xticks is not None:
        if theta_units.startswith('deg'):
            xticks = np.radians(xticks)
        ax_.set_xticks(xticks)
    if yticks is not None:
        ax_.set_yticks(yticks)
    if xticklabels is not None:
        ax_.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax_.set_yticklabels(yticklabels)

    ax_.tick_params(labelsize=fontsize, labelcolor=fontcolor)
    for t in ax_.get_xticklabels():
        t.set_family(font)
    for t in ax_.get_yticklabels():
        t.set_family(font)

    # Add gridlines?
    if grid:
        lw = float(grid)
        ax_.grid(True, lw=lw, color=grid_color)

    # Plot colorbar if requested
    if cbar:
        cbrng = np.linspace(vmin, vmax, cbar_nticks)
        if cbar_ticklabels is not None:
            if len(cbar_ticklabels) != cbar_nticks:
                raise ValueError('Number of colorbar tick labels must match '
                                 'number of colorbar ticks')
        else:
            cbar_ticklabels = ['{:.{}f}'.format(x, dp) for x in cbrng]

        cb = fig.colorbar(im, boundaries=None)
        cb.set_ticks(cbrng)
        cb.set_ticklabels(cbar_ticklabels)
        cb.set_label(cbarlabel, size=fontsize, family=font, color=fontcolor)
        cb.ax.tick_params(labelsize=fontsize, labelcolor=fontcolor)
        for t in cb.ax.get_yticklabels():
            t.set_family(font)

    # Return
    return fig, ax_
