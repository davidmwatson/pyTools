#!/usr/bin/env python3

"""
Script supports assorted image processing capabilities:

 * Phase scrambling (applyPhaseScram)

 * Applying soft windows (SoftWindowImage)

 * Making amplitude / phase image composites (combineAmplitudePhase)

 * Fourier filtering (makeFourierFilter, applyFourierFilter)

 * Make hybrid filtered images (makeHybridImage)

 * Making amplitude masks (makeAmplitudeMask)

 * Overlaying a fixation cross on an image (overlayFixation)

 * Plotting average amplitude spectra (plotAverageAmpSpec)


Also includes assorted utitlity functions which may come in handy when using
other functions in this script:

 * imread - Reads in an image and prepares for use by other functions and
   classes in this script (shouldn't need to use directly very often)

 * createFourierMaps - Returns maps of the spatial frequency and orientation
   domains of a Fourier spectrum.

 * gaussian - Gaussian function.

 * butterworth - Butterworth function.

 * fwhm2sigma, sigma2fwhm - Converts between full-width-half-maximum and sigma
   values used for defining bandwidths for some filters (e.g. Gaussians).
"""

### Import statements
import os, warnings, imageio
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.ndimage
from PIL import Image

RNG = np.random.default_rng()


##### UTILITY FUNCTION DEFINITIONS #####

# Misc functions
def imread(image, output_dtype=np.float64, pad_depth_channel=True,
           alpha_action='mask'):
    """
    Handles loading of image.

    Parameters
    ----------
    image : valid filepath, PIL Image instance, or numpy array - required
        Image to be loaded.

    output_dtype : numpy dtype, optional
        Datatype to cast output array to. If None, datatype left unchanged.

    pad_depth_channel : bool, optional
        If True and image is grayscale, will pad a trailing dimension to
        maintain compatibility with colour processing pipelines.

    alpha_action : str {'remove' | 'mask' | 'error'} or None, optional
        What to do if image has an alpha channel. If 'remove', channel is
        simply removed. If 'mask', channel is first used to mask image and
        then removed. If 'error', raise an error. If None, channel left
        unchanged (not recommended).

    Returns
    -------
    im : np.ndarray
        Loaded image.
    """
    # Load image into numpy array
    if isinstance(image, str) and os.path.isfile(image):
        im = imageio.imread(image)
    elif isinstance(image, Image.Image):
        im = np.asarray(image)
    elif isinstance(image, np.ndarray):
        im = image.copy()
    else:
        raise IOError('Image must be a valid filepath, PIL Image instance, or '
                      'numpy array')

    # Handle alpha channel
    if im.ndim == 3 and (im.shape[2] in [2,4]) and alpha_action:
        if alpha_action == 'remove':
            warnings.warn('Removing alpha channel')
            im = im[..., :-1]
        elif alpha_action == 'mask':
            warnings.warn('Masking by alpha channel')
            orig_dtype = im.dtype
            im = im.astype(np.float64)
            im, mask = np.split(im, [-1], axis=-1)
            im *= mask / mask.max()
            im = im.astype(orig_dtype)
        elif alpha_action == 'error':
            raise Exception('Image contains alpha channel')
        else:
            raise ValueError('Unrecognised alpha action')

    # If grayscale, pad a trailing dim so it works with colour pipelines
    if im.ndim < 3 and pad_depth_channel:
        im = np.expand_dims(im, axis=2)

    # Convert dtype
    if output_dtype:
        im = im.astype(output_dtype)

    # Return
    return im


def postproc_im(im, output_range=(0,255), output_dtype=np.uint8, squeeze=True):
    """
    Utility function for post-processing images. Squeezes trailing dims,
    clips pixel values to given range, and converts to specified datatype.

    Arguments
    ---------
    im : numpy array, required
        Input image, as numpy array.

    output_range : (min, max) tuple or None, optional
        Min and max values to clip image to. If None, image is not clipped.

    output_dtype : valid numpy datatype, optional
        Datatype to convert image to (e.g. uint8, float). If None, image is
        left as current datatype.

    squeeze : bool
        If True (default), squeeze colour dimension from grayscale images.

    Returns
    -------
    im : numpy array
        Processed image array.
    """
    if squeeze:
        im = im.squeeze()
    if output_range is not None:
        im = im.clip(*output_range)
    if output_dtype is not None:
        im = im.astype(output_dtype)
    return im


def createFourierMaps(imsize, map_type):
    """
    Returns spatial frequency and orienation maps for a Fourier spectrum.

    Parameters
    ----------
    imsize : tuple, required
        (length, width) tuple indicating size of spectrum. Any trailing values
        beyond the first 2 are ignored.

    map_type : str {'sf' | 'ori'}, required
        Whether to return spatial frequency or orientation map.

    Returns
    -------
    fmap - numpy array
        Array containing requested Fourier map.
    """
    # Get length and width
    L,W = imsize[:2]

    # Make meshgrid
    Wrng = ifftshift(np.arange(-np.floor(W/2), np.ceil(W/2)))
    Lrng = ifftshift(np.arange(-np.floor(L/2), np.ceil(L/2)))
    [fx,fy] = np.meshgrid(Wrng, Lrng)

    # Create maps, return
    if map_type == 'sf':
        return np.sqrt(fx**2 + fy**2)
    elif map_type == 'ori':
        return np.mod(np.arctan2(fx, fy), np.pi)
    else:
        raise ValueError('map_type must be one of \'sf\' or \'ori\'')


# Conversion functions
def fwhm2sigma(fwhm):
    """
    Converts a full-width-half-maximum value to a sigma value
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma2fwhm(sigma):
    """
    Converts a sigma value to a full-width-half-maximum value.
    """
    return sigma * (2 * np.sqrt(2 * np.log(2)))


# Filter functions
def butterworth(X, cutoff, order, mu=0, cutin=None):
    """
    Butterworth function.

    Parameters
    ----------
    X : numpy.ndarray, required
        Range of values to plot over.
    cutoff : float, required
        Cut-off value for function.  Note that if array X includes values both
        above and below the centre value (mu), then the cut-off value will
        determine the extent of the filter both above and below this point
        (this is different to a FWHM for instance in which the bandwidth is
        centred around the central value).
    order : int, required
        Order of function - higher values give steeper descent after cutoff.
    mu : float, optional
        Value to centre function about.
    cutin : float, optional
        If not None, provides a cut-in value for the function.  This allows
        construction of a bandpass filter.  The value should therefore be less
        than the value of the cut-off.
    """
    # Sub-func for butterworth function
    def func(X, cutoff, order, mu):
        return 1 / (1 + ( (X-mu) / cutoff)**(2*order))

    # Create cutoff filter
    B = func(X, cutoff, order, mu)

    # If cutin, create 2nd filter and subtract from 1st one
    if cutin is not None:
       B -= func(X, cutin, order, mu)

    # Return
    return B


def gaussian(X, sigma, mu=0):
    """
    Gaussian function.

    Parameters
    ----------
    X : numpy.ndarray, required
        Range of values to plot over.
    sigma : float, required
        Standard deviation of Gaussian.
    mu : float, optional
        Value to centre Gaussian on.
    """
    return np.exp(-0.5 * ((X-mu)/sigma)**2)



##### MAIN FUNCTION DEFINITIONS #####

def applyPhaseScram(image, coherence=0.0, rndphi=None, mask=None, nSegs=1,
                    **kwargs):
    """
    Applies phase scrambling to grayscale or colour images.

    Arguments
    ---------
    image : any valid filepath, PIL Image instance, or numpy array
        Image to apply phase scrambling to.

    coherence : float, optional
        Number in range 0-1 that determines amount of phase scrambling to
        apply, with 0 being fully scrambled (default) and 1 being not scrambled
        at all.

    rndphi : array, optional
        2D array of random phases - allows using a custom phase spectrum.
        Should be same size as image segments (or whole image if nSegs == 1).
        If omitted, a random one will be generated.

    mask : array, optional
        Mask of weights in range 0-1 that can be applied to random phase array
        (e.g. to scramble only certain parts of the spectrum).  Mask should be
        for an unshifted spectrum, and same size as image segments (or whole
        image if nSegs == 1).

    nSegs : int, optional
        Number of segments to split image into.  If nSegs is 1 (default),
        scrambling is performed across entire image (i.e. global scrambling).
        For values greater than 1, the image is split into an nSegs*nSegs grid
        and scrambling performed locally within each window of the grid. Length
        and width of image must be divisible by nSegs.  Note that if nSegs > 1
        then values passed to <rndphi> and <mask> arguments must be for size of
        each grid window, not the whole image.

    **kwargs
        Additional keyword arguments are passed to postproc_im function.

    Returns
    -------
    scram: numpy array
        Phase scrambled image as numpy array.

    Examples
    --------
    Phase scramble image with 0% coherence

    >>> im = imageio.imread('imageio:coffee.png')
    >>> scram1 = applyPhaseScram(im)

    Scramble with 40% phase coherence

    >>> scram2 = applyPhaseScram(im, coherence = .4)

    Use own random phase array

    >>> import numpy as np
    >>> myrndphi = np.angle(np.fft.fft2(np.random.rand(*im.shape)))
    >>> scram3 = applyPhaseScram(im, rndphi = myrndphi)

    Weight rndphi by mask.  Here we weight by an inverted horizontal-pass
    filter to scramble vertical orientations but preserve horizontals.

    >>> from imageprocessing import makeFourierFilter
    >>> filt = makeFourierFilter(
    ...     im, mode='ori', filtertype='gaussian', invert=True,
    ...     filter_kwargs = {'mu':np.radians(0),
    ...                      'sigma':fwhm2sigma(np.radians(45))}
    ...     )
    >>> scram4 = applyPhaseScram(im, mask = filt)

    Locally scrambled image within windows of an 8x8 grid

    >>> scram5 = applyPhaseScram(im, nSegs = 8)
    """
    # Read in image
    im = imread(image)
    L,W,D  = im.shape

    # Work out segments
    if L % nSegs or W % nSegs:
        raise ValueError(
            f'Image dimensions ({L},{W}) must be divisible by nSegs ({nSegs})'
            )
    segL = L // nSegs
    segW = W // nSegs

    # Pre-allocate array for scrambled image
    scram = np.empty_like(im)

    # Loop over segments in x and y direction (if nSegs = 1 each loop will
    # execute only once)
    for y1 in range(0, L, segL):
        y2 = y1 + segL
        for x1 in range(0, W, segW):
            x2 = x1 + segW

            # Slice out window
            win = im[y1:y2, x1:x2]

            # If no random phase array specified, make one
            if rndphi is None:
                rndphi_ = np.angle(fft2(RNG.uniform(size=[segL,segW])))
            else:
                rndphi_ = rndphi.copy()

            # Weight random phases by (1 - coherence)
            rndphi_ *= 1 - coherence

            # Weight rndphi by mask if one is provided
            if mask is not None:
                rndphi_ *= mask

            # Loop over colour channels (for greyscale this will execute once)
            for i in range(D):
                # Fourier transform this colour channel, calculate amplitude
                # and phase spectra
                F = fft2(win[:,:,i])
                ampF = np.abs(F)
                phiF = np.angle(F)
                # Calculate new phase spectrum
                phiF += rndphi_
                # Combine original amplitude spectrum with new phase spectrum
                F[:] = ampF * np.exp(phiF * 1j)
                # Inverse transform, assign into scram
                scram[y1:y2, x1:x2, i] = ifft2(F).real

    # Postproc and return
    return postproc_im(scram, **kwargs)


def combineAmplitudePhase(ampimage, phaseimage, **kwargs):
    """
    Produces composite image comprising power spectrum of powerimage and
    phase spectrum of phaseimage.  Images must be same size.

    Arguments
    ---------
    ampimage : any valid filepath, PIL Image instance, or numpy array
        Image to derive amplitude spectrum from.

    phaseimage : any valid filepath, PIL Image instance, or numpy array
        Image to derive phase spectrum from.

    **kwargs
        Additional keyword arguments are passed to postproc_im function.

    Returns
    -------
    combo : numpy array
        Processed image as numpy array. Mean luminance is scaled to
        approximate the mean of the input images.
    """
    # Read in images
    ampim = imread(ampimage)
    phaseim = imread(phaseimage)

    # Check images are same shape
    if ampim.shape !=  phaseim.shape:
        raise Exception('Images must be same shape')

    # Make note of image dimensions
    L,W,D = ampim.shape

    # Pre-allocate combined image array
    combo = np.empty([L,W,D], dtype = float)

    # Loop over colour channels (for grayscale will execute only once)
    for i in range(D):
        # Get power and phase of current colour channels in relevant images
        r = np.abs(fft2(ampim[:,:,i]))
        phi = np.angle(fft2(phaseim[:,:,i]))

        # Calculate new spectrum
        z = r * np.exp(phi * 1j)

        # Inverse transform, allocate to combo
        combo[:,:,i] = ifft2(z).real

    # Normalise mean luminance to mean of means of two images
    combo = (combo - combo.mean()) / combo.std()
    combo *= np.mean([ampim.std(), phaseim.std()])
    combo += np.mean([ampim.mean() , phaseim.mean()])

    # Postproc and return
    return postproc_im(combo, **kwargs)


def makeAmplitudeMask(imsize, rgb=False, beta=1, **kwargs):
    """
    Creates an amplitude mask of specified size.

    Arguments
    ---------
    imsize : tuple or list
        Desired size of mask as (L,W) tuple or [L,W] list. Any further trailing
        values are ignored.

    rgb : bool, optional
        If rgb = True, will create a colour mask by layering 3 amplitude masks
        into a RGB space.  Default is False.

    beta : int or float
        Power to raise frequency to, i.e. amplitude spectrum is given as
        1/(f**beta).  When beta = 1 (default) amplitude mask is pink noise.
        As beta increases, mask moves through red/brown noise and appears
        increasingly smooth.  When beta = 0, mask is equivalent to white
        noise.  Recommended values would generally be in range 0.5 to 2.

    **kwargs
        Additional keyword arguments are passed to postproc_im function.

    Returns
    -------
    ampmask : numpy array
        Requested amplitude mask as numpy array
    """
    ### Sub-function definitions
    def _run(ampF, L, W):
        """ Sub-function. Handles creation of amplitude mask """
        # Make random phase spectrum
        rndphi = np.angle(fft2(RNG.uniform(size=[L,W])))

        # Construct Fourier spectrum, inverse transform
        F = ampF * np.exp(1j * rndphi)
        ampmask = ifft2(F).real

        # ampmask is currently centred on zero, so rescale to range 0 - 255
        ampmask -= ampmask.min() # set min to 0
        ampmask *= 255.0 / ampmask.max() # set max to 255
        return ampmask.astype(np.uint8) # cast to uint8, return

    ### Begin main function
    # Get L and W
    L = imsize[0]
    W = imsize[1]

    # Get map of the spectrum
    SFmap = createFourierMaps([L,W], 'sf')

    # Make 1/f amplitude spectrum
    with np.errstate(divide = 'ignore'): # (ignore divide by 0 warning at DC)
        ampF = SFmap**(-beta)
    ampF[0,0] = 0 # Set DC to 0 - image will be DC-zeroed

    # Make amplitude mask according to rgb argument
    if rgb:
        # If rgb requested, layer 3 amplitude masks into RGB image
        ampmask = np.empty([L,W,3], dtype = np.uint8) # pre-allocate array
        for i in range(3):
            ampmask[:,:,i] = _run(ampF, L, W)
    else:
        # Otherwise, just make mask directly
        ampmask = _run(ampF, L, W)

    # Postproc and return
    return postproc_im(ampmask, **kwargs)


def overlayFixation(image=None, lum=255, offset=0, arm_length=12, arm_width=2,
                    imsize=(256,256), bglum=127, **kwargs):
    """
    Overlays fixation cross on specified image.

    Arguments
    ---------
    image : None, or any valid filepath, PIL Image instance, or numpy array
        Image to overlay fixation cross on.  If None (default), will overlay
        on a blank image instead (see also imsize and bglum arguments)

    lum : int or RGB tuple of ints, or 4 item tuple of either, optional
        Luminance of fixation cross.  If a single int or RGB tuple, this will
        be taken as the luminance for all 4 fixation arms.  If a 4 item tuple
        of ints or RGB tuples is given, a different luminance will be applied
        to each arm; the ordering should go clockwise from 12 o'clock,
        i.e. (upper, right, bottom, left).

    offset : int, optional
        Distance from center of the image to the nearest pixel of each arm

    arm_length : int, optional
        Length of each arm of fixation cross, specified in pixels

    arm_width : int, optional
        Thickness of each arm of fixation cross, specified in pixels (should be
        even number)

    imsize : (L,W) tuple, optional
        Desired size of blank image, defaults to 256x256 pixels.  Ignored if
        image is not None.

    bglum : int or RGB tuple of ints, optional
        Background luminance for blank image, defaults to mid-grey (127).
        Ignored if image is not None.

    **kwargs
        Additional keyword arguments are passed to postproc_im function.

    Returns
    -------
    im : numpy array
        Processed image as numpy array
    """
    AL, AW = arm_length, arm_width # for brevity

    # Assume lum is 4 item iterable giving different lum for each arm.  If
    # not iterable (single grayscale lum given) or is iterable but length is
    # 3 (single RGB tuple given), then repeat 4 times to match expected format
    if not hasattr(lum, '__iter__') or len(lum) == 3:
        lum = [lum] * 4

    # Create blank image if none provided
    if image is None:
        L, W = imsize[:2] # ignore any trailing dims beyond L and W
        # Work out if we need to make image in RGB or grayscale space,
        # dependent on whether any luminance args for cross or bg are RGB
        if any([hasattr(l, '__iter__') and len(l) == 3 for l in lum]) or \
           (hasattr(bglum, '__iter__') and len(bglum) == 3):
            D = 3 # RGB space
        else:
            D = 1 # grayscale
        # Create blank image of specified size and luminance
        im = np.ones((L,W,D), dtype = float) * bglum
    # If image is specified, read it in
    else:
        im = imread(image)

    # Determine midpoint of image
    hL, hW = im.shape[0]//2, im.shape[1]//2

    # Also need half of arm width
    hAW = AW//2

    # Overlay fixation cross
    # Upper arm
    im[hL-(AL+offset) : hL-offset, hW-hAW : hW+hAW, :] = lum[0]
    # Right arm
    im[hL-hAW : hL+hAW, hW+offset : hW+AL+offset, :] = lum[1]
    # Lower arm
    im[hL+offset : hL+AL+offset, hW-hAW : hW+hAW, :] = lum[2]
    # Left arm
    im[hL-hAW : hL+hAW, hW-(AL+offset) : hW-offset, :] = lum[3]

    # Postproc and return
    return postproc_im(im, **kwargs)


def plotAverageAmpSpec(indir, ext='png', nSegs=1, dpi=96, cmap=None):
    """
    Calculates and plots log-scaled Fourier average amplitude spectrum
    across a number of images.

    Spectra are calculated for all images in indir with specified extension.
    Outputs are saved into a directory called "AmplitudeSpectra" created
    inside the input directory.  Outputs are (1) the average amplitude
    spectrum across images stored in a numpy array and saved as .npy file,
    and (2) contour plots of the average spectrum saved as image files.

    Arguments
    ---------
    indir : str
        A valid filepath to directory containing the images to calculate
        the spectra of. All images in indir must have same dimensions.

    ext : str
        File extension of the input images (default = png).

    nSegs : int, optional
        Number of segments to window image by.  Spectra are calculated within
        each window separately.  If nSegs = 1 (default), the spectrum is
        calculated across the whole image.

    dpi : int, optional
        Resolution to save plots at (default = 96).

    cmap : any valid matplotlib cmap instance
        Colourmap for filled contour plot.
    """
    # Local imports just for this function
    import glob
    import matplotlib.pyplot as plt

    # Ensure . character not included in extension
    ext = ext.strip('.')

    # Glob for input files
    infiles = sorted(glob.glob(os.path.join(indir, f'*.{ext}')))
    if len(infiles) == 0:
        raise IOError('No images found! Check directory and extension')

    # Determine image dimensions from first image
    tmp = imageio.imread(infiles[0], as_gray=True)
    L, W = tmp.shape
    del(tmp)

    # Work out if we can segment image evenly, and dims of windows if we can
    if L % nSegs or W % nSegs:
        raise ValueError(
            f'Image dimensions ({L},{W}) must be divisible by nSegs ({nSegs})'
            )
    segL = L // nSegs
    segW = W // nSegs

    # Pre-allocate array for storing spectra
    spectra = np.empty([len(infiles), L, W], dtype=float)

    # Process inputs
    print('Processing...')
    # Loop over images
    for i, infile in enumerate(infiles):
        print('\t%s' %infile)

        # Read in, grayscale if RGB
        im = imageio.imread(infile, as_gray=True)

        # Calculate amplitude spectrum for current window
        for y in range(0, L, segL):
            for x in range(0, W, segW):
                # Slice out window
                win = im[y:y+segL, x:x+segW]
                # Calculate amplitude spectrum for window
                ampF = np.abs(fftshift(fft2(win)))
                # Log scale, assign relevant window of spectrum (we use ampF+1
                # to avoid any -ve values from log scaling values < 1)
                spectra[i, y:y+segL, x:x+segW] = np.log1p(ampF)
        spectra[i] /= spectra[i].max() # scale full array to range 0:1

    # Create average spectrum
    av_spectrum = spectra.mean(axis = 0)

    ### Save array, make and save plots
    print('Saving array and plots...')
    outdir = os.path.join(indir, 'AmplitudeSpectra')
    os.makedirs(outdir, exist_ok=True)

    # Main numpy array
    savename = os.path.join(outdir, f'win{nSegs}_array.npy')
    np.save(savename, av_spectrum)

    # Filled contour figure
    aspect_ratio = L / W
    figsize = (6.125, 6.125 * aspect_ratio) # 6.125 is default figure height
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1]) # add axes that fill figure
    ax.axis('off')
    ax.contour(av_spectrum, colors = 'k', origin = 'upper')
    ax.contourf(av_spectrum, origin='upper', cmap=cmap, vmin=0, vmax=1)
    savename = os.path.join(outdir, f'win{nSegs}_filled_contour.png')
    fig.savefig(savename, dpi = dpi)
    print('Saved ' + savename)
    plt.close(fig)

    # Line contour figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1]) # add axes that fill figure
    ax.axis('off')
    # Values of contour lines chosen to plot around mid-region of log-scaled
    # spectrum in range 0:1
    ax.contour(av_spectrum, [0.45, 0.55], colors = 'k', linewidths = 2,
               origin = 'upper')
    savename = os.path.join(outdir, f'win{nSegs}_line_contour.png')
    fig.savefig(savename, dpi = dpi)
    print('Saved ' + savename)
    plt.close(fig)


def makeFourierFilter(image_or_imsize, mode, filtertype, filter_kwargs={},
                      invert=False):
    """
    Makes filter of spatial frequency or orientation, to be applied in
    Fourier domain. Intended to be used in conjunction with applyFourierFilter.

    Arguments
    ---------
    image_or_imsize : int, (height,width) tuple, filepath, PIL Image, or array
        Can be image itself, or a (height, width) tuple giving image size. If
        a single integer, will take this as the size of a square image.

    mode : 'sf' or 'ori', required
        Use strings 'sf' or 'ori' to indicate to make a spatial frequency
        or an orientation filter respectively.

    filtertype : str or callable, required
        Type of filter to use.  Available options are:
         * 'gaussian' to use a Gaussian filter
         * 'butterworth' to use a Butterworth filter
        Alternatively, may specify a custom callable function.  This
        function should take a numpy arrary mapping the values (either
        spatial frequency or orientation) of the Fourier spectrum as its
        first argument (see also createFourierMaps function).  Spatial
        frequency values should be given in units of cycles/image, whilst
        orientation values should be given in radians and should be in the
        interval 0:pi.  Further keyword arguments to the function may be
        passed using the filter_kwargs argument of this function.

    filter_kwargs : dict or list of dicts, optional
        Dictionary of additional keyword arguments to be passed to filter
        function after the initial argument. Keys should indicate argument
        names, values should indicate argument values.  If a list of dicts
        is provided a separate filter will be created for each set of
        paramemters, and all of them summed together to create a composite
        filter - this is useful to create a filter with multiple components
        such as a cardinal or an oblique orientation filter.  In general,
        units should be given in cycles/image for a frequency filter and in
        radians in the interval 0:pi for an orientation filter.  If using a
        named filter (e.g. 'gaussian'), see help information of the
        relevant function in this script for available arguments.

    invert : bool, optional
        Set invert = True to invert filter, e.g. to make a high-pass filter
        (default = False).

    Returns
    -------
    filt : numpy array
        Requested filter as numpy array.

    Example usage
    -------------

    Low-pass Gaussian filter at FHWM = 30 cycles/image

    >>> im = imageio.imread('imageio:coffee.png')
    >>> from imageprocessing import fwhm2sigma
    >>> lowfilt = makeFourierFilter(
    ...     im, mode='sf', filtertype='gaussian',
    ...     filter_kwargs={'mu':0, 'sigma':fwhm2sigma(30)}
    ...     )

    High-pass Gaussian filter at FWHM = 50 cycles/image

    >>> highfilt = makeFourierFilter(
    ...     im, mode='sf', filtertype='gaussian', invert=True,
    ...     filter_kwargs={'mu':0, 'sigma':fwhm2sigma(50)}
    ...     )

    Vertical-pass Gaussian filter with at FWHM = 30 degrees

    >>> vertfilt = makeFourierFilter(
    ...     im, mode='ori', filtertype='gaussian',
    ...     filter_kwargs={'mu':np.radians(90),
    ...                    'sigma':np.radians(fwhm2sigma(30))}
    ...     )

    Oblique-pass 2nd-order Butterworth filter with cut-offs 15 degrees either
    side of centre orientations.

    >>> oblqfilt = makeFourierFilter(
    ...     im, mode='ori', filtertype='butterworth',
    ...     filter_kwargs=[ {'cutoff':np.radians(15), 'order':2,
    ...                      'mu':np.radians(45)},
    ...                     {'cutoff':np.radians(15), 'order':2,
    ...                      'mu':np.radians(135)} ]
    ...     )

    Low-pass filter using custom ideal-filter with cut-off 30 cycles/image

    >>> def ideal(X, cutoff):
    ...     return (X <= cutoff).astype(float)
    >>> idealfilt = makeFourierFilter(im, mode='sf', filtertype=ideal,
    ...                               filter_kwargs={'cutoff':30})

    See also
    --------
    * applyFourierFilter - Apply filter to image.
    * fwhm2sigma - Convert a full-width-half-maximum value to a sigma value
      (useful when making Gaussian filters).
    """
    # Get image size
    if isinstance(image_or_imsize, (list, tuple)):
        imsize = image_or_imsize[:2]
    elif isinstance(image_or_imsize, (int, float)):
        imsize = [int(image_or_imsize)] * 2
    else:
        imsize = imread(image_or_imsize).shape[:2]

    # Ensure appropriate mode
    if mode not in ['sf', 'ori']:
        raise ValueError('Mode must be \'sf\' or \'ori\'')

    # Assign appropriate filter function
    if isinstance(filtertype, str):
        filtertype = filtertype.lower() # ensure case insensitive
        if filtertype == 'gaussian':
            filter_func = gaussian
        elif filtertype == 'butterworth':
            filter_func = butterworth
        else:
            raise ValueError('Unrecognised filter type')
    elif callable(filtertype):
        filter_func = filtertype
    else:
        raise TypeError('Filter must be allowable string or callable')

    # If filter_kwargs is a dict, assume only one has been given and wrap
    # in list
    if isinstance(filter_kwargs, dict):
        filter_kwargs = [filter_kwargs]

    # Create spatial frequency or orientations map
    X = createFourierMaps(imsize, mode)

    # Pre-allocate filter, including trailing dimension for each sub-filter
    filt = np.zeros( X.shape + (len(filter_kwargs),) )

    # Loop through filter_kwargs
    for i, this_filter_kwargs in enumerate(filter_kwargs):
        # If doing frequency, just make filter and allocate to array
        if mode == 'sf':
            filt[..., i] = filter_func(X, **this_filter_kwargs)
        # If doing orientation, sum 3 filters to include +/- pi rad
        else:
            for offset in [-pi, 0, pi]:
                filt[..., i] += filter_func(X + offset, **this_filter_kwargs)

    # Sum filters along last dimension to create composite
    filt = filt.sum(axis=-1)

    # Invert if requested
    if invert:
        filt = 1 - filt

    # Add in DC
    filt[0,0] = 1

    # Return
    return filt


def applyFourierFilter(image, filt, **kwargs):
    """
    Apply filter to image. Intended to be used in conjunction with
    makeFourierFilter.

    Arguments
    ---------
    image : any valid filepath, PIL Image instance, or numpy array
        Image to apply phase scrambling to.

    filt : array
        Filter to apply to image. Can be created using makeFourierFilter
        function. Filter should be for an unshifted spectrum.

    **kwargs
        Further keyword arguments passed to postproc_im function.

    Returns
    -------
    filtim : numpy array
        Filtered image as numpy array.

    Example usage
    -------------

    Low-pass Gaussian filter image at FHWM = 30 cycles/image

    >>> im = imageio.imread('imageio:coffee.png')
    >>> from imageprocessing import fwhm2sigma
    >>> lowfilt = makeFourierFilter(
    ...     im, mode='sf', filtertype='gaussian',
    ...     filter_kwargs={'mu':0, 'sigma':fwhm2sigma(30)}
    ...     )
    >>> filtim = applyFourierFilter(image, filt)

    See also
    --------
    * makeFourierFilter - Makes filter.
    """
    # Load image
    im = imread(image)

    # Pre-allocate array for filtered image
    filtim = np.empty_like(im)

    # Loop over colour channels (will execute only once if grayscale)
    for i in range(im.shape[2]):
        F = fft2(im[:,:,i]) # into frequency domain
        F *= filt # apply filter
        filtim[:,:,i] = ifft2(F).real # back to image domain

    # Postproc and return
    return postproc_im(filtim, **kwargs)


def makeHybridImage(image1, image2, filter1_params, filter2_params, **kwargs):
    """
    Create hybrid image by filtering two images and compositing the results.

    Parameters
    ----------
    image1, image2: Any valid filepaths, PIL Image instances, or numpy arrays
        Input images. Must have same dimensions.

    filter1_params, filter2_params : dicts
        Dictionaries of keyword arguments to pass to ``makeFourierFilter``
        function, except for `image_or_imsize` argument which gets set by
        this function. Filters will be applied to image1 and image2
        respectively.

    **kwargs
        Additional keyword arguments are passed to postproc_im function.

    Returns
    -------
    hybrid : numpy array
        Hybrid image as numpy array.

    Example usage
    -------------

    Make a spatial frequency hybrid, with the first image low-pass filtered
    below FWHM = 30 cycles/image, and the second image high-pass filtered
    above FWHM = 50 cycles/image.

    >>> im1 = imageio.imread('imageio:camera.png')
    >>> im2 = imageio.imread('imageio:astronaut.png', as_gray=True)
    >>> from imageprocessing import fwhm2sigma
    >>> filter1_params = {'mode':'sf', 'filtertype':'gaussian', 'invert':False,
    ...                   'filter_kwargs':{'sigma':fwhm2sigma(30)}}
    >>> filter2_params = {'mode':'sf', 'filtertype':'gaussian', 'invert':True,
    ...                   'filter_kwargs':{'sigma':fwhm2sigma(50)}}
    >>> hybrid = makeHybridImage(
    ...     im1, im2, filter1_params, filter2_params
    ...     )
    """
    # Read images
    im1 = imread(image1)
    im2 = imread(image2)
    if im1.shape != im2.shape:
        raise Exception('Input images must have same dimensions')

    # Create filters, set DCs to 0.5
    filter1_params['image_or_imsize'] = im1
    filt1 = makeFourierFilter(**filter1_params)
    filt1[0,0] = 0.5

    filter2_params['image_or_imsize'] = im2
    filt2 = makeFourierFilter(**filter2_params)
    filt2[0,0] = 0.5

    # Filter images
    filtim1 = applyFourierFilter(im1, filt1, output_range=None,
                                 output_dtype=None, squeeze=False)
    filtim2 = applyFourierFilter(im2, filt2, output_range=None,
                                 output_dtype=None, squeeze=False)

    # Add filtered images to create hybrid
    hybrid = filtim1 + filtim2

    # Postproc and return
    return postproc_im(hybrid, **kwargs)



##### MAIN CLASS DEFINITIONS #####
class SoftWindowImage():
    """
    Class provides functions for applying a cosine-ramp soft window around
    edges of image.

    Many thanks to Dan Baker for providing the original version of this script!

    Arguments
    ---------
    mask_shape : str ('rect' | 'ellipse'), optional
        Shape of mask to apply - rectangular or elliptical
    mask_fwhm : float , optional
        Width of mask across the image between half-maximum points of the
        cosine ramp, specified as a proportion (between 0 and 1) of the
        image dimensions.

    Methods
    -------
    .maskImage
        Create and apply mask to image.

    Examples
    --------
    Apply a rectangular soft window

    >>> im = imageio.imread('imageio:astronaut.png')
    >>> windower = SoftWindowImage('rect')
    >>> winIm = windower.maskImage(im)

    The mask is created when the first image is processed, and stored within
    the class. The mask can be re-used for subsequent images of the same size

    >>> im2 = imageio.imread('imageio:wikkie.png')
    >>> winIm2 = windower.maskImage(im2)

    Create an elliptical mask with a fwhm of 0.8, and apply to image setting
    background to be white

    >>> windower = SoftWindowImage('ellipse', mask_fwhm=0.8)
    >>> winIm3 = windower.maskImage(im, bglum=255)
    """
    def __init__(self, mask_shape, mask_fwhm=0.9):
        if mask_shape not in ['ellipse', 'rect']:
            raise ValueError('mask_shape must be \'ellipse\' or \'rect\'')
        self.mask_shape = mask_shape
        self.mask_fwhm = mask_fwhm
        self.mask = None  # placeholder for mask

    def _createMask(self, imsize):
        """
        Create soft-window mask, assigns into class. Should get called
        automatically when first image is run.

        Arguments
        ---------
        imsize : tuple, required
            (L, W) tuple of image dimensions. Trailing dims are ignored.
        """
        # Extract image length and width for brevity
        L, W = imsize[:2]

        ### Create cosine smoothing kernel
        # Determine width of blur along horizontal / vertical edges of mask
        x_blurW = int(W * (1 - self.mask_fwhm))
        y_blurW = int(L * (1 - self.mask_fwhm))
        # Make cosine half cycles for horizontal / vertical dims
        x_cosine = np.cos(np.linspace(-pi/2, pi/2, x_blurW))
        y_cosine = np.cos(np.linspace(-pi/2, pi/2, y_blurW))

        ### Create mask for requested shape
        # Start with ideal mask (ones inside mask, zeros outside)
        if self.mask_shape == 'ellipse':
            # Work out radii of x and y dims
            x_radius = W * self.mask_fwhm / 2
            y_radius = L * self.mask_fwhm / 2
            # Make mask
            [fx,fy] = np.meshgrid(range(W),range(L))
            mask = ( ((fx - W/2)/x_radius)**2 + ((fy-L/2)/y_radius)**2 ) < 1
            mask = mask.astype(float)
        elif self.mask_shape == 'rect':
            # Initialise blank mask
            mask = np.zeros([L,W])
            # Work out border width along x and y dims
            x_bordW = x_blurW // 2
            y_bordW = y_blurW // 2
            # Fill in mask
            mask[y_bordW:-y_bordW, x_bordW:-x_bordW] = 1

        # Convolve mask with kernels to give blurred edge to mask
        # (doing two 1D convolutions along each axis is fastest)
        mask = scipy.ndimage.convolve1d(mask, x_cosine, axis=1)
        mask = scipy.ndimage.convolve1d(mask, y_cosine, axis=0)

        # Rescale to max of 1
        mask /= mask.max()

        # Assign to class
        self.mask = mask

    def maskImage(self, image, bglum='mean', **kwargs):
        """
        Create and apply mask to image.

        Arguments
        ---------
        image : any valid filepath, PIL Image instance, or numpy array
            Image to apply mask to.

        bglum : {'mean', int or float, RGB tuple of ints or floats}, optional
            Luminance to set background outside masked region to.
            If set to 'mean' (default) the mean image luminance is used.

        **kwargs
            Additional keyword arguments are passed to postproc_im function.

        Returns
        -------
        im : numpy array
            Masked image as numpy array with datatype uint8.
        """
        # Read image
        im = imread(image)

        # Create mask if not already done
        if self.mask is None or (self.mask.shape != im.shape[:2]):
            self._createMask(im.shape)

        # If no bglum provided, use mean luminance of image
        if bglum == 'mean':
            bglum = im.mean()

        # Subtract bglum
        im -= bglum

        # Apply mask to each channel in turn (executes only once if D == 1)
        for i in range(im.shape[2]):
            im[:,:,i] *= self.mask

        # Add bglum back in
        im += bglum

        # Postproc and return
        return postproc_im(im, **kwargs)
