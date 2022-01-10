#!/usr/bin/env python3
"""
Python port of MATLAB SHINE toolbox for controlling low-level image properties.


Reference
---------
Willenbockel, V., Sadr, J., Fiset, D., Horne, G. O., Gosselin, F.,
Tanaka, J. W. (2010). Controlling low-level image properties: The SHINE
toolbox. Behavior Research Methods, 42, 671-684.
https://doi.org/10.3758/BRM.42.3.671

SHINE toolbox, May 2010
(c) Verena Willenbockel, Javid Sadr, Daniel Fiset, Greg O. Horne,
Frederic Gosselin, James W. Tanaka

Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.


Dependencies
------------
* numpy
* scipy
* imageio
* Python Image Library (PIL) or Pillow
* matplotlib (only for specPlot & seperate funcs with qplot == True)
* Scikit-Image (only for SSIM optimisation of histogram matching)
"""

from __future__ import division
import os
import warnings
import itertools
import imageio
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

try:
    # pillow fork
    from PIL import Image
except ImportError:
    # original PIL
    import Image

have_ssim = False
try:
    # skimage >= 0.16
    from skimage.metrics import structural_similarity as ssim
    have_ssim = True
except ImportError:
    try:
        # skimage < 0.16
        from skimage.measure import compare_ssim as ssim
        have_ssim = True
    except ImportError:
        pass


### Hidden functions ###

def _load_images_and_masks(imlist, masklist=None, **kwargs):
    """
    Convenience function wrapping readImage; load images and mask images.
    Additional **kwargs passed to readImage function for main images only.
    """
    # Load images
    images = [readImage(im, **kwargs) for im in imlist]

    # Load masks
    if masklist is not None:
        masks = [None if m is None else readImage(m, dtype=bool) \
                 for m in masklist]
    else:
        masks = [None] * len(images)

    # Return
    return(images, masks)


def _hist_match_image(im, targ, mask, inplace=True):
    """
    Performs exact histogram matching between specified image and target
    histogram. See histMatch function for main user interface.
    """
    # Copy?
    if not inplace:
        im = im.copy()

    # Add a small amount of random noise to break ties for sorting in next
    # step.
    im += 0.1 * np.random.rand(*im.shape)

    # Sort image pixels (we actually only need indices of sort)
    if mask is None:
        idcs = np.argsort(im.flat)
    else:
        idcs = np.argsort(im[mask].flat)

    # Replace image histogram with target histogram, using idcs to place
    # pixels at correct positions
    svim = np.empty(len(idcs))
    svim[idcs] = targ
    if mask is None:
        im[:] = svim.reshape(im.shape)
    else:
        im[mask] = svim

    # Return?
    if not inplace:
        return im



### Main functions ###

def readImage(image, grayscale=True, dtype=float, alpha_action='mask'):
    """
    Function for loading an image into a numpy array

    Arguments
    ---------
    image - numpy array | PIL Image object | valid filepath
        Image to be loaded.
    grayscale - bool, optional
        If True (default) and image is RGB, will convert to grayscale.
    dtype - valid numpy datatype, optional
        Output images will be coerced to this datatype (default = float)
    alpha_action - 'remove'  | 'mask', optional
        What to do if image has an alpha channel. If 'remove', channel is
        simply removed. If 'mask' (default), channel is first used to mask
        image and then removed.

    Returns
    -------
    image - numpy array
        Image loaded into numpy array
    """
    # Load image
    if isinstance(image, np.ndarray):
        im = image.copy()
    elif isinstance(image, Image.Image):
        im = np.asarray(image)
    elif isinstance(image, str) and os.path.exists(image):
        im = np.asarray(imageio.imread(image))
    else:
        raise IOError(f'Cannot read image {image}')

    # Handle alpha channel
    if im.ndim == 3 and (im.shape[2] in [2,4]):
        if alpha_action == 'remove':
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn('Removing alpha channel')
            im = im[..., :-1]
        elif alpha_action == 'mask':
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn('Masking by alpha channel')
            orig_dtype = im.dtype
            im = im.astype(np.float64)
            im, mask = np.split(im, [-1], axis=-1)
            im *= mask / mask.max()
            im = im.astype(orig_dtype)
        else:
            raise ValueError('Unrecognised alpha action')

    # Grayscale-if necssary
    if grayscale and im.ndim > 2:
        im = np.array(Image.fromarray(im).convert('F'))

    # Convert to requested datatype
    im = im.astype(dtype)

    # Return
    return im


def rescale_images(images, rescale='basic', rng=(0,255), dtype='uint8'):
    """
    Rescales image to specfied range and casts to specified datatype.
    Used as post-processing function for other image processing functions.

    Arguments
    ---------
    images - array-like of numpy arrays
        Images to process - should be concatenated along first dimension.
    rescale - None | False | 'basic' | 'all' | 'average', optional
        * If None/False, will not perform any rescaling or clipping before
          returning image. Note - when using this option you should first
          make sure the image values are within a sensible range for the
          requested datatype (see below), e.g. within 0-255 for uint8.
        * If 'basic' (default), will simply ensure each image is clipped
          within allowable range.
        * If 'all', as per 'basic' but will also first scale all images so
          that min and max across all images are within the range.
        * If 'average', as per 'basic' but will also first scale all images
          so that the average min and max across images is within the range.
    rng - (min, max) tuple, optional
        Range to scale images into. Ignored if rescale == None.
    dtype - valid numpy datatype, optional
        Datatype to cast images to. NOTE - make sure that this is sensible for
        the requested value range.

    Returns
    -------
    images - list
        Processed images, concatenated along first dimension.
    """
    # Error check
    if rescale not in [None, False, 'basic', 'all', 'average']:
        raise TypeError("rescale must be None, 'basic', 'all', or 'average'")

    # Rescale / clip if requested
    if rescale:
        # Apply preliminary rescaling if necessary
        if rescale in ['all', 'average']:
            trg_min, trg_max = rng

            # rescale min
            if rescale == 'all':
                _min = np.min(images)
            else: # 'average'
                _min = np.mean([im.min() for im in images])

            # Temporarily rescale min to zero
            for im in images:
                im -= _min

            # rescale max
            if rescale == 'all':
                _max = np.max(images)
            else: # 'average'
                _max = np.mean([np.max(im) for im in images])

            # Rescale max, accounting for target min
            for im in images:
                im *= (trg_max - trg_min) / _max
                im += trg_min

        # Clip images to range, cast to requested datatype
        images = [np.clip(im, *rng).astype(dtype) for im in images]

    # If rescale is None, cast images to requested datatype without clipping
    else:
        images = [im.astype(dtype) for im in images]

    # Return
    return images


def getRMSE(image1, image2):
    """
    Get root-mean squared error between images.
    """
    im1 = readImage(image1, grayscale=False)
    im2 = readImage(image2, grayscale=False)
    return np.sqrt( ((im1 - im2)**2).mean() )


def getImstats(images, masks=None, bins=range(257)):
    """
    Calculates luminance and contrast statistics for images.

    Arguments
    ---------
    images - array-like
        List of input images in any format accepted by readImage function.
    masks - array-like, optional
        List of mask images in any format accepted by readImage function.
    bins - array-like, optional
        Bin edges for luminance histogram.

    Returns
    -------
    stats - dict
        Dictionary with following keys:
          * meanVec - vector of mean luminances for each image
          * stdVec - vector of image standard deviations
          * histArr - nImages x nBins array of histograms for each image
          * meanLum - mean of lums
          * meanStd - mean of contrasts
          * meanHist - mean of histArr across images
    """
    # Dict for appending results to
    stats = dict()
    stats['bins'] = bins

    # Load images and masks
    images, masks = _load_images_and_masks(images, masks)

    # Calculate stats
    stats['meanVec'] = np.empty(len(images))
    stats['stdVec'] = np.empty(len(images))
    stats['histArr'] = np.empty((len(images), len(bins)-1))
    for i, (im, m) in enumerate(zip(images, masks)):
        tmp = im[m] if m is not None else im.flatten()
        stats['meanVec'][i] = tmp.mean()
        stats['stdVec'][i] = tmp.std()
        stats['histArr'][i,:] = np.histogram(tmp, bins=bins)[0]

    # Calculate means
    stats['meanLum'] = stats['meanVec'].mean()
    stats['meanStd'] = stats['stdVec'].mean()
    stats['meanHist'] = stats['histArr'].mean(axis=0)

    # Return
    return stats


def separate(image, background=None, qplot=False):
    """
    Performs simple figure-ground segmentation.

    Arguments
    ---------
    image - any valid input to readImage
        Image to be segmented.
    background - None | int | float, optional
        Background luminance value - should be in range 0-255.  If None
        (default), will use most frequently occurring value in image.
    qplot - bool, optional
        If True, will make a plot of the background mask (default = False).

    Returns
    -------
    mask - numpy array
        Boolean mask of image (foreground == True, background == False).
    background - int | float
        Background luminance value.
    fig - (fig, ax) tuple
        Figure and axis handles. Only returned if qplot == True.
    """
    # Error check
    if not (background is None or 0 <= background <= 255):
        raise TypeError('Background must be None or value in range 0:255')

    # Read image
    im = readImage(image, dtype=np.uint8)

    # Define background as mode average of image if not provided
    if background is None:
        background = np.bincount(im.flat).argmax()

    # Mask image
    mask = im != background

    # De-noise
    mask = scipy.signal.medfilt(mask, kernel_size=(3,3)).astype(bool)

    # Plot if requested
    if qplot:
        fig, ax = plt.subplots()
        ax.imshow(mask, interpolation='nearest')

    # Return
    if qplot:
        return mask, background, (fig, ax)
    else:
        return mask, background


def specPlot(image, qplot=True):
    """
    Calculates (and can plot) the power spectrum of an image and it's
    corresponding rotational average.

    Arguments
    ---------
    image - any valid input to readImage
        Input image.
    qplot - bool, optional
        If True (default) will produce plots of the spectrum.

    Returns
    -------
    spec - numpy array
        Calculated power spectrum.
    rot_spec - numpy array
        Rotational average of power spectrum.
    (fig1, ax1) - figure and axis handles
        Plot of spectrum (only returned if qplot == True)
    (fig2, ax2) - figure and axis handles
        Plot of spectrum rotational average (only returned if qplot == True)
    """
    # Load image, get dimensions
    im = readImage(image)
    L, W = im.shape
    xmin, xmax = (-W//2, W//2)
    ymin, ymax = (-L//2, L//2)
    xrng = range(xmin, xmax)
    yrng = range(ymin, ymax)

    # Calculate power spectrum
    spec = np.abs(fftshift(fft2(im)))**2

    # Calculate rotational average of spectrum
    [fx, fy] = np.meshgrid(xrng, yrng)
    sf = (np.sqrt(fx**2 + fy**2)).round().astype(int).flatten()
    rot_spec = np.bincount(sf, weights=spec.flatten()) / np.bincount(sf)
    rot_spec = rot_spec[1:min(L,W)//2]

    # Make a plot if requested
    if qplot:
        fig1, ax1 = plt.subplots()
        h = ax1.imshow(np.log10(spec), extent=[xmin, xmax, ymin, ymax])
        ax1.axis('off')
        cb = fig1.colorbar(h)
        cb.set_label(r'$\log_{10}$(Energy)')

        fig2, ax2 = plt.subplots()
        ax2.loglog(np.arange(1, len(rot_spec)+1), rot_spec)
        ax2.set_xlabel('Spatial frequency (cycles/image)')
        ax2.set_ylabel('Energy')

    # Return
    if qplot:
        return spec, rot_spec, (fig1, ax1), (fig2, ax2)
    else:
        return spec, rot_spec


def lumMatch(images, masks=None, lum=None, contrast=None, grayscale=True,
             rescale_kwargs={}):
    """
    Perform basic mean luminance and contrast matching on images.

    Arguments
    ---------
    images - array-like
        List of input images in any format accepted by readImage function.
        Images should be concatenated along first dimension.
    masks - array-like
        List of mask images in any format accepted by readImage function.
        Masks should be concatenated along first dimension. Processing
        will occur within masked region only.  If None (default), will apply
        processing to whole image.
    lum - None | int | float, optional
        Desired mean luminance to match to.  If None (default), will use
        average luminance across images.
    contrast - None | int | float, optional
        Desired contrast (luminance standard deviation) to match to.
        If None (default), will use average standard deviation across images.
    grayscale - bool, optional
        If True (default), images will be converted to grayscale.
    rescale_kwargs - dict, optional
        Kwargs to pass to rescale_images function for image post-processing

    Returns
    -------
    lum_matched - list
        List of luminance equated images.
    """
    # Load images and masks
    images, masks = _load_images_and_masks(images, masks, grayscale=grayscale)

    # Get mean and std if not provided
    if lum is None:
        lum = np.mean([im[m].mean() if m is not None else im.mean() \
                       for im, m in zip(images, masks)])
    if contrast is None:
        contrast = np.mean([im[m].std() if m is not None else im.std() \
                            for im, m in zip(images, masks)])

    # Lum equate
    for im, m in zip(images, masks):
        if m is None:
            if im.std() != 0:
                # set mean = 0 and std  = 1
                im -= im.mean()
                im /= im.std()
                # set specified mean and std
                im *= contrast
                im += lum
            else:
                im = lum
        else:
            if im[m].std() != 0:
                # set mean = 0 and std  = 1
                im[m] -= im[m].mean()
                im[m] /= im[m].std()
                # set specified mean and std
                im[m] *= contrast
                im[m] += lum
            else:
                im[m] = lum

    # Postproc and return
    return rescale_images(images, **rescale_kwargs)


def histMatch(images, masks=None, hist=None, optim=False,
              optim_params={'niters':10, 'stepsize':67}, rescale_kwargs={}):
    """
    Perform exact histogram matching across images. Note - images will be
    converted to grayscale if necessary.

    Arguments
    ---------
    images - array-like
        List of input images in any format accepted by readImage function.
        Images should be concatenated along first dimension.
    masks - array-like, optional
        List of mask images in any format accepted by readImage function.
        Masks should be concatenated along first dimension. Processing
        will occur within masked region only.  If None (default), will apply
        processing to whole image.
    hist - (counts, bins) tuple, optional
        Target histogram to match to, specified as tuple of arrays giving
        counts and bin edges. If None (default), will use average histogram
        across images.
    optim - bool, optional
        If True, will optimise structural similarity (SSIM) index
        (default = False)
    optim_params - dict with keys 'niters' and 'stepsize', optional
        Dictionary specifying number of iterations and stepsize for SSIM
        optimisation. Ignored if optim == False.
    rescale_kwargs - dict, optional
        Kwargs to pass to rescale_images function for image post-processing

    Returns
    -------
    hist_matched - list
        List of histogram equated images.
    """
    # Error check
    if optim and not have_ssim:
        raise RuntimeError('SSIM optimisation requires scikit-image module')

    # Load images and masks
    images, masks = _load_images_and_masks(images, masks)

    # If hist not provided, obtain average histogram across images
    if hist is None:
        bins = range(257)
        allCounts = np.empty((len(images), len(bins)-1))
        for i, (im, m) in enumerate(zip(images, masks)):
            tmp = im[m] if m is not None else im.flatten()
            allCounts[i,:] = np.histogram(tmp, bins=bins)[0]
        counts = allCounts.mean(axis=0).round().astype(int)
    else:
        counts, bins = hist

    # Obtain flattened target histogram
    targ = np.asarray(list(itertools.chain.from_iterable(
            [ [lum] * count for lum, count in zip(bins, counts) ]
            )))

    # Hist equate
    for im, m in zip(images, masks):
        # Rounding errors when calculating histograms may lead to small
        # mismatches between length of idcs and targ.  If so, interpolate a
        # range of indices across targ that will make it right length
        sz = m.sum() if m is not None else im.size
        if len(targ) != sz:
            ix = np.linspace(0, len(targ)-1, sz).round().astype(int)
            t = targ[ix]
        else:
            t = targ

        # Do SSIM optimisation if requested
        if optim == True:
            for i in range(optim_params['niters']-1):
                tmp = _hist_match_image(im, t, m, inplace=False)
                mssim, grad = ssim(
                        im, tmp, data_range=255, use_sample_covariance=False,
                        gaussian_weights=True, sigma=1.5, gradient=True
                        )
                im[:] = tmp + optim_params['stepsize'] * im.size * grad

        # Do final histogram match
        _hist_match_image(im, t, m, inplace=True)

    # Return
    return rescale_images(images, **rescale_kwargs)


def specMatch(images, targmag=None, grayscale=True, rescale_kwargs={}):
    """
    Match amplitude spectra across images.  Function requires that all images
    are the same size.

    Arguments
    ---------
    images - array-like
        List of input images in any format accepted by readImage function.
        Images should be concatenated along first dimension.  All images must
        be same size.
    targmag - numpy array, optional
        Target amplitude spectrum. Must be same size as image and should be
        for an unshifted spectrum (i.e. DC should be at cell [0,0]). If None
        (default), will use average of image amplitude spectra.
    grayscale - bool, optional
        If True (default), images will be converted to grayscale.
    rescale_kwargs - dict, optional
        Kwargs to pass to rescale_images function for image post-processing

    Returns
    -------
    spec_matched - list
        List of spectrum matched images.
    """
    # Load images
    images = [readImage(im, grayscale=grayscale) for im in images]

    # Check all image are same size
    for im in images[1:]:
        if im.shape != images[0].shape:
            raise ValueError('All images must have same dimensions')

    # Calculate spectra
    amp_spectra = np.empty_like(images)
    phase_spectra = np.empty_like(images)
    for i, im in enumerate(images):
        F = fft2(im, axes=(0,1))
        amp_spectra[i] = np.abs(F)
        phase_spectra[i] = np.angle(F)

    # Calculate tarmag if needed
    if targmag is None:
        targmag = amp_spectra.mean(axis=0)

    # Match amplitude spectra to targmag
    for i in range(len(images)):
        F = targmag * np.exp(1j * phase_spectra[i])
        images[i] = ifft2(F, axes=(0,1)).real

    # Return images after rescaling
    return rescale_images(images, **rescale_kwargs)
