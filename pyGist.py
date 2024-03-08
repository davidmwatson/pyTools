#!/usr/bin/env python3

import os, imageio
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, fftshift
from PIL import Image


################################################################################


class LMgist():
    """
    Class provides a python port of the LMgist.m MATLAB script and associated
    functions written by Audo Oliva and Antonio Torralba for calculating
    a GIST descriptor from an image.

    For more info see Antonio Torralba's webiste on the GIST descriptor:
    http://people.csail.mit.edu/torralba/code/spatialenvelope/

    Please use this reference for the GIST descriptor:
    Oliva, A., Torralba, A. (2001) Modeling the Shape of the Scene: A Holistic
    Representation of the Spatial Envelope.
    International Journal of Computer Vision. 42(3), 145-175
    http://link.springer.com/article/10.1023/A:1011139631724


    Parameters
    ----------
    orientationsPerScale (optional) : list of ints, default = [8, 8, 8, 8]
        List indicating number of orientations and scales (frequencies) to
        sample from.  Each item denotes a spatial scale (from high to low
        frequency) such that the length of the list determines the number of
        scales to be sampled. The value of each item specifies the number of
        orientations to sample at that scale.  For example, the default of
        [8,8,8,8] will sample 4 scales each across 8 orientations.

    numberBlocks (optional) : int, default = 4
        Number of segments to divide image by, i.e. image will be divided
        into a numberBlocks * numberBlocks grid.

    imageSize (optional) : [L,W] list, default = size of provided image
        Length and width of image in pixels.

    fc_prefilt (optional) : int or float, default = 4
        Parameter for pre-filtering step, specified in cycles / image.

    G (optional) : L * W * nFilters array of Gabor filters, default = None
        Filters to be applied to image - will be created if not provided.

    gist (optional) : list, default = None
        Gist descriptor vector.  Useful if you have already calculated a gist
        descriptor and wish to load it in for plotting with showGist (but note
        that the other parameters must be correct for this vector for plotting
        to work).


    Returns
    -------
    Class instance for specified image and parameters.


    Functions
    -------
    .run(image) :
        Calculates GIST descriptor for specified image.
        See function help for more information.

    .showGist(mode) :
        Python port of the showGist.m Matlab script. See function help for
        more information.


    Examples
    --------
    Get an image

    >>> from scipy.misc import face
    >>> im = face(gray = True)

    Instantiate class with default parameters

    >>> from pyGist import LMgist
    >>> gist = LMgist()

    Calculate GIST descriptor for image

    >>> gist.run(im)

    Results get assigned back into class. Try plotting gist descriptor vector.

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.plot(gist.gist)
    >>> plt.show()

    Plot first filter

    >>> import numpy as np
    >>> plt.figure()
    >>> plt.imshow(np.fft.fftshift(gist.G[:,:,0]))
    >>> plt.show()

    Plot first filtered image

    >>> plt.figure()
    >>> plt.imshow(gist.filt_ims[:,:,0], cmap = 'gray')
    >>> plt.show()

    Plot downsampled version of above

    >>> plt.figure()
    >>> plt.imshow(gist.down_ims[:,:,0], cmap = 'gray', interpolation = 'nearest')
    >>> plt.show()

    Use .showGist() method to visualise gist descriptor in image space

    >>> fig = gist.showGist(mode = 'imshow')
    >>> fig.show()

    If you wish to run another image under the same parameters, you may pass
    it straight to the .run() method without having to re-instantiate the
    class.  If both images are the same size, the script will re-use the
    existing filters to save on computation time.

    >>> gist.run(\'/some/other/image.png\')

    """

    ### Main functions ###

    def __init__(self, orientationsPerScale = [8,8,8,8], numberBlocks = 4,
                    imageSize = None, fc_prefilt = 4, G = None, gist = None):

        # Handle imageSize argument
        if imageSize is not None: # case where imageSize is None is handled by .run() method
            # If a single value assume a square image
            if not hasattr(imageSize, '__iter__'):
                imageSize = [imageSize, imageSize]
            else:
                # Also assume square image for single-item list
                if len(imageSize) == 1:
                    imageSize = [imageSize[0], imageSize[0]]
                # If more than 2 values (e.g. depth of RGB image included),
                # only take first 2 values
                elif len(imageSize) > 2:
                    imageSize = imageSize[:2]

        # Assign arguments into class
        self.orientationsPerScale = orientationsPerScale
        self.numberBlocks = numberBlocks
        self.imageSize = imageSize
        self.fc_prefilt = fc_prefilt
        self.G = G
        self.gist = gist
        self.boundaryExtension = 32 # number of pixels to pad


    def run(self, image):
        """
        Calculate gist descriptor.

        Paramters
        ---------
        image (required) : any valid filepath, PIL Image instance, or numpy array.

        Ouput
        -----
        Results get assigned back into class under the following attributes:
         * gist : values of gist descriptor vector
         * G : array containing copies of the Gabor filters
         * filt_ims : array containing copies of each filtered image
         * down_ims : array containing copies of each filtered image down-sampled
           to a numberBlocks * numberBlocks grid

        """
        # Load image in
        self._imread(image)

        # Crop and rescale image to image size if necessary
        if self.imageSize is not None and not all(self.im.shape == \
                np.array(self.imageSize)):
            self._imresizecrop()

        # Create Gabor filters for filtering if we don't have them already
        # or if they are the wrong size (e.g. new image has been supplied)
        if self.G is None or not all(self.G.shape[:2] == \
                np.array(self.im.shape) + 2 * self.boundaryExtension):
            self._createGabor()

        # Rescale pixel luminances
        self._imrescale()

        # Pre-filter
        self._prefilt()

        # Calculate gist
        self._gistGabor()


    def showGist(self, mode = 'imshow', cmap = 'jet', signed = False):
        """
        Python port of the showGist.m Matlab script.  Note that this is a stripped
        down version of the Matlab script that does not assign a colour scale to
        the different frequency bands - colormap instead simply reflects the
        intensity of the pixel values.

        Note that if you want to use this method without first calling the
        .run() method on an image (e.g. you want to plot a pre-computed gist
        vector from somewhere else), then you will need to instantiate the
        class with both that vector supplied to the gist argument, and the
        correct image size supplied to the imageSize argument.

        Parameters
        ----------
        mode (optional) : str - 'imshow' (default) or 'contour'
            If 'imshow' will use make a standard image plot, if 'contour' will
            make a contour plot.
        cmap (optional) : any valid matplotlib colourmap, default = 'jet'
            Determines colourmap of plot
        signed (optional): bool, default = False
            If False, colourmap will run from min(data) to max (data). If True,
            colourmap will run from +/- max(abs(data)) and will be centred on
            zero, e.g. could use if plotting component loadings from a PCA.

        Returns
        -------
        fig : matplotlib figure instance
            Handle to generated figure

        Examples
        -------

        Calculate gist for an image, then make showGist plot.

        >>> from pyGist import LMgist
        >>> gist = LMgist()
        >>> gist.run('/some/image.png')
        >>> fig = gist.showGist()
        >>> fig.show()

        If you're not going to run the gist calculation first (e.g. you're
        plotting a gist vector you made earlier) you will need to instantiate
        the class with both that gist vector and the correct image size. In
        the following example, it is assumed that you have loaded the
        pre-computed gist vector into a variable called "gist_vec", and that
        this was originally calculated for a 256x256 pixel image.

        >>> gist = LMgist(gist = gist_vec, imageSize = (256,256))
        >>> fig = gist.showGist()
        >>> fig.show()

        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Some admin stuff
        if not mode in ['imshow','contour']:
            raise ValueError('mode should be \'imshow\' or \'contour\'')
        gist = self.gist
        nBlocks = self.numberBlocks
        nWindows = nBlocks**2

        # If we don't already have filters, try to make them (note - user must
        # specify an image or imageSize argument for this to work)
        if self.G is None:
            self._createGabor()

        # Extract filters into local variable for brevity
        G = self.G

        # Work out filter dims and number of filters
        x,y, nFilters = G.shape
        down_x, down_y = int(np.ceil(x/2.0)), int(np.ceil(y/2.0)) # size after downsampling

        # Indices describing which window and filter each point in gist relates to
        gistIdcs = np.arange(len(gist)).reshape(nFilters, nWindows)

        # Pre-allocate 4D array for storing plots
        gistPlots = np.empty([down_x,down_y,nWindows,nFilters], dtype = float)

        # Loop over filters
        for i in range(nFilters):
            # Get filter (down-sample to every other pixel to reduce memory load)
            filt = G[::2,::2,i]
            # Mirror about origin to give other 'tail' of filter, apply fftshift
            filt = fftshift(filt + np.rot90(filt,2))

            # Loop over windows
            for j in range(nWindows):
                gistIdx = gistIdcs[i,j] # position in gist for this window and filter
                # Weight filter by value in gist, allocate to gist plots
                gistPlots[:,:,j,i] = gist[gistIdx] * filt

        # Average across filters
        gistPlots = gistPlots.mean(axis = 3)

        ### Plot
        # Work out our colour limits
        if signed:
            vlim = np.abs(gistPlots).max()
            vmin, vmax = -vlim, vlim
        else:
            vmin, vmax = gistPlots.min(), gistPlots.max()

        # Create figure and ImageGrid
        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols = (nBlocks, nBlocks),
                direction = 'column', axes_pad = 0.04)

        # Loop over grid, fill in with requested plots
        for i, ax in enumerate(grid):
            thisPlot = gistPlots[:,:,i]
            if mode == 'imshow':
                ax.imshow(thisPlot, cmap = cmap, vmin = vmin, vmax = vmax)
            else: # if mode == 'contour'
                ax.contourf(thisPlot, cmap = cmap, origin = 'upper',
                        norm = Normalize(vmin = vmin, vmax = vmax))
            ax.set_axis_off()

        # Return figure handle
        return fig


    ### Hidden functions ###

    def _imread(self, image):
        """
        Handles loading of image.  Argument to function must be a valid filepath,
        PIL Image instance, or numpy array.  If image is in colour it will be
        converted to grayscale.
        Returns image as float64 numpy array
        """
        # Load image, convert to grayscale, put in numpy array
        if isinstance(image, str) and os.path.isfile(image):
            im = imageio.imread(image)
        elif isinstance(image, Image.Image):
            im = np.array(image)
        elif isinstance(image, np.ndarray):
            im = image
        else:
            raise IOError('Image must be a valid filepath, PIL Image instance, or numpy array')

        # If image is RGB, convert to grayscale
        if im.ndim  == 3:
            im = np.array(Image.fromarray(im).convert('F'))

        # Force cast to float64, assign back into class
        self.im = im.astype(np.float64)


    def _imresizecrop(self, method = Image.BILINEAR):
        """
        Resizes and crops image to be have L and W of (L,W) imageSize tuple
        """
        im = self.im
        L, W = np.array(im.shape[:2])
        newL, newW = np.array(self.imageSize)

        # Rescale so smallest dim matches equivalent dim of imageSize
        scaling = np.max([newL / L, newW / W])
        newsize = np.round([W * scaling, L * scaling]).astype(int) # (W,L) order for tuple as PIL uses this
        im = np.array(Image.fromarray(im).resize(newsize, method))

        # Longest dim may still exceed imageSize so crop where necessary
        sr = (im.shape[0] - newL) // 2
        sc = (im.shape[1] - newW) // 2
        im = im[sr:newL+sr, sc:newW+sc]

        # Assign back into class
        self.im = im


    def _imrescale(self):
        """
        Rescales luminance values of image into range 0:255
        """
        im = self.im
        im -= im.min()
        im *= 255.0 / im.max()
        self.im = im


    def _createGabor(self):
        """
        Creates Gabor filters (to be applied in Fourier domain)
        """
        ori = self.orientationsPerScale

        # Try to make filters from specified image size first - makes it
        # possible to generate filters without an actual image loaded
        if self.imageSize is not None:
            dims = self.imageSize
        # Otherwise, use actual size of image
        else:
            dims = self.im.shape

        # Add boundary extension margin to get required size of filters
        L,W = np.array(dims) + 2 * self.boundaryExtension

        # Work out number of scales and number of filters
        Nscales = len(ori)
        NFilters = sum(ori)

        # Make Nscales x 4 list of numbers, used later for making filters
        param = []
        for i in range(Nscales):
            for j in range(ori[i]):
                param.append( [0.35,
                               0.3/(1.85**(i)),
                               16.0*ori[i]**2 / 32.0**2,
                               pi/(ori[i])*(j) ] )
        param = np.array(param)

        # Generate maps of SF and orientations
        [fx,fy] = np.meshgrid(range(-W//2, W//2), range(-L//2, L//2))
        fr = fftshift(np.sqrt(fx**2 + fy**2)) # frequency map
        t = fftshift(np.arctan2(fy, fx)) # orientation map in range -pi:pi

        # Generate filters
        G = np.empty([L, W, NFilters], dtype = float)
        for i in range(NFilters):
            # "rotate" orientation map to match current ori
            tr = t + param[i, 3]
            # Wrap back into interval -pi:pi
            tr[tr < -pi] += 2 * pi
            tr[tr > pi] -= 2 * pi
            # Create Gabor filter, allocate to G
            G[:,:,i] = np.exp(-10 * param[i,0] * (fr/W/param[i,1] - 1)**2 - 2*param[i,2]*pi*tr**2)

        # Assign back into class
        self.G = G


    def _prefilt(self):
        """
        Apply pre-filtering to image.
        Input images are assumed to be floats in range 0:255
        """
        im = self.im.copy()
        fc = self.fc_prefilt
        w = 5 # padding width
        s1 = fc / np.sqrt(np.log(2))

        # Pad images to reduce boundary artifacts
        im = np.log(im+1)
        im = np.pad(im, w, 'symmetric')
        [sn, sm] = im.shape
        # Pad again till image dims are even
        n = max(sn, sm)
        n += n % 2
        im = np.pad(im, [[0, n-sn], [0,n-sm]], 'symmetric')

        # Filter
        meshrng = range(-n//2, n//2)
        fx, fy = np.meshgrid(meshrng, meshrng)
        gf = fftshift(np.exp(-(fx**2+fy**2)/(s1**2)))

        # Whiten
        im -= np.real(ifft2(fft2(im) * gf))

        # Local contrast normalisation
        localstd = np.sqrt(np.abs(ifft2(fft2(im**2) * gf)))
        im /= 0.2 + localstd

        # Crop to same size as input
        im = im[w:sn-w, w:sm-w]

        # Assign back into class
        self.prefilt_im = im


    def _gistGabor(self):
        """
        Apply filters to prefiltered image and downsample.  Return gist vector.
        """
        nBlocks = self.numberBlocks
        G = self.G
        be = self.boundaryExtension

        # Work out some other details
        W = nBlocks**2
        nrows, ncols = self.prefilt_im.shape
        ny, nx, nFilters = G.shape
        lenGist = W * nFilters

        # Pre-allocate gist vector
        gist = np.empty(lenGist, dtype = float)

        # Pre-allocate stores for filtered and down-sampled images
        filt_ims = np.empty([nrows, ncols, nFilters], dtype = float)
        down_ims = np.empty([nBlocks, nBlocks, nFilters], dtype = float)

        # Fourier transform padded image
        F = fft2(np.pad(self.prefilt_im, be, 'symmetric'))

        # Apply filters
        for i, k in enumerate(np.arange(0, lenGist, W)):
            ig = np.abs(ifft2(F * G[:,:,i])) # apply filter, ifft result
            ig = ig[be:-be, be:-be] # remove border
            v = self._downN(ig, nBlocks) # down sample
            gist[k:k+W] = v.flatten(order = 'F') # allocate flattened array to gist
            filt_ims[:,:,i] = ig # allocate filtered image to filt_ims
            down_ims[:,:,i] = v # allocate downsampled array to down_ims

        # Assign back into class
        self.gist = gist
        self.filt_ims = filt_ims
        self.down_ims = down_ims


    def _downN(self, x, N):
        """
        Average over non-overlapping square image blocks
        """
        nx = np.fix(np.linspace(0, x.shape[0], N+1)).astype(int)
        ny = np.fix(np.linspace(0, x.shape[1], N+1)).astype(int)
        y = np.empty([N, N], dtype = float)
        for xx in range(N):
            for yy in range(N):
                y[xx,yy] = x[nx[xx]:nx[xx+1], ny[yy]:ny[yy+1]].mean()
        return y


################################################################################
# If script called directly, run a demo
if __name__ == '__main__':
    print('#### Running Demo ####')
    import matplotlib.pyplot as plt
    plt.close('all')
    # Read in image
    print('Loading image...')
    im = imageio.imread('imageio:camera.png')

    # Run gist
    print('Calculating gist...')
    gist = LMgist()
    gist.run(im)

    # Display image
    print('Plotting...')
    plt.figure('Fig 1: Image')
    plt.imshow(gist.im, cmap = 'gray')
    plt.axis('off')

    # Plot gist vector
    plt.figure('Fig 2: GIST vector')
    plt.plot(gist.gist)

    # Plot filters
    plt.figure('Fig 3: Gabor Filters')
    for i in range(gist.G.shape[2]):
        f = fftshift(gist.G[:,:,i])
        plt.contour(f, [0.5, 0.75], colors = 'r', origin = 'upper') # contours at 1/2 and 3/4 maximum
    plt.axis('image')

    # Plot showGist
    fig = gist.showGist()
    fig.canvas.set_window_title('Fig 4: GIST visualised with showGIST')

    # Display
    plt.show()

    print('\nDone\n')

################################################################################
