#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated version of CIFTI tools. Simplifies masker by removing ability to
invert mask, while also adding some new functionality such as selecting
specific label IDs or using CIFTI structures as masks.

Requires nibabel >= v3.2, which isn't currently installed at YNiC. You might
need to pip install it yourself:
pip3 install --user --upgrade nibabel==3.2
"""

import os, warnings, inspect, copy
import numpy as np
import nibabel as nib

# Quick version check on nibabel
from packaging.version import parse as parse_version
if parse_version(nib.__version__) < parse_version('3.2.0'):
    raise ImportError('Nibabel version must be >= 3.2.0')


def data2cifti(data, template, dtype=None, cifti_type=None, axis_kwargs={}):
    """
    Construct new Cifti2Image object from data array.

    Arguments
    ---------
    data : ndarray
        [nSamples x nGrayordinates] array of data.

    template : nibabel.Cifti2Image
        Existing CIFTI image object to use as template. Should provide correct
        information for the columns axis, plus any necessary details for the
        NIFTI header and extra information.

    dtype : None or valid datatype
        Datatype to cast data to. If None (default), use existing datatype.

    cifti_type : None, str, or nibabel.cifti2.Axis class
        If provided, specifies a data type for the CIFTI image. Valid options
        are 'scalar', 'series', 'label', 'parcels', or a nibabel.cifti2.Axis
        class or child class or instance thereof. This can be useful if the
        size and/or datatype for the samples axis is different between the
        data array and template CIFTI image. If None (default), will just use
        whatever is specified in the template.

    axis_kwargs : dict
        Dictionary of keyword arguments to pass to axis class for the specified
        CIFTI data type. See nibabel documention for possible options. Ignored
        if cifti_type is None or an instance of a Axis class.

    Returns
    -------
    cifti : nibabel.Cifti2Image
        New CIFTI image object.
    """
    # Check cifti type
    if cifti_type is None:
        header = template.header
    else:
        if isinstance(cifti_type, nib.cifti2.Axis):
            ax0 = cifti_type
        else:
            if inspect.isclass(cifti_type) and issubclass(cifti_type, nib.cifti2.Axis):
                ax_class = cifti_type
            elif cifti_type == 'scalar':
                ax_class = nib.cifti2.ScalarAxis
            elif cifti_type == 'series':
                ax_class = nib.cifti2.SeriesAxis
            elif cifti_type == 'label':
                ax_class = nib.cifti2.LabelAxis
            elif cifti_type == 'parcels':
                ax_class = nib.cifti2.ParcelsAxis
            else:
                raise ValueError(f'Invalid cifti type: {cifti_type}')
            ax0 = ax_class(**axis_kwargs)

        ax1 = template.header.get_axis(1)
        header = nib.cifti2.Cifti2Header.from_axes([ax0, ax1])

    # Check dtype
    if dtype is None:
        dtype = data.dtype
    else:
        data = data.astype(dtype)

    # Create new CIFTI object
    new_cifti = nib.Cifti2Image(dataobj=data, header=header,
                                nifti_header=template.nifti_header,
                                extra=template.extra)
    new_cifti.update_headers()  # fixes nifti_header
    new_cifti.nifti_header.set_data_dtype(dtype)

    # Return
    return new_cifti



class CiftiHandler(object):
    """
    Provides tools for extracting surface and volume data from a nibabel CIFTI
    image. Can also use image as a template for creating new images from
    data arrays.

    Adapted from: https://nbviewer.jupyter.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb

    Arguments
    ---------
    img : str or nibabel.Cifti2Image object
        Path to CIFTI file, or CIFTI image object containing data.

    full_surface : bool
        If False (default), return only those vertices contained within the
        CIFTI file. If True, return all vertices for full surface, setting any
        missing vertices to zero (typically just the medial wall).

    Methods
    -------
    * get_volume_data : Extract volume data
    * get_surface_data : Extract surface data for a given hemisphere
    * get_all_data : Convenience function for extracting surface and volume data
    * create_new_cifti : Create new CIFTI image from provided data

    """
    def __init__(self, img, full_surface=False):
        # Load cifti
        if isinstance(img, nib.Cifti2Image):
            self.cifti = img
        elif isinstance(img, str) and os.path.isfile(img):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.cifti = nib.load(img)
        else:
            raise ValueError('img must be valid filepath or Cifti2Image object')

        self.axis0 = self.cifti.header.get_axis(0)
        self.axis1 = self.cifti.header.get_axis(1)

        # Assign remaining args to class
        self.full_surface = full_surface

    def _get_struct_info(self, struct_name):
        """
        Return structure indices and axis model for given structure
        """
        # Check name
        struct_name = self.axis1.to_cifti_brain_structure_name(struct_name)

        # Loop structures, return matching one
        for name, struct_indices, model in self.axis1.iter_structures():
            if name == struct_name:
                return struct_indices, model

        # If we reach here then structure doesn't exist - raise error
        raise Exception(f'No data found for structure: {struct_name}')

    def _get_volume_mask(self):
        """
        Return copy of volume mask
        """
        return self.axis1.volume_mask.copy()

    def _get_surface_mask(self):
        """
        Return copy of surface mask
        """
        return self.axis1.surface_mask.copy()

    def get_all_data(self, dtype=None, squeeze_data=False):
        """
        Return full data array from all available surface and volume blocks.

        Parameters
        ----------
        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True, squeeze singleton dimensions from data array. Default is
            False.

        Returns
        -------
        data : dict
            Dictionary of [samples x grayordinates] arrays containing data
            values for each block ('lh' for left surface, 'rh' for right
            surface, and 'volume' for sub-cortical structures)
        """
        data = {
            'lh':self.get_surface_data('lh', dtype, squeeze_data),
            'rh':self.get_surface_data('rh', dtype, squeeze_data),
            'volume':self.get_volume_data(dtype, squeeze_data),
            }
        return data

    def get_volume_data(self, dtype=None, squeeze_data=False):
        """
        Return all data from volume block.

        Arguments
        ---------
        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True, squeeze singleton dimensions from data array. Default is
            False.

        Returns
        -------
        vol_data : ndarray
            [samples x voxels] numpy array containing data values.
        """
        # Extract mask
        vol_mask = self._get_volume_mask()

        # Mask data
        data = self.cifti.get_fdata()
        vol_data = data[:, vol_mask]

        # Convert dtype?
        if dtype is not None:
            vol_data = vol_data.astype(dtype)

        # Squeeze?
        if squeeze_data:
            vol_data = vol_data.squeeze()

        # Return
        return vol_data

    def get_surface_data(self, hemi, dtype=None, squeeze_data=False):
        """
        Return all data from given hemisphere.

        Arguments
        ---------
        hemi : str { lh | rh }
            Name of hemisphere to load data from.

        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True, squeeze singleton dimensions from data array. Default is
            False.

        Returns
        -------
        surf_data : ndarray
            [samples x vertices] numpy array containing data values.
        """
        # Work out surface name
        if hemi == 'lh':
            surf_name = 'left_cortex'
        elif hemi == 'rh':
            surf_name = 'right_cortex'
        else:
            raise ValueError(f'Unrecognised hemisphere: \'{hemi}\'')

        # Extract data info
        n_samp = self.axis0.size

        # Search axis structures for requested surface
        struct_indices, model = self._get_struct_info(surf_name)

        # Extract surface data, pad to full set of vertices if necessary
        data = self.cifti.get_fdata()
        if self.full_surface:
            vtx_indices = model.vertex
            n_vtx = vtx_indices.max() + 1
            surf_data = np.zeros([n_samp, n_vtx], dtype=dtype)
            surf_data[:, vtx_indices] = data[:, struct_indices]
        else:
            surf_data = data[:, struct_indices]

        # Convert dtype?
        if dtype is not None:
            surf_data = surf_data.astype(dtype)

        # Squeeze?
        if squeeze_data:
            surf_data = np.squeeze(surf_data)

        # Return
        return surf_data

    def create_new_cifti(self, left_surface_data=None, right_surface_data=None,
                         volume_data=None, *args, **kwargs):
        """
        Create new CIFTI image object from provided datasets, using the
        original image as a template.

        Parameters
        ----------
        [left/right]_surface_data, volume_data : ndarrays
            [nSamples x nGrayordinates] arrays giving data for left and right
            surfaces and volume blocks respectively. Any datasets that are
            omitted will be replaced with zeros.

        *args, **kwargs
            Additional arguments are passed to data2cifti function.

        Returns
        -------
        new_cifti : nibabel.Cifti2Image object
            New Cifti2image object containing provided data
        """

        # We need at least one dataset to be provided
        if all([X is None for X in [left_surface_data, right_surface_data,
                                    volume_data]]):
            raise Exception('At least one of left or right surface or '
                            'volume data must be provided')

        # Get number of samples out of 1st available dataset
        for X in [left_surface_data, right_surface_data, volume_data]:
            if X is not None:
                n_samp = np.atleast_2d(X).shape[0]
                break

        # Get number of grayordinates from axis
        n_grayordinates = self.axis1.size

        # Pre-allocate data array
        data = np.zeros([n_samp, n_grayordinates])

        # Process left surface
        indices, model = self.get_struct_info('left_cortex')
        if left_surface_data is not None:
            if self.full_surface:
                left_surface_data = left_surface_data[..., model.vertex]
        else:
            left_surface_data = np.zeros([n_samp, indices.stop - indices.start])

        # Process right surface
        indices, model = self.get_struct_info('right_cortex')
        if right_surface_data is not None:
            if self.full_surface:
                right_surface_data = right_surface_data[..., model.vertex]
        else:
            right_surface_data = np.zeros([n_samp, indices.stop - indices.start])

        # Concat surface data over hemispheres and add to data array
        surface_data = np.hstack([left_surface_data, right_surface_data])
        surface_mask = self._get_surface_mask()
        data[..., surface_mask] = surface_data

        # Allocate volume data to array
        if volume_data is not None:
            vol_mask = self._get_volume_mask()
            data[..., vol_mask] = volume_data

        # Create CIFTI
        new_cifti = data2cifti(data, template=self.cifti, *args, **kwargs)

        # Return
        return new_cifti

    def uncache(self):
        """
        Uncache data from memory - good idea to call this when done.
        """
        self.cifti.uncache()


class CiftiMasker(object):
    """
    Vaguely modelled on nilearn's NiftiMasker / MultiNiftiMasker. Allows
    loading and remapping of CIFTI data while restricting to a mask.

    Arguments
    ---------
    mask_img : str, Cifti2Image, or CiftiHandler
        Path to input mask (likely a dlabel CIFTI file), long or short name
        of a CIFTI structure (e.g. 'CIFTI_STRUCTURE_LEFT_HIPPOCAMPUS' or
        'left hippocampus'), or Cifti2Image or CiftiHandler object containing
        mask data.

    Methods
    -------
    * fit : Load mask
    * transform() : Load dataset, applying mask
    * transform_multiple : Load multiple datasets, applying mask
    * fit_transform[_multiple] : Fit and transform in one go
    * inverse_transform : Create new CIFTI image from masked data
    * uncache : Clear cache
    """
    def __init__(self, mask_img):
        self.mask_img = mask_img
        self._is_fitted = False

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise Exception('This instance is not fitted yet. '
                            'Call .fit method first.')

    def _resample_to_data(self, dict_, data_handler, block='all'):
        """
        Re-sample full surface and volume arrays to just the gray-ordinates
        that exist in the data CIFTI image. Concat over left/right surf and
        volume blocks, replacing any omitted blocks with zeros.

        dict_ : Dictionary returned by CiftiHandler.get_all_data
        data_handler : CiftiHandler for data image
        block : 'all', 'surface', 'lh', 'rh', or 'volume'
        """
        # Error check
        if not any(X.size > 0 for X in dict_.values()):
            raise ValueError('CIFTI does not contain any data structures')

        # Pre-allocate list for structures
        array = []

        # Get dtype and number of samples from first available block - needed
        # when allocating zeros for missing blocks
        for X in dict_.values():
            if X.size > 0:
                nSamp = X.shape[0]
                dtype = X.dtype
                break

        # Left surface
        model = data_handler._get_struct_info('cortex_left')[1]
        if block in ['lh','surface','all'] and dict_['lh'].size > 0:
            array.append(dict_['lh'][..., model.vertex])
        else:
            nVtx = len(model.vertex)
            array.append(np.zeros([nSamp, nVtx], dtype))

        # Right surface
        model = data_handler._get_struct_info('cortex_right')[1]
        if block in ['rh','surface','all'] and dict_['rh'].size > 0:
            array.append(dict_['rh'][..., model.vertex])
        else:
            nVtx = len(model.vertex)
            array.append(np.zeros([nSamp, nVtx], dtype))

        # Volume
        if block in ['volume','all'] and dict_['volume'].size > 0:
            array.append(dict_['volume'])
        else:
            nVox = data_handler._get_volume_mask().sum()
            array.append(np.zeros([nSamp, nVox], dtype))

        # Concat arrays and return
        return np.hstack(array)

    def _parse_labelID(self, labelID, mapN):
        """
        Get numeric ID from label name, or pass through if already numeric
        """
        if isinstance(labelID, int):
            return labelID
        elif isinstance(labelID, str):
            if not isinstance(self.mask_handler.axis0, nib.cifti2.LabelAxis):
                raise TypeError('String label IDs only supported for dlabel masks')
            matches = [k for k,v in self.mask_handler.axis0.label[mapN].items() \
                       if v[0] == labelID]
            if len(matches) != 1:
                raise ValueError(f"No labels matching ID '{labelID}'")
            return matches[0]
        else:
            raise TypeError('Invalid label ID type')

    def fit(self):
        """
        Load mask
        """
        # Select CIFTI structure, or load from file, or pass through objects
        try:
            self.mask_struct = nib.cifti2.BrainModelAxis \
                                  .to_cifti_brain_structure_name(self.mask_img)
            self._mask_is_cifti_struct = True
        except ValueError:
            if isinstance(self.mask_img, CiftiHandler):
                self.mask_handler = copy.deepcopy(CiftiHandler)
                self.mask_handler.full_surface = True
            elif (isinstance(self.mask_img, str) and os.path.isfile(self.mask_img)) \
            or isinstance(self.mask_img, nib.Cifti2Image):
                self.mask_handler = CiftiHandler(self.mask_img, full_surface=True)
            else:
                raise ValueError('Invalid mask image')

            self.mask_dict = self.mask_handler.get_all_data(dtype=int)
            self._mask_is_cifti_struct = False

        # Return
        self._is_fitted = True
        return self

    def transform(self, img, mask_block='all', labelID=1, mapN=0, dtype=None):
        """
        Load data from CIFTI and apply mask

        Arguments
        ---------
        img : str, Cifti2Image, or CiftiHandler
            Path to CIFTI data file (likely a dscalar or dtseries), or a
            Cifti2Image or CiftiHandler object containing the data.

        mask_block : str {all | lh | rh | surface | volume}
            Which blocks from the CIFTI array to return data from. For example,
            could use to select data from only one hemisphere. Ignored if mask
            is a CIFTI structure. Default is 'all'.

        labelID : int or str
            ID of label to select if mask contains multiple labels. Can be
            the numeric index of the label (technically 0-indexed, though 0
            itself usually denotes unlabelled regions, so the first real label
            will have probably be denoted as 1). If the mask is stored in a
            dlabel file, can also be the name of the label. Ignored if mask is
            a CIFTI structure. Default is 1.

        mapN : int
            Index of map to select if mask contains multiple maps. Ignored if
            mask is a CIFTI structure. Default is 0 (first map).

        dtype : None or valid datatype
            Datatype to cast data to. If None (default), use existing datatype.

        Returns
        -------
        data_array : ndarray
            [nSamples x nGrayOrdinates] array of data values after masking
        """
        # Error check
        self._check_is_fitted()

        # Open handler for data file
        if isinstance(img, CiftiHandler):
            self.data_handler = copy.deepcopy(img)
            self.data_handler.full_surface = True
        elif (isinstance(img, str) and os.path.isfile(img)) \
        or isinstance(img, nib.Cifti2Image):
            self.data_handler = CiftiHandler(img, full_surface=True)
        else:
            raise ValueError('Invalid data image')

        # If mask is a CIFTI structure, index it out of data
        if self._mask_is_cifti_struct:
            slice_ = self.data_handler._get_struct_info(self.mask_struct)[0]
            data_array = self.data_handler.cifti.get_fdata()[..., slice_]
            if dtype is not None:
                data_array = data_array.astype(dtype)

        # Mask not cifti structure - load mask array and apply to data
        else:
            # Extract mask and data arrays for requested structures
            mask_array = self._resample_to_data(
                    self.mask_dict, self.data_handler, mask_block
                    )

            data_array = self._resample_to_data(
                    self.data_handler.get_all_data(dtype), self.data_handler,
                    mask_block
                    )

            # Convert mask array to boolean mask matching requested label
            mask_array = mask_array[mapN] == self._parse_labelID(labelID, mapN)

            # Apply mask to data
            data_array = data_array[..., mask_array]

        # Return
        return data_array

    def transform_multiple(self, imgs, vstack=False, *args, **kwargs):
        """
        Load data from multiple CIFTIs and apply mask

        imgs : list of strs, Cifti2Images, or CiftiHandlers
            List of input images

        vstack : bool
            If True, stack masked data arrays before returning. Default is
            False.

        *args, **kwargs
            Further arguments passed to transform method

        Returns
        -------
        data : ndarray or list of ndarrays
            If vstack is True, then an [(nImgs * nSamples) x nGrayordinates]
            array. If vstack is False, then an nImgs-length list of
            [nSamples x nGrayordinates] arrays.
        """
        data = [self.transform(img, *args, **kwargs) for img in imgs]
        if vstack:
            data = np.vstack(data)
        return data

    def fit_transform(self, *args, **kwargs):
        self.fit()
        return self.transform(*args, **kwargs)

    def fit_transform_multiple(self, *args, **kwargs):
        self.fit()
        return self.transform_multiple(*args, **kwargs)

    def inverse_transform(self, data_array, mask_block='all', labelID=1,
                          mapN=0, dtype=None, template_img=None,
                          return_as_cifti=True, *args, **kwargs):
        """
        "Unmask" data to return to original grayordinates array using provided
        CIFTI image as a template.

        Arguments
        ---------
        data_array : ndarray
            [nGrayordinates, ] 1D array or [nSamples x nGrayordinates] 2D
            array containing masked data.

        mask_block : str {all | lh | rh | surface | volume}
            Which blocks from the CIFTI array to return data from. Should
            match value supplied to forward transform.

        labelID : int or str
            ID or name of label to select if mask contains multiple labels.
            Should match value supplied to forward transform.

        mapN : int
            Index of map to select if mask contains multiple maps. Should match
            value supplied to forward transform.

        dtype : None or valid datatype
            Datatype to cast data to. If None (default), use existing datatype.

        template_img : None, str, Cifti2Image, or CiftiHandler
            Template to base new CIFTI on. If None (default) will use data
            CIFTI handler from most recent forward transform. Otherwise, can
            be path to data CIFTI file, or Cifti2Image or CiftiHandler object
            containing data.

        return_as_cifti : bool
            If True (default), return new Cifti2Image object. Otherwise,
            return numpy array.

        *args, **kwargs
            Additional arguments passed to data2cifti function.

        Returns
        -------
        new_data : Cifti2Image or ndarray
            Data reshaped to full set of grayordinates. Returned as Cifti2Image
            object if return_as_cifti is True, otherwise returned as array.
        """
        # Error check
        self._check_is_fitted()

        # Check dtype
        if dtype is None:
            dtype = data_array.dtype

        # Load template CIFTI
        if template_img is None:
            if self.data_handler is None:
                raise ValueError('Must supply template image or call '
                                 '.transform() method')
            else:
                template_handler = self.data_handler
        elif isinstance(template_img, CiftiHandler):
            template_handler = copy.deepcopy(template_img)
            template_handler.full_surface = True
        elif (isinstance(template_img, str) and os.path.isfile(template_img)) \
        or isinstance(template_img, nib.Cifti2Image):
            template_handler = CiftiHandler(template_img, full_surface=True)
        else:
            raise ValueError('Invalid template image')

        # Ensure data array 2D
        data_array = np.atleast_2d(data_array)

        # Get dimensions and allocate new array
        nSamples = data_array.shape[0]
        nOrds = template_handler.axis1.size
        new_array = np.zeros([nSamples, nOrds], dtype=dtype)

        # If mask is a CIFTI structure, allocate into its indices
        if self._mask_is_cifti_struct:
            slice_ = template_handler._get_struct_info(self.mask_struct)[0]
            new_array[..., slice_] = data_array
        # Mask not CIFTI structure - get mask array and allocate by it
        else:
            mask_array = self._resample_to_data(
                    self.mask_dict, template_handler, mask_block
                    )
            mask_array = mask_array[mapN] == self._parse_labelID(labelID, mapN)
            new_array[..., mask_array] = data_array

        # Return requested output
        if return_as_cifti:
            new_img = data2cifti(new_array, template_handler.cifti, dtype,
                                 *args, **kwargs)
            return new_img
        else:
            return new_array

    def uncache(self):
        """
        Clear mask and data from cache
        """
        self._check_is_fitted()
        if not self._mask_is_cifti_struct:
            self.mask_handler.uncache()
        self.data_handler.uncache()
