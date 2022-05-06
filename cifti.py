#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various tools for handling surface files.

Requires nibabel >= v3.2, which isn't currently installed at YNiC. You might
need to pip install it yourself:
pip3 install --user --upgrade nibabel==3.2
"""

import os, warnings, inspect
import numpy as np
import nibabel as nib

# Quick version check on nibabel
from packaging.version import parse as parse_version
if parse_version(nib.__version__) < parse_version('3.2.0'):
    raise ImportError('Nibabel verison must be >= 3.2.0')


def data2cifti(data, template, cifti_type=None, axis_kwargs={}):
    """
    Construct new Cifti2Image object from data array.

    Arguments
    ---------
    data : ndarray
        [nSamples x nGrayordinates] array of data.

    template : nibabel.Cifti2Image
        Existing CIFTI image object to use as template. Should provide correct
        information for the grayordinates axis, plus any necessary details
        for the NIFTI header and extra information.

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
        if cifti_type is None.

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

    # Create new CIFTI object
    new_cifti = nib.Cifti2Image(dataobj=data, header=header,
                                nifti_header=template.nifti_header,
                                extra=template.extra)
    new_cifti.update_headers()  # fixes nifti_header

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

    exclude_volume_structures : list
        List of volume structure labels to exclude from data. Can specify as
        full names (e.g. 'CIFTI_STRUCTURE_BRAIN_STEM') or short names
        (e.g. 'brain_stem') that can be matched to full names.

    Methods
    -------
    .get_volume_data : Extract volume data
    .get_surface_data : Extract surface data for a given hemisphere
    .get_all_data : Convenience function for extracting surface and volume data
    .create_new_cifti : Create new CIFTI image from provided data

    """
    def __init__(self, img, full_surface=False, exclude_volume_structures=[]):
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
        self.exclude_volume_structures = exclude_volume_structures

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
        Return copy of volume mask, first removing any requested structures
        """
        vol_mask = self.axis1.volume_mask.copy()
        if np.any(vol_mask):  # only check excluded structures if volume exists
            for struct_name in self.exclude_volume_structures:
                indices = self._get_struct_info(struct_name)[0]
                vol_mask[indices] = False
        return vol_mask

    def _get_surface_mask(self):
        """
        Return copy of surface mask
        """
        return self.axis1.surface_mask.copy()

    def get_all_data(self, dtype=None, squeeze_data=True):
        """
        Return full data array from all available surface and volume
        structures.

        Parameters
        ----------
        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True (default), squeeze singleton dimensions from data array.

        Returns
        -------
        data : dict
            Dictionary of [samples x grayordinates] arrays containing data
            values for each structure ('lh' for left surface, 'rh' for right
            surface, and 'volume' for sub-cortical structures)
        """
        data = {
            'lh':self.get_surface_data('lh', dtype, squeeze_data),
            'rh':self.get_surface_data('rh', dtype, squeeze_data),
            'volume':self.get_volume_data(dtype, squeeze_data),
            }
        return data

    def get_volume_data(self, dtype=None, squeeze_data=True):
        """
        Return all data from sub-cortical volume structures.

        Arguments
        ---------
        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True (default), squeeze singleton dimensions from data array.

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

    def get_surface_data(self, hemi, dtype=None, squeeze_data=True):
        """
        Return all data from given hemisphere.

        Arguments
        ---------
        hemi : str { lh | rh }
            Name of hemisphere to load data from.

        dtype : None or valid dtype
            Datatype to cast data to. If None (default), use existing datatype.

        squeeze_data : bool
            If True (default), squeeze singleton dimensions from data array

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
            surfaces and volume structures respectively. Any datasets that are
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
    mask_img : str or CiftiHandler
        Path to input mask (likely a dlabel CIFTI file), or a CiftiHandler
        instance containing the mask data.

    extra_mask_img : str, CiftiHandler, or None
        Only used if .[inverse_]transform() methods are called with invert_mask
        set to True. If supplied, will only invert mask within grayordinates
        included within the extra mask. If None (default), will invert entire
        mask.

    exclude_volume_structures : list
        List of volume structure labels to exclude from data. Can specify as
        full names (e.g. 'CIFTI_STRUCTURE_BRAIN_STEM') or short names
        (e.g. 'brain_stem') that can be matched to full names. Ignored if
        images are supplied as CiftiHandler instances.

    Methods
    -------
    * .fit() : Load mask
    * .transform() : Load dataset, applying mask
    * .transform_multiple() : Load multiple datasets, applying mask
    * .fit_transform[_multiple]() : Fit and transform in one go
    * .inverse_transform() : Create new CIFTI image from masked data
    """
    def __init__(self, mask_img=None, extra_mask_img=None,
                 exclude_volume_structures=[]):
        # Allocate vars to class
        self.mask_img = mask_img
        self.extra_mask_img = extra_mask_img
        self.exclude_volume_structures = exclude_volume_structures

        # Place holders to be filled later
        self.mask_handler = None
        self.mask_dict = None
        self.extra_mask_handler = None
        self.extra_mask_dict = None
        self.data_handler = None

    @staticmethod
    def _parse_struct(struct):
        """
        Parse structure label, returning list in ['lh','rh','volume'] order
        omitting levels as appropriate. Structs should be None, 'all',
        'surface', 'lh', 'rh', or 'volume':
            * If None or 'all' then return ['lh','rh','volume']
            * If 'surface' then return ['lh','rh']
            * If 'lh', 'rh', or 'volume' then return that value in a list
        """
        if struct not in [None, 'all', 'lh', 'rh', 'surface', 'volume']:
            raise ValueError(f'Invalid structure: {struct}')

        if struct in [None, 'all']:
            return ['lh','rh','volume']
        elif struct == 'surface':
            return ['lh','rh']
        else:
            return [struct]

    def _check_is_fitted(self):
        """
        Check that fit method has been called (a mask has been loaded) and
        error if not.
        """
        if self.mask_dict is None:
            raise Exception('This instance is not fitted yet. '
                            'Call .fit method first.')

    def _resample_masks_to_data(self, data_handler, mask_dict,
                                mask_struct=None, data_struct=None):
        """
        Resample masks in provided dict to match full data grayordinates and
        concatenate into array over lh, rh, and volume (in that order)
        """
        # Parse structures
        p_mask_structs = self._parse_struct(mask_struct)
        p_data_structs = self._parse_struct(data_struct)

        # Pre-allocate list for mask structures
        mask_array = []

        # Left surface
        if 'lh' in p_data_structs:
            indices, model = data_handler._get_struct_info('cortex_left')
            if 'lh' in p_mask_structs and mask_dict['lh'].size > 0:
                mask_array.append(mask_dict['lh'][..., model.vertex])
            else:
                mask_array.append(
                        np.zeros(indices.stop - indices.start, dtype=bool)
                        )

        # Right surface
        if 'rh' in p_data_structs:
            indices, model = data_handler._get_struct_info('cortex_right')
            if 'rh' in p_mask_structs and mask_dict['rh'].size > 0:
                mask_array.append(mask_dict['rh'][..., model.vertex])
            else:
                mask_array.append(
                        np.zeros(indices.stop - indices.start, dtype=bool)
                        )

        # Volume
        if 'volume' in p_data_structs:
            if 'volume' in p_mask_structs and mask_dict['volume'].size > 0:
                mask_array.append(mask_dict['volume'])
            else:
                vol_mask = data_handler._get_volume_mask()
                mask_array.append(np.zeros(vol_mask.sum(), dtype=bool))

        # Concat & return
        return np.hstack(mask_array)


    def _get_mask_array(self, data_handler, mask_struct, data_struct,
                        invert_mask):
        """
        Helper function for getting mask array. Resamples to data
        grayordinates, concats to array, and inverts if requested.
        """
        # Resample mask to data grayordinates
        mask_array = self._resample_masks_to_data(
            data_handler, self.mask_dict, mask_struct, data_struct
            )

        # Invert if requested. If extra mask supplied, re-mask the inverted
        # mask by it.
        if invert_mask:
            mask_array = ~mask_array
            if self.extra_mask_dict is not None:
                extra_mask_array = self._resample_masks_to_data(
                    data_handler, self.extra_mask_dict, mask_struct=None,
                    data_struct=data_struct
                    )
                mask_array *= extra_mask_array

        # Return
        return mask_array

    def fit(self):
        """
        Load mask, and extra mask (if provided).
        """
        # Load masks. Get full set of surface vertices - we'll resample to the
        # data vertices later
        if isinstance(self.mask_img, CiftiHandler):
            self.mask_handler = self.mask_img
        elif isinstance(self.mask_img, str) and os.path.isfile(self.mask_img):
            self.mask_handler = CiftiHandler(
                self.mask_img, full_surface=True,
                exclude_volume_structures=self.exclude_volume_structures
                )
        else:
            raise ValueError('Invalid mask image')

        self.mask_dict = self.mask_handler.get_all_data(dtype=bool)
        self.mask_handler.uncache()

        if self.extra_mask_img is not None:
            if isinstance(self.extra_mask_img, CiftiHandler):
                self.extra_mask_handler = self.extra_mask_img
            elif isinstance(self.extra_mask_img, str) \
                 and os.path.isfile(self.extra_mask_img):
                self.extra_mask_handler = CiftiHandler(
                    self.extra_mask_img, full_surface=True,
                    exclude_volume_structures=self.exclude_volume_structures
                    )
            else:
                raise ValueError('Invalid extra mask image')

            self.extra_mask_dict = self.extra_mask_handler.get_all_data(dtype=bool)
            self.extra_mask_handler.uncache()

        return self

    def transform(self, img, mask_struct=None, data_struct=None,
                  invert_mask=False, dtype=np.float32):
        """
        Load data from provided CIFTI file path, and return as
        [nSamples x nGrayOrdinates] array after applying mask.

        Arguments
        ---------
        img : str or CiftiHandler
            Path to CIFTI data file (likely a dscalar or dtseries), or a
            CiftiHandler instance containing the data.

        mask_struct : str { all | lh | rh | surface | volume } or None
            Structure within CIFTI to apply mask to. If 'lh' or 'rh' will
            apply to left or right cortical surfaces. If 'surface' will apply
            to both left and right cortical surfaces. If 'volume' will apply
            to sub-cortical volume. If None (default) or 'all', will apply to
            all available grayordinates.

        data_struct : str { all | lh | rh | surface | volume } or None
            Structure within CIFTI to extract data from. This is mostly useful
            for selecting what structures to load data from when the mask is
            inverted, e.g. setting invert_mask=True, mask_struct='lh', and
            data_struct='surface' will load only surface data outside the left
            hemisphere mask (i.e. excluding surface grayordinates within the
            left hemisphere mask and all volume grayordinates). Note that when
            not inverting the mask, if data and mask structures don't match
            then no data may be returned.

        invert_mask : bool
            If True, return data for grayordinates OUTSIDE mask. Data will
            still be restricted to extra_mask (if supplied).

        dtype : valid datatype
            Datatype to cast data to

        Returns
        -------
        data_array : ndarray
            [nSamples x nGrayordinates] array of data values after masking
        """
        # Error check
        self._check_is_fitted()

        # Open handler for data file and allocate to class
        if isinstance(img, CiftiHandler):
            self.data_handler = img
        elif isinstance(img, str) and os.path.isfile(img):
            self.data_handler = CiftiHandler(
                img, full_surface=False,
                exclude_volume_structures=self.exclude_volume_structures
                )
        else:
            raise ValueError('Invalid data image')

        # Load data from all structures, concat over structures
        data_dict = self.data_handler.get_all_data(dtype)
        data_array = np.hstack(
                [data_dict[S] for S in self._parse_struct(data_struct)]
                )

        # Get mask array
        mask_array = self._get_mask_array(self.data_handler, mask_struct,
                                          data_struct, invert_mask)

        # Apply mask to data
        data_array = data_array[..., mask_array]

        # Uncache data
        self.data_handler.uncache()

        # Return
        return data_array

    def transform_multiple(self, img_paths, vstack=False, *args, **kwargs):
        """
        Convenience function for applying transform to multiple CIFTI files

        Arguments
        ---------
        img_paths : list
            List of paths to CIFTI files or CiftiHandler instances

        vstack : bool
            If True, stack masked data arrays before returning

        *args, **kwargs
            Further arguments passed to .transform method

        Returns
        -------
        data : list or ndarray
            If <vstack> is False, then a list of [nSamples x nGrayordinates]
            arrays of data values after masking for each input. If <vstack> is
            True, then an [(nImgs * nSamples) x nGrayordinates] array.
        """
        self._check_is_fitted()
        data = [self.transform(img, *args, **kwargs) for img in img_paths]
        if vstack:
            data = np.vstack(data)
        return data

    def fit_transform(self, *args, **kwargs):
        self.fit()
        return self.transform(*args, **kwargs)

    def fit_transform_multiple(self, *args, **kwargs):
        self.fit()
        return self.transform_multiple(*args, **kwargs)

    def inverse_transform(self, data_array, mask_struct=None,
                          data_struct=None, invert_mask=False, dtype=None,
                          template_img=None, return_as_cifti=True,
                          *args, **kwargs):
        """
        "Unmask" data to return to original grayordinates array using provided
        CIFTI image as a template.

        Arguments
        ---------
        data_array : numpy array
            [nGrayordinates, ] 1D array or [nSamples x nGrayordinates] 2D
            array containing masked data.

        mask_struct : str { lh | rh | surface | volume } or None
            Structure within CIFTI that mask was applied to. Must match the
            option that was used during initial forward transformation (see
            .transform method).

        data_struct : str { lh | rh | surface | volume } or None
            Structure within CIFTI that data was extracted from. Much match
            the option was used during initial forward transformation (see
            .transform method).

        invert_mask : boolean
            Use inverted version of whole mask. Must match option that was
            used during initial forward transformation of data (see
            .transform method).

        dtype : valid datatype
            Datatype to convert data to. If None (default), use existing type.

        template_img : str, CiftiHandler, or None
            Filepath to template CIFTI, or CiftiHandler instance to be used as
            template. Used to determine grayordinates and extract header
            information. If None (default) can use details from previously
            transformed CIFTI handler (provided that .transform method has
            already been called).

        return_as_cifti : bool
            If True (default), return result as Cifti2Image object. If False,
            return data as numpy array.

        *args, **kwargs
            Additional arguments are passed to data2cifti function if
            return_as_cifti is True, or are ignored otherwise.

        Returns
        -------
        new_data : nibabel.Cifti2Image or ndarray
            Data reshaped to full set of grayordinates. Returned as Cifti2Image
            object if return_as_cifti is True, otherwise returned as array.
        """
        # Error check
        self._check_is_fitted()

        # Load template CIFTI
        if template_img is None:
            if self.data_handler is None:
                raise ValueError('Must supply template image or call '
                                 '.transform() method')
            else:
                template_handler = self.data_handler
        elif isinstance(template_img, CiftiHandler):
            template_handler = template_img
        elif isinstance(template_img, str) and os.path.isfile(template_img):
            template_handler = CiftiHandler(template_img)
        else:
            raise ValueError('Invalid template image')

        # Ensure data array 2D
        data_array = np.atleast_2d(data_array)

        # Calculate final data shape
        n_samples = data_array.shape[0]
        n_grayordinates = template_handler.axis1.size

        # Get mask array
        mask_array = self._get_mask_array(
                template_handler, mask_struct, data_struct=None,
                invert_mask=invert_mask
                )

        # If any volume structures were excluded, we need to add them back in
        if self.exclude_volume_structures:
            surf_mask = template_handler._get_surface_mask()
            vol_mask = template_handler._get_volume_mask()
            idcs = surf_mask | vol_mask
            tmp_mask_array = np.zeros(n_grayordinates, dtype=bool)
            tmp_mask_array[idcs] = mask_array
            mask_array = tmp_mask_array
            del tmp_mask_array

        # If data excludes any general structures, we need to exclude them
        # from the corresponding mask
        if data_struct:
            p_data_structs = self._parse_struct(data_struct)
            if 'lh' not in p_data_structs:
                indices = template_handler._get_struct_info('cortex_left')[0]
                mask_array[indices] = False
            if 'rh' not in p_data_structs:
                indices = template_handler._get_struct_info('cortex_right')[0]
                mask_array[indices] = False
            if 'volume' not in p_data_structs:
                indices = template_handler._get_volume_mask()
                mask_array[indices] = False

        # Resample data to new array
        new_array = np.zeros([n_samples, n_grayordinates], dtype=dtype)
        new_array[..., mask_array] = data_array

        # Return requested output
        if return_as_cifti:
            new_cifti = data2cifti(new_array, template_handler.cifti,
                                   *args, **kwargs)
            return new_cifti
        else:
            return new_array
