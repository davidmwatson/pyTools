#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for dealing with Nifti volumes
"""

import os
import numpy as np
import nibabel as nib

# Workaround for pickling error affecting nibabel, caused by bug in older
# versions of indexed_gzip
# https://github.com/pauldmccarthy/indexed_gzip/issues/28
# https://github.com/nipy/nibabel/issues/969#issuecomment-729206375
try:
    import indexed_gzip
except ImportError:
    pass
else:
    from packaging.version import parse as parse_version
    if parse_version(indexed_gzip.__version__) < parse_version('1.1.0'):
        import gzip
        nib.openers.HAVE_INDEXED_GZIP = False
        nib.openers.IndexedGzipFile = gzip.GzipFile


class QuickMasker(object):
    """
    Simple class for loading a mask and (inverse) transforming data through it.
    Loosely based on nilearn's MultiNiftiMasker, but without all the bells and
    whistles so it runs faster.

    Arguments
    ---------
    mask : str or Nifti1Image object
        Mask image. Can be a path to a NIFTI file or a nibabel Nifti1Image
        object.

    mask2 : str, NiftiImage object, or None
        Only used if .[inverse_]transform() methods are called with invert_mask
        set to True. If supplied, will cause this to take only voxels from
        this secondary mask that don't overlap with the primary mask2. If
        omitted, will take voxels from whole volume outside the primary mask.
    """
    def __init__(self, mask, mask2=None):
        self.mask = mask
        self.mask2 = mask2
        self._is_fitted = False

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise Exception('Must call .fit() method first')

    @staticmethod
    def _load_mask(mask):
        if isinstance(mask, str) and os.path.isfile(mask):
            mask_img = nib.load(mask)
        elif isinstance(mask, nib.Nifti1Image):
            mask_img = mask
        else:
            raise TypeError('Mask must be valid filepath or Nifti1Image object')

        mask_array = mask_img.get_fdata().astype(bool)
        mask_img.uncache()

        return mask_img, mask_array

    @staticmethod
    def _load_data(img, dtype):
        if isinstance(img, str) and os.path.isfile(img):
            data = nib.load(img).get_fdata(dtype=dtype)
        elif isinstance(img, nib.Nifti1Image):
            data = img.get_fdata(dtype=dtype)
        elif isinstance(img, np.ndarray):
            data = img.astype(dtype)
        else:
            raise TypeError('img must be valid filepath, Nifti1Image object, '
                            'or numpy array')
        return data

    def fit(self):
        """
        Load mask image
        """
        # Load primary mask
        self.mask_img, self.mask_array = self._load_mask(self.mask)

        # Load secondary mask?
        if self.mask2 is not None:
            _, self.mask_array2 = self._load_mask(self.mask2)
        else:
            self.mask_array2 = None

        # Finish up and return
        self._is_fitted = True
        return self

    def transform(self, imgs, invert_mask=False, vstack=False, dtype=np.float64):
        """
        Load data from imgfile and apply mask.

        Arguments
        ---------
        imgs : str, Nifti1Image object, ndarray, or list thereof
            Input data. Can be path(s) to a NIFTI file, nibabel Nifti1Image
            object(s), or 3/4D numpy array(s) containing data values.

        invert_mask : bool
            If True, load from vertices OUTSIDE of mask instead
            (default = False)

        vstack : bool
            If True, concatenate data arrays over input files (default = False)

        dtype : valid data type
            Type to cast data to (default is float64).

        Returns
        -------
        data : 1D or 2D ndarray
            Masked data, provided as an [nVoxels] 1D array if input is 3D,
            or an [nSamples x nVoxels] 2D array if input is 4D. If multiple
            inputs are provided, will be a list of each result if vstack is
            False, or an array concatenating the results over the samples
            axis if vstack is True.
        """
        # Setup
        self._check_is_fitted()

        if not isinstance(imgs, (tuple, list)):
            imgs = [imgs]

        if invert_mask:
            mask_array = ~self.mask_array
            if self.mask_array2 is not None:
                mask_array = mask_array & self.mask_array2
        else:
            mask_array = self.mask_array

        # Load data for each image and apply mask
        data = [self._load_data(img, dtype)[mask_array].T for img in imgs]

        # Stack data over images?
        if vstack:
            data = np.vstack(data)
        elif len(data) == 1:
            data = data[0]

        # Return
        return data

    def fit_transform(self, *args, **kwargs):
        self.fit()
        return self.transform(*args, **kwargs)

    def inverse_transform(self, data, invert_mask=False, dtype=np.float32,
                          return_as_nii=True):
        """
        "Unmask" data array

        Arguments
        ---------
        data : 1D or 2D ndarray
            [nVoxels] 1D or [nSamples x nVoxels] 2D nNumpy array containing
            data. Unmasked array will be 3D if data is 1D, or 4D if data is 2D.

        invert_mask : boolean
            Use inverted version of primary mask (default = False)

        dtype : valid data type
            Type to cast data to (default is float32).

        return_as_nii : bool
            If True (default), return as Nifti1Image object. If False, return
            numpy array.

        Returns
        -------
        new_img : Nifti1Image or ndarray
            Unmasked data in requested format.
        """
        # Setup
        self._check_is_fitted()

        if invert_mask:
            mask_array = ~self.mask_array
            if self.mask_array2 is not None:
                mask_array = mask_array & self.mask_array2
        else:
            mask_array = self.mask_array

        # Allocate new array and populate mask region with data
        dims = list(self.mask_img.shape)
        if data.ndim == 2:
            dims.append(data.shape[0])
        inv_data = np.zeros(dims, dtype=dtype)
        inv_data[mask_array] = data.T

        # Convert to NiftiImage if requested, and return
        if return_as_nii:
            new_img = nib.Nifti1Image(
                    inv_data, affine=self.mask_img.affine,
                    header=self.mask_img.header, extra=self.mask_img.extra
                    )
            return new_img
        else:
            return inv_data
