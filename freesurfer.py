#!/usr/bin/env python3

"""
Tools for dealing with Freesurfer surfaces
"""

import os, copy
import numpy as np
import nibabel as nib
from nibabel.freesurfer import read_label, read_annot


def load_label(label):
    """
    Load label from file or numpy array
    """
    if isinstance(label, str) and os.path.isfile(label):
        return read_label(label)
    elif isinstance(label, (tuple, list, np.ndarray)):
        return np.array(label)
    else:
        raise TypeError('Label must be valid path to label file or array ' \
                        'of vertices')

def load_data(data, dtype):
    """
    Load data from file, MGHImage object, or numpy array
    """
    if isinstance(data, np.ndarray):
        mgh = None
        data = data.astype(dtype, copy=True).squeeze().T
    else:
        if isinstance(data, str) and os.path.isfile(data):
            mgh = nib.load(data)
        elif isinstance(data, nib.MGHImage):
            mgh = data
        else:
            raise TypeError('Data must be valid filepath, MGHImage ' \
                            'object, or numpy array')
        data = mgh.get_fdata(dtype=dtype).squeeze().T
    return mgh, data


class MultiSurfaceMasker(object):
    """
    Vaguely modelled on nilearn's MultiNiftiMasker, allows loading and
    remapping of surface data via surface mask (stored as .label file). Data
    must use an mgh/mgz format.

    Arguments
    ---------
    mask: str or array-like
        Path to input mask .label file, or array of vertices within mask
    cortex_mask : str or array-like
        Only used if .[inverse_]transform() methods are called with invert_mask
        set to True. If supplied, will cause this to take only vertices from
        cortex_mask that don't overlap with mask. If omitted, will take
        vertices from whole cortex that don't overlap with mask.
    """
    def __init__(self, mask, cortex_mask=None):
        # Allocate vars to class
        self.mask = mask
        self.cortex_mask = cortex_mask

        # Place holders to be filled later
        self._is_fitted = False
        self.mask_idcs = None
        self.cortex_mask_idcs = None
        self.template_mgh = None

    def fit(self):
        """
        Load the mask.

        Returns
        -------
        self
            Class instance
        """
        self.mask_idcs = load_label(self.mask)
        if self.cortex_mask is not None:
            self.cortex_mask_idcs = load_label(self.cortex_mask)
        self._is_fitted = True
        return self

    def transform(self, imgs, invert_mask=False, vstack=False, dtype=np.float64):
        """
        Apply masks to data. Will also grab affine and header info out of first
        file and store in class (unless inputs are numpy arrays)

        Arguments
        ---------
        imgs : str, MGHImage object, ndarray, or list thereof
            Input MGH surface file(s). Can be specify as filepath to file,
            an MGHImage object, or a numpy array containing the data. If
            a numpy array, should be of shape [nVertices x 1 x 1 x nSamples] or
            of shape [nVertices x nSamples].

        invert_mask : bool
            If True, load from vertices OUTSIDE of mask instead
            (default = False)

        vstack : bool
            If True, concatenate data arrays over input files (default = False)

        dtype : valid datatype
            Datatype to cast values to (default = float64)

        Returns
        -------
        data : list or numpy array
            If vstack == False, then a list containing a set of
            [nSamples x nVertices] arrays for each input image.
            If vstack == True, then an [(nImgs * nSamples) x nVertices] array.
        """
        if not self._is_fitted:
            raise Exception('Must call .fit() method first')

        if not isinstance(imgs, (tuple, list)):
            imgs = [imgs]

        data = []
        for img in imgs:
            mgh, img_data = load_data(img, dtype)

            if (mgh is not None) and (self.template_mgh is None):
                self.template_mgh = copy.deepcopy(mgh)
                self.template_mgh.uncache()  # save space (don't need dataobj)

            if invert_mask:
                if self.cortex_mask_idcs is None:  # whole cortex
                    nVertices = img_data.shape[-1]
                    idcs = np.setdiff1d(np.arange(nVertices), self.mask_idcs)
                else:  # just within cortex mask
                    idcs = np.setdiff1d(self.cortex_mask_idcs, self.mask_idcs)
            else:
                idcs = self.mask_idcs

            data.append(img_data[...,idcs])

        if vstack:
            data = np.vstack(data)
        elif len(data) == 1:
            data = data[0]

        return data

    def fit_transform(self, *args, **kwargs):
        self.fit()
        return self.transform(*args, **kwargs)

    def inverse_transform(self, array, template_mgh=None, invert_mask=False,
                          dtype=np.float64, return_as_mgh=True):
        """
        Inverse transform data back to surface space.

        Arguments
        ---------
        array : numpy array
            Input data - [nVertices, ] 1D or [nSamples x nVertices] 2D array

        template_mgh : nibabel.MGHImage object or None
            Template MGH object to extract header and affine information from.
            If None, can use one stored from previous call to .transform method
            (assuming it has previously been called and inputs were not numpy
            arrays).

        invert_mask : boolean
            Use inverted version of whole mask (default = False)

        dtype : valid datatype
            Datatype to cast data to (default = float64)

        return_as_mgh: bool
            If True (default), return result as MGHImage object. If False,
            return data as numpy array.

        Returns
        -------
        new_data : nibabel.MGHImage or ndarray
            Data reshaped to full set of surface vertices. Returned as MGHImage
            object if return_as_mgh is True, otherwise returned as array.
        """
        # Get template info
        if template_mgh is None:
            if self.template_mgh is None:
                raise Exception('Must provide template MGH or call '
                                '.transform() method first')
            else:
                template_mgh = self.template_mgh

        affine = template_mgh.affine
        header = template_mgh.header
        extra = template_mgh.extra

        # Get total number of surface vertices
        nVertices = header['dims'][0]

        # Grab mask indices
        if invert_mask:
            if self.cortex_mask_idcs is None:  # whole cortex
                idcs = np.setdiff1d(np.arange(nVertices), self.mask_idcs)
            else:  # just within cortex mask
                idcs = np.setdiff1d(self.cortex_mask_idcs, self.mask_idcs)
        else:
            idcs = self.mask_idcs

        # Allocate new data array and fill in
        if array.ndim == 1:
            new_array = np.zeros(nVertices, dtype=dtype)
        else:
            nSamples = array.shape[0]
            new_array = np.zeros([nVertices, nSamples], dtype=dtype)
        new_array[idcs, ...] = array.T

        # Reshape to 3D/4D array (freesurfer seems to use this?)
        new_array = new_array[:, np.newaxis, np.newaxis, ...]

        # Return requested output
        if return_as_mgh:
            mgh = nib.MGHImage(new_array, affine=affine, header=header,
                               extra=extra)
            return mgh
        else:
            return new_array


class MultiSurfaceParcellation(object):
    """
    Vaguely modelled on MultiNiftiMasker, allows loading and remapping of
    surface data via parcellation (stored in .annot file), averaging over
    vertices within each parcel. Inputs must be in mgh/mgz format.

    Arguments
    ---------
    annot_path : str
        Path to input annotation file
    invalid_label_IDs : list of ints or strs
        List of label values or names considered invalid - corresponding
        parcels will be ignored.
    """
    def __init__(self, annot_path, invalid_label_IDs=None):
        # Allocate vars to class
        self.annot_path = annot_path
        self.invalid_label_IDs = invalid_label_IDs

        if (self.invalid_label_IDs is not None) and \
        (not isinstance(self.invalid_label_IDs, (tuple, list))):
            self.invalid_label_IDs = [self.invalid_label_IDs]

        # Placeholders
        self._is_fitted = False
        self.template_mgh = None

    def fit(self):
        """
        Load the annotation.

        Returns
        -------
        self
            Class instance
        """
        # Load annotation
        self.annot_labels, _, self.annot_names = read_annot(self.annot_path)

        # Work out unique labels, accounting for any labels to be removed
        if self.invalid_label_IDs is not None:
            isValid = None
            for ID in self.invalid_label_IDs:
                if isinstance(ID, (str, bytes)):
                    if isinstance(ID, str):
                        ID = ID.encode()
                    ID = self.annot_names.index(ID)

                _isValid = self.annot_labels != ID
                if isValid is None:
                    isValid = _isValid
                else:
                    isValid &= _isValid

            self.valid_indices = np.nonzero(isValid)[0]
            self.valid_labels = self.annot_labels[self.valid_indices]

            self.unique_labels, self.inverse_annot_indices = \
                np.unique(self.valid_labels, return_inverse=True)

        else:
            self.unique_labels, self.inverse_annot_indices = \
                np.unique(self.annot_labels, return_inverse=True)

        # Update internal fields
        self.nParcels = len(self.unique_labels)
        self._is_fitted = True

        # Return
        return self

    def transform(self, imgs, vstack=False, dtype=np.float64):
        """
        Load data, averaging over vertices within parcels.

        Arguments
        ---------
        imgs : str, MGHImage object, ndarray, or list thereof
            Input MGH surface file(s). Can be specify as filepath to file,
            an MGHImage object, or a numpy array containing the data. If
            a numpy array, should be of shape [nVertices x 1 x 1 x nSamples] or
            of shape [nVertices x nSamples].

        vstack : bool
            If True, concatenate data arrays over input files (default = False)

        dtype : valid datatype
            Datatype to cast values to (default = float64)

        Returns
        -------
        data : list or ndarray
            If vstack == False, then a list containing a set of
            [nSamples x nParcels] arrays for each input image.
            If vstack == True, then an [(nImgs * nSamples) x nParcels] array.
        """
        # Check class is fitted
        if not self._is_fitted:
            raise Exception('Must call .fit() method first')

        # Ensure images is a list
        if not isinstance(imgs, (tuple, list)):
            imgs = [imgs]

        # Load images, split by parcel and average over vertices within them
        data = []
        for img in imgs:
            mgh, img_data = load_data(img, dtype)
            nSamples = img_data.shape[0] if img_data.ndim > 1 else 1

            if (mgh is not None) and (self.template_mgh is None):
                self.template_mgh = copy.deepcopy(mgh)
                self.template_mgh.uncache()  # save space (don't need dataobj)

            parcel_data = np.empty([nSamples, self.nParcels], dtype=dtype)
            for i, label in enumerate(self.unique_labels):
                idcs = np.nonzero(self.annot_labels == label)[0]
                parcel_data[:, i] = img_data[..., idcs].mean(axis=-1)

            data.append(parcel_data.squeeze())

        # Concat images?
        if vstack:
            data = np.vstack(data)
        elif len(data) == 1:
            data = data[0]

        # Return
        return data

    def fit_transform(self, *args, **kwargs):
        self.fit()
        return self.transform(*args, **kwargs)

    def inverse_transform(self, array, template_mgh=None, dtype=np.float64,
                          return_as_mgh=True):
        """
        Inverse transform data back to surface space.

        Arguments
        ---------
        array : numpy array
            Input data - [nParcels, ] 1D or [nSamples x nParcels] 2D array

        template_mgh : nibabel.MGHImage object or None
            Template MGH object to extract header and affine information from.
            If None, can use one stored from previous call to .transform method
            (assuming it has previously been called and inputs were not numpy
            arrays).

        dtype : valid datatype
            Datatype to cast data to (default = float654)

        return_as_mgh: bool
            If True (default), return result as MGHImage object. If False,
            return data as numpy array.

        Returns
        -------
        new_data : nibabel.MGHImage or ndarray
            Data reshaped to full set of surface vertices. Returned as MGHImage
            object if return_as_mgh is True, otherwise returned as array.
        """
        # Get total number of surface vertices
        nVertices = len(self.annot_labels)

        # Inverse transform array
        new_array = array[..., self.inverse_annot_indices].T

        if self.invalid_label_IDs is not None:
            tmp = new_array.copy()
            if array.ndim == 1:
                new_array = np.full(nVertices, np.nan)
            else:
                nSamples = array.shape[0]
                new_array = np.full([nVertices, nSamples], np.nan)
            new_array[self.valid_indices, ...] = tmp
            del tmp

        new_array = new_array[:, np.newaxis, np.newaxis, ...]

        # Return requested output
        if return_as_mgh:
            # Get info from provided or previously loaded template
            if template_mgh is None:
                if self.template_mgh is None:
                    raise Exception('Must provide template MGH or call '
                                    '.transform() method first')
                else:
                    template_mgh = self.template_mgh

            affine = template_mgh.affine
            header = template_mgh.header
            extra = template_mgh.extra

            # Create new mgh object, return
            mgh = nib.MGHImage(new_array, affine=affine, header=header,
                               extra=extra)
            return mgh

        else:
            # Return array
            return new_array
