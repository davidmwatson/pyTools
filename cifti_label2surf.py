#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelled on FSL's label2surf command, allows transforming CIFTI or GIFTI label
files to FSL-compatible GIFTI files combining surface and label structures.
These are, for example, suitable for input to probtrackx.

Arguments
---------
-s|--surf
    Path to input surface GIFTI (probably white, midthickness, or pial surface)
-o|--out
    Path to output surface+label GIFTI
-l|--label
    Path to CIFTI/GIFTI label file (can be .dlabel.nii, .dscalar.nii,
    .label.gii, or .shape.gii). Can specify flag more than once to supply
    multiple label files - in this case, the script will take the union over
    all labels.
--hemi
    Surface hemisphere. Choose from L, R, lh, or rh. Required for CIFTI labels,
    but ignored for GIFTI labels.
--label-ids
    Name (for CIFTI dlabels only) or numeric value of specific label to use
    if multiple labels are encoded in same file. If multiple values given,
    will take union of all corresponding labels. If omitted (default), take
    union of all labels. Can also specify flag multiple times to apply
    different selection for each input label.
-m|--map
    Index of map/darray to select within CIFTI/GIFTI file (zero-indexed).
    The default is 0. Can also specify flag multiple times to apply different
    index to each input label.
-h|--help
    Print help and exit

"""

import sys, argparse
import numpy as np
import nibabel as nib


### Custom funcs ###

def get_struct_info(struct, brain_models_axis):
    for this_struct, slice_, model in brain_models_axis.iter_structures():
        if this_struct == struct:
            return slice_, model
    raise Exception(f"No information for structure '{struct}'")


### Parse args ###

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, usage=__doc__
        )

parser.add_argument('-s', '--surf', required=True, help='Input surface')
parser.add_argument('-o', '--out', required=True, help='Output surface')
parser.add_argument('-l', '--label', required=True, action='append',
                    help='CIFTI label file(s)')
parser.add_argument('--hemi', choices=['L','R','lh','rh'],
                    help='Hemisphere to extract label from')
parser.add_argument('--label-ids', action='append', nargs='+',
                    help='Numeric ID or name of label')
parser.add_argument('-m', '--map', action='append', type=int,
                    help='Index of map to extract label from')

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit()

args = parser.parse_args()
surf_file = args.surf
outfile = args.out
label_files = args.label
hemi = args.hemi
label_ids = args.label_ids
mapNs = args.map

if hemi is None:
    hemi_struct = None
elif hemi in ['L','lh']:
    hemi_struct = 'CIFTI_STRUCTURE_CORTEX_LEFT'
elif hemi in ['R','rh']:
    hemi_struct = 'CIFTI_STRUCTURE_CORTEX_RIGHT'

if label_ids is None:
    label_ids = [None] * len(label_files)
elif len(label_ids) == 1:
    label_ids *= len(label_files)

if mapNs is None:
    mapNs = [0] * len(label_files)
else:
    mapNs *= len(label_files)


### Begin ###

# Read surface
surf_gii = nib.load(surf_file)

coords_darray = surf_gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
if len(coords_darray) != 1:
    raise ValueError('Surface must contain exactly one pointset array')
coords_darray = coords_darray[0]
nVtcs_surf = coords_darray.dims[0]

triangle_darray = surf_gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
if len(triangle_darray) != 1:
    raise ValueError('Surface must contain exactly one triangle array')
triangle_darray = triangle_darray[0]
triangle_darray.coordsys = None  # only valid for pointset arrays

# Pre-allocate array for padded labels
pad_label_data = np.zeros(nVtcs_surf, dtype=bool)

# Loop labels
for i, label_file in enumerate(label_files):
    these_label_ids = label_ids[i]
    mapN = mapNs[i]

    # Open handle to label
    label_img = nib.load(label_file)

    # Handle CIFTI labels
    if isinstance(label_img, nib.Cifti2Image):
        # Need hemisphere for CIFTIs
        if hemi is None:
            raise ValueError('Must specify hemisphere for CIFTI labels')

        # Read label & data
        label_data = label_img.get_fdata()[mapN]

        # Convert label ID to numeric if necessary
        if these_label_ids is not None:
            for j, ID in enumerate(these_label_ids):
                try:
                    these_label_ids[j] = int(ID)
                except ValueError:
                    ax0 = label_img.header.get_axis(0)
                    if not isinstance(ax0, nib.cifti2.LabelAxis):
                        raise ValueError(
                                'Label ID names are only supported for ' \
                                'CIFTI dlabels. Use integer value instead.'
                                )
                    matches = [k for k,v in ax0.label[mapN].items() if v[0] == ID]
                    if len(matches) != 1:
                        raise ValueError(f"No matches in {label_file} for label '{ID}'")
                    these_label_ids[j] = matches[0]

        # Get brain models info for hemisphere
        ax1 = label_img.header.get_axis(1)
        slice_, model = get_struct_info(hemi_struct, ax1)
        nVtcs_label = ax1.nvertices[hemi_struct]
        if nVtcs_surf != nVtcs_label:
            raise Exception(('Number of vertices in label ({}) does not ' \
                             + 'match number of vertices on surface ({})') \
                            .format(nVtcs_label, nVtcs_surf))

        # Pad label to full set of vertices, add to array
        if these_label_ids is None:
            isLabel = label_data[slice_].astype(bool)
        else:
            isLabel = np.isin(label_data[slice_], these_label_ids)
        pad_label_data[model.vertex] |= isLabel

    # Handle GIFTI labels
    elif isinstance(label_img, nib.GiftiImage):
        # Get label data
        label_data = label_img.darrays[mapN].data
        nVtcs_label = len(label_data)
        if nVtcs_surf != nVtcs_label:
            raise Exception(('Number of vertices in label ({}) does not ' \
                             + 'match number of vertices on surface ({})') \
                            .format(nVtcs_label, nVtcs_surf))

        # Check label IDs
        for j, ID in enumerate(these_label_ids):
            try:
                these_label_ids[j] = int(ID)
            except ValueError:
                raise ValueError(
                        'Label ID names are only supported for ' \
                        'CIFTI dlabels. Use integer value instead.'
                        )

        # Add to array
        pad_label_data |= np.isin(label_data[slice_], these_label_ids)

    else:
        raise TypeError('Label must be CIFTI or GIFTI')

# Convert to darray
label_darray = nib.gifti.GiftiDataArray(
        pad_label_data.astype(np.float32), intent='NIFTI_INTENT_SHAPE',
        datatype='NIFTI_TYPE_FLOAT32'
        )
label_darray.coordsys = None # only valid for pointset arrays

# Create new GIFTI & save
new_gii = nib.GiftiImage(
        header=surf_gii.header, extra=surf_gii.extra, meta=surf_gii.meta,
        darrays=[coords_darray, triangle_darray, label_darray]
        )
new_gii.to_filename(outfile)
