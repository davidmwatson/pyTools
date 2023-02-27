#!/usr/bin/env python3
"""
Script uses a clustering algorithm to generate masks comprising a specified
number of spatially contiguous voxels clustered around a given seed point.

Arguments
---------

-i|--input (required)
    Path to input 3D statistical NIFTI volume (probably a zstat)

-n|--nvox (required)
    Desired number of voxels in cluster

-c|--seed-coord (required)
    Space delimited list of 3 values giving x, y, and z voxel (not mm)
    co-ordinates of seed voxel. This seed should fall within the desired
    cluster - it's a good idea to place it on or near the peak voxel within
    the region (it doesn't have to be exact).

-o|--outfile (required)
    Path to output NIFTI file to write mask to. Will append a ".nii.gz"
    extension if none provided.

-m|--initial-mask (optional)
    Path to mask NIFTI volume. If provided, will apply this mask to the data
    before performing the clustering - can be useful for preventing the
    cluster from growing into undesired regions (e.g. PPA bleeding into RSC).

--start-thr (optional)
    Space delimited list of one or more thresholds to use as starting points
    for optimisation. If floats, will use those thresholds directly. Can also
    specify as a percentage (e.g. 50%), which will choose the corresponding
    percentage value between the lower bound threshold (or zero if lower bound
    is disabled) and the value at the seed voxel. If more than one value
    provided, will run optimisation starting from each one, then select the
    solution yielding the lowest error. The default is 25%, 50%, and 75%.

--lower-bound (optional)
--upper-bound (optional)
    Lower and upper bounds for optimisation. If the optimised threshold falls
    outside this range, it will be clipped to this range and a warning will be
    raised. Either bound can be set to inf or nan to disable its use. The
    default is a lower bound of 1.64 (z-score giving one-tailed p < 0.05),
    and the upper bound is disabled.

Example usage
-------------
Define 100 voxel cluster from zstat around seed at [10,20,30]

> python3 defineROICluster.py -i  myZstat.nii.gz -n 100 -c 10 20 30

As above, but also apply initial mask and increase lower bound to Z = 3.1

> python3 defineROICluster.py -i  myZstat.nii.gz -n 100 -c 10 20 30 \\
>     -o myClusterMask.nii.gz -m myInitialMask.nii.gz --lower-bound 3.1

TODOs
-----
Newer versions of scipy allow specifying bounds as arguments to minimisation
function when using Nelder-Mead solver - this should be preferable to clipping
the values as we do now. We can currently specify bounds with other solvers
(e.g. L-BFGS-B), but these don't seem to work well here (maybe some difficulty
in calculating derivatives?). Might be worth updating usage once YNiC updates
scipy.

"""

import os, sys, warnings, argparse
import nibabel as nib
import numpy as np
import scipy.ndimage
from scipy.optimize import minimize


### Custom function definitions ###

def cluster(thr, vol, seed_coord):
    """
    Thesholds volume, then forms voxel cluster around seed voxel at given
    x,y,z co-ordinates.

    Arguments
    ---------
    thr : float
        Threshold to apply
    vol : 3D numpy array
        Statistical volume to be thresholded
    seed_coord : 3-item tuple of ints
        Co-ordinates of seed voxel, falling within desired cluster.

    Returns
    -------
    cluster_mask : 3D numpy array
        Boolean mask array indicating voxels falling within identified cluster
    cluster_size : int
        Number of voxels within cluster
    """
    # Threshold
    thr_vol = vol.copy()
    thr_vol[thr_vol < thr] = 0

    # Label
    labelled_array, n_features = scipy.ndimage.label(thr_vol)
    cluster_mask = labelled_array == labelled_array[tuple(seed_coord)]
    cluster_size = cluster_mask.sum()

    # Return
    return cluster_mask, cluster_size

def cost_func(thr, vol, seed_coord, target_nvox):
    """
    Cost function to be optimised. Applies clustering, then returns error
    between actual and target cluster size.

    Arguments
    ---------
    thr, vol, seed_coord
        All as per cluster function
    target_nvox : int
        Target cluster size

    Returns
    -------
    err : float
        Squared error between actual and target cluster size
    """
    # Cluster
    cluster_mask, cluster_size = cluster(thr, vol, seed_coord)

    # Return squared error of cluster size
    return float(cluster_size - target_nvox)**2


### Parse arguments ###

# Set up parser
class CustomFormatter(argparse.RawTextHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                                 description=__doc__)

parser.add_argument('-i', '--infile', required=True,
                    help='Path to input 3D statistical volume')
parser.add_argument('-n', '--nvox', required=True, type=int,
                    help='Desired cluster size')
parser.add_argument('-c', '--seed-coord', required=True, type=int, nargs=3,
                    help='x y z voxel (not mm) co-ordinates of seed voxel')
parser.add_argument('-o', '--outfile', required=True,
                    help='Path to desired output file')
parser.add_argument('-m', '--initial-mask',
                    help='Path to mask to apply before clustering')
parser.add_argument('--start-thr', nargs='+', default=['25%','50%','75%'],
                    help='Starting threshold(s) for optimisation')
parser.add_argument('--lower-bound', type=float, default=1.64,
                    help='Lower-bound threshold; set to inf or nan to disable')
parser.add_argument('--upper-bound', type=float, default='nan',
                    help='Upper-bound threshold; set to inf or nan to disable')

# If no arguments given, print help and exit
if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

# Parse args
args = parser.parse_args()
infile = args.infile
nvox = args.nvox
seed_coord = args.seed_coord
outfile = args.outfile
initial_mask = args.initial_mask
start_thresholds = args.start_thr
lower_bound = args.lower_bound
upper_bound = args.upper_bound

# Append outfile extension if none given
if '.' not in outfile:
    outfile += '.nii.gz'

# Treat inf or nan thresholds as None
if np.isinf(lower_bound) or np.isnan(lower_bound):
    lower_bound = None
if np.isinf(upper_bound) or np.isnan(upper_bound):
    upper_bound = None


### Begin main script ###

# Load infile
img = nib.load(infile)
vol = img.get_fdata()
hdr = img.header
affine = img.affine

# Apply mask if provided
if initial_mask is not None:
    maskvol = nib.load(initial_mask).get_fdata().astype(bool)
    vol *= maskvol

# Check lower bound appropriate for seed
if (lower_bound is not None) and (vol[tuple(seed_coord)] < lower_bound):
    raise ValueError('Lower bound cannot exceed value at seed voxel')

# Loop starting parameters
all_thr = []
all_errs = []
for x0 in start_thresholds:
    # If value specified as percentage, calcualate value from data range.
    # Otherwise, just convert to float.
    if x0.endswith('%'):
        prop = float(x0.rstrip('%')) / 100
        max_ = vol[tuple(seed_coord)]
        min_ = lower_bound if lower_bound is not None else 0
        x0 = min_ + prop * (max_ - min_)
    else:
        x0 = float(x0)

    # Run optimisation
    optimres = minimize(cost_func, x0=[x0], args=(vol, seed_coord, nvox),
                        method='Nelder-Mead')

    # Extract optimised threshold and error values, append to array
    all_thr.append(optimres.x[0])
    all_errs.append(optimres.fun)

# Choose threshold with lowest err
best_idx = np.argmin(all_errs)
opt_thr = all_thr[best_idx]

# Clip thresh to lower/upper bound if necessary
# TODO - try replacing this with bounds in minimize call once scipy updated
if (lower_bound is not None) and (opt_thr < lower_bound):
    warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to lower bound ({lower_bound:.2f})')
    opt_thr = lower_bound
elif (upper_bound is not None) and (opt_thr > upper_bound):
    warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to upper bound ({upper_bound:.2f})')
    opt_thr = upper_bound

# Run labelling algorithm once more with selected thresh to get final cluster
cluster_mask, cluster_size = cluster(opt_thr, vol, seed_coord)

# Map back to volume and save out
hdr['cal_min'] = 0
hdr['cal_max'] = 1
nib.Nifti1Image(cluster_mask, affine=affine, header=hdr).to_filename(outfile)

# Print results out
print('{} : Requested_Size={:d} Actual_Size={:d} Thr={:f}' \
      .format(os.path.basename(outfile), nvox, cluster_size, opt_thr))
