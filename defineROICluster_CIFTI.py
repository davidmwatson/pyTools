#!/usr/bin/env python3
#$ -j y
#$ -o /scratch/$HOME/logs/
#$ -N ROI_cluster
#$ -cwd

"""
Special version of clustering script for working with CIFTI files. Currently
only works for surface ROIs; to generate a volume ROI, export the volume
structures as a separate NIFTI and use normal clustering script.

Arguments
---------

-i|--infile (required)
    Path to input statistical CIFTI file (should be dscalar)

-o|--outfile (required)
    Path to output ROI CIFTI file (should be dscalar)

-a|--area (required)
    Desired target area in mm^2

-c|--seed-coord (required)
    Location of seed co-ordinate. Can either be specified as a single integer
    giving vertex number directly, or as 3 floats giving x, y, and z mm
    co-ordinates. Either co-ordinate can be obtained from the Connectome
    Workbench viewer by clicking at the desired surface location and reading
    the co-ordinate from the Info panel. If using x,y,z mm co-ordinates, these
    should be selected from the same surface that the cluster will be defined
    on (e.g. midthickness).

--hemi (required)
    Which hemisphere the seed is in - should be 'lh' or 'rh'

--use-group-surfaces
    If specified, use S1200 group average midthickness surfaces and vertex area
    metrics (paths are hard coded in script). Any specified surface and vertex
    area metric files will be ignored.

--[left/right]-surf
    Paths to left and right surface anatomy files (should be surf.gii GIFTIs).
    Can be any surface, but the midthickness surface is recommended. Ignored
    if --group-surfaces flag is specified, but required otherwise.

--[left/right]-area
    Paths to left and right vertex area metric files (should be shape.gii
    GIFTIs). These options are useful if you want to use a different area
    metric to the one provided by the surface itself (e.g. if using a group
    average surface). These should match the provided surfaces (e.g. they
    should give the midthickness vertex areas if using the midthickness
    surface). If not provided, areas will be calculated from specified
    surfaces. Ignored if --group-surfaces flag is specified.

--initial-roi
    Path to ROI CIFTI file (should be dscalar). If provided, will use this to
    mask data before applying clustering - this can be useful for constraining
    the cluster definition.

--start-thr
    Starting threshold for optimisation, specified as either a float giving the
    actual value (in units of the input statistical image) or as a percentage
    (e.g. 50%%). If a percentage, the script will set the starting value as
    this percentage of the difference between the lower bound (or zero if not
    provided) and the value at the seed vertex. The default is 50%%.

--[lower/upper]-bound
    Lower and upper bounds for optimisation. If the optimised threshold falls
    outside this range, it will be clipped to this range and a warning will be
    raised. Either bound can be set to inf or nan to disable its use. The
    default is a lower bound of 1.64 (z-score giving one-tailed p < 0.05),
    and the upper bound is disabled.

Example usage
-------------
Define a 500 mm^2 ROI using a seed in the right FFA on the group average
midthickness surfaces. Use manually defined initial ROI to constrain cluster.

> python3 defineROICluster_CIFTI.py \\
>     -i /path/to/face_zstat.dscalar.nii -o /path/to/rFFA.dscalar.nii \\
>     --initial-roi /path/to/loose_rFFA.dscalar.nii \\
>     -a 500 -c 45 -50 -25 --hemi rh --use-group-surfaces

Define a 400 mm^2 ROI using a seed in the left PPA of an individual's
midthickness surface.

> subj=100610
> anatdir=/mnt/hcpdata/Facelab/$subj/MNINonLinear/fsaverage_LR32k
> python3 defineROICluster_CIFTI.py \\
>    -i /path/to/place_zstat.dscalar.nii -o /path/to/lPPA.dscalar.nii \\
>    --initial-roi /path/to/loose_lPPA.dscalar.nii \\
>    -a 400 -c 23125 --hemi lh \\
>    --left-surf $anatdir/${subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii \\
>    --right-surf $anatdir/${subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii

"""

import os, sys, math, argparse, warnings, tempfile, subprocess
import nibabel as nib
from scipy.optimize import minimize


### Key vars ###

# Paths to default group anatomical surfaces
group_anatdir = '/groups/labs/facelab/Datasets/HCP/HCP_S1200_GroupAvg_v1/'
group_anatfiles = {
        'left_surf':os.path.join(
                group_anatdir, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
                ),
        'left_area':os.path.join(
                group_anatdir, 'S1200.L.midthickness_MSMAll_va.32k_fs_LR.shape.gii'
                ),
        'right_surf':os.path.join(
                group_anatdir, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
                ),
        'right_area':os.path.join(
                group_anatdir, 'S1200.R.midthickness_MSMAll_va.32k_fs_LR.shape.gii'
                )
        }


### Custom functions ###

def my_subprocess_run(*args, **kwargs):
    """
    Wrapper around subprocess.run that includes stderr stream in any error
    messages. All arguments as per subprocess.run, except `check` which is
    forced to False (function checks return code itself) and `capture_output`
    which is forced to True.
    """
    kwargs['check'] = False
    kwargs['capture_output'] = True

    rtn = subprocess.run(*args, **kwargs)

    if rtn.returncode:
        msg = f"Command '{rtn.args}' returned non-zero exit status {rtn.returncode:d}."
        stderr = rtn.stderr
        if stderr:
            if isinstance(stderr, bytes):
                stderr = stderr.decode()
            msg += f'\n\n{stderr.strip()}'
        raise OSError(msg)

    return rtn


def surface_closest_vertex(coords, surface):
    """
    Wrapper around wb_command -surface-closest-vertex. Returns nearest vertex
    for given set of co-ordinates

    Arguments
    ---------
    coords : list
        3-item list of [x, y, z] co-ordinates

    surface : str
        Path to surface file to find vertex for

    Returns
    -------
    vertex : int
        Matching vertex number for co-ordinate

    rtn : subprocess.CompletedProcess
        Object returned by subprocess.run
    """
    # Init some temp text files
    tmp_input = tempfile.mkstemp()[1]
    tmp_output = tempfile.mkstemp()[1]

    # Write co-ordinates to input file
    with open(tmp_input, 'w') as fd:
        fd.write(' '.join(str(c) for c in coords))

    # Build & run command
    # wb_command -surface-closest-vertex \
    #   <surface> <coord-list-file> <vertex-list-out>
    cmd = ['wb_command', '-surface-closest-vertex', surface,
           tmp_input, tmp_output]
    rtn = my_subprocess_run(cmd)

    # Read vertex from output file
    with open(tmp_output, 'r') as fd:
        vertex = int(fd.read().strip())

    # Clean up temporary files
    os.unlink(tmp_input)
    os.unlink(tmp_output)

    # Return
    return vertex, rtn


def cifti_find_clusters(stat_in, cluster_out, thr, anatfiles,
                        initial_ROI=None, surf_min_area=0, vol_min_size=0):
    """
    Wrapper around wb_command -cifti-find-clusters. Applies clustering to
    statisical image for given threshold.

    Arguments
    ---------
    stat_in : str
        Path to input statistical CIFTI file (should be dscalar.nii)

    cluster_out : str
        Path to output cluster CIFTI file (should be dscalar.nii)

    thr: float
        Statistical threshold to apply

    anatfiles : dict
        Dictionary with keys for 'left_surf' and 'right_surf', and optionally
        'left_area' and 'right_area'. Values should give paths to surface
        anatomical files (should be surf.gii) and optionally area metric files
        (should be shape.gii). If area metric files are omitted, command will
        calculate areas from surfaces directly.

    initial_ROI : str
        Path to ROI image to apply to statistical image before clustering.
        If omitted, will perform clustering on full set of grayordinates.

    surf_min_area, vol_min_size : float
        Minimum thresholds for clusters' surface area (in mm^2) or
        volume (mm^3). Defaults to zero.

    Returns
    -------
    rtn : subprocess.CompletedProcess
        Object returned by subprocess.run
    """
    # Convert numeric inputs to str
    thr = str(thr)
    surf_min_area = str(surf_min_area)
    vol_min_size = str(vol_min_size)

    # Build command
    # wb_command -cifti-find-clusters \
    #     <infile> <surf_thr> <surf_min_area> <vol_thr> <vol_min_size> \
    #     <direction> <outfile> \
    #    -merged-volume \
    #    -left-surface <left_surf> [-corrected-areas <left_area>] \
    #  - -right-surface <right_surf> [-corrected-areas <right_area>] \
    #    [-cifti-roi <roi>]
    cmd = ['wb_command', '-cifti-find-clusters', stat_in, thr,
           surf_min_area, thr, vol_min_size, 'COLUMN', cluster_out,
           '-merged-volume']

    cmd.extend(['-left-surface', anatfiles['left_surf']])
    if 'left_area' in anatfiles.keys():
        cmd.extend(['-corrected-areas', anatfiles['left_area']])

    cmd.extend(['-right-surface', anatfiles['right_surf']])
    if 'right_area' in anatfiles.keys():
        cmd.extend(['-corrected-areas', anatfiles['right_area']])

    if initial_ROI:
        cmd.extend(['-cifti-roi', initial_ROI])

    # Run and return
    return my_subprocess_run(cmd)


def cifti_label_import(cluster_in, label_out, label_list_file=''):
    """
    Wrapper around wb_command -cifti-label-import. Creates a CIFTI label file
    from a dense CIFTI file.

    Arguments
    ---------
    cluster_in : str
        Path to cluster image as a dense CIFTI file (should be dscalar.nii)

    label_out : str
        Path to output CIFTI label file (should be dlabel.nii)

    label_list_file : str
        Path to text file containing label names and colours. Alternatively,
        pass an empty string (default) to allow command to generate default
        names.

    Returns
    -------
    rtn : subprocess.CompletedProcess
        Object returned by subprocess.run
    """
    # Build command
    # wb_command -cifti-label-import \
    #     <input> <label-list-file or empty str> <output>
    cmd = ['wb_command', '-cifti-label-import', cluster_in, label_list_file,
           label_out]

    # Run and return
    return my_subprocess_run(cmd)


def cifti_label_to_roi(label_in, roi_out, key):
    """
    Wrapper around wb_command -cifti-label-to-roi. Extracts individual label
    from CIFTI label file and exports as ROI to dense CIFTI file. All
    grayordinates within ROI will be set to a value of 1.

    Arguments
    ---------
    label_in : str
        Path to input CIFTI label file (should be dlabel.nii)

    roi_out : str
        Path to output CIFTI ROI file (should be dscalar.nii)

    key : int
        ID of cluster to export

    Returns
    -------
    rtn : subprocess.CompletedProcess
        Object returned by subprocess.run
    """
    # Build command
    # wb_command -cifti-label-to-roi <infile> <outfile> -key <ID>
    cmd = ['wb_command', '-cifti-label-to-roi', label_in, roi_out,
           '-key', str(key)]

    # Run and return
    return my_subprocess_run(cmd)


def get_ROI_area(roi_in, anatfiles, hemi):
    """
    Calls wb_command -cifti-weighted-stats to calculate surface area of ROI.

    Arguments
    ---------
    roi_in : str
        Path to input CIFTI ROI file (should be dscalar.nii)

    anatfiles : dict
        Dictionary with keys for left/right_surf, and optionally
        left/right_area, and values as paths to those files. Area files will
        be used if provided, otherwise surface files will be used.

    hemi : str
        Which hemisphere ROI is in - should be 'lh' or 'rh'

    Returns
    -------
    area : float
        Surface area of ROI

    rtn : subprocess.CompletedProcess
        Object returned by subprocess.run
    """
    # Build command
    # wb_command -cifti-weighted-stats \
    #   <infile> \
    #   -spatial-weights \
    #     -left-area-[surf/metric] <left-[surf/metric]> \
    #     -right-area-[surf/metric] <right-[surf/metric]> \
    #   -sum
    cmd = ['wb_command', '-cifti-weighted-stats', roi_in, '-spatial-weights']

    if 'left_area' in anatfiles.keys():
        cmd.extend(['-left-area-metric', anatfiles['left_area']])
    else:
        cmd.extend(['-left-area-surf', anatfiles['left_surf']])

    if 'right_area' in anatfiles.keys():
        cmd.extend(['-right-area-metric', anatfiles['right_area']])
    else:
        cmd.extend(['-right-area-surf', anatfiles['right_surf']])

    cmd.append('-sum')

    # Run command
    rtn = my_subprocess_run(cmd, text=True)

    # Extract output into dict
    area_dict = {}
    for line in rtn.stdout.strip().split('\n'):
        struct, area = line.split(':\t')
        area_dict[struct] = float(area)

    # Extract area for this hemi
    if hemi == 'lh':
        area = area_dict['CORTEX_LEFT']
    else:
        area = area_dict['CORTEX_RIGHT']

    # Return area and command output
    return area, rtn


def get_cifti_structs(infile):
    """
    Extract structure fields from dense CIFTI file

    Arguments
    ---------
    infile : str
        Path to dense CIFTI file (e.g. dscalar)

    Returns
    -------
    cifti_structs : dict
        Dictionary keyed by structure names. Values are (slice, model) tuples,
        containing the slice objects for extracting the relevant elements from
        the full data array, and the BrainModelAxis object respectively.
    """
    cifti = nib.load(infile)
    ax1 = cifti.header.get_axis(1)
    cifti_structs = {struct:(slice_, model) for (struct, slice_, model) \
                     in ax1.iter_structures()}
    return cifti_structs


def get_value_at_vertex(infile, hemi_slice, idx):
    """
    Get value from CIFTI at specified vertex

    Arguments
    ---------
    infile : str
        Path to CIFTI file (dscalar, dlabel, etc.)

    hemi_slice : slice
        Slice object for extracting hemisphere grayordinates from full array

    idx : int
        Index of vertex within hemisphere array

    Returns
    -------
    value : float
        Value at specified vertex
    """
    data = nib.load(infile).get_fdata(caching='unchanged').squeeze()
    return data[hemi_slice][idx]


def run_clustering_pipeline(thr, infile, initial_ROI, cluster_out, label_out,
                            ROI_out, anatfiles, seed_idx, hemi, hemi_slice):
    """
    Convenience function, runs full clustering pipeline.

    Arguments
    ---------
    thr : float
        Threshold to apply

    initial_ROI : str
        As per cifti_find_cluster function

    cluster_out, label_out, ROI_out : str
        Paths to output files for cluster (should be dscalar.nii), label
        (should be dlabel.nii), and ROI files (should be dscalar.nii)

    anatfiles : dict
        Dictionary with keys for 'left_surf' and 'right_surf', and optionally
        'left_area' and 'right_area'. Values should give paths to surface
        anatomical files (should be surf.gii) and optionally area metric files
        (should be shape.gii)

    seed_idx : int
        Index for seed vertex within hemisphere data array

    hemi : str
        Which hemisphere seed is in - should be 'lh' or 'rh'

    hemi_slice : slice
        Slice object for extracting hemisphere grayordinates from full array

    Returns
    -------
    area : float
        Surface area of ROI
    """
    # Apply clustering
    cifti_find_clusters(infile, cluster_out, thr, anatfiles, initial_ROI)

    # Convert to label
    cifti_label_import(cluster_out, label_out)

    # Extract ID of label at seed
    labelID = int(get_value_at_vertex(label_out, hemi_slice, seed_idx))

    # Convert matching label to ROI
    cifti_label_to_roi(label_out, ROI_out, labelID)

    # Get surface area
    area = get_ROI_area(ROI_out, anatfiles, hemi)[0]

    # Return area
    return area


def cost_func(params, targ_area,  *args, **kwargs):
    """
    Wraps pipeline function, and returns estimation error. Pass to
    minimisation function.

    Arguments
    ---------
    params : list
        List of function parameters. Should contain a single parameter giving
        the statistical threshold.

    targ_area : float
        Target surface area for ROI

    *args, **kwargs
        Further arguments passed to run_clustering_pipeline function

    Returns
    -------
    RSS : float
        Residual sum of squares = (targ_area - actual_area)**2
    """
    # Unlist threshold
    thr = params[0]

    # Run pipeline
    area = run_clustering_pipeline(thr, *args, **kwargs)

    # Calculate RSS and return
    return (area - targ_area)**2


### Parse args ###

# Init parser
class CustomFormatter(argparse.RawTextHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=CustomFormatter
        )

parser.add_argument('-i', '--infile', required=True,
                    help='Path to input statistical CIFTI (dscalar)')
parser.add_argument('-o', '--outfile', required=True,
                    help='Path to output ROI CIFTI (dscalar)')
parser.add_argument('-a', '--area', required=True, type=float,
                    help='Desired cluster area in mm^2')
parser.add_argument('-c', '--seed-coord', required=True, type=float, nargs='+',
                    help='Either vertex number OR x y z co-ordinates of seed')
parser.add_argument('--hemi', required=True, choices=['lh','rh'],
                    help='Which hemisphere seed is in')
parser.add_argument('--use-group-surfaces', action='store_true',
                    help='Use S1200 group average midthickness surfaces')
parser.add_argument('--left-surf',
                    help='Path to left surface anatomy file (surf.gii)')
parser.add_argument('--right-surf',
                    help='Path to right surface anatomy file (surf.gii)')
parser.add_argument('--left-area',
                    help='Path to left vertex area metric file (shape.gii)')
parser.add_argument('--right-area',
                    help='Path to right vertex area metric file (shape.gii)')
parser.add_argument('-r', '--initial-roi',
                    help='Path to ROI CIFTI (dscalar) to apply before clustering')
parser.add_argument('--start-thr', default='50%',
                    help='Starting threshold for optimisation')
parser.add_argument('--lower-bound', type=float, default=1.64,
                    help='Lower-bound threshold; set to inf or nan to disable')
parser.add_argument('--upper-bound', type=float, default='nan',
                    help='Upper-bound threshold; set to inf or nan to disable')

# Parse args
if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()

infile = args.infile
outfile = args.outfile
targ_area = args.area
seed_coord = args.seed_coord
hemi = args.hemi
initial_ROI = args.initial_roi
start_thr = args.start_thr
lower_bound = args.lower_bound
upper_bound = args.upper_bound

# Use group surfaces if requested, otherwise extract specified surfaces
if args.use_group_surfaces:
    anatfiles = group_anatfiles
else:
    if not (args.left_surf and args.right_surf):
        raise Exception('Must specify surface files if not using group surfaces')
    anatfiles = {'left_surf':args.left_surf, 'right_surf':args.right_surf}
    if args.left_area:
        anatfiles['left_area'] = args.left_area
    if args.right_area:
        anatfiles['right_area'] = args.right_area

# Treat inf or nan thresholds as None
if not math.isfinite(lower_bound):
    lower_bound = None
if not math.isfinite(upper_bound):
    upper_bound = None

# Error check
if len(seed_coord) not in [1, 3]:
    raise Exception('Seed co-ord must be a single vertex or [x y z] triplet')

try:
    start_thr = float(start_thr)
except ValueError:
    if not start_thr.endswith('%'):
        raise Exception('Starting threshold must be float or percentage')


### Begin ###
# Init temporary files for intermediate cluster, label, and ROI images
tmp_cluster = tempfile.mkstemp(suffix='.dscalar.nii')[1]
tmp_label = tempfile.mkstemp(suffix='.dlabel.nii')[1]
tmp_ROI = tempfile.mkstemp(suffix='.dscalar.nii')[1]

# Find seed vertex from co-ords if necessary
if len(seed_coord) == 3:
    if hemi == 'lh':
        surf_file = anatfiles['left_surf']
    else:
        surf_file = anatfiles['right_surf']

    seed_vertex = surface_closest_vertex(seed_coord, surf_file)[0]
else:
    seed_vertex = int(seed_coord[0])

# Get data slice for hemisphere, and find surface index for seed vertex
cifti_structs = get_cifti_structs(infile)
if hemi == 'lh':
    hemi_slice, model = cifti_structs['CIFTI_STRUCTURE_CORTEX_LEFT']
else:
    hemi_slice, model = cifti_structs['CIFTI_STRUCTURE_CORTEX_RIGHT']
seed_idx = (model.vertex == seed_vertex).nonzero()[0][0]

# Work out start thr from percentage if necessary
if isinstance(start_thr, str):
    prop = float(start_thr.rstrip('%')) / 100
    max_ = get_value_at_vertex(infile, hemi_slice, seed_idx)
    min_ = lower_bound if lower_bound is not None else 0
    start_thr = min_ + prop * (max_ - min_)

# Run optimisation
pipeline_args = (infile, initial_ROI, tmp_cluster, tmp_label, tmp_ROI,
                 anatfiles, seed_idx, hemi, hemi_slice)

optim_args = (targ_area,) + pipeline_args

optimres = minimize(cost_func, x0=(start_thr,), args=optim_args,
                    method='Nelder-Mead')

opt_thr = optimres.x[0]

# Apply bounds
if (lower_bound is not None) and (opt_thr < lower_bound):
    warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to lower bound ({lower_bound:.2f})')
    opt_thr = lower_bound
elif (upper_bound is not None) and (opt_thr > upper_bound):
    warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to upper bound ({upper_bound:.2f})')
    opt_thr = upper_bound

# Run clustering one more time with optimised threshold, and set ROI output
# to final output file
cluster_area = run_clustering_pipeline(
        opt_thr, infile, initial_ROI, tmp_cluster, tmp_label, outfile,
        anatfiles, seed_idx, hemi, hemi_slice
        )

# Clean up temporary files
os.unlink(tmp_cluster)
os.unlink(tmp_label)
os.unlink(tmp_ROI)

# Print results out
print('{} : Requested_Area={:.3f} Actual_Area={:.3f} Thr={:.3f}' \
      .format(os.path.basename(outfile), targ_area, cluster_area, opt_thr))
