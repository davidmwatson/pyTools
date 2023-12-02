#!/usr/bin/env python3
"""
Uses clustering algorithm to generate labels of a specific size on a Freesurfer
surface from a given statistical surface image. Size may be specified in terms
of surface area or number of vertices.

Note - Freesurfer must be sourced before running this script.

Arguments
---------
-s|--subject
    Freesurfer subject ID

--hemi
    Surface hemisphere: 'lh' or 'rh'

-i|--input
    Path to input statistical surface image (probably a -log10(pstat))

-n|--cluster-size
    Desired size of cluster, in units of mm^2 or number of vertices depending
    on specified size metric.

-m|--size-metric
    Metric for cluster size. If 'area', measure surface area in mm^2.
    If 'nVtxs', measure number of vertices.

-v|--seed-vertex
    Location of the seed vertex. May be a single integer giving the vertex
    index, or a space delimited list of 3 floats giving the x, y, and z
    surfaceRAS co-ordinates. This seed should fall within the desired cluster;
    it's a good idea to place it on or near the peak voxel within the region
    (it doesn't have to be exact). Can specify the flag multiple times to run
    multiple ROIs - the number of times must match the number of outfiles.

-o|--outfile
    Path to output surface label file. A .label extension will be appended
    if not already included. Can specify multiple outfiles if running multiple
    ROIs - the number must match the number of seed coordinates.

--surf (optional)
    Which surface to use (default = white)

-l|--initial-label (optional)
    Path to surface label file. If provided, will apply this label to the data
    before performing the clustering - can be useful for preventing the
    cluster from growing into undesired regions (e.g. PPA bleeding into RSC).
    Can specify multiple labels if running multiple ROIs - the number must
    match the number of outfiles and seed co-ordinates.

--start-thr (optional)
    Space delimited list of one or more thresholds to use as starting points
    for optimisation. If floats, will use those thresholds directly. Can also
    specify as a percentage (e.g. 50%), which will choose the corresponding
    percentage value between the lower bound threshold (or zero if lower bound
    is disabled) and the value at the seed vertex. If more than one value
    provided, will run optimisation starting from each one, then select the
    solution yielding the lowest error. The default is 50%.

--lower-bound (optional)
--upper-bound (optional)
    Lower and upper bounds for optimisation. If the optimised threshold falls
    outside this range, it will be clipped to this range and a warning will be
    raised. Either bound can be set to inf or nan to disable its use. The
    default is a lower bound of 1.3 (-log10(0.05)), and the upper bound is
    disabled.

--sign (optional)
    Tail of statistical map to use. Can be abs, pos (default), or neg. Note
    that thresholds will be adjusted if a one-tailed pos/neg test is used, so
    that values will correspond to one-tailed p-values. For example, if sign is
    'pos' and the threshold is reported as 3.0 (one-tailed p = .001), the
    statistical map would actually be thresholded at 3.0 - log(2) ~= 2.7
    (two-tailed p = .002).

--subjects-dir (optional)
    Directory containing Freesurfer subject directories. If omitted, defaults
    to the value of the SUBJECTS_DIR environment variable.

Example usage
-------------
Define 1000mm^2 cluster from pstat using vertex at [10,20,30] surfaceRAS
co-ordinates, applying initial label.

> python3 defineSurfROICluster.py --subject fsaverage --hemi lh \\
    --infile sig.lh.mgh --cluster-size 1000 --size-metric area \\
    --seed-vertex 10 20 30 --outfile myCluster.lh.label \\
    --initial-label initialMask.lh.label

Create two clusters with seeds at [10,20,30] and [40,50,60] respectively

> python3 defineSurfROICluster.py --subject fsaverage --hemi lh \\
    --infile sig.lh.mgh --cluster-size 1000 --size-metric area \\
    --seed-vertex 10 20 30 --seed-vertex 40 50 60 \\
    --outfile myCluster1.lh.label myCluster2.lh.label \\
    --initial-label initialMask1.lh.label initialMask2.lh.label

"""

import os, sys, csv, warnings, argparse, subprocess, tempfile, shutil
import numpy as np
import nibabel as nib
from scipy.optimize import minimize


### Custom funcs ###


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


def RAS_to_vertex(subjects_dir, subject, hemi, vtx_coord):
    """
    Get index of closest vertex to surfaceRAS co-ordinate.

    Arguments
    ---------
    subjects_dir : str
        Path to Freesurfer subject directory
    subject : str
        Subject ID
    hemi : str
        lh or rh
    vtx_coord : array-like
        [x, y, z] surface RAS co-ordindate of vertex


    Returns
    -------
    vtx : int
        Index of vertex
    """
    geom_file = os.path.join(subjects_dir, subject, 'surf', hemi + '.white')
    surf_coords = nib.freesurfer.read_geometry(geom_file)[0]
    vtx_coord = np.asarray(vtx_coord, dtype=float)
    return np.linalg.norm(surf_coords - vtx_coord, axis=1).argmin()


def mri_surfcluster(thr, subjects_dir, subject, hemi, surf, infile,
                    clusterdir, sign, clabel=None, save_labels=False):
    """
    Run Freesurfer mri_surfcluster command via subprocess

    Arguments
    ---------
    thr : float
        Threshold to apply
    subjects_dir : str
        Path to Freesurfer subjects directory
    subject : str
        Subject ID
    hemi : str
        'lh' or 'rh'
    surf : str
        Which surface to use
    infile : str
        Path to input statistical surface file
    clusterdir : str
        Path to directory to save outputs. Must already exist.
    sign : str
        abs, pos, or neg

    clabel : str (optional)
        Path to label file to apply as pre-mask
    save_labels : bool (optional)
        If True, also save clusters out as label files. Default is False.

    Outputs
    -------
    Saves following to clusterdir:
    * cluster.mgh - Surface file labelling cluster IDs
    * cluster_summary.txt - Summary text file, includes cluster table
    * cluster-????.label - Labels for each cluster (if save_labels==True)
    """
    # Build command args
    cmd = ['mri_surfcluster',
           '--hemi', hemi,
           '--subject', subject,
           '--surf', surf,
           '--in', infile,
           '--thmin', str(thr),
           '--sign', sign,
           '--ocn', os.path.join(clusterdir, 'cluster.mgh'),
           '--sum', os.path.join(clusterdir, 'cluster_summary.txt'),
           '--sd', subjects_dir]
    if clabel:
        cmd.extend(['--clabel', clabel])
    if save_labels:
        cmd.extend(['--olab', os.path.join(clusterdir, 'cluster')])

    # Run command
    my_subprocess_run(cmd)


def get_cluster_at_vertex(clusterdir, vertex):
    """
    Extract clusterID at given vertex

    Arguments
    ---------
    clusterdir : str
        Directory containing outputs of mri_surfcluster
    vertex : int
        Index of vertex

    Returns
    -------
    ID : int
        Cluster ID
    """
    img = nib.load(os.path.join(clusterdir, 'cluster.mgh'))
    cluster_data = img.get_fdata().squeeze()
    clusterID = int(cluster_data[vertex])
    img.uncache()
    return clusterID


def get_cluster_size(clusterdir, clusterID, size_metric):
    """
    Read cluster summary file and extract given size metric for given cluster

    Arguments
    ---------
    clusterdir : str
        Directory containing outputs of mri_surfcluster
    clusterID : int
        ID of cluster
    size_metric : str
        'area' or 'nVtxs'

    Returns
    -------
    area : float
        Area of cluster in mm^2
    nVtxs : int
        Number of vertices contained in cluster
    """
    # Read contents of summary file
    with open(os.path.join(clusterdir, 'cluster_summary.txt'), 'r') as f:
        lines = f.readlines()

    # Work out which line cluster table starts on
    lineN = [line.startswith('# ClusterNo') for line in lines].index(True)

    # Extract header columns
    fieldnames = lines[lineN].lstrip('#').strip().split()

    # Read remaining lines into DictReader. Note that table includes multiple
    # spaces between columns - we set skipinitialspace=True to ignore them.
    reader = csv.DictReader(lines[lineN+1:], fieldnames, delimiter=' ',
                            skipinitialspace=True)

    # Loop till clusterID found, return relevant size metric
    for row in reader:
        if int(row['ClusterNo']) == clusterID:
            if size_metric == 'area':
                return float(row['Size(mm^2)'])
            elif size_metric == 'nVtxs':
                return int(row['NVtxs'])
            else:
                raise ValueError(f"Invalid metric: '{size_metric}'")

    # If we reach here, cluster wasn't found in table - error
    raise Exception('Cluster ID not found')


def cost_func(thr, subjects_dir, subject, hemi, surf, infile, clusterdir,
              sign, seed_vertex, target_size, size_metric, clabel=None):
    """
    Cost function to be optimised. Applies clustering, then returns error
    between actual and target cluster size.

    Arguments
    ---------
    thr, subjects_dir, subject, hemi, surf, clusterdir, surf, clabel
        All as per mri_surfcluster function. Threshold may be supplied as
        single-item array - this is for compatibility with minimize function.
    seed_vertex : int
        Index of seed vertex
    target_size : float or int
        Target size of cluster; either area (in mm^2) or num vertices
        depending on size metric.
    size_metric : str
        Size metric: 'area' or 'nVtxs'

    Returns
    -------
    err : float
        Squared error between actual and target cluster size
    """
    # Possibly extract threshold from array
    if hasattr(thr, '__iter__'):
        if len(thr) == 1:
            thr = thr[0]
        else:
            raise ValueError('thr must be float or single-item array')

    # Cluster
    mri_surfcluster(thr, subjects_dir, subject, hemi, surf, infile, clusterdir,
                    sign, clabel=clabel)

    # Find size of cluster
    clusterID = get_cluster_at_vertex(clusterdir, seed_vertex)
    if clusterID == 0:
        cluster_size = 0
    else:
        cluster_size = get_cluster_size(clusterdir, clusterID, size_metric)

    # Return squared error of cluster size
    return float(cluster_size - target_size)**2


### Parse arguments ###

# Check Freesurfer sourced
if 'FREESURFER_HOME' not in os.environ:
    raise OSError('Freesurfer must be sourced prior to running this script')

# Set up parser
class CustomFormatter(argparse.RawTextHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

parser = argparse.ArgumentParser(formatter_class=CustomFormatter,
                                 description=__doc__)

parser.add_argument('-s', '--subject', required=True, help='Subject ID')
parser.add_argument('--hemi', required=True, choices=['lh','rh'],
                    help='Surface hemisphere')
parser.add_argument('-i', '--infile', required=True,
                    help='Path to input surface image')
parser.add_argument('-n', '--cluster-size', required=True, type=float,
                    help='Desired cluster size')
parser.add_argument('-m', '--size-metric', required=True,
                    choices=['area', 'nVtxs'], help='Metric for cluster size')
parser.add_argument('-v', '--seed-vertex', required=True, type=float,
                    nargs='+', action='append',
                    help='surfaceRAS co-ordinates or index of seed vertex')
parser.add_argument('--surf', default='white', help='Surface to use')
parser.add_argument('-o', '--outfile', required=True, nargs='+',
                    help='Path to desired output label file')
parser.add_argument('-l', '--initial-label', nargs='+',
                    help='Path to label to apply before clustering')
parser.add_argument('--start-thr', nargs='+', default=['50%'],
                    help='Starting threshold(s) for optimisation')
parser.add_argument('--lower-bound', type=float, default=1.3,
                    help='Lower-bound threshold; set to inf or nan to disable')
parser.add_argument('--upper-bound', type=float, default='nan',
                    help='Upper-bound threshold; set to inf or nan to disable')
parser.add_argument('--sign', choices=['abs', 'pos', 'neg'], default='pos',
                    help='Tail of statistical map to use')
parser.add_argument('--subjects-dir', help='Freesurfer subjects directory')

# If no arguments given, print help and exit
if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit()

# Parse args
args = parser.parse_args()
subject = args.subject
hemi = args.hemi
infile = args.infile
target_size = args.cluster_size
size_metric = args.size_metric
seed_vertices = args.seed_vertex
outfiles = args.outfile
surf = args.surf
initial_labels = args.initial_label
start_thresholds = args.start_thr
lower_bound = args.lower_bound
upper_bound = args.upper_bound
sign = args.sign
subjects_dir = args.subjects_dir

# Allocate default subjects dir if necessary
if not subjects_dir:
    subjects_dir = os.environ['SUBJECTS_DIR']

# Treat inf or nan thresholds as None
if np.isinf(lower_bound) or np.isnan(lower_bound):
    lower_bound = None
if np.isinf(upper_bound) or np.isnan(upper_bound):
    upper_bound = None

# Check number of regions match across arguments
if len(outfiles) != len(seed_vertices):
    parser.error('Number of outfiles must match number of seed vertices')

if initial_labels is not None and (len(outfiles) != len(initial_labels)):
    parser.error('Number of outfiles must match number of initial labels')


### Begin ###

# Open temporary output directory
tmpdir = tempfile.TemporaryDirectory()

# Load surf data - needed for checking bounds, etc.
surf_data = nib.load(infile).get_fdata().squeeze()

# Loop regions
for i, (seed_vertex, outfile) in enumerate(zip(seed_vertices, outfiles)):
    # Grab initial label if provided
    initial_label = None if initial_labels is None else initial_labels[i]

    # If seed in scannerRAS coords, convert to vertex index. If already index,
    # extract value and convert float -> int
    if len(seed_vertex) == 3:
        seed_vertex = RAS_to_vertex(subjects_dir, subject, hemi, seed_vertex)
    elif len(seed_vertex) == 1:
        seed_vertex = int(seed_vertex[0])
    else:
        parser.error('seed_vertex must be single index or 3-item list of coords')

    # Ensure outfile .label extension
    if not outfile.endswith('.label'):
        outfile += '.label'

    # Check lower bound appropriate for seed
    if lower_bound is not None:
        seed_val = surf_data[seed_vertex]
        if sign in ['pos','neg']:
            seed_val -= np.log(2)
        if seed_val < lower_bound:
            raise ValueError('Lower bound cannot exceed value at seed voxel')

    # Loop starting parameters
    all_thr = []
    all_errs = []
    for x0 in start_thresholds:
        # If value specified as percentage, calcualate value from data range.
        # Otherwise, just convert to float.
        if x0.endswith('%'):
            prop = float(x0.rstrip('%')) / 100
            max_ = surf_data[seed_vertex]
            min_ = lower_bound if lower_bound is not None else 0
            x0 = min_ + prop * (max_ - min_)
            if sign in ['pos','neg']:
                x0 += np.log(2)
        else:
            x0 = float(x0)

        # Run optimisation
        args = (subjects_dir, subject, hemi, surf, infile, tmpdir.name, sign,
                seed_vertex, target_size, size_metric, initial_label)
        optimres = minimize(cost_func, x0=[x0], args=args, method='Nelder-Mead')

        # Extract optimised threshold and error values, append to array
        all_thr.append(optimres.x[0])
        all_errs.append(optimres.fun)

    # Choose threshold with lowest err
    best_idx = np.argmin(all_errs)
    opt_thr = all_thr[best_idx]

    # Clip thresh to lower/upper bound if necessary
    if (lower_bound is not None) and (opt_thr < lower_bound):
        warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to lower bound ({lower_bound:.2f})')
        opt_thr = lower_bound
    elif (upper_bound is not None) and (opt_thr > upper_bound):
        warnings.warn(f'Clipping threshold ({opt_thr:.2f}) to upper bound ({upper_bound:.2f})')
        opt_thr = upper_bound

    # Run clustering once more with selected thresh
    mri_surfcluster(opt_thr, subjects_dir, subject, hemi, surf, infile,
                    tmpdir.name, sign, clabel=initial_label, save_labels=True)
    clusterID = get_cluster_at_vertex(tmpdir.name, seed_vertex)
    cluster_size = get_cluster_size(tmpdir.name, clusterID, size_metric)

    # Copy label to outfile
    src = os.path.join(tmpdir.name, f'cluster-{clusterID:04d}.label')
    shutil.copyfile(src, outfile)

    # Print results
    print('{} : Metric={} Requested_Size={} Actual_Size={} Thr={:.2f}' \
          .format(os.path.basename(outfile), size_metric, target_size,
                  cluster_size, opt_thr))

# Cleanup tmpdir
tmpdir.cleanup()

