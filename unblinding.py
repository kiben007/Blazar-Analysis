import os, sys, glob, abc
import numpy as np, matplotlib, scipy, time
from astropy.io import fits
import random
import pickle
from matplotlib import pyplot as plt, colors
from scipy import stats, interpolate, optimize
import numpy.lib.recfunctions as rf
import multiprocessing
import argparse
import analysis_functions
import os

indir = '/data/i3store/users/mjlarson/ps_tracks/'
outdir = '/data/condor_builds/users/bbrinson/blazar'

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--number", help = "number of trials", default = 1000, type = int)
parser.add_argument("-J", "--job", help = "job number", default = 0, type = int)
parser.add_argument("-S", "--scheme", help = "flux weighting scheme", default = 0, type = int)
options = parser.parse_args()

number = options.number
job = options.job
scheme = options.scheme

data_files = indir + "/IC86_*exp.npy"
dfiles = glob.glob(data_files)
dfiles.remove("/data/i3store/users/mjlarson/ps_tracks/IC86_2011_exp.npy")
data = analysis_functions.read(dfiles)
data['ra'] = np.random.uniform(0, 2*np.pi, len(data))

dtype = np.dtype([('run', np.int),
                          ('event', np.int),
                          ('subevent', np.int), 
                          ('ra', np.float64), 
                          ('dec', np.float64),
                          ('azi', np.float64),
                          ('zen', np.float64),
                          ('time', np.float64),
                          ('logE', np.float64),
                          ('angErr', np.float64)])

fixed_data = np.zeros(len(data), dtype = dtype)

fixed_data['run'] = data['run']
fixed_data['event'] = data['event']
fixed_data['subevent'] = data['subevent']
fixed_data['ra'] = data['ra']
fixed_data['dec'] = data['dec']
fixed_data['azi'] = data['azi']
fixed_data['zen'] = data['zen']
fixed_data['time'] = data['time']
fixed_data['logE'] = data['logE']
fixed_data['angErr'] = data['angErr']

# Set the angular error floor to 0.2 degrees
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)

grl_files = indir + "GRL/IC86_*exp.npy"
grlf = glob.glob(grl_files)
grlf.remove("/data/i3store/users/mjlarson/ps_tracks/GRL/IC86_2011_exp.npy")
grl = analysis_functions.read(grlf)

test_data = np.load(outdir + '/prereqs/test_data.npy')

MJDREFF = 51910
MJDREFI = 7.428703703703703e-4

time_data = test_data['START'] / (24 * 3600) + (MJDREFF + MJDREFI)
flux_data = test_data['FLUX_100_300000']

flux_spline = scipy.interpolate.UnivariateSpline(time_data, flux_data, s = 0, k = 4)
time_vals = np.linspace(54684.00074287037, 59001.00074287037, 100000)

with open(outdir + '/prereqs/bg_p_dec.pkl', 'rb') as f:
    bg_p_dec = pickle.load(f)

gamma_points = np.load(outdir + '/prereqs/gamma_points.npy')
ratio_bins = np.load(outdir + '/prereqs/ratio_bins.npy')
sob_maps = np.load(outdir + '/prereqs/sob_maps.npy')

blazar_ra = 5.995042209084602
blazar_dec = 0.2818394575289712

analysis_times = np.load(outdir + '/prereqs/analysis_times.npy')

n_flares = 9

print('starting')

for i in range(0, n_flares):
    
    background_time_profile = analysis_functions.uniform_profile(analysis_times['start'][i], analysis_times['end'][i])
    signal_time_profile = analysis_functions.spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])
    
    args_unstacked = {"background_time_profile":background_time_profile,
        "signal_time_profile":signal_time_profile,
        "gamma_points":gamma_points,
        "ratio_bins":ratio_bins,
        "sob_maps":sob_maps,
        "data":fixed_data, 
        "bg_p_dec":bg_p_dec,
        "grl":grl,
        "test_ns":0.1,
        "minimize":True,
        "source_ra":blazar_ra,
        "test_ra":blazar_ra,
        "source_dec":blazar_dec,
        "test_dec":blazar_dec,
        "gauss":False,
        "spl":True}
    
    flare_info = analysis_functions.unblinded_unstacked(i, **args_unstacked)
    
    np.save(outdir + '/Unblinding/flare_' + str(i) + '_unblinded_test.npy', flare_info)
    
    print(str(i))

args_stacked = {"analysis_times":analysis_times,
        "flux_spline":flux_spline,
        "gamma_points":gamma_points,
        "ratio_bins":ratio_bins,
        "sob_maps":sob_maps,
        "data":fixed_data,
        "bg_p_dec":bg_p_dec,
        "grl":grl,
        "test_ns":0.1,
        "minimize":True,
        "source_ra":blazar_ra,
        "test_ra":blazar_ra,
        "source_dec":blazar_dec,
        "test_dec":blazar_dec,
        "gauss":False,
        "spl":True}

stacked_info = analysis_functions.unblinded_stacked(n_flares, **args_stacked)

np.save(outdir + '/Unblinding/stacked_unblinded_test.npy', flare_info)