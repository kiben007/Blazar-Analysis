import os, sys, glob, abc
import numpy as np, matplotlib, scipy, time
from tqdm.notebook import tqdm
from astropy.io import fits
import random
import pickle
from matplotlib import pyplot as plt, colors
from scipy import stats, interpolate, optimize
import numpy.lib.recfunctions as rf
import multiprocessing
import argparse
import analysis_functions

indir = '/data/i3store/users/mjlarson/ps_tracks/'
outdir = '/data/condor_builds/users/bbrinson/blazar'

def read(filelist):
    data = []
    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0:
            data = x.copy()
        else:
            data = rf.stack_arrays([data, x])
    return data

data_files = indir + "/IC86_*exp.npy"
dfiles = glob.glob(data_files)
dfiles.remove("/data/i3store/users/mjlarson/ps_tracks/IC86_2011_exp.npy")
data = read(dfiles)
data['ra'] = np.random.uniform(0, 2*np.pi, len(data))

sim_files = indir + 'IC86_2012_MC.npy'
sim = read(glob.glob(sim_files))

# Set the angular error floor to 0.2 degrees
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)

grl_files = indir + "GRL/IC86_*exp.npy"
grlf = glob.glob(grl_files)
grlf.remove("/data/i3store/users/mjlarson/ps_tracks/GRL/IC86_2011_exp.npy")
grl = read(grlf)

test_data = np.load(outdir + '/prereqs/test_data.npy')

MJDREFF = 51910
MJDREFI = 7.428703703703703e-4

time_data = test_data['START'] / (24 * 3600) + (MJDREFF + MJDREFI)
flux_data = test_data['FLUX_100_300000']

flux_spline = scipy.interpolate.UnivariateSpline(time_data, flux_data, s = 0, k = 4)
time_vals = np.linspace(54684.00074287037, 59001.00074287037, 100000)

with open('prereqs/bg_p_dec.pkl', 'rb') as f:
    bg_p_dec = pickle.load(f)

seed = None

gamma_points = np.load('prereqs/gamma_points.npy')
ratio_bins = np.load('prereqs/ratio_bins.npy')
sob_maps = np.load('prereqs/sob_maps.npy')

global_vars = {}

blazar_ra = 5.995042209084602
blazar_dec = 0.2818394575289712

analysis_times = np.load('prereqs/analysis_times.npy')

N_vals = np.logspace(-19, -16, 20)
plot_vals = np.logspace(-19, -16, 1000)

for i in range(0, n_flares):
    
    background_time_profile = analysis_functions.uniform_profile(analysis_times['start'][i], analysis_times['end'][i])
    signal_time_profile = analysis_functions.spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])

    args = {"test_ns":0.1,
            "minimize":True,
            "source_ra":blazar_ra,
            "test_ra":blazar_ra,
            "source_dec":blazar_dec,
            "test_dec":blazar_dec,
            "signal_time_profile":signal_time_profile,
            "background_time_profile":background_time_profile,
            "gauss":False,
            "spl":True,
            "ncpus":16,
            "angular_window":np.radians(180)}
    
    background_ts_analysis = np.load('prereqs/background_ts_' + str(i) + '_offset.npy')
    perc = np.zeros(len(N_vals))

    for j in range(0, len(N_vals)):
        signal_array = np.load('sensitivity_data_offset/analysis_job_' + str(j) + '_flare_' + str(i) + '_offset.npy')
        signal_ts_analysis = signal_array['ts']
        perc[j] = len(signal_ts_analysis[signal_ts_analysis > np.percentile(background_ts_analysis, 50)])/len(signal_ts_analysis)
        
    analysis_spl = interpolate.UnivariateSpline(N_vals, perc, s = 5E-4)
    
    fig1, ax1 = plt.subplots(figsize=(10,6))

    ax1.plot(N_vals, perc, '.', color = 'b')
    ax1.plot(plot_vals, analysis_spl(plot_vals), linewidth = 2, color = 'r')
    ax1.hlines(0.9, 1e-20, 1E-16)
    ax1.set_xlabel('Flux normalization N (GeV^-1 cm^-2 s^-1)')
    ax1.set_ylabel('Percentage of signal over background median')
    ax1.set_title('Spline estimate for sensitivity')
    ax1.set_xscale('log')

    fig1.savefig('plots2/spline_flare_' + str(i) + '_offset.png')
    
    add = (analysis_spl(plot_vals) >= 0.9)
    flux = plot_vals[add][0]
    
    analysis_spline_fits = produce_trials_multi(1000, N = flux, **args)
    
    spline_ts = np.array(analysis_spline_fits['ts'])
    
    full = np.concatenate([background_ts_analysis, spline_ts])
    xmin, xmax = full.min(), full.max()
    plot_bins = np.linspace(xmin, xmax, 50)

    fig2, ax2 = plt.subplots(figsize=(10,6))

    ax2.hist(background_ts_analysis,
            bins = plot_bins,
            histtype = 'step',
            color = 'k',
            linewidth = 3,
            alpha=0.5,)
    
    ax2.hist(spline_ts,
            bins = plot_bins,
            histtype = 'step',
            linewidth = 3,
            alpha=0.5,
            color = 'r',)
    
    ax2.vlines(np.percentile(background_ts_analysis, 50), 0, 1E3, color = 'k', linewidth = 3, linestyle = '--')
    ax2.set_xlabel('TS')
    ax2.set_ylabel('counts')
    ax2.legend(['background', 'N = ' + str(flux) + ' GeV^-1 cm^-2 s^-1', 'median of background'])
    ax2.set_yscale('log')
    ax2.set_title(r'Point source Sensitivity for 3C454.3 between MJD ' + str(analysis_times['start'][i]) + ' and ' + str(analysis_times['end'][i]))
    
    fig2.savefig('plots2/sensitivity_flare_' + str(i) + '_offset.png')
    
    np.save('sensitivities2/flare_' + str(i) + '_ts_offset.npy', spline_ts)