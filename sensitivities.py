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

indir = '/data/i3store/users/mjlarson/ps_tracks/'
outdir = '/data/condor_builds/users/bbrinson/blazar'

data_files = indir + "/IC86_*exp.npy"
dfiles = glob.glob(data_files)
dfiles.remove("/data/i3store/users/mjlarson/ps_tracks/IC86_2011_exp.npy")
data = analysis_functions.read(dfiles)
data['ra'] = np.random.uniform(0, 2*np.pi, len(data))

sim_files = indir + 'IC86_2012_MC.npy'
sim = analysis_functions.read(glob.glob(sim_files))

# Set the angular error floor to 0.2 degrees
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)

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

with open('prereqs/bg_p_dec.pkl', 'rb') as f:
    bg_p_dec = pickle.load(f)

seed = None

gamma_points = np.load(outdir + '/prereqs/gamma_points.npy')
ratio_bins = np.load(outdir + '/prereqs/ratio_bins.npy')
sob_maps = np.load(outdir + '/prereqs/sob_maps.npy')

global_vars = {}
n_flares = 9

blazar_ra = 5.995042209084602
blazar_dec = 0.2818394575289712

analysis_times = np.load(outdir + '/prereqs/analysis_times.npy')

N_vals = np.logspace(-19, -16, 20)
plot_vals = np.logspace(-19, -16, 1000)

print('starting')

for i in range(0, n_flares):
    
    background_time_profile = analysis_functions.uniform_profile(analysis_times['start'][i], analysis_times['end'][i])
    signal_time_profile = analysis_functions.spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])

    args = {"gamma_points":gamma_points,
        "ratio_bins":ratio_bins,
        "sob_maps":sob_maps,
        "data":data, 
        "sim":sim,
        "bg_p_dec":bg_p_dec,
        "grl":grl,
        "test_ns":0.1,
        "minimize":True,
        "source_ra":blazar_ra,
        "test_ra":blazar_ra,
        "source_dec":blazar_dec,
        "test_dec":blazar_dec,
        "signal_time_profile":signal_time_profile,
        "background_time_profile":background_time_profile,
        "gauss":False,
        "spl":True,
        "ncpus":4,
        "angular_window":np.radians(180)}
    
    background_ts_analysis = np.load(outdir + '/background_profiles_unstacked/background_ts_' + str(i) + '.npy')
    
    unblinded_values = np.load(outdir + '/Unblinding_test/flare_' + str(i) + '_unblinded_test.npy')
    
    unblinded_ts = unblinded_values['ts']
    
    greater_add = (background_ts_analysis >= unblinded_ts)
    
    greater_than_data = background_ts_analysis[greater_add]
    
    p_value_unstacked = len(greater_than_data)/len(background_ts_analysis)
    p_value_array = np.array([p_value_unstacked])
    
    np.save(outdir + '/Unblinding_limits_test/p-value_flare_' + str(i) + 'test_.npy', p_value_array)
    
    fig1, ax1 = plt.subplots(figsize=(10,6))

    full = np.concatenate([background_ts_analysis])
    xmin, xmax = full.min(), full.max()
    plot_bins = np.linspace(xmin, xmax, 50)
    
    ax1.hist(background_ts_analysis,
            bins = plot_bins,
            histtype = 'step',
            color = 'k',
            linewidth = 3,
            alpha=0.5,
            label = 'Background TS distribution')
    
    ax1.hist(greater_than_data,
            bins = plot_bins,
            histtype = 'step',
            color = 'r',
            linewidth = 3,
            alpha=0.5,
            label = 'Background above measured TS value')
    
    ax1.vlines(unblinded_ts, 0, 1E3, color = 'b', linewidth = 3, linestyle = '--', label = 'measured TS value')
    ax1.set_xlabel('TS')
    ax1.set_ylabel('counts')
    ax1.set_title('P-value = ' + str(p_value_unstacked) + ' for 3C454.3 flare between MJD ' + str(analysis_times['start'][i]) + ' and ' + str(analysis_times['end'][i]))
    ax1.legend()
    ax1.set_yscale('log')

    fig1.savefig(outdir + '/Unblinding_plots_test/p-value_flare_' + str(i) + '_test.png')
    
    perc = np.zeros(len(N_vals))

    for j in range(0, len(N_vals)):
        signal_array = np.load(outdir + '/sensitivity_data_offset/analysis_job_' + str(j) + '_flare_' + str(i) + '_offset.npy')
        signal_ts_analysis = signal_array['ts']
        perc[j] = len(signal_ts_analysis[signal_ts_analysis > unblinded_ts])/len(signal_ts_analysis)
        
    analysis_spl = interpolate.UnivariateSpline(N_vals, perc, s = 5E-4)
    
    fig2, ax2 = plt.subplots(figsize=(10,6))

    ax2.plot(N_vals, perc, '.', color = 'b')
    ax2.plot(plot_vals, analysis_spl(plot_vals), linewidth = 2, color = 'r')
    ax2.hlines(0.9, 1E-19, 1E-16)
    ax2.set_xlabel('Flux normalization N (GeV^-1 cm^-2 s^-1)')
    ax2.set_ylabel('Percentage of signal over measured TS value')
    ax2.set_title('Spline estimate for upper limit')
    ax2.set_xscale('log')

    fig2.savefig(outdir + '/Unblinding_plots_test/spline_flare_' + str(i) + '_unblinded_test.png')
    
    add = (analysis_spl(plot_vals) >= 0.9)
    flux = plot_vals[add][0]
    
    analysis_spline_fits = analysis_functions.produce_trials_multi(1000, N = flux, **args)
    
    spline_ts = np.array(analysis_spline_fits['ts'])
    
    full = np.concatenate([spline_ts])
    xmin, xmax = full.min(), full.max()
    plot_bins = np.linspace(xmin, xmax, 50)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    
    ax3.hist(spline_ts,
            bins = plot_bins,
            histtype = 'step',
            linewidth = 3,
            alpha=0.5,
            color = 'r',
            label = 'N = ' + str(flux) + ' GeV^-1 cm^-2 s^-1')
    
    ax3.vlines(unblinded_ts, 0, 1E3, color = 'k', linewidth = 3, linestyle = '--', label = 'Measured TS value')
    ax3.set_xlabel('TS')
    ax3.set_ylabel('counts')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_title(r'Flux Upper Limit for 3C454.3 flare between MJD ' + str(analysis_times['start'][i]) + ' and ' + str(analysis_times['end'][i]))
    
    fig3.savefig(outdir + '/Unblinding_plots_test/ul_flare_' + str(i) + '_test.png')
    
    np.save(outdir + '/Unblinding_limits_test/ul_flare_' + str(i) + '_ts_test.npy', spline_ts)
    
    print(str(i))
    
background_ts_analysis = np.load(outdir + '/background_profiles_stacked/background_ts_scheme_peak.npy')#using peak because I wrote the weighting function wrong
    
unblinded_values = np.load(outdir + '/Unblinding_test/stacked_unblinded_test.npy')
    
unblinded_ts = unblinded_values['ts']
    
greater_add = (background_ts_analysis >= unblinded_ts)
    
greater_than_data = background_ts_analysis[greater_add]
    
p_value_stacked = len(greater_than_data)/len(background_ts_analysis)
p_value_array = np.array([p_value_stacked])
    
np.save(outdir + '/Unblinding_limits_test/p-value_stacked_test_.npy', p_value_array)

fig4, ax4 = plt.subplots(figsize=(10,6))

full = np.concatenate([background_ts_analysis])
xmin, xmax = full.min(), full.max()
plot_bins = np.linspace(xmin, xmax, 50)
    
ax4.hist(background_ts_analysis,
            bins = plot_bins,
            histtype = 'step',
            color = 'k',
            linewidth = 3,
            alpha=0.5,
            label = 'Background TS distribution')
    
ax4.hist(greater_than_data,
            bins = plot_bins,
            histtype = 'step',
            color = 'r',
            linewidth = 3,
            alpha=0.5,
            label = 'Background above measured TS value')
    
ax4.vlines(unblinded_ts, 0, 1E3, color = 'b', linewidth = 3, linestyle = '--', label = 'measured TS value')
ax4.set_xlabel('TS')
ax4.set_ylabel('counts')
ax4.set_title('Stacked P-value = ' + str(p_value_stacked) + ' for 3C454.3 subflares')
ax4.legend()
ax4.set_yscale('log')

fig4.savefig(outdir + '/Unblinding_plots_test/p-value_stacked_test.png')

perc = np.zeros(len(N_vals))

for j in range(0, len(N_vals)):
    signal_array = np.load(outdir + '/stacked_outputs_final/analysis_job_' + str(j) + '_stacked_peak.npy') #using the peak because I wrote the weighting function wrong
    signal_ts_analysis = signal_array['ts']
    perc[j] = len(signal_ts_analysis[signal_ts_analysis > unblinded_ts])/len(signal_ts_analysis)
        
analysis_spl = interpolate.UnivariateSpline(N_vals, perc, s = 5E-4)
    
fig5, ax5 = plt.subplots(figsize=(10,6))

ax5.plot(N_vals, perc, '.', color = 'b')
ax5.plot(plot_vals, analysis_spl(plot_vals), linewidth = 2, color = 'r')
ax5.hlines(0.9, 1E-19, 1E-16)
ax5.set_xlabel('Flux normalization N (GeV^-1 cm^-2 s^-1)')
ax5.set_ylabel('Percentage of signal over measured TS value')
ax5.set_title('Spline estimate for stacked upper limit')
ax5.set_xscale('log')

fig5.savefig(outdir + '/Unblinding_plots_test/spline_stacked_unblinded_test.png')

add = (analysis_spl(plot_vals) >= 0.9)
flux = plot_vals[add][0]

flux_weights = analysis_functions.produce_flux(analysis_times, flux_spline, 2) #we're going with 2 because I wrote the weighting function wrong

args = {"gamma_points":gamma_points,
        "ratio_bins":ratio_bins,
        "sob_maps":sob_maps,
        "data":data, 
        "sim":sim,
        "bg_p_dec":bg_p_dec,
        "grl":grl,
        "analysis_times":analysis_times,
        "flux_spline":flux_spline,
        "flux_weights":flux_weights,
        "test_ns":0.1,
        "minimize":True,
        "source_ra":blazar_ra,
        "test_ra":blazar_ra,
        "source_dec":blazar_dec,
        "test_dec":blazar_dec,
        "gauss":False,
        "spl":True,
        "ncpus":4,
        "angular_window":np.radians(180)}

analysis_spline_fits = analysis_functions.produce_trials_stacked(number, N = flux, **args)

spline_ts = np.zeros(number)
for i in range(0, number):
    spline_ts[i] = sum(analysis_spline_fits[i][0])
    
full = np.concatenate([spline_ts])
xmin, xmax = full.min(), full.max()
plot_bins = np.linspace(xmin, xmax, 50)

fig6, ax6 = plt.subplots(figsize=(10,6))
    
ax6.hist(spline_ts,
            bins = plot_bins,
            histtype = 'step',
            linewidth = 3,
            alpha=0.5,
            color = 'r',
            label = 'N = ' + str(flux) + ' GeV^-1 cm^-2 s^-1')
    
ax6.vlines(unblinded_ts, 0, 1E3, color = 'k', linewidth = 3, linestyle = '--', label = 'Measured TS value')
ax6.set_xlabel('TS')
ax6.set_ylabel('counts')
ax6.legend()
ax6.set_yscale('log')
ax6.set_title(r'Flux Upper Limit for 3C454.3 Stacked Search')
    
fig6.savefig(outdir + '/Unblinding_plots_test/ul_flare_stacked_test.png')

np.save(outdir + '/Unblinding_limits_test/ul_flare_stacked_ts_test.npy', spline_ts)