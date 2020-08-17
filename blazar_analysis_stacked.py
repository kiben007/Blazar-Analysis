#!/usr/bin/env python

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
import timeit

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

with open(outdir + '/prereqs/bg_p_dec.pkl', 'rb') as f:
    bg_p_dec = pickle.load(f)

seed = None

gamma_points = np.load(outdir + '/prereqs/gamma_points.npy')
ratio_bins = np.load(outdir + '/prereqs/ratio_bins.npy')
sob_maps = np.load(outdir + '/prereqs/sob_maps.npy')

global_vars = {}

blazar_ra = 5.995042209084602
blazar_dec = 0.2818394575289712

analysis_times = np.load(outdir + '/prereqs/analysis_times.npy')

N_vals = np.logspace(-19, -16, 20)
flux_weights = analysis_functions.produce_flux(analysis_times, flux_spline, scheme)
schemes = ['uniform', 'length', 'peak', 'integrate']

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

analysis_fits = analysis_functions.produce_trials_stacked(number, N = N_vals[job], **args)
ts = np.zeros(number)
for i in range(0, number):
    ts[i] = sum(analysis_fits[i][0])
ntot = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][1]
    ntot[i] = list
ntot = np.concatenate(ntot)
ninj = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][2]
    ninj[i] = list
ninj = np.concatenate(ninj)
ns = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][3]
    ns[i] = list
ns = np.concatenate(ns)
gamma = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][4]
    gamma[i] = list
gamma = np.concatenate(gamma)
t_start = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][5]
    t_start[i] = list
t_start = np.concatenate(t_start)
t_end = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][6]
    t_end[i] = list
t_end = np.concatenate(t_end)
t_offset = [i for i in range(len(analysis_fits))]
for i in range(0, len(analysis_fits)):
    list = analysis_fits[i][7]
    t_offset[i] = list
t_offset = np.concatenate(t_offset)
np.save(outdir + '/stacked_outputs_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '.npy', ts)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_ntot.npy', ntot)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_ninj.npy', ninj)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_ns.npy', ns)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_gamma.npy', gamma)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_t_start.npy', t_start)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_t_end.npy', t_end)
np.save(outdir + '/stacked_parameters_final/analysis_job_' + str(job) + '_stacked_' + schemes[scheme] + '_t_offset.npy', t_offset)