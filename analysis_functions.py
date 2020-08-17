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

def read(filelist):
    # Reads data in
    data = []
    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0:
            data = x.copy()
        else:
            data = rf.stack_arrays([data, x])
    return data

def reduce_dataset(events, fraction=0.1):
    # Reduces fraction of events in dataset
    # Useful for speeding up computations if slow
    assert(fraction > 0)
    if fraction >= 1:
        return events.copy()
    N = len(events)
    n_keep = int(N*fraction)
    new_events = np.random.choice(events, n_keep, replace=False).copy()
    if 'ow' in events.dtype.names:
        new_events['ow'] /= fraction
    return new_events


def to_unit_vector(ra, dec):
    # These are some functions for converting ra/dec info into angular distance
    return np.array([np.cos(ra)*np.cos(dec),
                     np.sin(ra)*np.cos(dec),
                     np.sin(dec)])

def angular_distance(ra_A, dec_A, ra_B, dec_B):
    # These are some functions for converting ra/dec info into angular distance
    unit_A = to_unit_vector(ra_A, dec_A)
    unit_B = to_unit_vector(ra_B, dec_B)
    
    if len(unit_A.shape) != 1:
        return np.arccos(np.dot(unit_A.T, unit_B))
    else:
        return np.arccos(np.dot(unit_A, unit_B))

def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    # These are some functions for converting ra/dec info into angular distance
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).

    """
    def cross_matrix(x):
        r"""Calculate cross product matrix

        A[ij] = x_i * y_j - y_i * x_j

        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
    )

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
        + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1. - np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec

class generic_profile(object):
    r""" A generic base class to standardize the methods for the
    time profiles. While I'm only currently using scipy-based 
    probability distributions, you can write your own if you
    want. Just be sure to define these methods and ensure that
    # the PDF is normalized!"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,): pass
    
    @abc.abstractmethod
    def pdf(self, times): pass

    @abc.abstractmethod
    def logpdf(self, times): pass
    
    @abc.abstractmethod
    def random(self, n): pass

    @abc.abstractmethod
    def effective_exposure(self, times): pass
    
    @abc.abstractmethod
    def get_range(self): pass
    

class uniform_profile(generic_profile):
    r"""Time profile class for a uniform distribution. Use this
    for background or if you want to assume a steady signal from
    your source.
    """
    def __init__(self, start_time, end_time):
        r""" Constructor for the class.
        
        args:
            start_time, end_time: The bounds for the uniform 
                            distribution.
        """
        assert(end_time > start_time)
        self.start_time = start_time
        self.end_time = end_time
        self.norm = 1.0/(end_time-start_time)
        return
    
    def pdf(self, times):
        r""" Calculates the probability for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        output = np.zeros_like(times)
        output[(times>=self.start_time) &\
               (times<self.end_time)] = self.norm
        return output
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return np.log(self.pdf(times))
    
    def random(self, n=1): 
        r""" Return random values following the uniform distribution
        
        args:
            n: The number of random values to return
        """
        return np.random.uniform(self.start_time,
                                 self.end_time,
                                 n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [self.start_time, self.end_time]
    

class gauss_profile(generic_profile):
    r"""Time profile class for a gaussian distribution. Use this
    to produce gaussian-distributed times for your source.
    """
    def __init__(self, mean, sigma):
        r""" Constructor for the class.

        args:
            mean: The center form the distribution
            sigma: The width for the distribution
        """
        self.mean = mean
        self.sigma = sigma
        self.scipy_dist = scipy.stats.norm(mean, sigma)
        self.norm = 1.0/np.sqrt(2*np.pi*sigma**2)
        return
    
    def pdf(self, times):
        r""" Calculates the probability for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return self.scipy_dist.pdf(times)
    
    def logpdf(self, times):
        r""" Calculates the log(probability) for each time
            
        args:
            times: A numpy list of times to evaluate
        """
        return self.scipy_dist.logpdf(times)
        
    def random(self, n=1): 
        r""" Return random values following the gaussian distribution
        
        args:
            n: The number of random values to return
        """
        return self.scipy_dist.rvs(size=n)
    
    def effective_exposure(self): 
        r""" Calculate the weight associated with each
            event time. 
        """
        return 1.0/self.norm
    
    def get_range(self): 
        r""" Return the min/max values for the distribution 
        """
        return [-np.inf, np.inf]
    

class spline_profile(generic_profile):
    # Class for spline profile
    
    def __init__(self, spline, start, end):
        # Takes as parameters the univariate light-curve spline, and the start/end times of the flare
        self.spline = spline
        self.start_time = start
        self.end_time = end
        
        # Define a list of times between the start and end times for random sampling
        self.time_list = np.linspace(self.start_time, self.end_time, 100000)
        
        # These next few parameters are necesasry for calculating the effective exposure
        self.roots = self.spline.derivative().roots()
        self.roots = np.append(self.roots, [self.start_time])
        self.roots = np.append(self.roots, [self.end_time])
        index = (self.roots >= self.start_time)
        index &= (self.roots <= self.end_time)
        self.roots = self.roots[index]
        self.vals = self.spline(self.roots)
        self.norm = self.spline.integral(self.start_time, self.end_time)
        self.thing = np.max(self.vals)/self.norm
        
        # Define some parameters useful for random sampling
        self.hist = np.cumsum(self.pdf(self.time_list))
        self.hist /= self.hist.max()
        self.sample_spline = scipy.interpolate.UnivariateSpline(self.hist, self.time_list, k = 1)
        
        # This last parameter is used in some of the stacked weighting schemes
        self.max = np.max(self.vals)
        
    def pdf(self, times):
        # Calculates probability density for each time
        p = self.spline(times)/self.norm
        p[times<self.start_time] = 0
        p[times>self.end_time] = 0
        return p
        
    def logpdf(self, times):
        # Calculates log of the pdf for each time
        return np.log(self.pdf(times))
    
    def random(self, n):
        # randomly samples n times from spline
        self.uniform = np.random.uniform(0, 1, n)
        return self.sample_spline(self.uniform)
    
    def effective_exposure(self):
        # Return the effective exposure
        return 1.0/self.thing
    
    def get_range(self):
        # Return the start/end time of the flare
        return [self.start_time, self.end_time]

def select_and_weight(sim,
                      N=0,
                      gamma=-2,
                      source_ra = np.pi/2,
                      source_dec = np.pi/6,
                      time_profile = None,
                      sampling_width = np.radians(1),
                     ):
    # This weights the MC simulation based on a particular flux N and spectral index gamma
    assert('ow' in sim.dtype.names)
    assert(time_profile != None)

    # Did we already run this?
    if "weight" in sim.dtype.names:
        return sim
    
    # Pick out only those events that are close in
    # declination. We only want to sample from those.
    sindec_dist = np.abs(source_dec-sim['trueDec'])
    close = sindec_dist < sampling_width

    reduced_sim = rf.append_fields(sim[close].copy(),
                                   'weight',
                                   np.zeros(close.sum()),
                                   dtypes=np.float64)

    # Assign the weights using the newly defined "time profile"
    # classes above. If you want to make this a more complicated 
    # shape, talk to me and we can work it out.
    effective_livetime = time_profile.effective_exposure()
    reduced_sim['weight'] = reduced_sim['ow'] *\
                    N * (reduced_sim['trueE']/100.e3)**gamma *\
                    effective_livetime * 24 * 3600.

    # Apply the sampling width, which ensures that we
    # sample events from similar declinations. 
    # When we do this, correct for the solid angle 
    # we're including for sampling
    omega = 2*np.pi * (np.min([np.sin(source_dec+sampling_width), 1]) -\
                       np.max([np.sin(source_dec-sampling_width), -1]))
    reduced_sim['weight'] /= omega
    return reduced_sim

def produce_trial(data,
                  sim,
                  grl,
                  N=0,
                  gamma=-2,
                  source_ra = np.pi/2,
                  source_dec = np.pi/6,
                  signal_time_profile = None,
                  background_time_profile = None,
                  sampling_width = np.radians(1),
                  background_window = 14, # days
                  random_seed = None,
                  signal_weights = None,
                  return_signal_weights = False):
    #This produces trial datasets
    assert(background_window > 0)
    assert(signal_time_profile != None)
    assert(background_time_profile != None)

    if random_seed != None: np.random.seed(random_seed)
                
    # Start by calculating the background rate. For this, we'll
    # look at the number of events observed in runs just before
    # our start_time. We're picking this to exclude the start time, 
    # since we don't want to include our hypothesized signal in 
    # our background estimates
    start_time = background_time_profile.get_range()[0]
    fully_contained = (grl['start'] >= start_time-background_window) &\
                        (grl['stop'] < start_time)
    start_contained = (grl['start'] < start_time-background_window) &\
                        (grl['stop'] > start_time-background_window)

    background_runs = (fully_contained | start_contained)
    if not np.any(background_runs):
        print("ERROR: No runs found in GRL for calculation of "
              "background rates!")
        raise RuntimeError
    background_grl = grl[background_runs]
        
    # Get the number of events we see from these runs and scale 
    # it to the number we expect for our search livetime.
    n_background = background_grl['events'].sum()
    n_background /= background_grl['livetime'].sum()
    n_background *= background_time_profile.effective_exposure()
    n_background_observed = np.random.poisson(n_background)
    
    # How many events should we add in? This will now be based on the
    # total number of events actually observed during these runs
    background = np.random.choice(data, n_background_observed).copy()
    
    # Assign times to our background events 
    background['time'] = background_time_profile.random(len(background))
    
    # Randomize the background RA
    background['ra'] = np.random.uniform(0, 2*np.pi, len(background))
            
    # Do we want signal events?
    reduced_sim = sim
    if N > 0:
        reduced_sim = select_and_weight(N=N,
                                        gamma=gamma,
                                        source_ra = source_ra,
                                        source_dec = source_dec,
                                        time_profile = signal_time_profile,
                                        sim = sim,
                                        sampling_width = sampling_width,
                                        )

        # Pick the signal events
        total = reduced_sim['weight'].sum()
        n_signal_observed = scipy.stats.poisson.rvs(total)
        signal = np.random.choice(reduced_sim, n_signal_observed,
                                  p = reduced_sim['weight']/total,
                                  replace = False).copy()
        
        # Assign times to the signal using our time_profile class
        signal['time'] = signal_time_profile.random(len(signal))
        
        # And cut any times outside of the background range.
        bgrange = background_time_profile.get_range()
        contained_in_background = ((signal['time'] >= bgrange[0]) &\
                                   (signal['time'] < bgrange[1]))
        signal = signal[contained_in_background]
        
        # Update this number
        n_signal_observed = len(signal)
        
        if n_signal_observed > 0:
            ones = np.ones_like(signal['trueRa'])

            signal['ra'], signal['dec'] = rotate(signal['trueRa'], 
                                                 signal['trueDec'], 
                                                 ones*source_ra, 
                                                 ones*source_dec, 
                                                 signal['ra'], 
                                                 signal['dec'])
            signal['trueRa'], signal['trueDec'] = rotate(signal['trueRa'], 
                                                        signal['trueDec'], 
                                                        ones*source_ra, 
                                                        ones*source_dec, 
                                                        signal['trueRa'], 
                                                        signal['trueDec'])
                    
        # Because we want to return the entire event and not just the
        # number of events, we need to do some numpy magic. Specifically,
        # we need to remove the fields in the simulated events that are
        # not present in the data events. These include the true direction,
        # energy, and 'oneweight'.
        signal = rf.drop_fields(signal, [n for n in signal.dtype.names \
                                         if not n in background.dtype.names])
    else:
        signal = np.empty(0, dtype=background.dtype)
        signal_weights = None
        
    # Combine the signal background events and time-sort them.
    events = np.concatenate([background, signal])
    sorting_indices = np.argsort(events['time'])
    events = events[sorting_indices]
  
    # We need to check to ensure that every event is contained within
    # a good run. If the event happened when we had deadtime (when we
    # were not taking data), then we need to remove it.
    during_uptime = [np.any((grl['start'] <= t) & (grl['stop'] > t)) \
                        for t in events['time']]
    during_uptime = np.array(during_uptime, dtype=bool)
    events = events[during_uptime]

    if return_signal_weights:
        return events, reduced_sim
    else:
        return events

def signal_pdf(event,
               test_ra, 
               test_dec):
    # Defines the spatial signal pdf for eah event
    sigma = event['angErr']
    x = angular_distance(event['ra'], event['dec'], 
                         test_ra, test_dec)
    return (1.0/(2*np.pi*sigma**2))*np.exp(-x**2/(2*sigma**2))

def evaluate_interpolated_ratio(events,
                                bins,
                                ratio):
    # Get the bin that each event belongs to
    i = np.searchsorted(bins[0], np.sin(events['dec'])) - 1
    j = np.searchsorted(bins[1], events['logE']) - 1
    
    return ratio[i,j]

def get_energy_splines(events,
                       gamma_points,
                       ratio_bins,
                       sob_maps,
                      ):
    # Get the values for each event
    sob_ratios = evaluate_interpolated_ratio(events, 
                                            bins = ratio_bins,
                                            ratio = sob_maps)

    # These are just values at this point. We need
    # to interpolate them in order for this to give
    # us reasonable values. Let's spline them in log-space
    sob_splines = np.zeros(len(events), dtype=object)
    for i in range(len(events)):
        spline = scipy.interpolate.UnivariateSpline(gamma_points,
                                                    np.log(sob_ratios[i]),
                                                    k = 3,
                                                    s = 0,
                                                    ext = 'raise')
        sob_splines[i] = spline
    return sob_splines

def get_energy_sob(events,
                   gamma,
                   splines):
    # Given energy splines, returns signal-over-background
    # energy pdf for each event
    final_sob_ratios = np.ones_like(events, dtype=float)
    for i, spline in enumerate(splines):
        final_sob_ratios[i] = np.exp(spline(gamma))

    return final_sob_ratios

def background_pdf(event,
                   test_ra, 
                   test_dec,
                   bg_p_dec):
    # Calculates background spatial pdf
    background_likelihood = (1/(2*np.pi))*bg_p_dec(np.sin(event['dec']))
    return background_likelihood

def evaluate_ts_uniform(events,
                test_ra,
                test_dec,
                background_time_profile,
                signal_time_profile,
                gamma_points,
                ratio_bins,
                sob_maps,
                bg_p_dec,
                ns = 0,
                gamma = -2,
                tw = 100000.0/(24*3600.0),
                t0 = 56102.49,
                minimize = True):
    
    # structure to store our output
    output = {'ts':0,
                'ns':ns,
                'gamma':gamma,
                't_start':t0 - tw/2,
                't_end':t0 + tw/2}
    
    # If no flux, do nothing
    N = len(events)
    if N==0: 
        return output
    
    # Check: ns cannot be larger than N.
    if ns >= N: 
        ns = N - 0.00001

    # Define signal and background pdfs
    S = signal_pdf(events, test_ra, test_dec)
    B = background_pdf(events, test_ra, test_dec, bg_p_dec)
    keep = (S > 0)
    events = events[keep]
    S = S[keep]
    B = B[keep]
    
    #Define background time pdf from background time profile
    t_lh_background = background_time_profile.pdf(events['time'])
    
    #Get energy splines
    splines = get_energy_splines(events, gamma_points = gamma_points, ratio_bins = ratio_bins, sob_maps = sob_maps)
    
    #def constraint(x):
        #return (x[2] <= x[1]).astype(float)
    
    #Get range of background times
    T = background_time_profile.effective_exposure()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # In principle, we'd want to throw everything into
        # the minimizer at once and let it work out whatever
        # correlations there are. In practice, the ns parameter
        # isn't correlated with other parameters, so it's 
        # actually more efficient to fit it separately after
        # the other parameters are found. Doing it this way
        # speeds things up by a factor of ~2.5x
        def get_inner_ts(params):
            # Minimize parameters other than ns
            gamma, param1, param2 = params
            param1 += background_time_profile.start_time
            param2 += background_time_profile.start_time
            e_lh_ratio = get_energy_sob(events, gamma, splines)
            if param1 >= param2:
                return 100000
            if ~np.isfinite(param1) or ~np.isfinite(param2):
                return 100000
            t_lh_signal = uniform_profile(param1, param2).pdf(events['time'])
            tw = param2 - param1
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            ts = (1/N*(sob - 1))+1
            ts = ts[ts != 0]
            if len(ts) == 0:
                return 100000
            return -2*np.sum(np.log(T/tw * ts))

        # This one will only fit ns
        def get_ts(ns, params):
            sob = params[0]
            t = params[1]
            ts = (ns/N*(sob - 1))+1
            return -2*np.sum(np.log(ts))#t * ts))
        
        if minimize:
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters (just gamma, in this case).
            t0 -= background_time_profile.start_time
            x0 = [gamma, t0 - tw/2, t0 + tw/2,]
            bounds = [[-4, -1], #gamma [min, max]
                        [0, T], #t_start [min, max]
                        [0, T], #t_end [min, max]
                        ]
            #constraint_dict = {"type":'eq', "fun":constraint}
            bf_params = scipy.optimize.minimize(get_inner_ts,
                                                x0 = x0,
                                                bounds = bounds,
                                                method = 'L-BFGS-B')

            # and now set up the fit for ns
            x0 = [ns,]
            bounds = [[0, N],] 
            e_lh_ratio = get_energy_sob(events, bf_params.x[0], splines)
            t_lh_signal = uniform_profile(bf_params.x[1] + background_time_profile.start_time, 
                                          bf_params.x[2] + background_time_profile.start_time).pdf(events['time'])
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            t = (bf_params.x[2] - bf_params.x[1])/T

            result = scipy.optimize.minimize(get_ts,
                                             x0 = x0,
                                             args = [sob, t],
                                             bounds = bounds,
                                             method = 'L-BFGS-B',
                                            )
            
            # Store the results in the output array
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = bf_params.x[0]
            output['t_start'] = bf_params.x[1] + background_time_profile.start_time
            output['t_end'] = bf_params.x[2] + background_time_profile.start_time

        else:
            #fix this later
            e_lh_ratio = get_energy_sob(events, gamma, splines)
            t_lh_signal = uniform_profile(t0 - tw/2, t0 + tw/2).pdf(events['time'])
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            output['ts'] = -1*get_ts(ns, [sob, tw/T])
            output['ns'] = ns
            output['gamma'] = gamma
            output['t_start'] = t0 - tw/2
            output['t_end'] = t0 + tw/2

        return output 

def evaluate_ts_gauss(events,
                test_ra,
                test_dec,
                background_time_profile,
                signal_time_profile,
                gamma_points,
                ratio_bins,
                sob_maps,
                bg_p_dec,
                ns = 0,
                gamma = -2,
                minimize = True):
    
    # structure to store our output
    output = {'ts':0,
                'ns':ns,
                'gamma':gamma,
                't_mean':0,
                't_sigma':0}
        
    N = len(events)
    if N==0: 
        return output
    
    # Check: ns cannot be larger than N.
    if ns >= N: 
        ns = N - 0.00001

    S = signal_pdf(events, test_ra, test_dec)
    B = background_pdf(events, test_ra, test_dec, bg_p_dec)
    
    t_lh_background = background_time_profile.pdf(events['time'])
    
    splines = get_energy_splines(events, gamma_points = gamma_points, ratio_bins = ratio_bins, sob_maps = sob_maps)
    
    def constraint(x):
        return np.atleast_1d(x[2] - x[1])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # In principle, we'd want to throw everything into
        # the minimizer at once and let it work out whatever
        # correlations there are. In practice, the ns parameter
        # isn't correlated with other parameters, so it's 
        # actually more efficient to fit it separately after
        # the other parameters are found. Doing it this way
        # speeds things up by a factor of ~2.5x
        def get_inner_ts(params):
            gamma, param1, param2 = params
            e_lh_ratio = get_energy_sob(events, gamma, splines)
            t_lh_signal = gauss_profile(param1, param2).pdf(events['time'])
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            ts = (1/N*(sob - 1))+1
            ts = ts[ts != 0]
            if len(ts) == 0:
                return 100000
            # Adjust ts calculation for Tw/T
            return -2*np.sum(np.log(ts))

        # This one will only fit ns
        def get_ts(ns, params):
            sob = params[0]
            ts = (ns/N*(sob - 1))+1
            return -2*np.sum(np.log(ts))
        
        if minimize:
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters (just gamma, in this case).
            x0 = [gamma, np.average(events['time']), np.std(events['time']),]
            bounds = [[-4, -1], #gamma [min, max]
                        background_time_profile.get_range(), #mean [min, max]
                        [0, background_time_profile.effective_exposure()], #sigma [min, max]
                        ]
            constraint_dict = ()
            bf_params = scipy.optimize.minimize(get_inner_ts,
                                                x0 = x0,
                                                bounds = bounds,
                                                method = 'SLSQP',
                                                constraints = constraint_dict)

            # and now set up the fit for ns
            x0 = [ns,]
            bounds = [[0, N],] 
            e_lh_ratio = get_energy_sob(events, bf_params.x[0], splines)
            t_lh_signal = gauss_profile(bf_params.x[1], bf_params.x[2]).pdf(events['time'])
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)

            result = scipy.optimize.minimize(get_ts,
                                             x0 = x0,
                                             args = [sob,],
                                             bounds = bounds,
                                             method = 'L-BFGS-B',
                                            )
            
            # Store the results in the output array
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = bf_params.x[0]
            output['t_mean'] = bf_params.x[1]
            output['t_sigma'] = bf_params.x[2]

        else:
            output['ts'] = -1*get_ts(ns, [gamma,])
            output['ns'] = ns
            output['gamma'] = gamma
            #fix this later
            output['t_mean'] = 0
            output['t_sigma'] = 0

        return output

def evaluate_ts_spline(events,
                test_ra,
                test_dec,
                background_time_profile,
                signal_time_profile,
                gamma_points,
                ratio_bins,
                sob_maps,
                bg_p_dec,
                ns = 0,
                gamma = -2,
                offset = 0,
                minimize = True):
    
    assert(signal_time_profile is not None)
    
    # structure to store our output
    output = {'ts':0,
                'ns':ns,
                'gamma':gamma,
                't_start':signal_time_profile.start_time,
                't_end':signal_time_profile.end_time,
                't_offset':offset}
        
    # If no events, don't bother minimizing
    N = len(events)
    if N==0: 
        return output
    
    # Check: ns cannot be larger than N.
    if ns >= N: 
        ns = N - 0.00001

    # Define your signal and background spatial pdfs
    S = signal_pdf(events, test_ra, test_dec)
    B = background_pdf(events, test_ra, test_dec, bg_p_dec)
    keep = (S > 0)
    events = events[keep]
    S = S[keep]
    B = B[keep]
    
    # Define background time pdf
    t_lh_background = background_time_profile.pdf(events['time'])
    
    # Define signal-over-background energy splines
    splines = get_energy_splines(events, gamma_points = gamma_points, ratio_bins = ratio_bins, sob_maps = sob_maps)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # In principle, we'd want to throw everything into
        # the minimizer at once and let it work out whatever
        # correlations there are. In practice, the ns parameter
        # isn't correlated with other parameters, so it's 
        # actually more efficient to fit it separately after
        # the other parameters are found. Doing it this way
        # speeds things up by a factor of ~2.5x
        def get_inner_ts(params):
            # Minimize secondary parameters (gamma, offset)
            gamma = params[0]
            e_lh_ratio = get_energy_sob(events, gamma, splines) # signal-over-background energy pdf
            t_lh_signal = signal_time_profile.pdf(events['time'] + params[1]) # signal time pdf
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            ts = (1/N*(sob - 1))+1
            ts = ts[ts != 0] # eliminate zeroes from ts values
            if len(ts) == 0:
                #If no ts values remaining, penalize heavily in minimizer
                return 100000
            return -2*np.sum(np.log(ts))

        # This one will only fit ns
        def get_ts(ns, params):
            sob = params[0]
            ts = (ns/N*(sob - 1))+1
            return -2*np.sum(np.log(ts))
        
        if minimize:
            # Set the seed values, which tell the minimizer
            # where to start, and the bounds. First do the
            # shape parameters
            t_range = signal_time_profile.get_range()[1] - signal_time_profile.get_range()[0]
            x0 = [gamma, offset] # initial values
            bounds = [[-4, -1], # gamma [min, max]
                      [-2 * t_range, 2 * t_range]] # offset [min, max]
            bf_params = scipy.optimize.minimize(get_inner_ts,
                                                x0 = x0,
                                                bounds = bounds,
                                                method = 'L-BFGS-B')

            # and now set up the fit for ns
            x0 = [ns,]
            bounds = [[0, N],] 
            e_lh_ratio = get_energy_sob(events, bf_params.x[0], splines) # energy likelihood for minimized parameters
            t_lh_signal = signal_time_profile.pdf(events['time'] + bf_params.x[1]) #time likelihood for minimized parameters
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)

            result = scipy.optimize.minimize(get_ts,
                                             x0 = x0,
                                             args = [sob,],
                                             bounds = bounds,
                                             method = 'L-BFGS-B',
                                            )
            
            # Store the results in the output array
            output['ts'] = -1*result.fun
            output['ns'] = result.x[0]
            output['gamma'] = bf_params.x[0]
            output['t_start'] = signal_time_profile.start_time
            output['t_end'] = signal_time_profile.end_time
            output['t_offset'] = bf_params.x[1]

        else:
            #fix this later
            e_lh_ratio = get_energy_sob(events, gamma, splines)
            sob = S/B*e_lh_ratio * (t_lh_signal/t_lh_background)
            output['ts'] = -1*get_ts(ns, [sob,])
            output['ns'] = ns
            output['gamma'] = gamma
            output['t_start'] = signal_time_profile.start_time
            output['t_end'] = signal_time_profile.end_time
            output['t_offset'] = offset

        return output
    
def produce_trials_multi(ntrials,
                        
                     # Time profiles for signal/background generation
                     background_time_profile,
                     signal_time_profile,
                     
                     gamma_points,
                     ratio_bins,
                     sob_maps,
                     data, 
                     sim,
                     bg_p_dec,
                     grl,
                     
                     # Signal flux parameters
                     N = 0, 
                     gamma = -2,
                     t0 = 56102.5,
                     tw = 1000.0/(3600.0 * 24),
                     offset = 0,
                     source_ra = np.pi/2, 
                     source_dec = np.pi/6, 
                     sampling_width = np.radians(5),
                     angular_window = np.radians(10),
                     reduced_sim = None,

                     # Estimate the background rate over this many days
                     background_window = 21, 
                     
                     # Parameters to control where/when you look
                     test_ns = 1,
                     test_gamma = -2,
                     test_ra = np.pi/2,
                     test_dec = np.pi/6, 
                     
                     minimize = True,
                     gauss = True,
                     spl = False,
                     random_seed = None,
                     verbose = True,
                     ncpus = 4):
      
    if random_seed:
        np.random.seed(random_seed)

    if background_window < 1:
        print("WARN: Your window for estimating the backgroud rate is"
              " {} and is less than 1 day. You may run into large"
              " statistical uncertainties on the background rates which"
              " may lead to unreliable trials. Increase your" 
              " background_window to at least a day or a week to reduce"
              " these issues.")

    if background_time_profile.effective_exposure() > background_window:
        print("WARN: Going to estimate the background from a window"
              " of {} days, but producing a trial of {} days. Upscaling"
              " can be a bit dangerous, since you run the risk of missing"
              " impacts from seasonal fluctuations. Just keep it in mind"
              " as you run.")
        
    # Cut down the sim. We're going to be using the same
    # source and weights each time, so this stops us from
    # needing to recalculate over and over again.
    if reduced_sim is None:
        reduced_sim = select_and_weight(N = N,
                                        gamma = gamma,
                                        source_ra = source_ra,
                                        source_dec = source_dec,
                                        time_profile = signal_time_profile,
                                        sim = sim,
                                        sampling_width = sampling_width,
                                        )
    
    else:
        reduced_sim = select_and_weight(N = N,
                                        gamma = gamma,
                                        source_ra = source_ra,
                                        source_dec = source_dec,
                                        time_profile = signal_time_profile,
                                        sim = reduced_sim,
                                        sampling_width = sampling_width,
                                        )
    
    # Build a place to store information for the trial
    if gauss:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int), 
                          ('ns', np.float64), 
                          ('gamma', np.float64),
                          ('t_mean', np.float64),
                          ('t_sigma', np.float64)])
    elif spl:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int),
                          ('ns', np.float64),
                          ('gamma', np.float64),
                          ('t_start', np.float64),
                          ('t_end', np.float64),
                          ('t_offset', np.float64)])
    else:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int), 
                          ('ns', np.float64), 
                          ('gamma', np.float64),
                          ('t_start', np.float64),
                          ('t_end', np.float64)])

    # We're going to cache the signal weights, which will
    # speed up our signal generation significantly. 
    signal_weights = None
    
    global global_vars
    global_vars = {}
    global_vars.clear()
    global_vars.update(locals())
    
    with multiprocessing.Pool(ncpus) as pool:
        results = list(pool.imap(complete_trial, range(ntrials)))
        fit_info = np.array(results, dtype = dtype)
        return fit_info

def complete_trial(ntrials):
    
    N = global_vars['N']
    gamma = global_vars['gamma']
    background_time_profile = global_vars['background_time_profile']
    signal_time_profile = global_vars['signal_time_profile']
    t0 = global_vars['t0']
    tw = global_vars['tw']
    source_ra = global_vars['source_ra']
    source_dec = global_vars['source_dec']
    sampling_width = global_vars['sampling_width']
    background_window = global_vars['background_window']
    test_ns = global_vars['test_ns']
    test_gamma = global_vars['test_gamma']
    test_ra = global_vars['test_ra']
    test_dec = global_vars['test_dec']
    data = global_vars['data'] 
    sim = global_vars['sim']
    grl = global_vars['grl']
    gamma_points = global_vars['gamma_points']
    ratio_bins = global_vars['ratio_bins']
    sob_maps = global_vars['sob_maps']
    minimize = global_vars['minimize']
    gauss = global_vars['gauss']
    spl = global_vars['spl']
    verbose = global_vars['verbose']
    ncpus = global_vars['ncpus']
    reduced_sim = global_vars['reduced_sim']
    angular_window = global_vars['angular_window']
    bg_p_dec = global_vars['bg_p_dec']
    offset = global_vars['offset']
    
    fit_info = []
    np.random.seed()
    
    # Produce a trial
    trial = produce_trial(data,
                          reduced_sim,
                          grl,
                          N = N, 
                          gamma = gamma, 
                          source_ra = source_ra, 
                          source_dec = source_dec, 
                          background_time_profile = background_time_profile,
                          signal_time_profile = signal_time_profile,
                          sampling_width = sampling_width,
                          background_window = background_window,
                          random_seed = None,
                          signal_weights = None, 
                          return_signal_weights = False)
    
    # Cut trial to certain angular window
    keep = (np.abs(trial['ra'] - test_ra) <  angular_window) & (np.abs(trial['dec'] - test_dec) < angular_window)
    trial = trial[keep]
    
    # And get the weights
    if gauss:
        bestfit = evaluate_ts_gauss(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  minimize = minimize,
                                  )
    elif spl:
        bestfit = evaluate_ts_spline(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  offset = offset,
                                  minimize = minimize
                                  )
    else:
        bestfit = evaluate_ts_uniform(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  t0 = t0,
                                  tw = tw,
                                  minimize = minimize
                                  )
    if gauss:
        return bestfit['ts'], len(trial), (trial['run']>200000).sum(), bestfit['ns'], bestfit['gamma'], bestfit['t_mean'], bestfit['t_sigma']
    elif spl:
        return bestfit['ts'], len(trial), (trial['run']>200000).sum(), bestfit['ns'], bestfit['gamma'], bestfit['t_start'], bestfit['t_end'], bestfit['t_offset']
    else:
        return bestfit['ts'], len(trial), (trial['run']>200000).sum(), bestfit['ns'], bestfit['gamma'], bestfit['t_start'], bestfit['t_end']

def produce_flux(analysis_times, flux_spline, scheme):
    # Obtain weights for stacked fluxes
    fluxes = np.zeros(len(analysis_times))
    if scheme == 0: #uniform weighting
        for i in range(0, len(analysis_times)):
            fluxes[i] = 1/len(analysis_times)
    elif scheme == 1: #length weighting
        for i in range(0, len(analysis_times)):
            sig = spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])
            flux = sig.get_range()[1] = sig.get_range()[0]
            fluxes[i] = flux
    elif scheme == 3: #peak flux weighting
        for i in range(0, len(analysis_times)):
            sig = spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])
            flux = sig.max
            fluxes[i] = flux
    else: #time-integrated flux weighting
        for i in range(0, len(analysis_times)):
            sig = spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])
            flux = sig.norm
            fluxes[i] = flux
    total_flux = np.sum(fluxes)
    fluxes /= total_flux
    return fluxes
    
def produce_trials_stacked(ntrials,
                     
                     gamma_points,
                     ratio_bins,
                     sob_maps,
                     data, 
                     sim,
                     bg_p_dec,
                     grl,
                     analysis_times,
                     flux_spline,
                     flux_weights,
                     
                     # Signal flux parameters
                     N = 0, 
                     gamma = -2,
                     t0 = 56102.5,
                     tw = 1000.0/(3600.0 * 24),
                     offset = 0,
                     source_ra = np.pi/2, 
                     source_dec = np.pi/6, 
                     sampling_width = np.radians(5),
                     angular_window = np.radians(10),
                     reduced_sim = None,

                     # Estimate the background rate over this many days
                     background_window = 21, 
                     
                     # Parameters to control where/when you look
                     test_ns = 1,
                     test_gamma = -2,
                     test_ra = np.pi/2,
                     test_dec = np.pi/6, 
                     
                     minimize = True,
                     gauss = True,
                     spl = False,
                     random_seed = None,
                     verbose = True,
                     ncpus = 4,
                     nflares = 9):
    
    # Does Stacked trials instead of unstacked trials
    if random_seed:
        np.random.seed(random_seed)

    if background_window < 1:
        print("WARN: Your window for estimating the backgroud rate is"
              " {} and is less than 1 day. You may run into large"
              " statistical uncertainties on the background rates which"
              " may lead to unreliable trials. Increase your" 
              " background_window to at least a day or a week to reduce"
              " these issues.")
    
    # Build a place to store information for the trial
    if gauss:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int), 
                          ('ns', np.float64), 
                          ('gamma', np.float64),
                          ('t_mean', np.float64),
                          ('t_sigma', np.float64)])
    elif spl:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int),
                          ('ns', np.float64),
                          ('gamma', np.float64),
                          ('t_start', np.float64),
                          ('t_end', np.float64),
                          ('t_offset', np.float64)])
    else:
        dtype = np.dtype([('ts', np.float64),
                          ('ntot', np.int),
                          ('ninj', np.int), 
                          ('ns', np.float64), 
                          ('gamma', np.float64),
                          ('t_start', np.float64),
                          ('t_end', np.float64)])

    # We're going to cache the signal weights, which will
    # speed up our signal generation significantly. 
    signal_weights = None
    
    global global_vars
    global_vars = {}
    global_vars.clear()
    global_vars.update(locals())
    
    with multiprocessing.Pool(ncpus) as pool:
        results = list(pool.imap(stacked_trial, range(ntrials)))
        print(results)
        return results
    
def stacked_trial(ntrials):
    
    N = global_vars['N']
    gamma = global_vars['gamma']
    t0 = global_vars['t0']
    tw = global_vars['tw']
    source_ra = global_vars['source_ra']
    source_dec = global_vars['source_dec']
    sampling_width = global_vars['sampling_width']
    background_window = global_vars['background_window']
    test_ns = global_vars['test_ns']
    test_gamma = global_vars['test_gamma']
    test_ra = global_vars['test_ra']
    test_dec = global_vars['test_dec']
    data = global_vars['data'] 
    sim = global_vars['sim']
    grl = global_vars['grl']
    gamma_points = global_vars['gamma_points']
    ratio_bins = global_vars['ratio_bins']
    sob_maps = global_vars['sob_maps']
    minimize = global_vars['minimize']
    gauss = global_vars['gauss']
    spl = global_vars['spl']
    verbose = global_vars['verbose']
    ncpus = global_vars['ncpus']
    reduced_sim = global_vars['reduced_sim']
    angular_window = global_vars['angular_window']
    bg_p_dec = global_vars['bg_p_dec']
    offset = global_vars['offset']
    nflares = global_vars['nflares']
    dtype = global_vars['dtype']
    analysis_times = global_vars['analysis_times']
    flux_spline = global_vars['flux_spline']
    flux_weights = global_vars['flux_weights']
    
    fit_info = np.zeros(nflares, dtype = dtype)
    
    for i in range(0, nflares):
    # Do this for every flare, add ts values at the end
        np.random.seed()
        
        # Define time profiles for flare
        background_time_profile = uniform_profile(analysis_times['start'][i], analysis_times['end'][i])
        signal_time_profile = spline_profile(flux_spline, analysis_times['start'][i], analysis_times['end'][i])
        
        if background_time_profile.effective_exposure() > background_window:
            print("WARN: Going to estimate the background from a window"
                  " of {} days, but producing a trial of {} days. Upscaling"
                  " can be a bit dangerous, since you run the risk of missing"
                  " impacts from seasonal fluctuations. Just keep it in mind"
                  " as you run.")
        
        # Cut down the sim. We're going to be using the same
        # source and weights each time, so this stops us from
        # needing to recalculate over and over again.
        reduced_sim = select_and_weight(N = N * flux_weights[i],
                                        gamma = gamma,
                                        source_ra = source_ra,
                                        source_dec = source_dec,
                                        time_profile = signal_time_profile,
                                        sim = sim,
                                        sampling_width = sampling_width,
                                        )
        # Produce a trial
        trial = produce_trial(data,
                          reduced_sim,
                          grl,
                          N = N * flux_weights[i], 
                          gamma = gamma, 
                          source_ra = source_ra, 
                          source_dec = source_dec, 
                          background_time_profile = background_time_profile,
                          signal_time_profile = signal_time_profile,
                          sampling_width = sampling_width,
                          background_window = background_window,
                          random_seed = None,
                          signal_weights = None, 
                          return_signal_weights = False)
    
        # Cut the trial dataset
        keep = (np.abs(trial['ra'] - test_ra) <  angular_window) & (np.abs(trial['dec'] - test_dec) < angular_window)
        trial = trial[keep]
        
        # And get the weights
        if gauss:
            bestfit = evaluate_ts_gauss(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  minimize = minimize,
                                  )
        elif spl:
            bestfit = evaluate_ts_spline(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  offset = offset,
                                  minimize = minimize
                                  )
        else:
            bestfit = evaluate_ts_uniform(trial, 
                                  test_ra, 
                                  test_dec,
                                  background_time_profile,
                                  signal_time_profile,
                                  gamma_points,
                                  ratio_bins,
                                  sob_maps,
                                  bg_p_dec,
                                  ns = test_ns,
                                  gamma = test_gamma,
                                  t0 = t0,
                                  tw = tw,
                                  minimize = minimize
                                  )
        fit_info[i]['ts'] = bestfit['ts']
        fit_info[i]['ntot'] = len(trial)
        fit_info[i]['ninj'] = (trial['run']>200000).sum()
        fit_info[i]['ns'] = bestfit['ns']
        fit_info[i]['gamma'] = bestfit['gamma']
        if gauss:
            fit_info[i]['t_mean'] = bestfit['t_mean']
            fit_info[i]['t_sigma'] = bestfit['t_sigma']
        elif spl:
            fit_info[i]['t_start'] = bestfit['t_start']
            fit_info[i]['t_end'] = bestfit['t_end']
            fit_info[i]['t_offset'] = bestfit['t_offset']
        else:
            fit_info[i]['t_start'] = bestfit['t_start']
            fit_info[i]['t_end'] = bestfit['t_end']
            
    #ts values need to be summed together outside this function
    ts = fit_info['ts']
    ntot = fit_info['ntot']
    ninj = fit_info['ninj']
    ns = fit_info['ns']
    gamma = fit_info['gamma']
    
    if gauss:
        t_mean = fit_info['t_mean']
        t_sigma = fit_info['t_sigma']
        return ts, ntot, ninj, ns, gamma, t_mean, t_sigma
    elif spl:
        t_start = fit_info['t_start']
        t_end = fit_info['t_end']
        t_offset = fit_info['t_offset']
        return ts, ntot, ninj, ns, gamma, t_start, t_end, t_offset
    else:
        t_start = fit_info['t_start']
        t_end = fit_info['t_end']
        return ts, ntot, ninj, ns, gamma, t_start, t_end       