import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, gamma
from collections import namedtuple

state_labels = ['S', 'E', 'I', 'R']
State = namedtuple('State', state_labels)
def make_state(S=0, E=0, I=0, R=0):
    return State(S=S, E=E, I=I, R=R)

def v2s(v):
    return State(*v)

def s2v(s):
    return np.array([*s])

class SEIRModel:
    def __init__(self,
                 R_0=3.3,
                 T_inc=4.6,
                 T_inf=1.4):
        self.R_0 = R_0 if callable(R_0) else (lambda _: R_0)
        self.T_inc = T_inc
        self.T_inf = T_inf
        
    def __call__(self, t, y):
        y = v2s(y)
        N = y.S + y.E + y.I + y.R
        beta = self.R_0(t) / (N * self.T_inf)
        Sd = -beta * y.S * y.I
        Ed = beta * y.S * y.I - y.E / self.T_inc
        Id = y.E / self.T_inc - y.I / self.T_inf
        Rd = y.I / self.T_inf        
        return s2v(make_state(S=Sd, E=Ed, I=Id, R=Rd))
    
class EmissionModel:
    def __init__(self,
                 f_cdr=0.044,
                 f_cfr = 0.009,
                 n_detect = 4.,
                 T_detect = 11.,
                 n_resolve=9,
                 T_resolve=8.,
                 n_death=4,
                 T_death=14.):
        self.p_symptoms = f_cdr * dg_weights(n_detect,
                                              T_detect/n_detect,
                                              int(T_detect*30))
        self.p_resolve = (1-f_cfr/f_cdr) * dg_weights(n_resolve,
                                                  T_resolve/n_resolve,
                                                  int(T_resolve*30))
        self.p_death = (f_cfr / f_cdr) * dg_weights(n_death,
                                                    T_death/n_death,
                                                    int(T_death*30))
        
    def add_observables(self, sim):
        sim['All exposed'] = sim.E + sim.I + sim.R
        sim['Daily exposed'] = sim['All exposed'].diff()
        sim['Daily exposed'].iloc[0] = 0.
        sim['Daily cases'] = dp_convolve(sim['Daily exposed'],
                                            self.p_symptoms)
        sim['Daily recoveries'] = dp_convolve(sim['Daily cases'],
                                              self.p_resolve)
        sim['Daily deaths'] = dp_convolve(sim['Daily cases'],
                                          self.p_death)
        sim['All cases'] = sim['Daily cases'].cumsum()
        sim['All recoveries'] = sim['Daily recoveries'].cumsum()
        sim['All deaths'] = sim['Daily deaths'].cumsum()
        sim['Active cases'] = (sim['All cases'] -
                               sim['All recoveries'] -
                               sim['All deaths'])
        return sim
        
def calibrate_transmission_model(early_growth_rate,
                                 mean_generation_time,
                                 lat_fraction):
    """
    Returns R_0, T_inc and T_inf for a SEIR model calibrated to
    the specified initial daily case/death growth rate, the infection
    mean generation time, and the fraction of this time that the
    infection is assumed to be latent (i.e. non-infectious, E stage).
    """
    T_inc = lat_fraction * mean_generation_time
    T_inf = mean_generation_time - T_inc
    pos_eigenvalue = np.log(1+early_growth_rate)
    R_0 = (1 + pos_eigenvalue * T_inc) * (1 + pos_eigenvalue * T_inf)
    return {'R_0':R_0, 'T_inc': T_inc, 'T_inf': T_inf}
    
def run_outbreak(transmission_model,
                 emission_model,
                 sim_time_days, initial_state):
    sim_time_days = int(sim_time_days)
    t_eval = np.linspace(0, sim_time_days, sim_time_days+1)
    ivp = solve_ivp(transmission_model, (0, sim_time_days),
                    s2v(initial_state), t_eval=t_eval)
    if ivp.status != 0:
        print(ivp.message)
    result = pd.DataFrame(ivp.y.T,
                          index=ivp.t,
                          columns=state_labels)
    result = emission_model.add_observables(result)
    return result

def calibrate_timing_to_cases(sim, confirmed_cases):
    """
    Assumes confirmed_cases is a date-indexed Series. Currently assumes
    confirmed cases.
    """
    ref_date = confirmed_cases.index[0]
    confirmed_cases = confirmed_cases.copy()
    confirmed_cases.index = (confirmed_cases.index -
                             confirmed_cases.index[0]).to_series().dt.days
    def mape(shift):
        C = np.interp(confirmed_cases.index+shift, sim.index,
                      sim['All cases'].values)
        return (C / confirmed_cases - 1).abs().mean()
    opt = minimize_scalar(mape)
    if not opt.success:
        print(opt)
        raise RuntimeError('Optimiser failed')
    opt_shift = round(opt.x)
    result = sim.copy()
    result.index = ref_date + pd.to_timedelta(sim.index - opt_shift, unit='D')
    result.index.name=('Date')
    return result

def identify_date(sim, confirmed_cases, target_date='2020/03/14'):
    """
    Assumes confirmed_cases is a date-indexed Series. Currently assumes
    confirmed cases.
    """
    target_date = pd.to_datetime(target_date)
    target_value = confirmed_cases[target_date]
    try:
        date_offset = sim.index[sim['All cases']>target_value][0]
    except IndexError: 
        raise ValueError("Cannot find suitable date offset")
    result = sim.copy()
    result.index = target_date + pd.to_timedelta(sim.index - date_offset,
                                                 unit='D')
    return result

def dg_weights(n, mean, n_days):
    """ Discretise a gamma distribution over n_days timesteps
    and return the weights """
    dist = gamma(n, scale=mean)
    edges = np.linspace(-0.5, n_days-0.5, n_days+1)
    p = np.diff(dist.cdf(edges))
    p /= np.sum(p)
    return p

def dp_convolve(signal, p):
    """ Convolve kernel p over a signal forward in time, returning
    a vector of equal length to signal """
    return np.convolve(signal, p, mode='full')[:len(signal)]

def piecewise_linear_R_0_profile(dates, R_0s, baseline_simulation):
    dates = (pd.to_datetime(dates) - baseline_simulation.index[0]).days.values
    return lambda t: np.interp(t, dates, R_0s)
