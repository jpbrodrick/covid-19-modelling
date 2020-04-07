import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, gamma
from collections import namedtuple

class SEIR(namedtuple('SEIR', ['S', 'E', 'I', 'R'])):
    @staticmethod
    def make_state(S=0, E=0, I=0, R=0):
        return SEIR(S=S, E=E, I=I, R=R)

    @staticmethod
    def from_a(a):
        return SEIR(*a)

    def to_a(self):
        return np.array([*self])

    @property
    def state_labels(self):
        return self._fields

class SEIRModel:
    def __init__(self,
                 initial_state,
                 R_0=None,
                 T_inc=None,
                 T_inf=None,
                 early_growth_rate=None,
                 mean_generation_time=None,
                 lat_fraction=None):
        #TODO: input validation
        alt_parm = R_0 is None or T_inc is None or T_inf is None
        std_parm = (early_growth_rate is None or mean_generation_time is None or
                    lat_fraction is None)
        if not (alt_parm or std_parm):
            raise ValueError('You have to either provide all of R_0, T_inc '
                             'and T_inf or all of early_growth_rate, '
                             'mean_generation_time and lat_frac')
        if alt_parm:
            params = SEIRModel.calibrate_parameters(early_growth_rate,
                                                    mean_generation_time,
                                                    lat_fraction)
            R_0, T_inc, T_inf = params['R_0'], params['T_inc'], params['T_inf']
        self.R_0 = R_0 if callable(R_0) else (lambda _: R_0)
        self.T_inc = T_inc
        self.T_inf = T_inf
        self.initial_state = initial_state
        self.reset_cache()

    @property
    def R_0(self):
        return self._R_0

    @R_0.setter
    def R_0(self, value):
        self._R_0 = value if callable(value) else (lambda _: value)
        self.reset_cache()

    @property
    def T_inc(self):
        return self._T_inc

    @T_inc.setter
    def T_inc(self, value):
        self._T_inc = value
        self.reset_cache()

    @property
    def T_inf(self):
        return self._T_inf

    @T_inf.setter
    def T_inf(self, value):
        self._T_inf = value
        self.reset_cache()

    def reset_cache(self):
        self._path = None

    @staticmethod
    def calibrate_parameters(early_growth_rate,
                             mean_generation_time,
                             lat_fraction):
        """
        Calculate SEIR parameters R_0, T_inc and T_inf calibrated to
        the specified initial daily case/death growth rate, the infection
        mean generation time, and the fraction of this time that the
        infection is assumed to be latent (i.e. non-infectious, E stage).
        """
        T_inc = lat_fraction * mean_generation_time
        T_inf = mean_generation_time - T_inc
        pos_eigenvalue = np.log(1+early_growth_rate)
        R_0 = (1 + pos_eigenvalue * T_inc) * (1 + pos_eigenvalue * T_inf)
        return {'R_0':R_0, 'T_inc':T_inc, 'T_inf':T_inf}
         
    def _ydot(self, t, y):
        y = SEIR.from_a(y)
        N = y.S + y.E + y.I + y.R
        # Under the assumption that population size does not affect dynamics on a relative basis
        # we enforce that a uniform R_0 across population segments implies a uniform beta
        beta = self.R_0(t) / (N.sum() * self.T_inf)
        Sd = -beta * y.S * y.I
        Ed = beta * y.S * y.I - y.E / self.T_inc
        Id = y.E / self.T_inc - y.I / self.T_inf
        Rd = y.I / self.T_inf        
        return SEIR.make_state(S=Sd, E=Ed, I=Id, R=Rd).to_a()

    def predict(self, sim_days, T_start=0):
        if (self._path is not None and len(self._path)>sim_days):
            # Use cached result if possible
            result = self._path.iloc[:sim_days+1]
            result.index = result.index - result.index[0] + T_start
            return result
        else:
            sim_days = int(sim_days)
            t_eval = np.linspace(T_start, T_start+sim_days, sim_days+1)
            ivp = solve_ivp(self._ydot, (0, sim_days),
                            self.initial_state.to_a(), t_eval=t_eval)
            if ivp.status != 0:
                print(ivp.message)
            self._path = pd.DataFrame(ivp.y.T,
                                      index=ivp.t,
                                      columns=self.initial_state.state_labels)
            return self._path.copy()
    
class SEIRObsModel(SEIRModel):
    def __init__(self, f_cdr, f_cfr, cv_detect, T_detect, cv_resolve, T_resolve,
                 cv_death, T_death, start_date=0, **SEIR_params):
        #TODO: input validation
        n_detect, n_resolve, n_death = [cv**(-2) for cv in
                                        (cv_detect, cv_resolve, cv_death)]
        self._fullpath = None
        self.p_symptoms = f_cdr * dg_weights(n_detect,
                                              T_detect/n_detect,
                                              int(T_detect*30))
        self.p_resolve = (1-f_cfr) * dg_weights(n_resolve,
                                                T_resolve/n_resolve,
                                                int(T_resolve*30))
        self.p_death = f_cfr * dg_weights(n_death,
                                          T_death/n_death,
                                          int(T_death*30))
        self.start_date = start_date
        super().__init__(**SEIR_params)

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, start_date):
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        self._start_date = start_date
        self._offset = (pd.Timedelta(1, 'D')
                        if isinstance(self._start_date, pd.Timestamp)
                        else 1.)

    def reset_cache(self):
        self._fullpath = None
        super().reset_cache()
        
    def score(self, day_zero, cases, recovered, deaths, weights=None,
              predictions=None):
        raise NotImplementedError()

    def fit(self, cases, recovered, deaths, sim_days, obs_threshold=10,
            weights=None):
        """
        Fit day 0 of the simulation to observed cases, recovered and 
        deaths. Weights is a 3-tuple specifying the relative weights to
        place on case data, recovery data and death data respectively when
        calculating the score.
        """
        if self._fullpath is None:
            self.predict(sim_days)
            
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        w_c, w_r, w_d = weights

        cases, recovered, deaths = [s.dropna()
                                     for s in (cases, recovered, deaths)]

        total_weight = (len(cases) * w_c + len(recovered) * w_r +
                        len(deaths) * w_d)

        def error_ts(days_shift, s_obs, s_pred):
            if len(s_obs)==0:
                return 0.
            obs_days = (s_obs.index - self.start_date) / self._offset
            s_pred = np.interp(obs_days + days_shift, s_pred.index,
                               s_pred.values)
            return np.sum(np.abs((np.log((obs_threshold+s_obs.values)/
                                         (obs_threshold+s_pred)))))
            
        def score(days_shift):
            result = (w_c * error_ts(days_shift, cases,
                                   self._fullpath['All cases']) +
                    w_r * error_ts(days_shift, recovered,
                                   self._fullpath['All recovered']) +
                    w_d * error_ts(days_shift, deaths,
                                   self._fullpath['All deaths']))
            return result

        opt = minimize_scalar(score)
        if not opt.success:
            print(opt)
            raise RuntimeError('Optimiser failed')
        opt_shift = round(opt.x)
        self.start_date -= opt_shift * self._offset
        return self
    
    def predict(self, sim_days):
        if self._fullpath is not None and len(self._fullpath)>sim_days:
            sim =  self._fullpath.iloc[:sim_days+1]
        else:
            sim = super().predict(sim_days)
            sim['All exposed'] = sim.E + sim.I + sim.R
            sim['Daily exposed'] = sim['All exposed'].diff()
            sim['Daily exposed'].iloc[0] = 0.
            sim['Daily cases'] = dp_convolve(sim['Daily exposed'],
                                             self.p_symptoms)
            sim['Daily recovered'] = dp_convolve(sim['Daily cases'],
                                                  self.p_resolve)
            sim['Daily deaths'] = dp_convolve(sim['Daily cases'],
                                              self.p_death)
            sim['All cases'] = sim['Daily cases'].cumsum()
            sim['All recovered'] = sim['Daily recovered'].cumsum()
            sim['All deaths'] = sim['Daily deaths'].cumsum()
            sim['Active cases'] = (sim['All cases'] -
                                   sim['All recovered'] -
                                   sim['All deaths'])
            self._fullpath = sim
        result = sim.copy()
        if isinstance(self.start_date, pd.Timestamp):
            result.index = self.start_date + pd.TimedeltaIndex(sim.index, 'D')
        else:
            result.index = self.start_date + sim.index.values
        result.index.name = 'Date'
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
