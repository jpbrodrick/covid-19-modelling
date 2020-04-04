import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
from ipywidgets import interactive, FloatSlider, fixed, Text, BoundedIntText
from outbreak_modelling import *

def sims_to_longform(sims):
    """
    Convert one or more simulations to long-form format for plotting
    with seaborn. The input `sims` should take the form of a dict:
    keys should be strings labelling the simulations; values should
    be the sim DataFrames themselves.
    """
    result = pd.concat({k: v.rename_axis('PROJECTION', axis=1).stack().rename('Value')
                        for k, v in sims.items()})
    result.index.rename('SIMULATION NAME', level=0, inplace=True)
    result = result.to_frame().reset_index()
    result['PROJECTION'] = result['PROJECTION'].astype('category')
    return result


def plot_simulations(sims, observations, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.sca(ax)
    plot_data = sims_to_longform(sims)
    sns.lineplot(data=plot_data,
                 x='Date', y='Value', hue='PROJECTION',
                 style='SIMULATION NAME',
                 hue_order=['All cases', 'All deaths', 'Daily cases',
                            'Daily deaths'],
                 dashes = ['', (2, 4)])
    ax.plot([], [], ' ', label='OBSERVATIONS')
    ax.set_prop_cycle(None)
    observations[['confirmed', 'deaths']].plot(ax=ax, marker='x', ls='')
    ax.set_yscale('log')
    ax.set_ylim(1, None)
    ax.yaxis.set_major_locator(ticker.LogLocator(10., (1.,), 15, 15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(10., range(10), 15, 15))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(which='minor', axis='y', alpha=0.2)
    ax.legend().set_title('')
    ax.legend()
    ax.set_ylabel('')
    ax.set_xlabel('Date')
    #plt.xticks(rotation=90)
    return plot_data

def explore_simulation(initial_growth_rate,
		       serial_interval,
		       latent_fraction,
		       f_cdr, f_cfr,
		       T_detect, T_resolve, T_death,
		       n_detect, n_resolve, n_death,
		       R_0_lockdown,
                       lockdown_release_date,
                       lockdown_release_timeframe_weeks,
                       sim_time_weeks,
                       observations):
    initial_growth_rate /= 100
    f_cdr /= 100
    f_cfr /= 100
    
    try:
        lockdown_release_date = pd.to_datetime(lockdown_release_date)
    except TypeError:
        print('Error understanding lockdown release date')
        return
    lockdown_release_end = (lockdown_release_date +
                            pd.to_timedelta(7*lockdown_release_timeframe_weeks,
                                            'D'))
    if lockdown_release_date < pd.to_datetime('20/04/01'):
        print('Lockdown cannot be released before April 2020')
        return
        
    transmission_parameters = calibrate_transmission_model(initial_growth_rate,
                                                           serial_interval,
                                                           latent_fraction)
    t_model_c = SEIRModel(**transmission_parameters)
    e_model_c = EmissionModel(f_cdr, f_cfr,
                              n_detect, T_detect,
                              n_resolve, T_resolve,
                              n_death, T_death)
    sim_baseline = run_outbreak(t_model_c, e_model_c, 7*sim_time_weeks,
                                SEIR.make_state(S=6.64e7, I=1))
    sim_baseline = calibrate_timing_to_cases(sim_baseline,
                                             observations.confirmed[observations['phase']=='unrestricted'])
    t_model_ld = SEIRModel(**transmission_parameters)
    R_0_ld = piecewise_linear_R_0_profile(['2020/03/10', '2020/03/26',
                                           lockdown_release_date,
                                           lockdown_release_end],
                                          [transmission_parameters['R_0'],
                                           R_0_lockdown, R_0_lockdown,
                                           transmission_parameters['R_0']],
                                          sim_baseline)
    t_model_ld.R_0 = R_0_ld
    sim_ld = run_outbreak(t_model_ld, e_model_c, 7*sim_time_weeks,
                          SEIR.make_state(S=6.64e7, I=1))
    sim_ld = calibrate_timing_to_cases(sim_ld,
                                       observations.confirmed[observations['phase']=='unrestricted'])
    assert ((sim_baseline.index - sim_ld.index).to_series().dt.days==0).all()
    _, (axt, axb) = plt.subplots(2, 1, figsize=(12, 16),
                                 gridspec_kw={'height_ratios': [1, 3]})
    plt.sca(axt)
    plt.plot(sim_baseline.index, R_0_ld(range(len(sim_baseline))))
    plt.ylim(0, None)
    plt.ylabel('$R_0(t)$')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.title('$R_0$ profile')
    plot_data = plot_simulations({'UK lockdown': sim_ld,
                                  'Unconstrained baseline': sim_baseline},
                                 observations,
                                 ax=axb)
    axb.set_title('Projections')
    for ax in (axt, axb):
        ax.axvspan(pd.to_datetime('2020/03/10'),
                   pd.to_datetime('2020/03/26'), color='orange', alpha=0.03)
        ax.axvspan(pd.to_datetime('2020/03/26'),
                   lockdown_release_date, color='r', alpha=0.04)
        ax.axvspan(lockdown_release_date, lockdown_release_end, color='g', alpha=0.05)
    plt.subplots_adjust(hspace=0.5)
    plot_data.to_csv('simdata/last-simulation.csv')
    return plot_data

def my_slider(value, mymin, mymax, step, description):
    return FloatSlider(value=value, min=mymin, max=mymax, step=step,
                       description=description,
                       style={'description_width': 'initial'})

def my_text_box(value, mymin, mymax, step, description):
    return BoundedIntText(value=value, min=mymin, max=mymax, step=step,
                          description=description,
                          style={'description_width': 'initial'})

def interactive_simulation(observations):
    return interactive(explore_simulation,
                       {'manual':True},
                       initial_growth_rate = my_slider(30, 5, 50, 2, 'Initial growth rate, %'),
                       serial_interval = my_slider(6.5, 2, 10, 0.5, 'Mean serial interval, days'),
                       latent_fraction = my_slider(0.71, 0.1, 0.9, 0.1, 'Latent period fraction'),
                       f_cdr = my_slider(4.4, 0.1, 10, 0.1, 'Case detection rate, %'),
                       f_cfr = my_slider(0.9, 0.1, 5, 0.1, 'Case fatality rate, %'),
                       T_detect = my_slider(11, 1, 30, 1, 'Time to detection, days'),
                       T_resolve = my_slider(8, 1, 30, 1, 'Time to recovery, days'),
                       T_death = my_slider(10, 1, 30, 1, 'Time to death, days'),
                       n_detect = my_slider(4, 1, 30, 1, 'Detection time shape'),
                       n_resolve = my_slider(9, 1, 30, 1, 'Recovery time shape'),
                       n_death = my_slider(4, 1, 30, 1, 'Death time shape'),
		       R_0_lockdown = my_slider(0.7, 0.1, 4, 0.1, '$R_0$ during lockdown'),
                       lockdown_release_date = Text(value='2020/06/30',
                                                    description='Lockdown release date',
                                                    style={'description_width': 'initial'}),
                       
                       lockdown_release_timeframe_weeks = my_text_box(26, 1, 9999, 1, 'Number of weeks for lockdown release'),
                       sim_time_weeks = my_text_box(52, 1, 999, 1, 'Simulation length, weeks'),
                       observations = fixed(observations))

