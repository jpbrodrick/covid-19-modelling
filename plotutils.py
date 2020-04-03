import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
from ipywidgets import interactive, FloatSlider
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


