import numpy as np
import pandas as pd

def load_data():
    data = {'confirmed': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
            'deaths': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
            'recovered': 'jhdata/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'}
    for k in data:
        data[k] = pd.read_csv(data[k])
    data = pd.concat(data, axis=0)
    data.index.names=['statistic', 'old_index']
    data = data.reset_index().drop('old_index', axis=1)
    return data

def single_country_data(data, country, unrestricted_dates):
    """ unrestricted_dates assumed to be a (start_date, end_date) tuple """
    ts = data[(data['Country/Region']==country) & (data['Province/State'].isnull())]
    ts = ts.set_index('statistic').drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T
    ts.index.name='Date'
    ts.index = pd.to_datetime(ts.index)
    ts['active_cases'] = ts.confirmed - ts.deaths - ts.recovered
    ts['phase'] = np.nan
    ts.loc[ts.index[0], 'phase'] = 'isolated'
    ts.loc[unrestricted_dates[0], 'phase'] = 'unrestricted'
    ts.loc[pd.to_datetime(unrestricted_dates[1])+pd.to_timedelta(1, 'D'), 'phase'] = 'suppressed'
    ts['phase'].fillna(method='ffill', inplace=True)
    ts['phase'] = pd.Categorical(ts['phase'])
    return ts
