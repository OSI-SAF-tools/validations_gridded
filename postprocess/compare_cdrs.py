import platform
from os.path import join

import matplotlib.dates as mdates
import matplotlib.pylab as plt
import xarray as xr
import yaml

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


def get_config():
    with open("config.yml", 'r') as stream:
        base_url = yaml.load(stream)
    return base_url


config = get_config()
direc = join(config['MachineConfigs'][platform.node()]['results'])

vs = ['intermediate_bias', 'intermediate_stddev', 'ice_bias', 'ice_stddev', 'water_bias', 'water_stddev']


def dataframes():
    for hem in ['NH', 'SH']:
        ds1 = xr.open_dataset(
            join(direc, 'ice_conc_polstere-100_cont-reproc_{hem}_20150101_20190301.nc'.format(hem=hem)))
        ds2 = xr.open_dataset(join(direc, 'ice_conc_ease2-250_icdr-v2p0_{hem}_20150101_20190301.nc'.format(hem=hem)))
        df1 = ds1[vs].to_dataframe()
        df2 = ds2[vs].to_dataframe()
        yield hem, df1.merge(df2, on='time', suffixes=('_430', '_430b'))


for hem, df in dataframes():
    for v in vs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6 / 1.618))
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)

        plt.plot(df[v + '_430'], '-bo', ms=2, lw=0.8, color='tab:blue')
        plt.plot(df[v + '_430b'], '-bo', ms=2, lw=0.8, color='tab:red')
        plt.plot(df[v + '_430'] - df[v + '_430b'], '-bo', ms=2, lw=0.8, color='0.3')
        ax.axhline(color='k')
        ax.legend(['430', '430b', '430 - 430b'], loc=4)
        ax.set_title(v + ' ' + hem)
        plt.grid()
        plt.savefig('/home/jol/Documents/temp/compare_osi430_430b/' + '{v}_{hem}'.format(v=v, hem=hem))
        # plt.show()
