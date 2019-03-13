import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import yaml
from os.path import join
import os
from glob import glob
import platform
import pandas as pd


def get_config():
    with open("config.yml", 'r') as stream:
        base_url = yaml.load(stream)
    return base_url


def seasonal_ds(fname, ds):
    ds_seaonal = ds.groupby('time.season').mean(dim='time')
    var_names = [v for v in ds_seaonal.data_vars]
    encoding = {v: ds[v].encoding for v in ds if v in var_names}
    try:
        ds_seaonal.to_netcdf(fname.replace('.nc', '_seasonal.nc'), encoding=encoding)
    except ValueError:
        print('ValueError with ', fname)
        ds_seaonal.to_netcdf(fname.replace('.nc', '_seasonal.nc'))


def aggregated_stats(fname, ds):
    ds_sub = ds[['ice_bias', 'ice_stddev', 'water_bias', 'water_stddev', 'intermediate_bias', 'intermediate_stddev']]
    ds_sub = ds_sub.to_dataframe()
    del ds_sub['source']
    fname_excel = fname.replace('.nc', '.xlsx')
    writer = pd.ExcelWriter(fname_excel, engine='xlsxwriter')
    ds_sub.to_excel(writer, sheet_name='data')
    for sheet in ['1M', '1Y']:
        stats = ds_sub.resample(sheet).mean()
        stats.to_excel(writer, sheet_name=sheet)
    writer.save()
    writer.close()


# ds = xr.open_dataset('/data/jol/validation/ice_conc_ease2-250_icdr-v2p0_SH_20150101_20190301.nc')
# ds.ice_bias.plot()
# plt.show()
#
#
# worst_time_i = int(np.argwhere((ds.ice_bias == ds.ice_bias.min()).values))
#
# ds.conc_error.isel(time=worst_time_i - 1).plot()
# plt.show()
# ds.conc_error.isel(time=worst_time_i).plot()
# plt.show()
#
# ds.ice_chart_standard.isel(time=worst_time_i -1).plot()
# plt.show()
# ds.ice_chart_standard.isel(time=worst_time_i).plot()
# plt.show()


if __name__ == '__main__':
    config = get_config()
    direc = join(config['MachineConfigs'][platform.node()]['results'])
    for fname in filter(lambda fn: not 'seasonal' in fn, glob(direc + '*.nc')):
        print(fname)
        ds = xr.open_dataset(fname)
        # seasonal_ds(fname, ds)
        aggregated_stats(fname, ds)





