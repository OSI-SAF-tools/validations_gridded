from os.path import join
import xarray as xr


for hem in ['NH', 'SH']:
    ds1 = xr.open_dataset(join('/data2/validations',
                               'ice_conc_polstere-100_cont-reproc_{hem}_20150101_20170426.nc'.format(hem=hem)))
    ds2 = xr.open_dataset(join('/data2/validations',
                               'ice_conc_polstere-100_cont-reproc_{hem}_20170427_20190301.nc'.format(hem=hem)))
    ds1['xc'] = ds2['xc']
    ds1['yc'] = ds2['yc']


    ds_merged = xr.concat([ds1, ds2], dim='time')
    ds_merged = ds_merged.rename({'marginal_bias': 'intermediate_bias', 'marginal_stddev': 'intermediate_stddev'})

    ds_merged.to_netcdf(join('/data2/validations',
                             'ice_conc_polstere-100_cont-reproc_{hem}_20150101_20190301.nc'.format(hem=hem)))
