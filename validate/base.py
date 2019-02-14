# from dask.distributed import Client
import datetime
import logging
import os
import platform
import sys
import warnings
from datetime import datetime, timedelta
from ftplib import FTP, error_perm
from glob import glob
from os.path import join, isfile, isdir

import numpy as np
import xarray as xr
import yaml

log = logging.getLogger()
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.getcwd())



class Validate:

    def __init__(self, url, icechart_dir, hemisphere, start_date, end_date, store_test_files=False):
        """
        :param source_glob: OSI SAF glob string
        :param reference_glob: Ice Chart glob string
        :param start_date: start date string of dataset to use
        :param end_date: end date string of dataset to use
        """
        self.test_delay = None  # TODO: Use if there a delay between the ref and test data
        # Set Attributes
        self.hemisphere = hemisphere
        self.start_date = start_date
        self.end_date = end_date
        self.url = url
        self.ice_charts_dir = icechart_dir
        self.store_test_files = store_test_files

    def __call__(self):
        # Load Data
        self.test_files, self.ref_glob, self.save_name = self.get_data()
        log.info("Loading test data...")
        self.ds_test = self.load_test_data(self.test_files)
        self.ds_test = self.ds_test[self.test_variables]
        log.info("Loading reference data...")
        self.ds_ref = self.load_reference_data(self.ref_glob)

        self.ds_ref['xc'] = self.ds_test.xc
        self.ds_ref['yc'] = self.ds_test.yc

        # Check that the coordinates for the reference and test dataset are the same
        assert np.all(self.ds_ref.xc.data == self.ds_test.xc.data)
        assert np.all(self.ds_ref.yc.data == self.ds_test.yc.data)

    @staticmethod
    def get_config():
        current_dir = os.path.dirname(__file__)
        with open(join(current_dir, "config.yml"), 'r') as stream:
            base_url = yaml.load(stream)
        return base_url

    def generate_timeseries(self):
        start = datetime.strptime(self.start_date, '%Y%m%d')
        end = datetime.strptime(self.end_date, '%Y%m%d')
        step = timedelta(days=1)
        dt = start
        while dt <= end:
            yield dt
            dt += step

    @staticmethod
    def download_test_data_http(urls, temp_dir):
        try:
            os.makedirs(temp_dir)
        except FileExistsError:
            pass

        for url in urls:
            filename = url.split('/')[-1]
            full_filename = join(temp_dir, filename)
            if not isfile(full_filename):
                try:
                    ds = xr.open_dataset(url)
                    ds.to_netcdf(full_filename)
                    try:
                        assert not os.stat(full_filename).st_size == 0
                    except AssertionError:
                        os.remove(full_filename)
                        raise IOError("file is empty")
                except RuntimeError:
                    log.error('This url does not exist: {0}'.format(url))
                    continue
            yield full_filename

    @staticmethod
    def download_test_data_ftp(urls, temp_dir):
        try:
            os.makedirs(temp_dir)
        except FileExistsError:
            pass

        if not urls:
            raise ValueError('No urls found on remote server')

        ftp_server = urls[0].split('/')[2]
        ftp = FTP(ftp_server, timeout=300)
        ftp.login()

        for url in urls:
            filename = url.split('/')[-1]
            full_filename = join(temp_dir, filename)
            if not isfile(full_filename):
                try:
                    with open(full_filename, 'wb') as file:
                        ftp.retrbinary('RETR ' + url.split(ftp_server)[-1], file.write)
                        file.close()
                        try:
                            assert not os.stat(full_filename).st_size == 0
                        except AssertionError:
                            os.remove(full_filename)
                            raise IOError("file is empty")
                except error_perm:
                    log.error('This url does not exist: {0}'.format(url))
                    continue
                except (RuntimeError, TimeoutError, IOError) as err:
                    log.error('Cannot get: {0}\n {1}'.format(url, err))
                    continue
            yield full_filename


    @staticmethod
    def get_chunk_size(len_time):
        """
        Find and chunk size which is divisible by the time-length of the array. It should not be
        too big, so the time-length may need to be reduced, in which case we reduce the size of the
        data set.
        :param len_time: time length of data array
        :return: new time-length, chunk size
        """
        for lt in range(len_time, 0, -1):
            for i in range(5, 12):
                if (lt % i) == 0:
                    break
            else:
                continue  # only executed if the inner loop did NOT break
            break
        return lt, i

    @staticmethod
    def _get_xy_coords(da):
        df = da.to_dataframe()
        # convert to a list of coordinate pairs
        return df[[df.keys()[-1], ]][df[df.keys()[-1]].values].to_records()[['xc', 'yc']].tolist()

    def get_chart_fnames(self, time):
        return glob(join(self.ice_charts_dir,
                         '{0}*_{1}_*.nc'.format(time.strftime('%Y%m%d'), self.hemisphere)))

    def get_data(self):
        # if there is no delay in the test data relative to the reference then use the same time series as
        # ice chart
        ts = [dt for dt in self.generate_timeseries()]
        ts_ice_chart = [dt for dt in ts if len(self.get_chart_fnames(dt)) > 0]
        if self.test_delay:
            ts_test = ts
        else:
            ts_test = ts_ice_chart
            if not ts_test:
                raise ValueError('There are no ice charts with the same dates as the OSI SAF data')

        test_files = list(self.url.format(Y=dt.year, y=dt.year, m=dt.month, d=dt.day, hem=self.hemisphere.lower())
                          for dt in ts_test)

        save_name = self.url.split('/')[-1].format(hem='', Y=0, m=0, d=0).replace('_000001200.nc', '').replace('__',
                                                                                                               '_')
        if 'ftp' in self.url:
            test_files = self.download_test_data_ftp(test_files, join('/tmp', save_name))
        elif self.store_test_files and ('http' in self.url):
            test_files = self.download_test_data_http(test_files, join('/tmp', save_name))

        # Get the ice chart files covering the desired interval
        ref_glob = [self.get_chart_fnames(dt)[0] for dt in ts_ice_chart]
        return test_files, ref_glob, save_name

    def variable_info(self, ds):
        info = {'lat': {'attr': {'least_significant_digit': 3,
                                 'standard_name': 'latitude',
                                 'long_name': 'latitude',
                                 'units': 'degrees_north'},
                        },
                'lon': {'attr': {'least_significant_digit': 3,
                                 'standard_name': 'longitude',
                                 'long_name': 'longitude',
                                 'units': 'degrees_east'}}}
        for v in ['lat', 'lon']:
            ds[v].attrs = info[v]['attr']
        return ds

    def load_test_data(self, source_glob, preprocess=None):
        """
        :param source_glob: OSI SAF glob string
        """

        ds_test = xr.open_mfdataset(source_glob, autoclose=True, decode_cf=True, data_vars=self.test_variables,
                                    parallel=True, preprocess=preprocess)

        # Remove the time from the date so that it corresponds to the reference data set
        ds_test['time'] = ds_test['time'].to_series().apply(lambda dt:
                                                            datetime(dt.year, dt.month, dt.day, 0))
        # ds_test = ds_test[[self.test_variables]].astype(np.float32)
        len_time, i = self.get_chunk_size(len(ds_test.time))
        ds_test = ds_test.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        ds_test = self.variable_info(ds_test)
        return ds_test

    @staticmethod
    def get_ice_charts_and_sources(fname_charts):
        sources = [fn.split('_')[-1].replace('.nc', '') for fn in fname_charts]
        return fname_charts, sources

    def load_reference_data(self, reference_glob):
        """
        :param source_glob: OSI SAF glob string
        """
        fname_charts, sources = self.get_ice_charts_and_sources(reference_glob)

        sources_dict = {s: i for i, s in enumerate(np.unique(list(sources)))}

        ds_ref = xr.open_mfdataset(fname_charts,
                                   autoclose=True, decode_cf=True,
                                   data_vars=['codes', 'lower', 'upper'], parallel=True)
        ds_ref = ds_ref.assign_coords(source=xr.DataArray([sources_dict[s] for s in sources], dims='time'))

        attr_str = 'The ice charts are are from the following source:\n'
        for k, v in sources_dict.items():
            attr_str += '{k}: {v}\n'.format(k=k, v=v)
        ds_ref.attrs = {'description': attr_str}

        ds_ref = ds_ref.sortby('time')
        try:
            ds_ref = ds_ref.drop(['lat', 'lon'])
        except ValueError:
            pass

        ds_ref['time'] = ds_ref['time'].to_series().apply(lambda dt:
                                                          datetime(dt.year, dt.month, dt.day, 0))
        ds_ref = ds_ref.isel(time=~ds_ref.time.to_pandas().duplicated())
        # try:
        #     ds_ref = ds_ref.reindex(source=source]).squeeze()
        # except AttributeError:

        ds_ref = ds_ref.squeeze()
        len_time, i = self.get_chunk_size(len(ds_ref.time))
        ds_ref = ds_ref.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        ds_ref['codes'] = ds_ref['codes'].astype(np.int16)
        return ds_ref

    def merge(self, ds_ref, ds_test, test_variables):
        dataset = xr.merge([ds_ref, ds_test[test_variables]], join='inner')
        if self.start_date:
            dataset = dataset.sel(time=slice(self.start_date, self.end_date))
        len_time, i = self.get_chunk_size(len(dataset.time))
        dataset = dataset.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        return dataset

    @staticmethod
    def aggregate_stats(dataset, variable):
        return dataset[variable].resample(time='6M', closed='left').mean()

    def to_netcdf(self, results_dir):
        encoding_float = {"least_significant_digit": 4, "zlib": True, "complevel": 6}
        encodings = {v: encoding_float for v, ds in self.dataset.items() if np.issubdtype(ds.dtype, np.floating)}

        v = self.hemisphere, self.start_date, self.end_date
        fname = join(results_dir, self.save_name + '_{0}_{1}_{2}.nc'.format(*v))
        self.dataset.to_netcdf(fname, encoding=encodings)  # , unlimited_dims=['time'])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            for f in self.test_files:
                try:
                    if ((not self.store_test_files) and ('ftp' in self.url)) or os.stat(f).st_size == 0:
                        os.remove(f)
                except OSError:
                    pass
        except AttributeError:
            pass
        except Exception as exc:
            log.error('Error {} occurred while exiting'.format(exc))
