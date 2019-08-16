# from dask.distributed import Client
import datetime
import logging
import numpy as np
import os
import sys
import warnings
import xarray as xr
import yaml
from datetime import datetime, timedelta
from ftplib import FTP, error_perm
from functools import lru_cache
from glob import glob
from itertools import chain
from os.path import dirname
from os.path import join, isfile, isdir

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(fmessage)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.getcwd())


class Validate:
    name = 'validation'

    def __init__(self, url, icechart_dir, hemisphere, start_date, end_date, store_product_files=False):
        """
        :param url: url to OSI SAF data. e.g. http://.../{Y}/{m:02d}/...{Y}{m:02d}{d:02d}1200.nc
                    where {Y} is the year
                          {m} is the month
                          {d} is the day
        :param icechart_dir: path to ice charts
        :param hemisphere: either NH|SH
        :param start_date: format: YYYYmmdd
        :param end_date: format: YYYYmmdd
        :param store_product_files: True stores the results as a netCDF file
        """

        self.product_delay = None  # TODO: Use if there a delay between the ref and product data
        # Set Attributes
        self.hemisphere = hemisphere
        self.start_date = start_date
        self.end_date = end_date
        self.url = url
        self.ice_charts_dir = icechart_dir
        self.store_product_files = store_product_files
        self.delete_list = []

        self.save_name = self.url.split('/')[-1].format(hem='', Y=0, m=0, d=0).replace('.nc', '').\
            replace('_00000', '').replace('1200', '').replace('__', '_').replace('*', '_all')

        # TODO: Ask Luis why had you changed it to this self.save_dir = join('/data/ice_val/tmp', self.save_name)
        self.save_dir = join('/tmp', self.save_name)

        try:
            os.makedirs(self.save_dir)
        except FileExistsError:
            pass

    def __call__(self):
        log.info("Processing the {0} hemisphere".format(self.hemisphere))
        # Load Data
        self.product_files, self.ref_glob = self.get_data()
        log.info("Loading product data...")
        self.ds_product = self.load_product_data(self.product_files)
        self.ds_product = self.ds_product[self.product_variables]
        log.info("Loading reference data...")
        self.ds_ref = self.load_reference_data(self.ref_glob)
        # self.ds_ref = self.ds_ref.isel(source=0, time=self.ds_ref.source == 0)

        # Check that the coordinates for the reference and product dataset are the same
        assert np.all(np.isclose(self.ds_ref.xc.data, self.ds_product.xc.data, atol=10))
        assert np.all(np.isclose(self.ds_ref.yc.data, self.ds_product.yc.data, atol=10))
        self.ds_ref['xc'] = self.ds_product.xc
        self.ds_ref['yc'] = self.ds_product.yc

    # TODO: Delete if not used
    # @staticmethod
    # def get_config():
    #     current_dir = os.path.dirname(__file__)
    #     with open(join(current_dir, "config.yml"), 'r') as stream:
    #         base_url = yaml.load(stream)
    #     return base_url

    def generate_timeseries(self):
        start = datetime.strptime(self.start_date, '%Y%m%d')
        end = datetime.strptime(self.end_date, '%Y%m%d')
        step = timedelta(days=1)
        dt = start
        while dt <= end:
            yield dt
            dt += step

    def download_product_data_http(self, urls):
        for url in urls:
            filename = url.split('/')[-1]
            full_filename = join(self.save_dir, filename)
            if isfile(full_filename):  # yield the locally stored file, if it is found
                yield full_filename
                continue
            try:
                ds = xr.open_dataset(url)
                ds.to_netcdf(full_filename)
                assert not os.stat(full_filename).st_size == 0
            except AssertionError:
                raise IOError("File {} is empty".format(full_filename))
                continue
            except RuntimeError:
                log.error('This url does not exist: {0}'.format(url))
                continue
            finally:
                if isfile(full_filename) and os.stat(full_filename).st_size == 0:
                    os.remove(full_filename)
                if not self.store_product_files:
                    self.delete_list.append(full_filename)
            yield full_filename

    def download_product_data_ftp(self, urls):

        # if not urls:
        #     raise ValueError('No urls found on remote server')

        ftp_server = self.url.split('/')[2]
        ftp = FTP(ftp_server, timeout=60 * 60 * 12)
        ftp.login()

        # check that the directory exists

        @lru_cache(maxsize=32)
        def _remote_dir_exists(directory):
            if ftp.nlst(directory):
                return True
            else:
                return False

        @lru_cache(maxsize=1)
        def remote_dirs_exists(directory):
            direct_part = ''
            for d in directory.split('/'):
                direct_part += d
                if not _remote_dir_exists(direct_part):
                    return False
                direct_part += '/'
            return True

        for url in urls:
            filename = url.split('/')[-1]
            full_filename = join(self.save_dir, filename)
            if isfile(full_filename):  # yield the locally stored file, if it is found
                yield full_filename
                continue
            server_direc = os.path.dirname(url.split(ftp_server)[1])
            if not remote_dirs_exists(server_direc):
                print('Directory: ', server_direc, ' does not exist.')
                continue
            try:
                with open(full_filename, 'wb') as file:
                    ftp.retrbinary('RETR ' + url.split(ftp_server)[-1], file.write)
                    file.close()
            except error_perm:
                os.remove(full_filename)
                log.error('This url does not exist: {0}'.format(url))
                continue
            except (RuntimeError, TimeoutError, IOError) as err:
                os.remove(full_filename)
                log.error('Cannot get: {0}\n {1}'.format(url, err))
                continue
            finally:
                if not self.store_product_files:
                    self.delete_list.append(full_filename)
            yield full_filename

    def get_product_data_local(self, urls):
        return chain.from_iterable(glob(url) for url in urls if glob(url))

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

    def get_chart_fnames(self, times):
        for time in times:
            time_str = time.strftime('%Y%m%d')
            fname_glob = join(self.ice_charts_dir, '{0}_{1}_*.nc'.format(time_str, self.hemisphere))
            for f in glob(fname_glob):
                if os.path.isfile(f):
                    yield f, time

    def get_data(self):
        # if there is no delay in the product data relative to the reference then use the same time series as
        # ice chart
        ts_desired = list(self.generate_timeseries())
        ice_charts_times = list(zip(*self.get_chart_fnames(ts_desired)))
        if not ice_charts_times:
            raise IOError('There are no ice charts with the same dates as the OSI SAF data')
        ice_charts, times = ice_charts_times[0], ice_charts_times[1]

        if self.product_delay:
            times = ts_desired

        urls = (self.url.format(Y=dt.year, m=dt.month, d=dt.day, hem=self.hemisphere.lower()) for dt in
                sorted(set(times)))

        if 'ftp' in self.url:
            product_files_local = self.download_product_data_ftp(urls)
        elif self.store_product_files and ('http' in self.url):
            product_files_local = self.download_product_data_http(urls)
        elif isdir(dirname(self.url)):
            product_files_local = self.get_product_data_local(urls)

        product_files_local = list(product_files_local)

        if not product_files_local:
            raise IOError('No product files found')

        # Get the ice chart files covering the desired interval
        return product_files_local, ice_charts

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

    def load_product_data(self, source_glob, preprocess=None):
        """
        :param source_glob: OSI SAF glob string
        """

        ds_product = xr.open_mfdataset(source_glob, autoclose=True, decode_cf=True, data_vars=self.product_variables,
                                    parallel=True, preprocess=preprocess)

        # Remove the time from the date so that it corresponds to the reference data set
        ds_product['time'] = ds_product['time'].to_series().apply(lambda dt:
                                                            datetime(dt.year, dt.month, dt.day, 0))
        # ds_product = ds_product[[self.product_variables]].astype(np.float32)
        len_time, i = self.get_chunk_size(len(ds_product.time))
        ds_product = ds_product.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        ds_product = self.variable_info(ds_product)
        return ds_product

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

        ds_ref = ds_ref.isel(source=0)
        len_arr = len(ds_ref.time)
        # if len_arr == (1, ):
        #     len_arr = 1
        len_time, i = self.get_chunk_size(len_arr)
        ds_ref = ds_ref.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        ds_ref['codes'] = ds_ref['codes'].astype(np.int16)
        ds_ref.attrs['sources'] = sources_dict
        return ds_ref

    def merge(self, ds_ref, ds_product, product_variables):
        dataset = xr.merge([ds_ref, ds_product[product_variables]], join='inner')
        if self.start_date:
            dataset = dataset.sel(time=slice(self.start_date, self.end_date))
        len_time, i = self.get_chunk_size(len(dataset.time))
        dataset = dataset.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        return dataset

    # TODO: Delete if not used
    @staticmethod
    def aggregate_stats(dataset, variable):
        return dataset[variable].resample(time='6M').mean()

    def to_netcdf(self, results_dir):
        encoding_float = {"least_significant_digit": 4, "zlib": True, "complevel": 6}
        encoding_int = {"zlib": True, "complevel": 6}
        encodings = {v: encoding_float if np.issubdtype(ds.dtype, np.floating) else encoding_int for v, ds in self.dataset.items()}
        class_name = str(self.__class__.__dict__['__module__']).split('.')[-1]
        v = self.hemisphere, self.start_date, self.end_date
        fname = join(results_dir, class_name + '_' + self.save_name + '_{0}_{1}_{2}.nc'.format(*v))
        self.dataset.to_netcdf(fname, encoding=encodings, unlimited_dims=['time'])

    def __enter__(self):
        for f in glob(join(self.save_dir, '*.nc')):
            if os.stat(f).st_size == 0:
                print('Removing empty file: {0}'.format(f))
                os.remove(f)
        return self

    def __str__(self):
        return 'validation'

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            for f in self.delete_list:
                try:
                    # removes downloaded OSI SAF files if requested
                    os.remove(f)
                except OSError:
                    pass
        except AttributeError:
            pass
        except Exception as exc:
            log.error('Error {}\noccurred while exiting'.format(exc))
