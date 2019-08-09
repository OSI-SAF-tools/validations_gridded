#!/usr/bin/python3


import logging
import numpy as np
import pandas as pd
import sys
import xarray as xr
# from distributed import Client
from collections import OrderedDict

from validate import base

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)


class ValidateConc(base.Validate):
    product_variables = ['ice_conc', 'status_flag']

    def standardise_ref(self):
        """
        :return:
        """
        nan = np.logical_or(np.isnan(self.dataset.lower), np.isnan(self.dataset.upper))
        below = np.less(self.dataset.ice_conc, self.dataset.lower)
        above = np.greater(self.dataset.ice_conc, self.dataset.upper)
        within = np.logical_and(np.greater_equal(self.dataset.ice_conc, self.dataset.lower),
                                np.less_equal(self.dataset.ice_conc, self.dataset.upper))
        conditions = [nan, below, above, within]
        results = [np.nan, self.dataset.lower, self.dataset.upper, self.dataset.ice_conc]
        standardised = np.select(conditions, results)
        return xr.DataArray(standardised, dims=self.dataset.ice_conc.dims, coords=self.dataset.ice_conc.coords)

    def conc_error(self, ice_conc, ice_chart):
        error = ice_chart - ice_conc
        valid_ref = np.logical_and(self.dataset.ice_conc >= 0,
                                   self.dataset.ice_conc <= 100)
        valid_src = np.logical_and(self.dataset.ice_chart_standard >= 0,
                                   self.dataset.ice_chart_standard <= 100)
        da = xr.DataArray(error, dims=self.dataset.ice_conc.dims, coords=self.dataset.ice_conc.coords)
        da = da.where(valid_ref).where(valid_src)
        return da

    def load_product_data(self, source_glob, preprocess=None):
        product = super().load_product_data(source_glob, preprocess)
        product['status_flag'] = product['status_flag'].astype(np.uint8)
        return product

    """
    Validation Functions
    """

    def has_flag(self, flag):
        return np.bitwise_and(self.dataset.status_flag, flag) == flag

    def bias_std(self, error):
        bias = error.mean(dim=('xc', 'yc'))
        std = error.std(dim=('xc', 'yc'))
        return bias, std

    def match_pct(self, error, threshold):
        error = np.abs(error)
        count_total = error.count(dim=('xc', 'yc'))
        count_above_threshold = error.where(error <= threshold).count(dim=('xc', 'yc'))
        return count_total, 100 * count_above_threshold / count_total

    def compute_stats(self):
        chart_standard = self.dataset.ice_chart_standard
        conc_error = self.dataset.conc_error

        total_bias, total_stddev = self.bias_std(conc_error)
        ice = chart_standard >= 99
        ice_bias, ice_stddev = self.bias_std(conc_error.where(ice))
        water = chart_standard <= 1
        water_bias, water_stddev = self.bias_std(conc_error.where(water))
        marginal = np.logical_and(chart_standard > 1, chart_standard < 99)
        marginal_bias, marginal_stddev = self.bias_std(conc_error.where(marginal))

        count, within_10pct = self.match_pct(self.dataset.conc_error, 10)
        count, within_20pct = self.match_pct(self.dataset.conc_error, 20)

        stats = OrderedDict(reference_time=self.dataset.time,

                            total_bias=total_bias,
                            total_stddev=total_stddev,

                            intermediate_bias=marginal_bias,
                            intermediate_stddev=marginal_stddev,

                            ice_bias=ice_bias,
                            ice_stddev=ice_stddev,

                            water_bias=water_bias,
                            water_stddev=water_stddev,

                            within_10pct=within_10pct,
                            within_20pct=within_20pct,

                            count=count)
        self.dataset = self.dataset.update(stats)

        self.dataset.total_bias.attrs = {'unit': '-', 'long_name': 'total_bias', 'x_label': 'Date',
                                         'y_label': 'Bias', 'plot_type': 'line'}
        self.dataset.total_stddev.attrs = {'unit': '-', 'long_name': 'total_SD', 'x_label': 'Date',
                                           'y_label': '$\sigma$', 'plot_type': 'line'}
        self.dataset.ice_bias.attrs = {'unit': '-', 'long_name': 'ice_bias', 'x_label': 'Date',
                                       'y_label': 'Bias', 'plot_type': 'line'}
        self.dataset.ice_stddev.attrs = {'unit': '-', 'long_name': 'ice_SD', 'x_label': 'Date',
                                         'y_label': '$\sigma$', 'plot_type': 'line'}
        self.dataset.water_bias.attrs = {'unit': '-', 'long_name': 'water_bias', 'x_label': 'Date',
                                         'y_label': 'Bias', 'plot_type': 'line'}
        self.dataset.water_stddev.attrs = {'unit': '-', 'long_name': 'water_SD', 'x_label': 'Date',
                                           'y_label': '$\sigma$', 'plot_type': 'line'}
        self.dataset.within_10pct.attrs = {'unit': '%', 'long_name': 'percentage_of_grid_points',
                                           'x_label': 'Date', 'y_label': 'Match', 'plot_type': 'bar'}
        self.dataset.within_20pct.attrs = {'unit': '%', 'long_name': 'water_SD', 'x_label': 'Date',
                                           'y_label': 'Match', 'plot_type': 'bar'}

        return pd.DataFrame(stats)

    def __call__(self):
        super().__call__()

        log.info("Merging...")
        self.dataset = self.merge(self.ds_ref, self.ds_product, self.product_variables)
        self.dataset.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        log.info("Loading into memory...")
        self.dataset = self.dataset.compute()
        log.info("Standardising...")
        self.dataset['ice_chart_standard'] = self.standardise_ref()
        log.info("Ice conc difference...")
        self.dataset['conc_error'] = self.conc_error(self.dataset.ice_chart_standard, self.dataset.ice_conc)
        log.info("Selecting using mask...")
        self.dataset['conc_error'] = self.select_using_flags(self.dataset['conc_error'])
        log.info("Finished initialising.")

        self.compute_stats()

        out_vars = ['ice_bias', 'ice_stddev', 'water_bias', 'water_stddev', 'within_10pct', 'within_20pct']
        json_str = self.dataset[out_vars].to_dataframe().transpose().to_json(double_precision=3)
        return json_str


class ValidateConcCDR(ValidateConc):
    def select_using_flags(self, data_array):
        # Draft E
        mask_flags = self.has_flag(1) | self.has_flag(2) | \
                     self.has_flag(8) | self.has_flag(32) | \
                     self.has_flag(64) | self.has_flag(128)
        da = data_array.where(np.logical_not(mask_flags))
        return da


class ValidateConcHYR(ValidateConc):
    def select_using_flags(self, data_array):
        da = data_array.where(self.has_flag(0))
        return da


class ValdiateConcL2(ValidateConc):

    def select_using_flags(self, data_array):
        da = data_array.where(self.has_flag(0))
        return da

    def merge(self, ds_ref, ds_product, product_variables):
        try:
            source = ds_ref.attrs['sources']['NIC-SHP']
            ds_ref = ds_ref.isel(source=source, time=ds_ref.source == source)
        except ValueError:
            pass

        ds_ref = ds_ref.reindex(time=self.ds_product.time, method='ffill')
        dataset = xr.merge([ds_ref, ds_product[product_variables]], join='inner')
        if self.start_date:
            dataset = dataset.sel(time=slice(self.start_date, self.end_date))
        len_time, i = self.get_chunk_size(len(dataset.time))
        dataset = dataset.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        return dataset

    def load_product_data(self, product_glob, preprocess=None):
        """
        :param product_glob: OSI SAF glob string
        """

        ds_product = xr.open_mfdataset(product_glob, autoclose=True, decode_cf=True, data_vars=self.product_variables,
                                       parallel=True, preprocess=preprocess)

        # Remove the time from the date so that it corresponds to the reference data set
        # ds_product['time'] = ds_product['time'].to_series().apply(lambda dt:
        #                                                     datetime(dt.year, dt.month, dt.day, 0))
        # ds_product = ds_product[[self.product_variables]].astype(np.float32)
        len_time, i = self.get_chunk_size(len(ds_product.time))
        ds_product = ds_product.isel(time=slice(0, len_time)).chunk(chunks={'time': i})
        ds_product = self.variable_info(ds_product)
        return ds_product

