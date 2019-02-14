"""
validate

Usage:
    validate_conc <validator> <start_date> <end_date> <url> <icechart_dir> [<save_dir>]

"""

import inspect
import logging
import sys
# from distributed import Client
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from docopt import docopt

from validate import base, run

log = logging.getLogger()
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)


class ValidateConc(base.Validate):
    test_variables = ['ice_conc', 'status_flag']

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

    def load_test_data(self, source_glob, preprocess=None):
        test = super().load_test_data(source_glob, preprocess)
        test['status_flag'] = test['status_flag'].astype(np.uint8)
        return test

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

        count, within_10pct = self.match_pct(self.dataset.conc_error, 10)
        count, within_20pct = self.match_pct(self.dataset.conc_error, 20)

        stats = OrderedDict(reference_time=self.dataset.time,

                            total_bias=total_bias,
                            total_stddev=total_stddev,

                            ice_bias=ice_bias,
                            ice_stddev=ice_stddev,

                            water_bias=water_bias,
                            water_stddev=water_stddev,

                            within_10pct=within_10pct,
                            within_20pct=within_20pct,

                            count=count)
        self.dataset = self.dataset.update(stats)

        for variable in ['ice_bias', 'ice_stddev', 'water_bias', 'water_stddev']:
            data = self.aggregate_stats(self.dataset, variable)
            self.dataset[variable + '_6monthly'] = xr.DataArray(data,
                                                                dims='times_6monthly',
                                                                coords={'times_6monthly': data.time.values})

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

        self.dataset.ice_bias_6monthly.attrs = {'unit': '-', 'long_name': 'ice_bias_6monthly', 'x_label': 'Date',
                                                'y_label': 'Bias', 'plot_type': '',
                                                'description': '6 monthly mean of ice bias at end of period'}
        self.dataset.ice_stddev_6monthly.attrs = {'unit': '-', 'long_name': 'ice_SD_6monthly', 'x_label': 'Date',
                                                  'y_label': '$\sigma$', 'plot_type': '',
                                                  'description': '6 monthly mean of ice SD at end of period'}
        self.dataset.water_bias_6monthly.attrs = {'unit': '-', 'long_name': 'water_bias_6monthly', 'x_label': 'Date',
                                                  'y_label': 'Bias', 'plot_type': '',
                                                  'description': '6 monthly mean of water bias at end of period'}
        self.dataset.water_stddev_6monthly.attrs = {'unit': '-', 'long_name': 'water_SD_6monthly', 'x_label': 'Date',
                                                    'y_label': '$\sigma$', 'plot_type': '',
                                                    'description': '6 monthly mean of water SD at end of period'}

        return pd.DataFrame(stats)

    def __call__(self):
        super().__call__()

        log.info("Merging...")
        self.dataset = self.merge(self.ds_ref, self.ds_test, self.test_variables)
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
        json_str = self.dataset.drop('source')[out_vars].to_dataframe().transpose().to_json(double_precision=3)
        return json_str


class ValidateConcCDR450(ValidateConc):
    def select_using_flags(self, data_array):
        # Draft E
        mask_flags = self.has_flag(1) | self.has_flag(2) | \
                     self.has_flag(8) | self.has_flag(32) | \
                     self.has_flag(64) | self.has_flag(128)
        da = data_array.where(np.logical_not(mask_flags))
        return da


class ValidateConcICDR430(ValidateConc):
    def load_test_data(self, source_glob, preprocess=None):
        test = super().load_test_data(source_glob, preprocess)
        test['status_flag'] = test['status_flag'].astype(np.uint8)
        return test


class ValidateConcHYR(ValidateConc):
    def select_using_flags(self, data_array):
        da = data_array.where(self.has_flag(0))
        return da


# Get a dictionary of all the validation classes
validators = {name: obj for name, obj in
              inspect.getmembers(sys.modules[__name__], inspect.isclass)
              if 'merge' in dir(obj)}

if __name__ == "__main__":
    # Client()
    args = docopt(__doc__)
    run.run_val(validators,
                args['<validator>'],
                args['<url>'],
                args['<icechart_dir>'],
                args['<start_date>'],
                args['<end_date>'],
                args['<save_dir>'],
                __doc__,
                __file__)
