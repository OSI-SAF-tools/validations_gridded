#!/usr/bin/env python3

import warnings
from functools import partial

from validate import base
import dask.array as da
import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from skimage import feature
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

warnings.filterwarnings('ignore')


class ValidateEdge(base.Validate):
    test_variables = ['ice_edge', ]

    @staticmethod
    def standardise_func_ice_edge(ice_edge):
        """
        Where the ice edge has a flag of > 1 set the value to 100% so that we can make a comparison
        with the ice chart
        :param ice_edge: the ice edge
        :return:
        """
        result = np.select([np.isnan(ice_edge), ice_edge > 1, ice_edge == 1, ice_edge == 0],
                           [np.nan, 100, 0, np.nan])
        chunks = list(result.shape)
        chunks[0] = 1
        return da.from_array(result.astype(np.float32), chunks=chunks)

    def standardise_func_ice_chart(self, ref_low, ref_upp):
        """
        We need either ice or no ice, so if the ice concentration in the ice chart is above 35%
        set to 100 otherwise 0%
        :param ref_low: the lower limit of ice concentration in the ice chart
        :param ref_upp: the upper limit of ice concentration in the ice chart
        :return:
        """
        ice_threshold = 35
        ref_low = ref_low.where(np.logical_not(self.exclude_from_icechart(ref_low, ref_upp)))
        result = np.select([np.isnan(ref_low), ref_low > ice_threshold, ref_low <= ice_threshold],
                           [np.nan, 100, 0])
        chunks = list(result.shape)
        chunks[0] = 1
        return da.from_array(result.astype(np.float32), chunks=chunks)

    @staticmethod
    def has_edge_occurred(arr, window=3):
        """
        For a given location, if the ice concentration has changed in the last 'window' days,
        the edge must have occurred during those days. This is done because the ice chart is
        assumed to be made with data from the last 'window' days.
        :param arr:
        :param window:
        :return: DataArray which is True where the edge occurred
        """
        ice = (arr > 1)  # Get the ice from the flag
        roll_arr = ice.rolling(min_periods=1, time=window, center=False)
        ice_edge_roll_min = roll_arr.min()
        ice_edge_roll_max = roll_arr.max()
        result = np.not_equal(ice_edge_roll_min, ice_edge_roll_max)
        return result

    @staticmethod
    def exclude_from_icechart(ref_low, ref_upp):
        """
        If the ice concentration given in the ice chart is an interval which includes 35% then it is not
        possible to know where the edge is, we say the edge has occurred in all these pixels
        :param ref_low: lower ice concentration limit
        :param ref_upp: upper ice concentration limit
        :return:
        """
        return np.logical_and(ref_low <= 35, ref_upp >= 35)

    def standardise_data(self, dataset):
        """
        Put the ice chart and satellite data on a common reference so that we can compare them
        """
        dataset['ice_edge_standard'] = xr.DataArray(self.standardise_func_ice_edge(dataset['ice_edge']),
                                                    dims=dataset.ice_edge.dims,
                                                    coords=dataset.ice_edge.coords)

        dataset['ice_chart_standard'] = xr.DataArray(self.standardise_func_ice_chart(dataset['lower'],
                                                                                     dataset['upper']),
                                                     dims=dataset.lower.dims,
                                                     coords=dataset.lower.coords)

        dataset['ice_edge_standard'] = dataset['ice_edge_standard'].where(
            ~np.isnan(dataset['ice_chart_standard']))
        dataset['ice_chart_standard'] = dataset['ice_chart_standard'].where(
            ~np.isnan(dataset['ice_edge_standard']))
        return dataset

    """
    Validate based on area
    """

    def compute_percent_area_of_diff(self):
        self.dataset['diff_standard'] = (self.dataset.ice_chart_standard - self.dataset.ice_edge_standard)
        diff = self.dataset['diff_standard'].astype(np.int)
        self.dataset['false_ice'] = diff.where(diff == -100).count(
            dim=['xc', 'yc'])
        self.dataset['false_water'] = diff.where(diff == 100).count(
            dim=['xc', 'yc'])
        self.dataset['true_ice'] = diff.where(np.logical_and(diff == 0,
                                                             self.dataset['ice_chart_standard'] == 100)).count(
            dim=['xc', 'yc'])

        self.dataset['false_ice'].attrs = {'unit': '', 'long_name': 'Num_of_false_ice_pixels',
                                           'x_label': 'Date',
                                           'y_label': 'Pixels (%)', 'plot_type': 'line'}
        self.dataset['false_water'].attrs = {'unit': '', 'long_name': 'Num_of_false_ice_pixels',
                                             'x_label': 'Date',
                                             'y_label': 'Pixels (%)', 'plot_type': 'line'}
        self.dataset['true_ice'].attrs = {'unit': '', 'long_name': 'Num_of_true_ice_pixels',
                                          'x_label': 'Date',
                                          'y_label': 'Pixels (%)', 'plot_type': 'line'}

    """
    Validate based on distance between the edges 
    """

    @staticmethod
    def edge(arr, sigma=0):
        """
        Get the edge from the ice-edge
        :param arr:
        :param sigma:
        :return:
        """
        return feature.canny(arr.astype('float32').values, sigma=sigma)

    @staticmethod
    def get_xy_coords(da):
        df = da.to_dataframe()
        # convert to a list of coordinate pairs
        return df[[df.keys()[-1], ]][df[df.keys()[-1]].values].to_records()[['xc', 'yc']].tolist()

    def distance_between_edges(self, ds):
        """
        For each ice-chart-edge point, look up the nearest ice-edge-product point
        :param ds:
        :return:
        """
        try:
            xy_test = self.get_xy_coords(ds['ice_edge_line'])
            kdt = KDTree(xy_test)
            xy_chart = self.get_xy_coords(ds['ice_chart_line'])
            v = np.array(kdt.query(xy_chart)[0])
            result = xr.DataArray([v.mean(), np.median(v)])
            # pl.scatter(*list(zip(*xy_test)), s=0.01, c='b')
            # pl.scatter(*list(zip(*xy_chart)), s=0.01, c='r')
            # pl.show()
            return result
        except ValueError:
            return xr.DataArray(np.nan)

    def compute_distance_between_edges(self):
        self.dataset['ice_chart_line'] = self.dataset['ice_chart_standard']. \
            groupby('time').apply(partial(self.edge, sigma=0))
        self.dataset['ice_edge_line'] = self.dataset['ice_edge_standard'].groupby('time').apply(self.edge)
        # self.dataset['ice_edge_line'] = np.logical_or(self.dataset['ice_edge_line'], self.dataset['edge_occured'])
        log.info('loading arrays')
        # self.dataset.persist()
        ds_edges = self.dataset[['ice_edge_line', 'ice_chart_line']].load()
        log.info('computing distances')
        da_distances = ds_edges.groupby('time').apply(self.distance_between_edges)

        self.dataset['distance_between_edges_mean'] = da_distances.isel(dim_0=0)
        self.dataset['distance_between_edges_median'] = da_distances.isel(dim_0=1)

        self.dataset['distance_between_edges_mean'].attrs = {'unit': 'km', 'long_name': 'distance_between_edges_mean',
                                                             'x_label': 'Date',
                                                             'y_label': 'Mean Distance', 'plot_type': 'line'}
        self.dataset['distance_between_edges_median'].attrs = {'unit': 'km',
                                                               'long_name': 'distance_between_edges_median',
                                                               'x_label': 'Date',
                                                               'y_label': 'Median Distance', 'plot_type': 'line'}

    def __call__(self):
        super().__call__()

        # ds_test['edge_occured'] = self.has_edge_occurred(ds_test['ice_edge'])
        log.info("Merging...")
        dataset = self.merge(self.ds_ref, self.ds_test, self.test_variables)
        log.info("Standardising...")
        self.dataset = self.standardise_data(dataset)
        self.dataset.compute()

        del self.ds_ref, self.ds_test, dataset

        self.compute_distance_between_edges()
        self.compute_percent_area_of_diff()

        for variable in ['distance_between_edges_mean', ]:
            data = self.aggregate_stats(self.dataset, variable)
            self.dataset[variable + '_6monthly'] = xr.DataArray(data,
                                                                dims='times_6monthly',
                                                                coords={'times_6monthly': data.time.values})

        out_vars = ['distance_between_edges_mean', 'false_ice', 'false_water', 'true_ice']
        json_str = self.dataset[out_vars].to_dataframe().transpose().to_json(double_precision=3)
        return json_str



