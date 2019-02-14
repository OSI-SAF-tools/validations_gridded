#!/usr/bin/python3

"""
validation_runner

Usage:
    validation_runner.py <validation_set> [--save_full_results] [--save_osisaf_files]
    validation_runner.py --available_validators
    validation_runner.py -h | --help

Options:
    --save_full_results     Saves the results to a netCDF file, in the results directory given in config.yml
    --save_osisaf_files     Saves the OSI SAF files in /tmp for reuse

<validation_set>    Should be one included in config.yml under ValidationsToRun

Example:
    python3 validation_runner.py DMI_production
"""

import datetime
import logging
import platform
import sys
from os.path import join, dirname

import yaml
from docopt import docopt

from validate import validate, available_validators

log = logging.getLogger()
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)


def get_config():
    current_dir = dirname(__file__)
    with open(join(current_dir, "config.yml"), 'r') as stream:
        base_url = yaml.load(stream)
    return base_url


def run_over_dates(start, stop, config, validation_set, save_full_results, save_osisaf_files):
    machine_cfg = config['MachineConfigs'][platform.node()]
    validations_list = config['ValidationsToRun'][validation_set]
    for val_name in validations_list:
        val_cfg = config['Validations'][val_name]
        try:

            ice_chart_dir = join(machine_cfg['ice_charts'], val_cfg['icechart_dir'])
            if save_full_results:
                results_dir = machine_cfg['results']
            else:
                results_dir = None
            yield validate(val_cfg['validator'], val_cfg['url'], ice_chart_dir, start, stop, results_dir,
                           save_osisaf_files)

        except Exception as err:
            log.error('Expection {0} with {1}'.format(err, val_name))
            raise


def to_database(config, validation_set, save_full_results, save_osisaf_files):
    machine_cfg = config['MachineConfigs'][platform.node()]

    start = '20180101'  # TODO: get this from database
    today = datetime.datetime.now().strftime('%Y%m%d')
    # stop = today
    stop = '20180201'

    for json_str in run_over_dates(start, stop, config, validation_set, save_full_results, save_osisaf_files):
        print(json_str)
        if machine_cfg['mongodb_ip']:
            pass  # TODO: Insert result into a database
        else:
            pass  # save to json file in results directory


if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--available_validators']:
        print('Available validators:')
        available_validators()
    else:
        cfg = get_config()
        to_database(cfg, args['<validation_set>'], args['--save_full_results'], args['--save_osisaf_files'])
