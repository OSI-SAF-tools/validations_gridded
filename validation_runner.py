#!/usr/bin/python3

"""
validation_runner

Usage:
    validation_runner.py <validation_names> <start> <end>
    validation_runner.py to_database <validation_names>
    validation_runner.py --available_validators
    validation_runner.py -h | --help

Options:
    -h --help               Show this screen
    --available_validators  List the available validators to use for <validation_names>

Arguments:
    to_database        Save to results to a database and do NOT store OSI SAF files locally or save the results in a netCDF file
    <validation_names> Should be one included in config.yml under ValidationLists or Validations
    <start>            Start date of validation period, in the format YYYYmmdd
    <end>              End date of validation period, in the format YYYYmmdd

With 'validation_runner.py to_database <validation_names>' the latest results are stored in a database, whereas with
'validation_runner.py <validation_names> <start> <end>' the results are stored as a netCDF to the directory
MachineConfigs/results in config.yml.

Examples:
    Run validation for the list ProductionDMI and the results in a database (used in production):
        python3 validation_runner.py to_database ProductionDMI

    Run validation and save the full results as a netCDF file and saves the OSI SAF files for reuse
    (used to look further into the results):
        Run validation for the list ProductionDMI in config.yml under Validations:
            python3 validation_runner.py ProductionDMI 20150101 20170101
        Run single validation, ice_conc_edge_http, given in config.yml ValidationLists:
            python3 validation_runner.py ice_conc_edge_http 20150101 20170101

    Documentation
        python3 validation_runner.py --help
"""

import datetime
import logging
import platform
import sys
import yaml
from docopt import docopt
from os.path import join, dirname

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


def validation(config, validation_set, start, end, save_full_results, save_osisaf_files):
    machine_cfg = config['MachineConfigs'][platform.node()]

    try:
        validations_list = config['ValidationLists'][validation_set]
    except KeyError:
        validations_list = [validation_set, ]

    if save_full_results:
        results_dir = machine_cfg['results']
    else:
        results_dir = None

    for val_name in validations_list:
        val_cfg = config['Validations'][val_name]
        try:
            ice_chart_dir = join(machine_cfg['ice_charts'], val_cfg['icechart_dir'])
            results = validate(val_cfg['validator'], val_cfg['url'], ice_chart_dir, start, end,
                               results_dir, save_osisaf_files)
            yield val_name, results
        except Exception as err:
            log.error('Expection {0} with {1}'.format(err, val_name))
            raise


def to_database(config, validation_set):
    start  # TODO: get from db
    end  # TODO: get from db
    if machine_cfg['mongodb_ip']:
        pass  # TODO: Insert result into a database
    for name, res in validation(config, validation_set, start, end, False, False):
        pass  # TODO: insert into database


if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--available_validators']:
        print('Available validators:')
        for av in available_validators():
            print(av)
    else:
        cfg = get_config()
        if args['to_database']:
            to_database(cfg, args['<validation_names>'])
        else:
            save_full_results = save_osisaf_files = True
            for name, result in validation(cfg, args['<validation_names>'], args['<start>'], args['<end>'],
                                           save_full_results, save_osisaf_files):
                print(name)
                print(result)
