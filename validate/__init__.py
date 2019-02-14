
from validate import validate_conc
from validate import validate_edge

from validate.validate_conc import validators as validators_conc_dict
from validate.validate_edge import validators as validators_edge_dict
from validate.run import run_val, _available_validators

validators = {**validators_conc_dict, **validators_edge_dict}


def validate(validator, url, icechart_dir, start_date, end_date, netcdf_dir=None, save_osisaf_files=False):
    """
    :param validator: name of class used to do the validation
    :param url:  url of OSI SAF data
    :param icechart_dir: relative directory of ice chart
    :param start_date: start date in format YYYYMMDD
    :param end_date: end date in format YYYYMMDD
    :return: A dictionary as follows: {'NH': NH results as JSON string, 'SH', ... }
    """

    name_str = '{0} or {1}'.format(validate_conc.__name__, validate_edge.__name__)
    return run_val(validators, validator, url, icechart_dir, start_date, end_date, validate.__doc__,  name_str,
                   netcdf_dir, save_osisaf_files)


def available_validators():
    return _available_validators(validators)


