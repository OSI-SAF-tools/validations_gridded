#!/usr/bin/python3

"""
validate

Runs a single validation and stores the results as a netCDF file.

Usage:
    validate <validator> <start_date> <end_date> <url> <icechart_dir> <save_dir>
    validate -h | --help

Attributes:
    validator: The name of the validation. It should be on of the validations in
               validate.py/config.yml
    start_date: Start date of validation interval, in the format %Y%m%d
    end_date: End date of validation interval, in the format %Y%m%d
    url: url of osi saf data source
    icechart_dir: Relative path to ice chart directory
    save_dir: Directory to save netCDF file containing results


Example:

python3 validate ValidateConcHYR 20100101 20100301 \
ftp://osisaf.met.no/archive/ice/conc/{Y}/{m:02d}/ice_conc_{hem}_polstere-100_multi_{Y}{m:02d}{d:02d}1200.nc \
/path/to/ice_charts/directory/ /path/to/results/directory/
\

"""

import os
import sys

from docopt import docopt

path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, '..')
sys.path.insert(0, path)
from validate import validate

if __name__ == "__main__":
    args = docopt(__doc__)
    result = validate(args['<validator>'],
                      args['<url>'],
                      args['<icechart_dir>'],
                      args['<start_date>'],
                      args['<end_date>'],
                      args['<save_dir>']
                      )
    print(result)
