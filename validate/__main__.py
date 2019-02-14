"""
validate

Usage:
    validate <validator> <start_date> <end_date> <url> <icechart_dir> [<save_dir>]

Attributes:
    validator: The name of the validation. It should be on of the validations in
               validate.py/config.yml
    start_date: Start date of validation interval, in the format %Y%m%d
    end_date: End date of validation interval, in the format %Y%m%d
    url: url of osi saf data source
    icechart_dir: Relative path to ice chart directory
    save_dir: Directory to save netCDF file containing results

"""

from docopt import docopt

from validate import validate

if __name__ == "__main__":
    # Client()
    args = docopt(__doc__)
    validate(args['<validator>'],
             args['<url>'],
             args['<icechart_dir>'],
             args['<start_date>'],
             args['<end_date>'],
             args['<save_dir>']
             )
