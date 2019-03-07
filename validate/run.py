from validate import validate_conc
from validate import validate_edge
import os


def get_validation_classes(modules):
    'Get the validation classes from the submodules'
    for mod in modules:
        attrs = dir(mod)  # This will list a lot of things, but we only want the validation classes
        for attr in attrs:
            try:
                cls = getattr(mod, attr)
                # The validation classes have the attribute 'validation'
                if getattr(cls, 'name') == 'validation':
                    yield cls
            except AttributeError:
                pass


validators = {cls.__name__: cls for cls in get_validation_classes((validate_conc, validate_edge))}


def validate(validator, url, icechart_dir, start_date, end_date, netcdf_dir=None, save_osisaf_files=False):
    """
    :param validator: name of class used to do the validation
    :param url:  url of OSI SAF data
    :param icechart_dir: relative directory of ice chart
    :param start_date: start date in format YYYYMMDD
    :param end_date: end date in format YYYYMMDD
    :param netcdf_dir: directory to store the results
    :param save_osisaf_files: save the OSI SAF files for reuse

    :return: A dictionary as follows: {'NH': NH results as JSON string, 'SH', ... }
    """

    try:
        assert os.path.isdir(icechart_dir)
    except AssertionError:
        raise AssertionError('The ice charts directory {} does not exist'.format(icechart_dir))

    results = {}
    for hem in ['NH', 'SH']:
        try:

            try:
                validator_cls = validators[validator]
            except KeyError:
                print("Error: class '{cls}' was not found!\n".format(cls=validator))
                print('<validator> should be one of the following:')
                for av in available_validators(validators):
                    print(av)
                raise ()
            with validator_cls(url, icechart_dir, hem, str(start_date), str(end_date), save_osisaf_files) as val:
                results[hem] = val()
                if netcdf_dir:
                    val.to_netcdf(netcdf_dir)
        except Exception as e:
            print("Error with hemisphere {0}:\n {1}".format(hem, e))
            # log.error("Error {0} with year, hemisphere {1}".format(e, hem))  # TODO: put logging in
            raise
    return results


def available_validators(validators):
    for v in validators.keys():
        yield '    ' + v
    yield """
    You can make a new validator creating a subclass in validate_conc.py or validate_edge.py and  
    adding it to the config.yml, as follows:

     Validations:
      # Template set validation arguments
      Example: # Name of validation set
        <url>: http//example.com/path/to/netcdf.nc # http or ftp address to OSI SAF data to validate
        <icechart_dir>: relative_path_{hem} # The relative path to the ice charts relative_path_NH and relative_path_SH,
                                            # where hem is the hemisphere (NH|SH). The base path is given above in MachineConfigs
        <validator>: ValidationClass # The validation class in validate_conc.py or validate_edge.py.
                                     # Use and existing class or make a subclass.

    Then add it to config.yml under
    ValidationsToRun:
      <NameOfValidationList>:
        - ...
        - ...
        - <validator>

     """
