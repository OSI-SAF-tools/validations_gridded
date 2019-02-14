def run_val(validators, validator, url, icechart_dir, start_date, end_date, doc_str, file_str, netcdf_dir=None, store_test_files=False):
    results = {}
    for hem in ['NH', 'SH']:
        try:

            try:
                validator_cls = validators[validator]
            except KeyError:
                print(doc_str)
                print("Error: class '{cls}' was not found!\n".format(cls=validator))
                print('<validator> should be one of the following:')
                _available_validators(validators)
                raise
            with validator_cls(url, icechart_dir, hem, str(start_date), str(end_date), store_test_files) as val:
                results[hem] = val()
                if netcdf_dir:
                    val.to_netcdf(netcdf_dir)
        except Exception as e:
            print("Error with hemisphere {0}:\n {1}".format(hem, e))
            # log.error("Error {0} with year, hemisphere {1}".format(e, hem))  # TODO: put logging in
            # raise
    return results


def _available_validators(validators):
    for v in validators.keys():
        print('  ', v)
    print("""
    You can make a new validator creating a subclassing in validate_conc.py or validate_edge.py and  
    adding to the config.yml in 'validator', as follows
     Validations:
      # Template set validation arguments
      Example: # Name of validation set
        url: http//example.com/path/to/netcdf.nc # http or ftp address to OSI SAF data to validate
        icechart_dir: relative_path_{hem} # The relative path to the ice charts relative_path_NH and relative_path_SH,
                                          # where hem is the hemisphere (NH|SH). The base path is given above in MachineConfigs
        validator: ValidationClass # The validation class in validate_conc.py or validate_edge.py.
                                   # Use and existing class or make a subclass.
                                   
    Then add it to config.yml under
    ValidationsToRun:
      NameOfValidationList:
        - ...
        - ...
        - validator

     """)
