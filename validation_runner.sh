#!/usr/bin/env bash


####################################################################
## Run a single validation. It saves the results to a netCDF file.##
####################################################################

python3 validate ValidateConcHYR 20100101 20100301 \
                 ftp://osisaf.met.no/archive/ice/conc/{Y}/{m:02d}/ice_conc_{hem}_polstere-100_multi_{Y}{m:02d}{d:02d}1200.nc \
                 /data/jol/ice_charts/gridded/OSISAF/ /data/jol/results/

## Documentation
python3 validate --help


######################################################
## Run a set of validations specified in config.yml ##
######################################################

# Run validation for the list ProductionDMI and the results in a database (used in production):
python3 validation_runner.py to_database ProductionDMI

# Run validation and save the full results as a netCDF file and saves the OSI SAF files for reuse
# (used to look further into the results):

# Run validation for the list ProductionDMI in config.yml under Validations:
python3 validation_runner.py ice_conc_edge_http 20150101 20170101

# Run single validation, ice_conc_edge_http, given in config.yml ValidationLists:
python3 validation_runner.py ProductionDMI 20150101 20170101

# Documentation
python3 validation_runner.py --help
