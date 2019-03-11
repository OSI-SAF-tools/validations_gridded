#!/usr/bin/env bash


####################################################################
## Run a single validation. It saves the results to a netCDF file.##
####################################################################

python3 validate ValidateConcHYR 20100101 20100301 \
                 ftp://osisaf.met.no/archive/ice/conc/{Y}/{m:02d}/ice_conc_{hem}_polstere-100_multi_{Y}{m:02d}{d:02d}1200.nc \
                 /data/jol/ice_charts/gridded/OSISAF/ /data/jol/results/
                 \

## Documentation
#python3 validate --help


######################################################
## Run a set of validations specified in config.yml ##
######################################################

# Put the results in a database. Used in production.
python3 validation_runner.py to_database ProductionDMI

# Put the results in a database and save the full results as a netCDF file and saves the OSI SAF files for reuse.
# Used to look further into the results.
python3 validation_runner.py ProductionDMI 20150101 20170101

# Documentation
python3 validation_runner.py --help
