#!/usr/bin/env bash

# Run a single validation
#python3 validate ValidateConcHYR 20180101 20190101 http://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/amsr2_conc/{Y}/{m:02d}/ice_conc_{hem}_polstere-100_amsr2_{Y}{m:02d}{d:02d}1200.nc /data2/ice_charts/gridded/OSISAF_NH/

# Run a set of validations specified in config.yml
python3 validation_runner.py DMI

