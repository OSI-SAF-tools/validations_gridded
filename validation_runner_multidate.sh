#!/usr/bin/env bash

# Does the validation in parallel with one day per netCDF file output.
# Used for level 2 validation.
#
# $1 = validation name
# $2 = start date (e.g.: yyyy-mm-dd)
# $3 = end date
#
# Example: ./validation_runner_multidate.sh ice_conc_amsr2_level2_tud 20180101 20190101

# create an array of dates
end_date=$(date -d "$3" +%Y%m%d)
iter_date=$(date -d "$2" +%Y%m%d)
dates=""
until [[ ${iter_date} == ${end_date} ]]; do
    dates+=" ${iter_date}" # append to dates
    iter_date=$(date -d "${iter_date} +1 day" +%Y%m%d) # advance the date
done

parallel python3 validation_runner.py "$1" {} {} ::: $dates
