#
# Helperfile for netcdf related functions. Currently flexibly enables
# compression according to Python version and netCDF4 version.
#
# (C) 2024, sberendsen@soton.ac.uk
#
# TODO make the compression level a parameter
# TODO make it determine the python & netCDF4 version only once
# TODO make it automatically determine a chunksize based on the variable shape

import sys

import netCDF4

# Settings ---------------------------------------------------------------------
max_chunksize = 1000000
min_chunksize = 1000


# Functions --------------------------------------------------------------------
def create_netcdf_variable(nc_group,
                           var_name: str,
                           var_type: str,
                           dimensions: tuple,
                           compress: bool = False,
                           least_significant_digit: int = None):
    
    if compress and sys.version_info[0] == 3 and sys.version_info[1] >= 12:
        var = nc_group.createVariable(var_name,
                                         var_type,
                                         dimensions,
                                         least_significant_digit=least_significant_digit,
                                         compression="zstd")
    elif compress:
        var = nc_group.createVariable(var_name,
                                         var_type,
                                         dimensions,
                                         least_significant_digit=least_significant_digit,
                                         zlib=True,
                                         complevel=6)
    else:
        var = nc_group.createVariable(var_name, var_type, dimensions)

    return var


if __name__ == "__main__":

    raise Exception("This file is not meant to be run standalone.")
