import datetime
from dateutil.relativedelta import relativedelta
import os
import sys
# import pickle

import HydroBlocks  # as HB


def Read_Metadata_File(file):

    import json
    # metadata = json.load(open(file,'r'))['HydroBlocks']
    with open(file, 'r') as f:
        metadata = json.load(f)['HydroBlocks']

    return metadata


# Read in the metadata file
metadata_file = sys.argv[1]
metadata = Read_Metadata_File(metadata_file)
info = metadata

# Define idate and fdate
idate = datetime.datetime(metadata['startdate']['year'],
                          metadata['startdate']['month'],
                          metadata['startdate']['day'], 0)
fdate = datetime.datetime(
    metadata['enddate']['year'], metadata['enddate']['month'],
    metadata['enddate']['day'], 0) + datetime.timedelta(days=1)

spin_date = datetime.datetime(metadata['spin_date']['year'],
                              metadata['spin_date']['month'],
                              metadata['spin_date']['day'], 0)
if metadata['spin_date']['month'] == 12:
    spin_date = spin_date + datetime.timedelta(days=1)

# setup debug stuff, if needed
if "debug_dir" in metadata.keys():
    flag_log = True
    debug_dir = metadata["debug_dir"]
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    f_log = open(debug_dir + '/_runtime_HydroBlocks.log', 'w')
    f_log.write('Runtime Log-file for HydroBlocks\n')
else:
    flag_log = False

# pickle.dump(metadata, open(debug_dir + '/metadata.p', 'wb'))

# timing setups
t_overall = datetime.datetime.now()

# Run the segments for the model
sidate = idate
sfdate = idate
while sidate < fdate:

    # prep for timing measurement
    t0 = datetime.datetime.now()
    t_start = t0
    if flag_log:
        f_log.write('\nStarting segment for ' + str(sidate) + '\n')

    sfdate = sidate + relativedelta(
        years=metadata['segment']['years_per_segment'])
    if sfdate > fdate:
        sfdate = fdate
    # Set the parameters
    info['idate'] = sidate
    info['fdate'] = sfdate
    info['spin_date'] = spin_date

    # Run the model
    # Initialize
    t_start = datetime.datetime.now()
    if flag_log:
        f_log.write('Initialization:\n')
    HB = HydroBlocks.initialize(info, flag_log, f_log)
    if flag_log:
        f_log.write('   Initialization Time: ' +
                    str(datetime.datetime.now() - t_start) + '\n')

    # Run the model
    t_start = datetime.datetime.now()
    if flag_log:
        f_log.write('Model Compute:\n')
    HB.run(info, flag_log, f_log)
    if flag_log:
        f_log.write('   Model Run Time: ' +
                    str(datetime.datetime.now() - t_start) + '\n')

    # Finalize
    t_start = datetime.datetime.now()
    if flag_log:
        f_log.write('Finalisation:\n')
    HB.finalize()
    if flag_log:
        f_log.write('   Finalization Time: ' +
                    str(datetime.datetime.now() - t_start) + '\n')

    # Update initial time step
    sidate = sfdate

# Close the log file
if flag_log:
    f_log.write('Overall runtime: ' +
                str(datetime.datetime.now() - t_overall) + '\n')
    f_log.close()
