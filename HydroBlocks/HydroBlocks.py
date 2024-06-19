import os
import numpy as np
import netCDF4 as nc
import h5py
import datetime
import time
# import sys
import scipy.sparse as sparse
# import pickle
# from joblib import cpu_count

import netcdf_utils as ncutil



def assign_string(nelem, pstring):
    # def assign_string(dtype,pstring):

    # print(str(dtype))
    # nelem = int(str(dtype)[2:])
    # nelem = len)
    tmp = np.chararray(shape=(nelem, ))
    tmp[:] = ' '
    tmp2 = []
    for letter in pstring:
        # print(letter)
        tmp2.append(letter)
    tmp[0:len(tmp2)] = tmp2

    return tmp


def initialize(info, flag_log: bool, f_log):

    HB = HydroBlocks(info, flag_log, f_log)

    return HB


class HydroBlocks:

    def __init__(self, info, flag_log: bool, f_log):

        # Read in general information
        self.general_information(info)

        # Initialize Noah-MP
        print("Initializing Noah-MP", flush=True)
        t_start = datetime.datetime.now()
        if flag_log:
            f_log.write(f"   - NOAH-MP Init:\n")
        self.initialize_noahmp(flag_log, f_log)
        if flag_log:
            f_log.write(
                f"     Init Overall: {datetime.datetime.now() - t_start}\n")

        # Initialize subsurface module
        print("Initializing subsurface module", flush=True)
        t_start = datetime.datetime.now()
        if flag_log:
            f_log.write(f"   - Subsurface Init:\n")
        self.initialize_subsurface()
        if flag_log:
            f_log.write("     Subsurface Init Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # -- Needs to be completed -- Noemi
        # Initialize human water use module
        # print("Initializing Human Water Management",flush=True)
        # self.initialize_hwu(info)

        # Other metrics
        t_start = datetime.datetime.now()
        if flag_log:
            f_log.write(f"   - Other Metrics Init:\n")
        self.dE = 0.0
        self.r = 0.0
        self.dr = 0.0
        self.et = 0.0
        self.etran = 0.0
        self.esoil = 0.0
        self.ecan = 0.0
        self.prcp = 0.0
        self.q = 0.0
        self.errwat = 0.0
        self.erreng = 0.0
        self.dzwt0 = np.zeros(self.nhru, dtype=np.float32)
        self.beg_wb = np.zeros(self.nhru, dtype=np.float32)
        self.end_wb = np.zeros(self.nhru, dtype=np.float32)
        self.itime = 0
        if flag_log:
            f_log.write("     Other Metrics Init Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Restart from initial conditions?
        t_start = datetime.datetime.now()
        if flag_log:
            f_log.write(f"   - Restart:\n")
        self.restart()
        if flag_log:
            f_log.write("     Restart Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        return

    def restart(self, ):

        file_restart = '%s/%s.h5' % (self.metadata['restart']['dir'],
                                     self.idate.strftime('%Y-%m-%d'))
        if (os.path.exists(file_restart) is False
                or self.metadata['restart']['flag'] is False):
            print("Cold startup", flush=True)
            return

        # Read in the restart information
        fp = h5py.File(file_restart, 'r')
        self.noahmp.smceq[:] = fp['smceq'][:]
        self.noahmp.albold[:] = fp['albold'][:]
        self.noahmp.sneqvo[:] = fp['sneqvo'][:]
        self.noahmp.stc[:] = fp['stc'][:]
        self.noahmp.sh2o[:] = fp['sh2o'][:]
        self.noahmp.smc[:] = fp['smc'][:]
        self.noahmp.tah[:] = fp['tah'][:]
        self.noahmp.eah[:] = fp['eah'][:]
        self.noahmp.fwet[:] = fp['fwet'][:]
        self.noahmp.canliq[:] = fp['canliq'][:]
        self.noahmp.canice[:] = fp['canice'][:]
        self.noahmp.tv[:] = fp['tv'][:]
        self.noahmp.tg[:] = fp['tg'][:]
        self.noahmp.qsfc1d[:] = fp['qsfc1d'][:]
        self.noahmp.qsnow[:] = fp['qsnow'][:]
        self.noahmp.isnow[:] = fp['isnow'][:]
        self.noahmp.zsnso[:] = fp['zsnso'][:]
        self.noahmp.sndpth[:] = fp['sndpth'][:]
        self.noahmp.swe[:] = fp['swe'][:]
        self.noahmp.snice[:] = fp['snice'][:]
        self.noahmp.snliq[:] = fp['snliq'][:]
        self.noahmp.zwt[:] = fp['zwt'][:]
        self.noahmp.wa[:] = fp['wa'][:]
        self.noahmp.wt[:] = fp['wt'][:]
        self.noahmp.wslake[:] = fp['wslake'][:]
        self.noahmp.lfmass[:] = fp['lfmass'][:]
        self.noahmp.rtmass[:] = fp['rtmass'][:]
        self.noahmp.stmass[:] = fp['stmass'][:]
        self.noahmp.wood[:] = fp['wood'][:]
        self.noahmp.stblcp[:] = fp['stblcp'][:]
        self.noahmp.fastcp[:] = fp['fastcp'][:]
        self.noahmp.plai[:] = fp['plai'][:]
        self.noahmp.psai[:] = fp['psai'][:]
        self.noahmp.cm[:] = fp['cm'][:]
        self.noahmp.ch[:] = fp['ch'][:]
        self.noahmp.tauss[:] = fp['tauss'][:]
        self.noahmp.smcwtd[:] = fp['smcwtd'][:]
        self.noahmp.deeprech[:] = fp['deeprech'][:]
        self.noahmp.rech[:] = fp['rech'][:]
        fp.close()
        del fp

        return

    def general_information(self, info):

        # Define the metadata
        self.dt = info['dt']
        self.dtt = self.dt  # info['dtt']
        self.nsoil = len(info['dz'])  # ['nsoil']
        # import multiprocessing
        # self.ncores = multiprocessing.cpu_count()
        self.ncores = info['ncores']  # cpu_count() # Used in OMP in Noah-MP
        print("Threads in NOAH-MP", self.ncores, flush=True)
        self.idate = info['idate']
        self.fdate = info['fdate']
        self.spin_date = info['spin_date']
        self.mkl_flag = info['mkl_flag']
        self.dt_timedelta = datetime.timedelta(seconds=self.dt)
        self.input_fp = nc.Dataset(info['input_file'])
        self.dx = self.input_fp.groups['metadata'].dx
        self.nhru = len(self.input_fp.dimensions['hru'])
        self.surface_flow_flag = info['surface_flow_flag']
        self.subsurface_module = info['subsurface_module']
        # self.hwu_flag = info['water_management']['hwu_flag']
        self.pct = self.input_fp.groups['parameters'].variables[
            'area_pct'][:] / 100
        self.pct = self.pct / np.sum(self.pct)
        self.metadata = info
        self.m = self.input_fp.groups['parameters'].variables['m'][:]  # Noemi
        # self.m[:] = 0.1 # m
        self.input_fp_meteo_time = self.input_fp.groups[
            'meteorology'].variables['time']

        # Create a list of all the dates
        dates = []
        date = self.idate
        while date < self.fdate:
            dates.append(date)
            date = date + self.dt_timedelta
        self.dates = np.array(dates)

        # Read in all the meteorology for the given time period
        # determine the first time step for the meteorology
        var = self.input_fp['meteorology'].variables['time']
        ndates = var[:]
        # convert current date to num
        # ndate = nc.date2num(self.idate, units=var.units, calendar=var.calendar)
        # minitial_time = np.where(ndates == ndate)[0][0]
        # define period to extract
        idate = nc.date2num(self.idate, units=var.units, calendar=var.calendar)
        fdate = nc.date2num(self.fdate, units=var.units, calendar=var.calendar)
        idx = np.where((ndates >= idate) & (ndates < fdate))[0]
        self.meteodata = {}
        for var in [
                'lwdown', 'swdown', 'psurf', 'wind', 'tair', 'spfh', 'precip'
        ]:
            self.meteodata[var] = self.input_fp['meteorology'][var][idx, :]
        # Define subtimestep
        dt = self.dt
        dt_meteo = (ndates[1] - ndates[0]) * 3600.0
        self.nt_timestep = int(dt_meteo / dt)

        return

    def initialize_noahmp(self, flag_log: bool, f_log):

        # Initialize noahmp
        import pyNoahMP.NoahMP
        self.noahmp = pyNoahMP.NoahMP

        # Initialize parameters
        self.noahmp.ncells = self.nhru
        self.noahmp.nsoil = self.nsoil
        self.noahmp.nsnow = 3
        self.noahmp.dt = self.dt

        # Allocate memory
        # 1d,integer
        t_start = datetime.datetime.now()
        vars = [
            'isnow', 'isurban', 'slopetyp', 'soiltyp', 'vegtyp', 'ice', 'isc',
            'ist', 'root_depth'
        ]
        for var in vars:
            exec(f'self.noahmp.{var} = np.zeros(self.nhru).astype(np.int32)')
        if flag_log:
            f_log.write("     Allocate Memory (1d, int) Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # 1d,real
        t_start = datetime.datetime.now()
        vars = [
            'z_ml', 'lwdn', 'swdn', 'p_ml', 'psfc', 'prcp', 't_ml', 'q_ml',
            'u_ml', 'v_ml', 'fsh', 'ssoil', 'salb', 'fsno', 'swe', 'sndpth',
            'emissi', 'qsfc1d', 'tv', 'tg', 'canice', 'canliq', 'eah', 'tah',
            'cm', 'ch', 'fwet', 'sneqvo', 'albold', 'qsnow', 'wslake', 'zwt',
            'dzwt', 'wa', 'wt', 'smcwtd', 'deeprech', 'rech', 'lfmass',
            'rtmass', 'stmass', 'wood', 'stblcp', 'fastcp', 'plai', 'psai',
            'tauss', 't2mv', 't2mb', 'q2mv', 'q2mb', 'trad', 'nee', 'gpp',
            'npp', 'fvegmp', 'runsf', 'runsb', 'ecan', 'etran', 'esoil', 'fsa',
            'fira', 'apar', 'psn', 'sav', 'sag', 'rssun', 'rssha', 'bgap',
            'wgap', 'tgv', 'tgb', 'chv', 'chb', 'irc', 'irg', 'shc', 'shg',
            'evg', 'ghv', 'irb', 'shb', 'evb', 'ghb', 'tr', 'evc', 'chleaf',
            'chuc', 'chv2', 'chb2', 'cosz', 'lat', 'lon', 'fveg', 'fvgmax',
            'fpice', 'fcev', 'fgev', 'fctr', 'qsnbot', 'ponding', 'ponding1',
            'ponding2', 'fsr', 'co2pp', 'o2pp', 'foln', 'tbot', 'smcmax',
            'smcdry', 'smcref', 'errwat', 'si0', 'si1', 'zwt0', 'minzwt',
            'co2air', 'o2air', 'bb0', 'drysmc0', 'f110', 'maxsmc0', 'refsmc0',
            'satpsi0', 'satdk0', 'satdw0', 'wltsmc0', 'qtz0'
        ]
        for var in vars:
            exec(f'self.noahmp.{var} = np.zeros(self.nhru).astype(np.float32)')
        if flag_log:
            f_log.write("     Allocate Memory (1d, real) Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # 2d,real
        t_start = datetime.datetime.now()
        # vars = ['stc','sh2o','smc','smceq','zsnso','snice','snliq','ficeold','zsoil','sldpth','hdiv']
        vars = ['sh2o', 'smc', 'smceq', 'zsoil', 'sldpth', 'hdiv']
        for var in vars:
            exec(f'self.noahmp.{var} = '
                 'np.zeros((self.nhru,self.nsoil),order=\'F\')'
                 '.astype(np.float32)')
        self.noahmp.zsnso = np.zeros(
            (self.nhru, self.noahmp.nsoil + self.noahmp.nsnow),
            order='F').astype(np.float32)
        self.noahmp.stc = np.zeros(
            (self.nhru, self.noahmp.nsoil + self.noahmp.nsnow),
            order='F').astype(np.float32)
        self.noahmp.snice = np.zeros((self.nhru, self.noahmp.nsnow),
                                     order='F').astype(np.float32)
        self.noahmp.snliq = np.zeros((self.nhru, self.noahmp.nsnow),
                                     order='F').astype(np.float32)
        self.noahmp.ficeold = np.zeros((self.nhru, self.noahmp.nsnow),
                                       order='F').astype(np.float32)
        if flag_log:
            f_log.write("     Allocate Memory (2d, real) Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Define data
        t_start = datetime.datetime.now()
        self.noahmp.llanduse = 'MODIFIED_IGBP_MODIS_NOAH'
        # assign_string(self.noahmp.llanduse.dtype,'MODIFIED_IGBP_MODIS_NOAH')
        self.noahmp.lsoil = 'CUST'
        # self.noahmp['lsoil'] = assign_string(self.noahmp.lsoil.dtype,'CUST')
        fdir = os.path.dirname(os.path.abspath(__file__))
        VEGPARM = '%s/pyNoahMP/data/VEGPARM.TBL' % fdir
        GENPARM = '%s/pyNoahMP/data/GENPARM.TBL' % fdir
        MPTABLE = '%s/pyNoahMP/data/MPTABLE.TBL' % fdir
        self.noahmp.vegparm_file = VEGPARM
        self.noahmp.genparm_file = GENPARM
        self.noahmp.mptable_file = MPTABLE
        # self.noahmp['vegparm_file'] = assign_string(self.noahmp.vegparm_file.dtype,VEGPARM)
        # self.noahmp['genparm_file'] = assign_string(self.noahmp.genparm_file.dtype,GENPARM)
        # self.noahmp['mptable_file'] = assign_string(self.noahmp.mptable_file.dtype,MPTABLE)
        # Define the options
        self.noahmp.idveg = 1  # dynamic vegetation (1 -> off ; 2 -> on)
        self.noahmp.iopt_crs = 1  # 2 canopy stomatal resistance (1-> Ball-Berry; 2->Jarvis)
        self.noahmp.iopt_btr = 1  # 1 # soil moisture factor for stomatal resistance (1-> Noah; 2-> CLM; 3-> SSiB)
        # Runoff 5 is really messed up
        self.noahmp.iopt_run = 5  # 2 # runoff and groundwater (1->SIMGM; 2->SIMTOP; 3->Schaake96; 4->BATS)
        self.noahmp.iopt_sfc = 1  # 2# 1# 1 # surface layer drag coeff (CH & CM) (1->M-O; 2->Chen97)
        self.noahmp.iopt_frz = 2  # 1# 2 # supercooled liquid water (1-> NY06; 2->Koren99)
        # Inf=1 improves infiltration on fronze conditions, but leads to numerical instability in normal conditions
        self.noahmp.iopt_inf = 2  # 1# 2 # frozen soil permeability (1-> NY06; 2->Koren99)
        self.noahmp.iopt_rad = 1  # 3# 3# 2 radiation transfer (1->gap=F(3D,cosz); 2->gap=0; 3->gap=1-Fveg)
        self.noahmp.iopt_alb = 2  # 2 snow surface albedo (1->BATS; 2->CLASS)
        self.noahmp.iopt_snf = 2  # 1# 3 rainfall & snowfall (1-Jordan91; 2->BATS; 3->Noah)]
        self.noahmp.iopt_tbot = 1  # 2# 2# 1 # lower boundary of soil temperature (1->zero-flux; 2->Noah)
        self.noahmp.iopt_stc = 1  # 1# 1# 2 snow/soil temperature time scheme (only layer 1) 1 -> semi-implicit; 2 -> full implicit (original Noah)

        self.noahmp.iz0tlnd = 0
        self.noahmp.sldpth[:] = np.array(self.metadata['dz'])
        self.noahmp.z_ml[:] = 10.0
        self.noahmp.zsoil = -np.cumsum(self.noahmp.sldpth[:], axis=1)
        self.noahmp.zsnso[:] = 0.0
        self.noahmp.zsnso[:, self.noahmp.nsnow:] = self.noahmp.zsoil[:]
        if flag_log:
            f_log.write("     Define Data, Main Settings Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Set all to land (can include lake in the future...)
        t_start = datetime.datetime.now()
        lc = self.input_fp.groups['parameters'].variables['land_cover'][:]
        ist = np.copy(lc).astype(np.int32)
        ist[:] = 1
        ist[lc == 17] = 2
        self.noahmp.ist[:] = ist[:]  # 1
        self.noahmp.isurban[:] = 13
        if flag_log:
            f_log.write("     Define Data, Landcover Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Set soil color
        t_start = datetime.datetime.now()
        self.noahmp.isc[:] = 4  # 4
        self.noahmp.ice[:] = 0

        # Initialize snow info
        self.noahmp.isnow[:] = 0

        # Others
        self.noahmp.dx = self.dx
        self.noahmp.ice[:] = 0
        self.noahmp.foln[:] = 1.0
        self.noahmp.ficeold[:] = 0.0
        self.noahmp.albold[:] = 0.65
        self.noahmp.sneqvo[:] = 0.0
        self.noahmp.ch[:] = 0.0001
        self.noahmp.cm[:] = 0.0001
        self.noahmp.canliq[:] = 0.0
        self.noahmp.canice[:] = 0.0
        self.noahmp.sndpth[:] = 0.0
        self.noahmp.swe[:] = 0.0
        self.noahmp.snice[:] = 0.0
        self.noahmp.snliq[:] = 0.0
        self.noahmp.wa[:] = 0.0
        self.noahmp.wt[:] = self.noahmp.wa[:]
        self.noahmp.zwt[:] = 0.0
        self.noahmp.wslake[:] = 0.0
        self.noahmp.lfmass[:] = 9.0
        self.noahmp.rtmass[:] = 500.0
        self.noahmp.stmass[:] = 3.33
        self.noahmp.wood[:] = 500.0
        self.noahmp.stblcp[:] = 1000.0
        self.noahmp.fastcp[:] = 1000.0
        self.noahmp.plai[:] = 0.5
        self.noahmp.tauss[:] = 0.0
        self.noahmp.smcwtd[:] = self.noahmp.sh2o[:, 1]
        self.noahmp.deeprech[:] = 0.0
        self.noahmp.rech[:] = 0.0
        self.noahmp.eah[:] = 1000.0
        self.noahmp.fwet[:] = 0.0
        self.noahmp.psai[:] = 0.1
        self.noahmp.stc[:] = 285.0
        self.noahmp.slopetyp[:] = 3
        self.noahmp.albold[:] = 0.5
        if flag_log:
            f_log.write("     Define Data, Soils and other Params Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Define the data
        t_start = datetime.datetime.now()
        self.noahmp.vegtyp[:] = self.input_fp.groups['parameters'].variables[
            'land_cover'][:]
        self.noahmp.soiltyp[:] = np.arange(1, self.noahmp.ncells + 1)
        self.noahmp.clay_pct = self.input_fp.groups['parameters'].variables[
            'clay'][:]  # Noemi
        self.noahmp.smcmax[:] = self.input_fp.groups['parameters'].variables[
            'MAXSMC'][:]
        self.noahmp.smcref[:] = self.input_fp.groups['parameters'].variables[
            'REFSMC'][:]
        self.noahmp.smcdry[:] = self.input_fp.groups['parameters'].variables[
            'DRYSMC'][:]
        for ilayer in range(self.noahmp.sh2o.shape[1]):
            self.noahmp.sh2o[:, ilayer] = self.input_fp.groups[
                'parameters'].variables['MAXSMC'][:]
            # self.noahmp.sh2o[:,ilayer] = self.input_fp.groups['parameters'].variables['REFSMC'][:] # Noemi, start at REFSMC
        self.noahmp.smc[:] = self.noahmp.sh2o[:]
        self.noahmp.smcwtd[:] = self.noahmp.sh2o[:, 0]
        if flag_log:
            f_log.write("     Define Data, Input Params Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Initialize the soil parameters
        t_start = datetime.datetime.now()
        self.noahmp.bb0[:] = self.input_fp.groups['parameters'].variables[
            'BB'][:]
        self.noahmp.drysmc0[:] = self.input_fp.groups['parameters'].variables[
            'DRYSMC'][:]
        self.noahmp.f110[:] = 0.0  # self.input_fp.groups['parameters'].variables['F11'][:]
        self.noahmp.maxsmc0[:] = self.input_fp.groups['parameters'].variables[
            'MAXSMC'][:]
        self.noahmp.refsmc0[:] = self.input_fp.groups['parameters'].variables[
            'REFSMC'][:]
        self.noahmp.satpsi0[:] = self.input_fp.groups['parameters'].variables[
            'SATPSI'][:]
        self.noahmp.satdk0[:] = self.input_fp.groups['parameters'].variables[
            'SATDK'][:]
        self.noahmp.satdw0[:] = self.input_fp.groups['parameters'].variables[
            'SATDW'][:]
        self.noahmp.wltsmc0[:] = self.input_fp.groups['parameters'].variables[
            'WLTSMC'][:]
        self.noahmp.qtz0[:] = self.input_fp.groups['parameters'].variables[
            'QTZ'][:]
        if flag_log:
            f_log.write("     Initialize Soil Parameters Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Set lat/lon (declination calculation)
        t_start = datetime.datetime.now()
        self.noahmp.lat[:] = 0.0174532925 * self.input_fp.groups[
            'metadata'].latitude
        self.noahmp.lon[:] = 0.0174532925 * self.input_fp.groups[
            'metadata'].longitude
        if flag_log:
            f_log.write("     Set lat/lon Overall: "
                        f"{datetime.datetime.now() - t_start}\n")

        # Initialize output
        t_start = datetime.datetime.now()
        self.noahmp.tg[:] = 285.0
        self.noahmp.tv[:] = 285.0
        for ilayer in range(self.noahmp.nsnow, self.noahmp.stc.shape[1]):
            self.noahmp.stc[:, ilayer] = 285.0
        self.noahmp.tah[:] = 285.0
        self.noahmp.t2mv[:] = 285.0
        self.noahmp.t2mb[:] = 285.0
        self.noahmp.runsf[:] = 0.0
        self.noahmp.runsb[:] = 0.0
        if flag_log:
            f_log.write("     Initialize Derived Param Output Arrays: "
                        f"{datetime.datetime.now() - t_start}\n")

        # forcing
        self.noahmp.fveg[:] = 1.0
        self.noahmp.fvgmax[:] = 1.0
        self.noahmp.tbot[:] = 285.0

        # Define the parameters
        t_start = datetime.datetime.now()
        noah = self.noahmp
        self.noahmp.initialize_parameters(
            noah.llanduse, noah.lsoil, noah.vegparm_file, noah.genparm_file,
            noah.iopt_crs, noah.iopt_btr, noah.iopt_run, noah.iopt_sfc,
            noah.iopt_frz, noah.iopt_inf, noah.iopt_rad, noah.iopt_alb,
            noah.iopt_snf, noah.iopt_tbot, noah.iopt_stc, noah.idveg,
            noah.mptable_file, noah.bb0, noah.drysmc0, noah.f110, noah.maxsmc0,
            noah.refsmc0, noah.satpsi0, noah.satdk0, noah.satdw0, noah.wltsmc0,
            noah.qtz0)
        if flag_log:
            f_log.write("     Initialize NOAH internal Params: "
                        f"{datetime.datetime.now() - t_start}\n")

        return

    def initialize_subsurface(self, ):

        if self.subsurface_module == 'richards':
            self.initialize_richards()

        return

    def initialize_richards(self, ):

        from pyRichards import richards

        # Initialize richards
        self.richards = richards.richards(self.nhru, self.nsoil)

        # Set other parameters
        self.richards.dx = self.dx
        self.richards.nhru = self.nhru
        # print(self.nhru)
        self.richards.m[:] = self.input_fp.groups['parameters'].variables[
            'm'][:]
        # self.richards.dem[:] = self.input_fp.groups['parameters'].variables['dem'][:]
        self.richards.dem[:] = self.input_fp.groups['parameters'].variables[
            'hand'][:]
        # self.richards.dem[:] = (self.input_fp.groups['parameters'].variables['dem'][:]+self.input_fp.groups['parameters'].variables['hand'][:])/2.
        self.richards.slope[:] = self.input_fp.groups['parameters'].variables[
            'slope'][:]
        # self.richards.hand[:] = self.input_fp.groups['parameters'].variables['hand'][:]
        self.richards.area[:] = self.input_fp.groups['parameters'].variables[
            'area'][:]
        self.richards.width = sparse.csr_matrix(
            (self.input_fp.groups['wmatrix'].variables['data'][:],
             self.input_fp.groups['wmatrix'].variables['indices'][:],
             self.input_fp.groups['wmatrix'].variables['indptr'][:]),
            shape=(self.nhru, self.nhru),
            dtype=np.float64)
        # self.richards.width_dense = np.array(self.richards.width.todense())
        # with np.errstate(invalid='ignore',divide='ignore'):tmp = self.richards.width_dense/self.richards.area
        # self.richards.dx = (tmp + tmp.T)/2
        self.richards.I = self.richards.width.copy()
        self.richards.I[self.richards.I != 0] = 1
        self.richards.w = np.array(self.richards.width.todense())
        with np.errstate(invalid='ignore', divide='ignore'):
            tmp = self.richards.area / self.richards.w
        dx = (tmp + tmp.T) / 2
        self.richards.dx = dx

        return

    def initialize_hwu(self, info):

        # from pyHWU.Human_Water_Use import Human_Water_Use as hwu
        # self.hwu = hwu(self,info)
        # self.hwu.area = self.input_fp.groups['parameters'].variables['area'][:]

        # if self.hwu.hwu_flag == True:
        # print("Initializing Human Water Management")
        # self.hwu.initialize_allocation(self,)

        return

    def run(self, info, flag_log: bool, f_log):

        # Run the model
        date = self.idate
        tic = time.time()
        self.noahmp.dzwt[:] = 0.0

        time_update_input = datetime.timedelta(seconds=0)
        time_init_water_balance = datetime.timedelta(seconds=0)
        time_update = datetime.timedelta(seconds=0)
        time_finalize_water_balance = datetime.timedelta(seconds=0)
        time_calculate_water_balance_error = datetime.timedelta(seconds=0)
        time_calculate_energy_balance_error = datetime.timedelta(seconds=0)
        time_update_output = datetime.timedelta(seconds=0)

        t_start_compute = datetime.datetime.now()
        while date < self.fdate:

            # Update input data
            # tic0 = time.time()
            t_start = datetime.datetime.now()
            self.update_input(date)
            # print('update input',time.time() - tic0,flush=True)
            if flag_log:
                time_update_input += datetime.datetime.now() - t_start

            # Save the original precip
            precip = np.copy(self.noahmp.prcp)

            # Calculate initial NOAH water balance
            t_start = datetime.datetime.now()
            self.initialize_water_balance()
            if flag_log:
                time_init_water_balance += datetime.datetime.now() - t_start

            # Update model
            # tic0 = time.time()
            t_start = datetime.datetime.now()
            self.update(date)
            # print('update model',time.time() - tic0,flush=True)
            if flag_log:
                time_update += datetime.datetime.now() - t_start

            # Return precip to original value
            self.noahmp.prcp[:] = precip[:]

            # Calculate final water balance
            t_start = datetime.datetime.now()
            self.finalize_water_balance()
            if flag_log:
                time_finalize_water_balance += datetime.datetime.now(
                ) - t_start

            # Update the water balance error
            t_start = datetime.datetime.now()
            self.calculate_water_balance_error()
            if flag_log:
                time_calculate_water_balance_error += datetime.datetime.now(
                ) - t_start

            # Update the energy balance error
            t_start = datetime.datetime.now()
            self.calculate_energy_balance_error()
            if flag_log:
                time_calculate_energy_balance_error += datetime.datetime.now(
                ) - t_start

            # Update time and date
            self.date = date
            info['date'] = date

            # Update output
            # tic0 = time.time()
            if self.date >= self.spin_date:
                t_start = datetime.datetime.now()
                self.update_output(date)
                if flag_log:
                    time_update_output += datetime.datetime.now() - t_start
                # print('update output',time.time() - tic0,flush=True)

            # Update time step
            date = date + self.dt_timedelta
            self.itime = self.itime + 1

            # Output some statistics
            if (date.hour == 0) and (date.day == 1):
                # print('cid: %d' % info['cid'],date.strftime("%Y-%m-%d"),'%10.4f'%(time.time()-tic),'et:%10.4f'%self.et,'prcp:%10.4f'%self.prcp,'q:%10.4f'%self.q,'WB ERR:%10.6f' % self.errwat,'ENG ERR:%10.6f' % self.erreng,flush=True)
                print(date.strftime("%Y-%m-%d"),
                      'time:%10.4f' % (time.time() - tic),
                      'et:%10.4f' % self.et,
                      'prcp:%10.4f' % self.prcp,
                      'q:%10.4f' % self.q,
                      'WB ERR:%10.6f' % self.errwat,
                      'ENG ERR:%10.6f' % self.erreng,
                      flush=True)

        # Output some statistics
        if flag_log:
            f_log.write(
                f"     - Update Input: {time_update_input}\n"
                f"     - Initialize Water Balance: {time_init_water_balance}\n"
                f"     - Update: {time_update}\n"
                f"     - Finalize Water Balance: {time_finalize_water_balance}\n"
                f"     - Calculate Water Balance Error: {time_calculate_water_balance_error}\n"
                f"     - Calculate Energy Balance Error: {time_calculate_energy_balance_error}\n"
                f"     - Update Output: {time_update_output}\n"
                f"     Compute Loop: {datetime.datetime.now() - t_start_compute}\n"
            )

        pass

    def update_input(self, date):

        self.noahmp.itime = self.itime
        # dt = self.dt
        self.noahmp.nowdate = assign_string(19,
                                            date.strftime('%Y-%m-%d_%H:%M:%S'))
        # self.noahmp.nowdate = date.strftime('%Y-%m-%d_%H:%M:%S')
        self.noahmp.julian = (date -
                              datetime.datetime(date.year, 1, 1, 0)).days
        self.noahmp.yearlen = (
            datetime.datetime(date.year + 1, 1, 1, 0) -
            datetime.datetime(date.year, 1, 1, 1, 0)).days + 1

        # Update meteorology
        # meteorology = self.input_fp.groups['meteorology']
        '''if date == self.idate:
            # determine the first time step for the meteorology
            var = meteorology.variables['time']
            ndates = var[:]
            # convert current date to num
            ndate = nc.date2num(date,units=var.units,calendar=var.calendar)
            # identify the position
            self.minitial_itime = np.where(ndates == ndate)[0][0]
        i = self.itime + self.minitial_itime'''
        '''
        # Update meteorology
        meteorology = self.input_fp.groups['meteorology']
        # convert current date to num
        ndate = nc.date2num(date,units=self.input_fp_meteo_time.units,calendar=self.input_fp_meteo_time.calendar)
        # identify the position
        i = np.argmin(np.abs(self.input_fp_meteo_time[:]-ndate))

        # Downscale meteorology
        self.noahmp.lwdn[:] = meteorology.variables['lwdown'][i,:] # W/m2
        self.noahmp.swdn[:] = meteorology.variables['swdown'][i,:] # W/m2
        self.noahmp.psfc[:] = meteorology.variables['psurf'][i,:] # Pa
        self.noahmp.p_ml[:] = self.noahmp.psfc[:] # Pa
        self.noahmp.u_ml[:] = (meteorology.variables['wind'][i,:]**2/2)**0.5 # m/s
        self.noahmp.v_ml[:] = self.noahmp.u_ml[:] # m/s
        self.noahmp.t_ml[:] = meteorology.variables['tair'][i,:] # K
        self.noahmp.q_ml[:] = meteorology.variables['spfh'][i,:] # Kg/Kg
        self.noahmp.qsfc1d[:] = self.noahmp.q_ml[:] # Kg/Kg
        self.noahmp.prcp[:] = meteorology.variables['precip'][i,:] # mm/s
        '''

        # Update meteorology
        i = int(np.floor(self.itime / self.nt_timestep))
        self.noahmp.lwdn[:] = self.meteodata['lwdown'][i, :]  # W/m2
        self.noahmp.swdn[:] = self.meteodata['swdown'][i, :]  # W/m2
        self.noahmp.psfc[:] = self.meteodata['psurf'][i, :]  # Pa
        self.noahmp.p_ml[:] = self.noahmp.psfc[:]  # Pa
        self.noahmp.u_ml[:] = (self.meteodata['wind'][i, :]**2 / 2)**0.5  # m/s
        self.noahmp.v_ml[:] = self.noahmp.u_ml[:]  # m/s
        self.noahmp.t_ml[:] = self.meteodata['tair'][i, :]  # K
        self.noahmp.q_ml[:] = self.meteodata['spfh'][i, :]  # Kg/Kg
        self.noahmp.qsfc1d[:] = self.noahmp.q_ml[:]  # Kg/Kg
        self.noahmp.prcp[:] = self.meteodata['precip'][i, :]  # mm/s

        # Set the partial pressure of CO2 and O2
        self.noahmp.co2air[:] = 355.E-6 * self.noahmp.psfc[:]  # ! Partial pressure of CO2 (Pa) ! From NOAH-MP-WRF
        self.noahmp.o2air[:] = 0.209 * self.noahmp.psfc[:]  # ! Partial pressure of O2 (Pa)  ! From NOAH-MP-WRF

        # Update water demands
        '''if self.hwu.hwu_flag == True:
   if (date.hour*3600)%self.hwu.dta == 0:
    water_use = self.input_fp.groups['water_use']
    if self.hwu.hwu_indust_flag == True:
     self.hwu.demand_indust[:]  = water_use.variables['industrial'][i,:] # m/s
     self.hwu.deficit_indust[:] = np.copy(self.hwu.demand_indust[:])
    if self.hwu.hwu_domest_flag == True:
     self.hwu.demand_domest[:]  = water_use.variables['domestic'][i,:] # m/s
     self.hwu.deficit_domest[:] = np.copy(self.hwu.demand_domest[:])
    if self.hwu.hwu_lstock_flag == True:
     self.hwu.demand_lstock[:]  = water_use.variables['livestock'][i,:] # m/s
     self.hwu.deficit_lstock[:] = np.copy(self.hwu.demand_lstock[:])'''

        return

    def update(self, date):

        # Apply irrigation
        # self.hwu.Human_Water_Irrigation(self,date)

        # Update subsurface
        self.update_subsurface()

        # Update NOAH
        noah = self.noahmp
        self.noahmp.run_model(
            self.ncores, noah.yearlen, noah.idveg, noah.iopt_crs,
            noah.iopt_btr, noah.iopt_run, noah.iopt_sfc, noah.iopt_frz,
            noah.iopt_inf, noah.iopt_rad, noah.iopt_alb, noah.iopt_snf,
            noah.iopt_tbot, noah.iopt_stc, noah.itime, noah.iz0tlnd, noah.dt,
            noah.dx, noah.julian, noah.isnow, noah.isurban, noah.slopetyp,
            noah.soiltyp, noah.vegtyp, noah.ice, noah.isc, noah.ist,
            noah.root_depth, noah.nowdate, noah.z_ml, noah.lwdn, noah.swdn,
            noah.p_ml, noah.psfc, noah.prcp, noah.t_ml, noah.q_ml, noah.u_ml,
            noah.v_ml, noah.fsh, noah.ssoil, noah.salb, noah.fsno, noah.swe,
            noah.sndpth, noah.emissi, noah.qsfc1d, noah.tv, noah.tg,
            noah.canice, noah.canliq, noah.eah, noah.tah, noah.cm, noah.ch,
            noah.fwet, noah.sneqvo, noah.albold, noah.qsnow, noah.wslake,
            noah.zwt, noah.dzwt, noah.wa, noah.wt, noah.smcwtd, noah.deeprech,
            noah.rech, noah.lfmass, noah.rtmass, noah.stmass, noah.wood,
            noah.stblcp, noah.fastcp, noah.plai, noah.psai, noah.tauss,
            noah.t2mv, noah.t2mb, noah.q2mv, noah.q2mb, noah.trad, noah.nee,
            noah.gpp, noah.npp, noah.fvegmp, noah.runsf, noah.runsb, noah.ecan,
            noah.etran, noah.esoil, noah.fsa, noah.fira, noah.apar, noah.psn,
            noah.sav, noah.sag, noah.rssun, noah.rssha, noah.bgap, noah.wgap,
            noah.tgv, noah.tgb, noah.chv, noah.chb, noah.irc, noah.irg,
            noah.shc, noah.shg, noah.evg, noah.ghv, noah.irb, noah.shb,
            noah.evb, noah.ghb, noah.tr, noah.evc, noah.chleaf, noah.chuc,
            noah.chv2, noah.chb2, noah.cosz, noah.lat, noah.lon, noah.fveg,
            noah.fvgmax, noah.fpice, noah.fcev, noah.fgev, noah.fctr,
            noah.qsnbot, noah.ponding, noah.ponding1, noah.ponding2, noah.fsr,
            noah.co2pp, noah.o2pp, noah.foln, noah.tbot, noah.smcmax,
            noah.smcdry, noah.smcref, noah.errwat, noah.si0, noah.si1,
            noah.zwt0, noah.minzwt, noah.co2air, noah.o2air, noah.bb0,
            noah.drysmc0, noah.f110, noah.maxsmc0, noah.refsmc0, noah.satpsi0,
            noah.satdk0, noah.satdw0, noah.wltsmc0, noah.qtz0, noah.stc,
            noah.sh2o, noah.smc, noah.smceq, noah.zsnso, noah.snice,
            noah.snliq, noah.ficeold, noah.zsoil, noah.sldpth, noah.hdiv,
            noah.nsoil, noah.nsnow)

        # Calculate water demands and supplies, and allocate volumes
        # self.hwu.Calc_Human_Water_Demand_Supply(self,date)

        # Abstract Surface Water and Groundwater
        # self.hwu.Water_Supply_Abstraction(self,date)

        return

    def update_subsurface(self, ):

        self.noahmp.dzwt[:] = 0.0

        if self.subsurface_module == 'richards':

            # Assign noahmp variables to subsurface module
            self.richards.theta[:] = self.noahmp.smc[:]
            self.richards.thetar[:] = self.noahmp.drysmc0[:]
            self.richards.thetas[:] = self.noahmp.maxsmc0[:]
            self.richards.b[:] = self.noahmp.bb0[:]
            self.richards.satpsi[:] = self.noahmp.satpsi0[:]
            self.richards.ksat[:] = self.noahmp.satdk0[:]
            self.richards.dz[:] = self.noahmp.sldpth[:]

            # Update subsurface module
            # self.richards.update()
            self.richards.update_numba()

            # Assign subsurface module variables to noahmp
            self.noahmp.hdiv[:] = self.richards.hdiv[:]

        return

    def initialize_water_balance(self, ):

        smw = np.sum(1000 * self.noahmp.sldpth * self.noahmp.smc, axis=1)
        self.beg_wb = np.copy(self.noahmp.canliq + self.noahmp.canice +
                              self.noahmp.swe + self.noahmp.wa + smw)
        self.dzwt0 = np.copy(self.noahmp.dzwt)

        return

    def finalize_water_balance(self, ):

        NOAH = self.noahmp
        smw = np.sum(1000 * NOAH.sldpth * NOAH.smc, axis=1)
        self.end_wb = np.copy(NOAH.canliq + NOAH.canice + NOAH.swe + NOAH.wa +
                              smw)

        return

    def calculate_water_balance_error(self, ):

        NOAH = self.noahmp
        # HWU = self.hwu
        dt = self.dt
        if self.subsurface_module == 'richards':
            tmp = np.copy(
                self.end_wb - self.beg_wb - NOAH.dt *
                (NOAH.prcp - NOAH.ecan - NOAH.etran - NOAH.esoil - NOAH.runsf -
                 NOAH.runsb - np.sum(NOAH.hdiv, axis=1)))
        else:
            tmp = np.copy(self.end_wb - self.beg_wb - NOAH.dt *
                          (NOAH.prcp - NOAH.ecan - NOAH.etran - NOAH.esoil -
                           NOAH.runsf - NOAH.runsb))
        self.errwat += np.sum(self.pct * tmp)
        self.q = self.q + dt * np.sum(self.pct * NOAH.runsb) + dt * np.sum(
            self.pct * NOAH.runsf)
        self.et = self.et + dt * np.sum(self.pct *
                                        (NOAH.ecan + NOAH.etran + NOAH.esoil))
        self.etran += dt * np.sum(self.pct * NOAH.etran)
        self.ecan += dt * np.sum(self.pct * NOAH.ecan)
        self.esoil += dt * np.sum(self.pct * NOAH.esoil)
        self.prcp = self.prcp + dt * np.sum(self.pct * NOAH.prcp)

        return

    def calculate_energy_balance_error(self, ):

        NOAH = self.noahmp
        tmp = np.copy(NOAH.sav + NOAH.sag - NOAH.fira - NOAH.fsh - NOAH.fcev -
                      NOAH.fgev - NOAH.fctr - NOAH.ssoil)
        self.erreng += np.sum(self.pct * tmp)

        return

    def update_output(self, date):

        NOAH = self.noahmp
        # HWU = self.hwu
        HB = self
        itime = self.itime

        # Create the netcdf file
        if date == self.idate:
            self.create_netcdf_file()

        # General info
        grp = self.output_fp.groups['metadata']
        dates = grp.variables['date']
        dates[itime] = nc.date2num(date,
                                   units=dates.units,
                                   calendar=dates.calendar)

        # Update the variables
        grp = self.output_fp.groups['data']

        tmp = {}
        # NoahMP
        tmp['smc'] = np.copy(NOAH.smc)  # m3/m3
        tmp['g'] = np.copy(NOAH.ssoil)  # W/m2
        tmp['sh'] = np.copy(NOAH.fsh)  # W/m2
        tmp['lh'] = np.copy(NOAH.fcev + NOAH.fgev + NOAH.fctr)  # W/m2
        tmp['shb'] = np.copy(NOAH.shb)  # W/m2
        tmp['evb'] = np.copy(NOAH.evb)  # W/m2
        tmp['swdn'] = np.copy(NOAH.swdn)  # W/m2
        tmp['t2mv'] = np.copy(NOAH.t2mv)  # W/m2
        tmp['t2mb'] = np.copy(NOAH.t2mb)  # W/m2
        tmp['qbase'] = NOAH.dt * np.copy(NOAH.runsb)  # mm
        tmp['qsurface'] = NOAH.dt * np.copy(NOAH.runsf)  # mm
        tmp['runoff'] = NOAH.dt * (np.copy(NOAH.runsf) + np.copy(NOAH.runsb)
                                   )  # mm
        tmp['prcp'] = NOAH.dt * np.copy(NOAH.prcp)  # mm
        tmp['trad'] = np.copy(NOAH.trad)  # K
        tmp['stc'] = np.copy(NOAH.stc[:, 3])  # K
        tmp['tv'] = np.copy(NOAH.tv)  # K
        tmp['salb'] = np.copy(NOAH.salb)
        tmp['wtd'] = np.copy(NOAH.zwt)  # m
        tmp['swe'] = np.copy(NOAH.swe)  # m ? mm?
        # tmp['totsmc'] = np.sum(NOAH.sldpth*NOAH.smc,axis=1)/np.sum(NOAH.sldpth[0]) # m3/m3
        tmp['hdiv'] = np.copy(NOAH.hdiv)

        # root zone
        cs = np.cumsum(NOAH.sldpth[0, :])
        mask = cs <= 0.5
        pct = NOAH.sldpth[0, mask] / np.sum(NOAH.sldpth[0, mask])
        tmp['smc_root'] = np.sum(pct * NOAH.smc[:, mask], axis=1)  # m3/m3

        # top soil layer
        tmp['smc1'] = np.copy(NOAH.smc[:, 0])
        tmp['relsmc1'] = (np.copy(NOAH.smc[:, 0]) -
                          NOAH.drysmc0) / (NOAH.maxsmc0 - NOAH.drysmc0)

        # mask water bodies to saturated soil moisture # Noemi
        mask_water = self.noahmp.vegtyp == 17
        tmp['smc1'][mask_water] = NOAH.maxsmc0[mask_water]
        tmp['relsmc1'][mask_water] = NOAH.maxsmc0[mask_water]
        tmp['smc_root'][mask_water] = NOAH.maxsmc0[mask_water]

        # total soil moisture / soil water storage -- only until depth to bedrock
        dl = [np.argmin(np.abs(cs - self.m[i])) for i in range(self.nhru)]
        tmp['totsmc'] = [
            np.sum(NOAH.sldpth[i, :dl[i]] * NOAH.smc[i, :dl[i]]) /
            np.sum(NOAH.sldpth[i, :dl[i]]) for i in range(self.nhru)
        ]

        # Snow
        tmp['albold'] = np.copy(NOAH.albold)  # snow albedo at last time step
        tmp['ficeold'] = np.copy(NOAH.ficeold)  # snow layer ice fraction
        tmp['fsno'] = np.copy(NOAH.fsno)  # snow cover fraction
        tmp['ponding'] = np.copy(NOAH.ponding)  # snowmelt with no pack [mm]
        tmp['ponding1'] = np.copy(NOAH.ponding1)  # snowmelt with no pack [mm]
        tmp['ponding2'] = np.copy(NOAH.ponding2)  # snowmelt with no pack [mm]
        tmp['qsnbot'] = np.copy(
            NOAH.qsnbot)  # snowmelt out bottom of pack [mm/s]
        tmp['qsnow'] = np.copy(NOAH.qsnow)  # snowfall on the ground (mm/s)
        tmp['sndpth'] = np.copy(NOAH.sndpth)  # snow depth (m)
        tmp['sneqvo'] = np.copy(
            NOAH.sneqvo)  # snow mass at last time step (mm h2o)
        tmp['snice'] = np.copy(NOAH.snice)  # snow layer ice (mm)
        tmp['snliq'] = np.copy(NOAH.snliq)  # snow layer liquid water (mm)
        tmp['swe'] = np.copy(NOAH.swe)  # snow water equivalent (mm)
        tmp['zsnso'] = np.copy(NOAH.zsnso)  # snow layer depth (m)

        # General
        tmp['errwat'] = np.copy(HB.errwat)

        # Water Management
        '''if self.hwu.hwu_agric_flag == True:
            tmp['demand_agric'] = np.copy(HWU.demand_agric)*NOAH.dt       # m
            tmp['deficit_agric'] = np.copy(HWU.deficit_agric)*NOAH.dt     # m
            tmp['irrig_agric'] = np.copy(HWU.irrigation)*(NOAH.dt/1000.0)   # m

        if self.hwu.hwu_flag == True:
            if self.hwu.hwu_indust_flag == True:
                tmp['demand_indust'] = np.copy(HWU.demand_indust)*NOAH.dt # m
                tmp['deficit_indust'] = np.copy(HWU.deficit_indust)*NOAH.dt # m
                tmp['alloc_indust'] = np.copy(HWU.demand_indust-HWU.deficit_indust)*NOAH.dt # m
            if self.hwu.hwu_domest_flag == True:
                tmp['demand_domest'] = np.copy(HWU.demand_domest)*NOAH.dt # m
                tmp['deficit_domest'] = np.copy(HWU.deficit_domest)*NOAH.dt # m
                tmp['alloc_domest'] = np.copy(HWU.demand_domest-HWU.deficit_domest)*NOAH.dt # m
            if self.hwu.hwu_lstock_flag == True:
                tmp['demand_lstock'] = np.copy(HWU.demand_lstock)*NOAH.dt # m
                tmp['deficit_lstock'] = np.copy(HWU.deficit_lstock)*NOAH.dt # m
                tmp['alloc_lstock'] = np.copy(HWU.demand_lstock-HWU.deficit_lstock)*NOAH.dt # m
            if self.hwu.hwu_sf_flag == True:
                tmp['alloc_sf'] = np.copy(HWU.alloc_sf) # m
            if self.hwu.hwu_gw_flag == True:
                tmp['alloc_gw'] = np.copy(HWU.alloc_gw) # m
        '''

        # Output the variables
        sep = 100
        if itime == 0:
            self.output = {}
            for var in self.metadata['output']['vars']:
                shp = grp.variables[var].shape
                self.output[var] = np.zeros((sep, shp[1]))
        # Fill the data (MISSING!)
        val = itime % sep
        try:
            for var in self.metadata['output']['vars']:
                self.output[var][val, :] = tmp[var]
        except ValueError as e:
            print(f'Valueerror for {var} at {val}')
            raise ValueError(e)
        # self.output[itime] =
        if (itime + 1) % sep == 0:
            for var in self.metadata['output']['vars']:
                grp.variables[var][itime - sep + 1:itime +
                                   1, :] = self.output[var][:]
        if (itime + 1) == self.ntime:
            val = int(sep * np.ceil((itime - sep) / sep))
            for var in self.metadata['output']['vars']:
                grp.variables[var][val:itime +
                                   1, :] = self.output[var][0:itime - val +
                                                            1, :]

        return

    def create_netcdf_file(self, ):

        # Create the output directory if necessary
        os.system('mkdir -p %s' % self.metadata['output']['dir'])

        # Extract the pointers to both input and output files
        ofile = '%s/%s.nc' % (self.metadata['output']['dir'],
                              self.idate.strftime('%Y-%m-%d'))
        self.output_fp = nc.Dataset(ofile, 'w', format='NETCDF4')
        fp_out = self.output_fp
        fp_in = self.input_fp

        # Define the metadata
        # TODO replace with reading from e.g. a csv file in directory for easy reuse
        # TODO put in a check vs all the input variables whether they're supported
        metadata = {
            'g': {
                'description': 'Ground heat flux',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'sh': {
                'description': 'Sensible heat flux',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'lh': {
                'description': 'Latent heat flux',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            # 'lwnet':{'description':'Net longwave radiation','units':'W/m2','dims':('time','hru',),'precision':4},
            # 'swnet':{'description':'Absorbed shortwave radiation','units':'W/m2','dims':('time','hru',),'precision':4},
            "t2mv": {
                'description': 'Vegetated air temperature',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            "t2mb": {
                'description': 'Bare air temperature',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            "shb": {
                'description': 'Bare sensible heat flux',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            "evb": {
                'description': 'Bare latent heat flux',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            "swdn": {
                'description': 'Shortwave down',
                'units': 'W/m2',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'trad': {
                'description': 'Land surface skin temperature',
                'units': 'K',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 2
            },
            'stc': {
                'description': 'Snow/soil temperature',
                'units': 'K',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 2
            },
            'tv': {
                'description': 'Canopy temperature',
                'units': 'K',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 2
            },
            'salb': {
                'description': 'Land surface albedo',
                'units': ' ',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qbase': {
                'description': 'Excess runoff',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qsurface': {
                'description': 'Surface runoff',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'runoff': {
                'description': 'Total runoff',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'prcp': {
                'description': 'Precipitation',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'smc': {
                'description': 'Soil water content',
                'units': 'm3/m3',
                'dims': ('time', 'hru', 'soil'),
                'precision': 4
            },
            'swd': {
                'description': 'Soil water deficit',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'sstorage': {
                'description': 'Surface storage',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            # 'sndpth':{'description':'Snow depth','units':'mm','dims':('time','hru',)},
            'swe': {
                'description': 'Snow water equivalent',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qout_subsurface': {
                'description': 'Subsurface flux',
                'units': 'm2/s',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qout_surface': {
                'description': 'Surface flux',
                'units': 'm2/s',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'wtd': {
                'description': 'WTD',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 2
            },
            'errwat': {
                'description': 'errwat',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'totsmc': {
                'description': 'totsmc',
                'units': 'm3/m3',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 2
            },
            'hdiv': {
                'description': 'hdiv',
                'units': 'mm/s',
                'dims': ('time', 'hru', 'soil'),
                'precision': 4
            },
            'smc1': {
                'description': 'Soil water content at the top soil layer',
                'units': 'm3/m3',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'relsmc1': {
                'description':
                'Relative soil water content at the top soil layer',
                'units': 'm3/m3',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'smc_root': {
                'description': 'Soil water content at the root zone',
                'units': 'm3/m3',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },

            # Water Management
            'demand_agric': {
                'description': 'Irrigation water demand',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'deficit_agric': {
                'description': 'Irrigation water deficit',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'irrig_agric': {
                'description': 'Irrigated water volume',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'demand_indust': {
                'description': 'Industrial water demand',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                )
            },
            'precision': 4,
            'deficit_indust': {
                'description': 'Industrial water deficit',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'alloc_indust': {
                'description': 'Industrial water allocated',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'demand_domest': {
                'description': 'Domestic demand',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'deficit_domest': {
                'description': 'Domestic deficit',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'alloc_domest': {
                'description': 'Domestic water allocated',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'demand_lstock': {
                'description': 'Livestock demand',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'deficit_lstock': {
                'description': 'Livestock deficit',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'alloc_lstock': {
                'description': 'Livestock water allocated',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'alloc_sf': {
                'description': 'Surface water allocated',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'alloc_gw': {
                'description': 'Groundwater water allocated',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },

            # Snow Related
            'fsno': {
                'description': 'snow cover fraction',
                'units': '',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'sndpth': {
                'description': 'snow depth',
                'units': 'm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'sneqvo': {
                'description': 'snow mass at last time step',
                'units': 'mm h2o',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'albold': {
                'description': 'snow albedo at last time step',
                'units': '',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qsnow': {
                'description': 'snowfall on the ground',
                'units': 'mm/s',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'qsnbot': {
                'description': 'snowmelt out bottom of pack',
                'units': 'mm/s',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'ponding': {
                'description': 'snowmelt with no pack',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'ponding1': {
                'description': 'snowmelt with no pack',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            },
            'ponding2': {
                'description': 'snowmelt with no pack',
                'units': 'mm',
                'dims': (
                    'time',
                    'hru',
                ),
                'precision': 4
            }  # ,
            # 'zsnso': {
            #     'description': 'snow layer depth',
            #     'units': 'm',
            #     'dims': (
            #         'time',
            #         'hru',
            #     ),
            #     'precision': 4
            # },
            # 'snice': {
            #     'description': 'snow layer ice',
            #     'units': 'mm',
            #     'dims': (
            #         'time',
            #         'hru',
            #     ),
            #     'precision': 4
            # },
            # 'snliq': {
            #     'description': 'snow layer liquid water',
            #     'units': 'mm',
            #     'dims': (
            #         'time',
            #         'hru',
            #     ),
            #     'precision': 4
            # },
            # 'ficeold': {
            #     'description': 'snow layer ice fraction',
            #     'units': '',
            #     'dims': (
            #         'time',
            #         'hru',
            #     ),
            #     'precision': 4
            # }
        }

        # Create the dimensions
        print('Creating the dimensions', flush=True)
        ntime = 24 * 3600 * ((self.fdate - self.idate).days) / self.dt
        self.ntime = ntime
        nhru = len(fp_in.dimensions['hru'])
        fp_out.createDimension('hru', nhru)
        fp_out.createDimension('time', ntime)
        fp_out.createDimension('soil', self.nsoil)

        # Create the output
        print('Creating the data group', flush=True)
        grp = fp_out.createGroup('data')
        for var in self.metadata['output']['vars']:
            try:
                # ncvar = grp.createVariable(
                #     var,
                #     'f4',
                #     metadata[var]['dims'],
                #     least_significant_digit=metadata[var]['precision'])  # ,
                # compression="zstd")  # ,zlib=True)
                ncvar = ncutil.create_netcdf_variable(grp, var, 'f4', metadata[var]['dims'], 
                                                       least_significant_digit=metadata[var]['precision'], compress=True)
            except KeyError as e:
                print(f'\nVariable {var} not found in metadata')
                print(metadata.keys(), '\n')
                raise KeyError(e)

            ncvar.description = metadata[var]['description']
            ncvar.units = metadata[var]['units']

        # Create the metadata
        print('Creating the metadata group', flush=True)
        grp = fp_out.createGroup('metadata')
        # Time
        grp.createVariable('time', 'f8', ('time', ))
        dates = grp.createVariable('date', 'f8', ('time', ))  #,
        # compression="zstd")
        dates.units = 'hours since 1900-01-01'
        dates.calendar = 'standard'

        # HRU percentage coverage
        print('Setting the HRU percentage coverage', flush=True)
        pcts = grp.createVariable('pct', 'f4',
                                  ('hru', ))  #, compression="zstd")
        pcts[:] = fp_in.groups['parameters'].variables['area_pct'][:]
        pcts.description = 'hru percentage coverage'
        pcts.units = '%'

        # HRU area
        print('Setting the HRU areal coverage', flush=True)
        area = grp.createVariable('area', 'f4',
                                  ('hru', ))  #, compression="zstd")
        area[:] = fp_in.groups['parameters'].variables['area'][:]
        area.units = 'meters squared'
        print('Defining the HRU ids', flush=True)
        hru = grp.createVariable('hru', 'i4',
                                 ('hru', ))  # , compression="zstd")
        hrus = []
        for value in range(nhru):
            hrus.append(value)
        hru[:] = np.array(hrus)
        hru.description = 'hru ids'

        return

    def finalize(self, ):

        # Create the restart directory if necessary
        os.system('mkdir -p %s' % self.metadata['restart']['dir'])

        # Save the restart file
        file_restart = '%s/%s.h5' % (self.metadata['restart']['dir'],
                                     self.fdate.strftime('%Y-%m-%d'))
        fp = h5py.File(file_restart, 'w')
        fp['smceq'] = self.noahmp.smceq[:]
        fp['albold'] = self.noahmp.albold[:]
        fp['sneqvo'] = self.noahmp.sneqvo[:]
        fp['stc'] = self.noahmp.stc[:]
        fp['sh2o'] = self.noahmp.sh2o[:]
        fp['smc'] = self.noahmp.smc[:]
        fp['tah'] = self.noahmp.tah[:]
        fp['eah'] = self.noahmp.eah[:]
        fp['fwet'] = self.noahmp.fwet[:]
        fp['canliq'] = self.noahmp.canliq[:]
        fp['canice'] = self.noahmp.canice[:]
        fp['tv'] = self.noahmp.tv[:]
        fp['tg'] = self.noahmp.tg[:]
        fp['qsfc1d'] = self.noahmp.qsfc1d[:]
        fp['qsnow'] = self.noahmp.qsnow[:]
        fp['isnow'] = self.noahmp.isnow[:]
        fp['zsnso'] = self.noahmp.zsnso[:]
        fp['sndpth'] = self.noahmp.sndpth[:]
        fp['swe'] = self.noahmp.swe[:]
        fp['snice'] = self.noahmp.snice[:]
        fp['snliq'] = self.noahmp.snliq[:]
        fp['zwt'] = self.noahmp.zwt[:]
        fp['wa'] = self.noahmp.wa[:]
        fp['wt'] = self.noahmp.wt[:]
        fp['wslake'] = self.noahmp.wslake[:]
        fp['lfmass'] = self.noahmp.lfmass[:]
        fp['rtmass'] = self.noahmp.rtmass[:]
        fp['stmass'] = self.noahmp.stmass[:]
        fp['wood'] = self.noahmp.wood[:]
        fp['stblcp'] = self.noahmp.stblcp[:]
        fp['fastcp'] = self.noahmp.fastcp[:]
        fp['plai'] = self.noahmp.plai[:]
        fp['psai'] = self.noahmp.psai[:]
        fp['cm'] = self.noahmp.cm[:]
        fp['ch'] = self.noahmp.ch[:]
        fp['tauss'] = self.noahmp.tauss[:]
        fp['smcwtd'] = self.noahmp.smcwtd[:]
        fp['deeprech'] = self.noahmp.deeprech[:]
        fp['rech'] = self.noahmp.rech[:]
        fp.close()
        del fp

        # Close the LSM
        del self.noahmp

        # Close the files
        self.input_fp.close()
        del self.input_fp
        if self.date >= self.spin_date:
            self.output_fp.close()
            del self.output_fp

        return
