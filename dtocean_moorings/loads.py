"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter

Mathew Topper <dataonlygreater@gmail.com>, 2017
"""

# Built in modulesrad
import math
import logging
import operator
from itertools import izip

# External module import
import numpy as np
from scipy import interpolate, optimize, special

# Start logging
module_logger = logging.getLogger(__name__)


class Loads(object):
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Loads class
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    System and environmental loads submodule
    
        Args:
            bathygrid
            sysorig
            deviceid
            foundloc
            prefootrad
            bathygriddeltax
            bathygriddeltay
            lineangs
            gravity
            seaden
            submass
            subvol
            sysmass
            sysvol
            systyp
            Clen
            currentprof
            currentvel
            hubheight
            thrustcurv
            syswidth
            syslength
            dragcoefcyl
            windvel 
            winddir
            sysheight
            sysorienang
            sysrough
            airden
            sysdraft
            currentdir
            currentdragcoefrect
            hs
            tp
            gamma
            driftcoeffloatrect
            
            
            
        Attributes:
            bathyint (float) [-]: interpolation parameter
            bathysysorig (float) [m]: bathymetry at system origin
            rotorload (float) [N]: calculated rotor load
            fairlocglob (list) [-]: global coordinates for N fairleads: X coordinate (float) [m],
                                                                        Y coordinate (float) [m]
            foundlocglob (list) [-]: global coordinates for N foundations: X coordinate (float) [m],
                                                                           Y coordinate (float) [m]                                                                
            griddist (numpy.ndarray):   UTM coordinate X (float) [m],
                                        UTM coordinate Y (float) [m],
                                        bathymetry Z (float) [m],
                                        distance to foundation point (float) [m]
            gpsort (numpy.ndarray): sorted version of griddist by distance to foundation point
            gpnearind (list) [-]: indices of nearest grid point for N foundations 
            gpnearestinds (list) [-]: deep copy of gpnearind
            gpnear (list) [-]: indices of four nearest grid points
            gpnearlocs (list) [-]: copy of gpnear
            soiltyp (list) [-]: soil type at each foundation point
            soildep (list) [-]: soil depth at each foundation point
            soilgroup (list) [-]: soil group at each foundation point
            j, gpind, gp, x, y, tpind, s, tind, mode (int, float): temporary integers and values
            totsubstatloads (list): substation static loads:    X component (float) [N],
                                                                Y component (float) [N],
                                                                Z component (float) [N] 
            totsubsteadloads (list): substation steady loads:    X component (float) [N],
                                                                 Y component (float) [N],
                                                                 Z component (float) [N]
            totsysstatloads (list): system static loads:    X component (float) [N],
                                                            Y component (float) [N],
                                                            Z component (float) [N] 
            numstrips (float) [-]: number of strips for load decomposition
            rotorsa (float) [m2]: rotor swept area
            currentvelhub (float) [m/s]: current velocity at hub height
            rotorload (list): rotor load:    X component (float) [N],
                                             Y component (float) [N],
                                             Z component (float) [N]
            winddragcoef (list): wind drag coefficient:    front direction [-],
                                                           beam direction (for rectangular bodies) [-]
            dryprojarea (list): dry projected area:    front direction [-],
                                                       beam direction (for rectangular bodies) [-]
            windpress (list): wind pressure:    front direction [-],
                                                beam direction (for rectangular bodies) [-]
            windgustpress (list): wind gust pressure:    front direction [-],
                                                         beam direction (for rectangular bodies) [-]
            wlratio (float) [-]: width/length ratio   
            dragcoefint (float) [-]: drag coefficient interpolation parameters
            dryheight (float) [m]: system dry height
            windvel (float) [m/s]: wind velocity magnitude
            winddir (float) [rad]: wind direction
            windvelmax (float) [m/s]: wind velocity magnitude at reference height
            windangattk (float) [rad]: wind angle of attack
            windrn (float) [-]: wind induced reynolds number
            winddragcoefint (float) [-]: wind drag coefficient interpolation parameters
            winddragcoef (float) [-]: wind drag coefficient
            hwratio (float) [-]: system height/width ratio
            hlratio (float) [-]: system height/length ratio
            steadywind (list): steady wind load:     X component (float) [N],
                                                     Y component (float) [N],
                                                     Z component (float) [N]
            gustwind (list): gust wind load:    X component (float) [N],
                                                Y component (float) [N],
                                                Z component (float) [N]
            windload (list): maximum wind load:    X component (float) [N],
                                                   Y component (float) [N],
                                                   Z component (float) [N]
            wetheight (float) [m]: system wet height
            currentangattk (float) [rad]: current angle of attack
            currentload (list) [m/s]: current load for each strip
            currentloadloc (list) [N]: current load location for each strip
            currentdragcoef (list): current drag coefficient:    front direction [-],
                                                                 beam direction (for rectangular bodies) [-]
            deltahwet (float) [m]: strip height
            currentvelstrp (list) [m/s]: current velocity for each strip
            currentrn (float) [-]: current induced reynolds number
            currentdragcoefint (float) [-]: wind drag coefficient interpolation parameters
            wetprojarea (float) [-]: wet projected area
            eqcurrentload (float) [N]: equivalent current load over structure
            eqcurrentloadloc (float) [m]: equivalent current load location
            steadycurrent (list): steady current load:     X component (float) [N],
                                                           Y component (float) [N],
                                                           Z component (float) [N]
            meanwavedrift (list): mean wave drift load:     X component (float) [N],
                                                            Y component (float) [N],
                                                            Z component (float) [N]                                                           
            totsyssteadloads (list): total system steady load:  X component (float) [N],
                                                                Y component (float) [N],
                                                                Z component (float) [N]
            omegawave (list) [rad/s]: angular wave frequency for all analysed wave conditions
            hmax (list) [m]: maximum wave height for all analysed wave conditions
            hb (list) [m]: breaking wave height for all analysed wave conditions
            wavelength (list) [m]: wavelength for all analysed wave conditions
            wavenumber (list) [-]: wavenumber for all analysed wave conditions
            wnumerr (list) [-]: wavenumber error for all analysed wave conditions
            waveangattk (list) [-]: wave angle of attack for all analysed wave conditions
            wavnum (float) [-]: wnumfunc function input
            bandwidth (float) [-]: stationary seastate bandwidth
            tz (list) [s]: zero crossing period for all analysed wave conditions
            meandriftwaveload (float) [N]: mean wave drift load
            refleccoefint (float) [-]: reflection coefficient interpolation parameters
            refleccoef (float) [-]: reflection coefficient 
            syswaveloadmax (list): maxmium system wave load:    X component (float) [N],
                                                                Y component (float) [N],
                                                                Z component (float) [N]
            diffractrig (bool) [-]: diffraction flag
            syswaveload (numpy.ndarray): system wave load for all analysed wave conditions: :    X component (float) [N],
                                                                                                 Y component (float) [N],
                                                                                                 Z component (float) [N]
            ressyswaveload (list) [N]: resultant system wave load for all analysed wave conditions
            bessel1 (list) [-]: bessel function of the first kind of order 1 for all analysed wave conditions
            inlinecurrentvelstrp (list) [m/s]: inline current velocity for each strip
            wavedragcoefsteady (list): wave steady drag coefficient:    X component [-],
                                                                        Y component [-],
                                                                        Z component [-]
            waveaddmasscoef (list): wave added mass coefficient:    X component [-],
                                                                    Y component [-],
                                                                    Z component [-]
            wakeampfactorpi (list): wake amplification factor (Cpi):    X component [-],
                                                                        Y component [-],
                                                                        Z component [-]
            wakeampfactor (list): wake amplification factor:    X component [-],
                                                                Y component [-],
                                                                Z component [-]
            wavern (list): wave induced reynolds number:    X component [-],
                                                            Y component [-],
                                                            Z component [-]
            wavekc (list): wave induced keulegan-carpenter number:    X component [-],
                                                                      Y component [-],
                                                                      Z component [-]
            deltat (float) [s]: timestep
            timevec (list) [s]: time vector
            horzpartvelmax (numpy.ndarray) [m/s]: horizontal particle velocity for each timestep and strip
            vertpartvelmax (numpy.ndarray) [m/s]: vertical particle velocity for each timestep and strip
            horzpartaccmax (numpy.ndarray) [m/s]: horizontal particle acceleration for each timestep and strip
            vertpartaccmax (numpy.ndarray) [m/s]: vertical particle acceleration for each timestep and strip
            combhorzvel (numpy.ndarray) [m/s]: combined horizontal particle velocity for each timestep and strip
            projwidth (float) [m]: projected system width
            projlength (float) [m]: projected system length
            midpoint (float) [-]: vertical midpoint
            z (float) [m]: temporary vertical level
            t (float) [s]: temporary time value
            eqvertwaveload (float) [N]: equivalent vertical wave load
            waterlinearea  (float) [m2]: system waterline area
            eqhorzwaveload (float) [N]: equivalent horizontal wave load
            eqsysdraft (float) [m]: equivalent system draft
            horzwaveload (numpy.ndarray) [N]: horizontal wave load for each timestep and strip
            vertwaveload (numpy.ndarray) [N]: vertical wave load for each timestep and strip
            vertwaveloadloc (numpy.ndarray) [m]: vertical wave load location for each timestep and strip
            horzwaveloadmax (list) [N]: maximum horizontal wave load for each strip
            horzwaveloadloc (list) [m]: maximum horizontal wave load location for each strip
            wavedragcoefint (float) [-]: wave drag coefficient interpolation parameters
            wakeampfactorint (float) [-]: wake amplification factor interpolation parameters
            wakeampfactorinterp (float) [-]: interpolated wake amplification factor
            wavedragcoef (list): wave drag coefficient:    X component [-],
                                                           Y component [-],
                                                           Z component [-]
            syswaveloadmaxind (int) [-]: maximum wave load index
            eqhorzwaveloadloc (float) [m]: equivalent location of horizontal wave load
            ndsysrough (float) [-]: non-dimensional system surface roughness
                                          
        Functions:
            gpnearloc: locates nearest grid points to foundation locations to determine bathymetry and soil type
            substat: calculates substation static loads
            subststead: calculates substation steady loads
            sysstat: calculates system static loads
            sysstead: calculates system steady loads            
            syswave: calculates system wave loads 
            wnumfunc: calculates wavenumber using dispersion relation

    """

    def __init__(self, variables):
        
        self._variables = variables
        self.foundradnew = None
        self.numlines = None
        self.quanfound = None
        self.fairlocglob = []
        self.fairloc = None
        self.gpnear = []
        self.soiltyp = []
        self.soildep = [] 
        self.soilgroup = []
        self.foundlocglob = []
        self.foundloc = None
        self.mindevdist = None
        self.maxdisp = None
        self.lineangs = []
        self.bathysysorig = None
        
        return
        
    def gpnearloc(self, deviceid, systype, foundloc, sysorig, sysorienang):   

        # Get number of foundations required
        self.quanfound = self._get_foundation_quantity(systype, foundloc)
        
        # Set type specific variables
        if systype in ('wavefloat', 'tidefloat'):
            
            # Fairleads
            self._set_fairloc(self.quanfound)
            
            # Minimum distance between devices
            self.mindevdist = self._get_minimum_distance(sysorig)
            
            # Max displacement of floating devices
            self.maxdisp = self._get_maximum_displacement()
            
            # Foundation locations
            if self.foundradnew:
                self.foundloc = foundloc
            else:
                self.foundloc = self._get_foundation_locations(
                                                    self.lineangs,
                                                    self.quanfound)
            
        elif systype in ('wavefixed', 'tidefixed','substation'):
            
            self.foundloc = np.array(foundloc)
            
        else:
            
            errStr = "System type '{}' is not recognised.".format(systype)
            raise ValueError(errStr)
        
        foundlocglobs = []
        gpnears = []
        soil_types = []
        soil_depths = []
        soil_groups = []
        
        # Collect data for foundations
        for foundloc in self.foundloc:
            
            # Start foundlocglob by rounding x, y values
            foundlocglob = [round(foundloc[0], 3), round(foundloc[1], 3)]
            
            (closest_idx,
             closest_point) = self._get_closest_grid_point(sysorig,
                                                           foundloc)
            
            gpnearestinds, gpnear = self._get_neighbours(sysorig,
                                                         closest_point,
                                                         foundloc)

            founddepth = self._get_depth(foundloc,
                                         gpnearestinds)
            
            foundlocglob.append(founddepth)
            
            # Get soil data
            soiltype, soildepth = self._get_soil_type_depth(closest_idx)
            soilgroup = self._get_soil_group(soiltype)
                                    
            foundlocglobs.append(np.array(foundlocglob))
            gpnears.append(gpnear)
            soil_types.append(soiltype)
            soil_depths.append(soildepth)
            soil_groups.append(soilgroup)
            
        # Add data for device
        _, closest_point = self._get_closest_grid_point(sysorig)
        gpnearestinds, gpnear = self._get_neighbours(sysorig,
                                                     closest_point)
        sysdepth = self._get_depth(sysorig,
                                   gpnearestinds)
        sysloc = np.array([sysorig[0], sysorig[1], sysdepth])
        
        foundlocglobs.append(sysloc)

        # Update attributes
        self.foundlocglob = foundlocglobs
        self.gpnear = gpnears
        self.soiltyp = soil_types
        self.soildep = soil_depths
        self.soilgroup = soil_groups
        self.bathysysorig = -sysdepth
        
        return
        
    def sysstat(self,sysvol,sysmass):
        """ Calculate static system loads """
        
        """ Total system static loads (vertical upward = positive) """
        self.totsysstatloads = [0.0, 0.0, self._variables.gravity 
                                * (self._variables.seaden 
                                * sysvol - sysmass)]
                         
    def sysstead(self,systype,
                     syswidth,
                     syslength,
                     sysheight,
                     sysorienang,
                     sysprof,
                     sysdryfa,
                     sysdryba,
                     syswetfa,
                     syswetba,
                     sysvol,
                     sysrough):
        """ Calculate steady system loads """
        if (systype == "tidefixed" and sysheight < self.bathysysorig):
            sysheight = sysheight - self._variables.Clen[0] / 2.0
        """ Number of strips to decompose each section """
        self.numstrips = 10
        """ Estimate rotor loads based on depth distribution. It is assumed 
            that the nacelle is orientated to the flow direction """    
        """ Note: It is assumed that for twin axis designs each rotor has the 
            same loading (i.e. no yaw moment) """  
        self.currentangattk = (self._variables.currentdir 
                    - sysorienang) 
        if  self.currentangattk < 0.0:
             self.currentangattk = 360.0 +  self.currentangattk
        if systype in ("tidefixed","tidefloat"):
            if (self._variables.Clen[1] == 0.0 or
                self._variables.Clen[1] is None):
                """ Single axis rotor """
                self.rotorsa = (math.pi / 4.0) * self._variables.Clen[0] ** 2.0
            elif self._variables.Clen[1] > 0.0:
                """ Twin-axis rotor """
                self.rotorsa = (2.0 * (math.pi / 4.0) 
                                * self._variables.Clen[0] ** 2.0)
            else:
                errStr = ("Invalid turbine interdistance given. Must be "
                          "None or positive float")
                raise ValueError(errStr)
                
            if self._variables.currentprof == "1/7 power law": 
                if systype == 'tidefixed':
                    self.currentvelhub = (self._variables.currentvel 
                        * (self._variables.hubheight / self.bathysysorig) 
                        ** (1.0/7.0))
                elif systype == 'tidefloat':
                    self.currentvelhub = (self._variables.currentvel 
                        * ((self.bathysysorig - (self._variables.sysdraft 
                        + self._variables.hubheight)) / self.bathysysorig) 
                        ** (1.0/7.0))
                    
            elif self._variables.currentprof == "uniform":
                self.currentvelhub = self._variables.currentvel 
            self.thrustcoefint = interpolate.interp1d(
                            self._variables.thrustcurv[:,0], 
                            self._variables.thrustcurv[:,1],
                            bounds_error=False) 
            if (self.currentvelhub 
                >= min(self._variables.thrustcurv[:,0]) 
                and self.currentvelhub
                <= max(self._variables.thrustcurv[:,0])):
                self.thrustcoef = self.thrustcoefint(
                    self.currentvelhub)
            elif (self.currentvelhub
                < min(self._variables.thrustcurv[:,0])):
                self.thrustcoef = self._variables.thrustcurv[0,1]
                module_logger.warn('WARNING: Hub velocity out of data range')
            elif (self.currentvelhub
                > max(self._variables.thrustcurv[:,0])):
                self.thrustcoef = self._variables.thrustcurv[-1,1] 
                module_logger.warn('WARNING: Hub velocity out of data range')
            self.rotorload = [0.5 * self._variables.seaden * self.rotorsa 
                * self.thrustcoef 
                * self.currentvelhub ** 2.0 
                * math.sin(self.currentangattk 
                * math.pi / 180.0), 
                0.5 * self._variables.seaden * self.rotorsa 
                * self.thrustcoef 
                * self.currentvelhub ** 2.0 
                * math.cos(self.currentangattk * math.pi / 180.0), 0.0]
        elif systype in ("wavefixed","wavefloat", 'substation'):
            self.rotorload = [0.0, 0.0, 0.0]
            self._variables.hubheight = 0.0
        
        self.winddragcoef = [0.0, 0.0]
        self.dryprojarea = [0.0, 0.0]
        self.windpress = [0.0, 0.0]
        self.windgustpress = [0.0, 0.0]  
        self.wlratio = syswidth / syslength
           
        self.dragcoefint = interpolate.interp2d(self._variables.dragcoefcyl[0,1:],
                                                self._variables.dragcoefcyl[1:,0],
                                                self._variables.dragcoefcyl[1:,1:])
        """ Calculate steady wind loads if device is surface piercing """  
        
        if self._variables.windvel > 0.0:
            if systype in ('tidefixed','wavefixed','substation'): 
                if sysheight > self.bathysysorig: 
                    """ Calculate vector of steady wind loads using power law 
                        wind profile. Note: lift forces not considered """                    
                    self.dryheight = (sysheight - self.bathysysorig)
                else: 
                    self.dryheight = 0.0
                    self.steadywind = [0.0, 0.0, 0.0]
                    self.gustwind = [0.0, 0.0, 0.0]
                    self.windload = [0.0, 0.0, 0.0]
                    
            elif systype in ("wavefloat","tidefloat"):
                self.dryheight = (sysheight - self._variables.sysdraft)  
            if self.dryheight > 0.0:
                for w in range(0,2):
                    """ Two runs: i) steady wind load and ii) gust load """
                    if w == 0:
                        windvel = self._variables.windvel
                        winddir = self._variables.winddir
                    elif w == 1:
                        windvel = self._variables.windgustvel
                        winddir = self._variables.windgustdir
                    """ Reference height of 10m """
                    self.windvelmax = windvel * (self.dryheight/10.0) ** 0.12 
                    """ Angle of attack in local system axis. Direction from 
                        which wind is blowing """
                    self.windangattk = winddir - sysorienang
                    if self.windangattk < 0.0:
                        self.windangattk = 360.0 + self.windangattk
                    if sysprof == "cylindrical":
                        """ Shape coefficient based on Reynolds number. Smooth 
                            surface assumed """
                        self.windrn = (syswidth 
                                        * self.windvelmax / 1.45e-5)
                        ndsysrough = sysrough / syswidth
                        if self.windrn > 0.0:                        
                            """ Use end points for out of range values """
                            if self.windrn > self._variables.dragcoefcyl[-1,0]:
                                self.winddragcoefint = interpolate.interp1d(
                                    self._variables.dragcoefcyl[0,1:len(
                                    self._variables.dragcoefcyl[0,:])+1], 
                                    self._variables.dragcoefcyl[-1,1:len(
                                    self._variables.dragcoefcyl[0,:])+1],
                                    bounds_error=False) 
                                self.winddragcoef[0] = self.winddragcoefint(
                                    ndsysrough)
                                self.winddragcoef[1] = self.winddragcoef[0]
                                module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                            elif self.windrn < self._variables.dragcoefcyl[1,0]:
                                self.winddragcoefint = interpolate.interp1d(
                                    self._variables.dragcoefcyl[0,1:len(
                                    self._variables.dragcoefcyl[0,:])+1], 
                                    self._variables.dragcoefcyl[1,1:len(
                                    self._variables.dragcoefcyl[0,:])+1],
                                    bounds_error=False) 
                                self.winddragcoef[0] = self.winddragcoefint(
                                    ndsysrough)
                                self.winddragcoef[1] = self.winddragcoef[0]
                                module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                            else: 
                                self.winddragcoef[0] = self.dragcoefint(
                                    self.windrn, ndsysrough)
                                self.winddragcoef[1] = self.winddragcoef[0]
                    elif sysprof == "rectangular":
                        """ Wind coefficient values taken from Table 5-5 in 
                            DNV-RP-C205 """   
                        if self.wlratio == 1.0: 
                            self._variables.winddragcoefrect[0,1] = 0.5
                        self.winddragcoefint = interpolate.interp2d(
                            self._variables.winddragcoefrect[0,1:],
                            self._variables.winddragcoefrect[1:,0],
                            self._variables.winddragcoefrect[1:,1:])                                          
                        self.hwratio = self.dryheight / syswidth
                        self.hlratio = self.dryheight / syslength
                        self.winddragcoef[0] = self.winddragcoefint(
                            self.hlratio,self.wlratio ** -1.0) 
                        self.winddragcoef[1] = self.winddragcoefint(
                            self.hwratio,self.wlratio)  
                    if w == 0:
                        self.steadywind = [(0.5 * self._variables.airden 
                                * self.windvelmax ** 2.0) * self.winddragcoef[0] 
                                * sysdryba * math.sin(
                                self.windangattk * math.pi / 180.0), (0.5 
                                * self._variables.airden * self.windvelmax ** 2.0) 
                                * self.winddragcoef[1] * sysdryfa 
                                *  math.cos(self.windangattk * math.pi 
                                / 180.0), 0.0]
                    elif w == 1:
                        self.gustwind = [(0.5 * self._variables.airden 
                                * self.windvelmax ** 2.0) * self.winddragcoef[0] 
                                * sysdryba * math.sin(
                                self.windangattk * math.pi / 180.0), (0.5 
                                * self._variables.airden * self.windvelmax ** 2.0) 
                                * self.winddragcoef[1] * sysdryfa 
                                *  math.cos(self.windangattk * math.pi 
                                / 180.0), 0.0]                
        else: 
            self.steadywind = [0.0, 0.0, 0.0]
            self.gustwind = [0.0, 0.0, 0.0]
            self.windload = [0.0, 0.0, 0.0]
            self.dryheight = 0.0     
            
        """ Calculate steady current loads local to system. Wet height to MSL """
        if systype in ('tidefixed','wavefixed','substation'): 
            if sysheight >= self.bathysysorig:
                self.wetheight = self.bathysysorig
            else: 
                self.wetheight = sysheight
                
        elif systype in ("wavefloat","tidefloat"):
            self.wetheight = self._variables.sysdraft 
        self.currentload = [[0 for col in range(0,self.numstrips)] 
                                for row in range(0, 2)]
        self.currentloadloc = [0 for row in range(0, self.numstrips)]
        self.currentvelstrp = [0 for row in range(0, self.numstrips)]    
        self.currentdragcoef = [0.0, 0.0]
        self.deltahwet = self.wetheight / float(self.numstrips)

        """ Decritise structure into strips """
        for s in range(0, self.numstrips):            
            if self._variables.currentprof == "1/7 power law":
                if systype in ("wavefloat","tidefloat"):
                    self.currentvelstrp[s] = (self._variables.currentvel 
                        * (((self.bathysysorig - self.wetheight) 
                        + (0.5 * self.deltahwet + self.wetheight 
                        * (s / float(self.numstrips)))) 
                        / self.bathysysorig) ** (1.0/7.0))
                elif systype in ('tidefixed','wavefixed','substation'): 
                    self.currentvelstrp[s] = (self._variables.currentvel 
                        * (((0.5 * self.deltahwet + self.wetheight 
                        * (s / float(self.numstrips)))) 
                        / self.bathysysorig) ** (1.0/7.0))                             
            elif self._variables.currentprof == "uniform":
                self.currentvelstrp[s] = self._variables.currentvel    
            if sysprof == "cylindrical":
                self.currentrn = (syswidth * math.fabs(
                self.currentvelstrp[s])) / 1.35e-6     
                ndsysrough = sysrough / syswidth
                """ Drag coefficients from DNV-RP-C205. Note: for Re ~ 5e4 """                
                """ Use end points for out of range values """
                if self.currentrn > self._variables.dragcoefcyl[-1,0]:
                    self.currentdragcoefint = (interpolate.interp1d(
                        self._variables.dragcoefcyl[0,1:len(
                        self._variables.dragcoefcyl[0,:])+1], 
                        self._variables.dragcoefcyl[-1,1:len(
                        self._variables.dragcoefcyl[0,:])+1],
                        bounds_error=False))
                    self.currentdragcoefintp = self.currentdragcoefint(
                                            ndsysrough)
                    self.currentdragcoef[0] = self.currentdragcoefintp
                    self.currentdragcoef[1] = self.currentdragcoefintp
                    module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                elif self.currentrn < self._variables.dragcoefcyl[1,0]:
                    self.currentdragcoefint = (interpolate.interp1d(
                        self._variables.dragcoefcyl[0,1:len(
                        self._variables.dragcoefcyl[0,:])+1], 
                        self._variables.dragcoefcyl[1,1:len(
                        self._variables.dragcoefcyl[0,:])+1],
                        bounds_error=False))
                    self.currentdragcoefintp = self.currentdragcoefint(
                                            ndsysrough)
                    self.currentdragcoef[0] = self.currentdragcoefintp
                    self.currentdragcoef[1] = self.currentdragcoefintp
                    module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                else: 
                    self.currentdragcoefintp = self.dragcoefint(self.currentrn, 
                                                    ndsysrough)
                    self.currentdragcoef[0] = self.currentdragcoefintp[0] 
                    self.currentdragcoef[1] = self.currentdragcoefintp[0]                                                 

            elif sysprof == "rectangular":
                """ Use end points for out of range values """
                self.currentdragcoefint = interpolate.interp1d(
                        self._variables.currentdragcoefrect[:,0], 
                        self._variables.currentdragcoefrect[:,1],
                        bounds_error=False)                
                if (self.wlratio ** -1.0 <= self._variables.currentdragcoefrect[0,0]
                    and self.wlratio ** -1.0 >= self._variables.currentdragcoefrect[-1,0]):
                    self.currentdragcoef[0] = self.currentdragcoefint(
                                                self.wlratio ** -1.0)
                elif self.wlratio ** -1.0 > self._variables.currentdragcoefrect[0,0]:
                    self.currentdragcoef[0] = self._variables.currentdragcoefrect[0,1]
                    module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                elif self.wlratio ** -1.0 < self._variables.currentdragcoefrect[-1,0]:
                    self.currentdragcoef[0] = self._variables.currentdragcoefrect[-1,1]
                    module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                if (self.wlratio <= self._variables.currentdragcoefrect[0,0]
                    and self.wlratio >= self._variables.currentdragcoefrect[-1,0]):
                    self.currentdragcoef[1] = self.currentdragcoefint(
                                                self.wlratio)
                elif self.wlratio > self._variables.currentdragcoefrect[0,0]:
                    self.currentdragcoef[1] = self._variables.currentdragcoefrect[0,1]
                    module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                elif self.wlratio < self._variables.currentdragcoefrect[-1,0]:
                    self.currentdragcoef[1] = self._variables.currentdragcoefrect[-1,1]
                    module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
            """ Current load for each strip """
            self.currentload[0][s] = ((0.5 * self._variables.seaden 
                * self.currentdragcoef[0] * self.currentvelstrp[s] ** 2.0) 
                * syslength * self.deltahwet
                * math.sin(self.currentangattk * math.pi / 180.0))
            self.currentload[1][s] = ((0.5 * self._variables.seaden 
                * self.currentdragcoef[1] * self.currentvelstrp[s] ** 2.0)
                * syswidth * self.deltahwet
                * math.cos(self.currentangattk * math.pi / 180.0))
            self.currentloadloc[s] = (0.5 * self.deltahwet + self.wetheight 
                                * (s / float(self.numstrips))
                                * math.sqrt((self.currentload[0][s]) ** 2.0 
                                + (self.currentload[1][s]) ** 2.0))             
#       
        if self._variables.currentvel == 0.0:
            self.eqcurrentload = 0.0
            self.eqcurrentloadloc = 0.0
            self.steadycurrent = [0.0, 0.0, 0.0]
        else: 
            self.eqcurrentload = math.sqrt(sum(self.currentload[0]) ** 2.0 
                                            + sum(self.currentload[1]) ** 2.0)
            if math.fabs(self.eqcurrentload) > 0.0:
                self.eqcurrentloadloc = sum(self.currentloadloc) / self.eqcurrentload 
            else:
                self.eqcurrentloadloc = 0.0
            self.steadycurrent = [sum(self.currentload[0]), 
                                  sum(self.currentload[1]),
                                  0.0]     
        
        if self.gustwind > self.steadywind:
            self.windload = self.gustwind
        else:
            self.windload = self.steadywind
            
        if not self._variables.hs:
            self.eqhorzwaveloadloc = 0.0
            self.meanwavedrift = [0.0, 0.0, 0.0] 
            self.meanwavedriftsort = [0.0, 0.0, 0.0]
            self.syswaveload = [0.0, 0.0, 0.0]
            self.totsyssteadloads = map(sum,zip(self.rotorload,
                                                self.windload,
                                                self.steadycurrent))
        else:
            self.omegawave = [0 for row in range(0,len(self._variables.tp))]
            self.hmax = [0 for row in range(0,len(self._variables.tp))]
            self.hb = [0 for row in range(0,len(self._variables.tp))]
            self.wavelength = [0 for row in range(0,len(self._variables.tp))]
            self.wavenumber = [0 for row in range(0,len(self._variables.tp))]
            self.tz = [0 for row in range(0,len(self._variables.tp))]
            wnumerr = [0 for row in range(0,len(self._variables.tp))]
            self.meanwavedrift = [0 for row in range(0,len(self._variables.tp))]
            self.meanwavedriftsort = [0 for row in range(0,len(self._variables.tp))]
            self.totsyssteadloads = [0 for row in range(0,len(self._variables.tp))]
            self.waveangattk = [0 for row in range(0,len(self._variables.tp))]
       
            
            for tpind in range(0, len(self._variables.tp)): 
                self.waveangattk[tpind] = (self._variables.wavedir[tpind] 
                                            - sysorienang)
                if self.waveangattk[tpind] < 0.0:
                    self.waveangattk[tpind] = 360.0 + self.waveangattk[tpind]
                self.omegawave[tpind] = 2 * math.pi / self._variables.tp[tpind] 
                """ Find wavenumber using dispersion relation """
                def wnumfunc(wavnum):
                    return ((self.omegawave[tpind] ** 2.0) 
                            / (self._variables.gravity 
                            * math.tanh(wavnum * self.bathysysorig)))
                self.wavenumber[tpind] = optimize.fixed_point(wnumfunc, 
                                                                1.0e-10)
                wnumerr[tpind] = self.wavenumber[tpind] - (
                    self.omegawave[tpind] ** 2.0) / (self._variables.gravity 
                    * math.tanh(self.wavenumber[tpind] * self.bathysysorig))
                self.wavelength[tpind] = 2.0 * math.pi / self.wavenumber[tpind]
                """ Maximum breaking wave height DNV-RP-C205. Stationary 
                    sea-state lasting 3 hours is assumed """
                bandwidth = (-0.000191 * self._variables.gamma[tpind] ** 3.0 
                    + 0.00488 * self._variables.gamma[tpind] ** 2.0 - 0.0525 
                    * self._variables.gamma[tpind] - 0.605)
                self.tz[tpind] = np.divide(self._variables.tp[tpind],1.4049)
                self.hmax[tpind] = 0.5 * self._variables.hs[tpind] * math.sqrt(
                    (1.0 - bandwidth) * math.log(3.0 * 3600 / self.tz[tpind])) 
                """ Breaking wave limit (DNV-RP-C205) """
                self.hb[tpind] = self.wavelength[tpind] * 0.142 * math.tanh(2.0 * math.pi 
                                    * self.bathysysorig / self.wavelength[tpind])
                if self.hmax[tpind] > self.hb[tpind]:
                   self.hmax[tpind] = self.hb[tpind]
                
                """ Calculate vector of wave drift loads using far-field 
                    approach Note: i) Only horizontal drift forces are 
                    considered ii) Viscous drift force accounted for in 
                    Morison's equation iii) For surface piercing bodies 
                    only """
                if ((systype in ('tidefixed','wavefixed','substation')
                    and sysheight >= self.bathysysorig) 
                    or systype in ("tidefloat","wavefloat")):                    
                    if sysprof == "cylindrical":    
                        if systype in ('tidefixed','wavefixed','substation'):                            
                            """ McIver 1987 fixed cylinder """                            
                            self.meandriftwaveload = (5.0 
                                * self._variables.seaden 
                                * self._variables.gravity * math.pi ** 2.0 * (0.5 
                                * self.hmax[tpind]) ** 2.0 
                                * 0.5 * syswidth 
                                * (self.wavenumber[tpind] 
                                * 0.5 * syswidth) ** 3 
                                * (1.0 + (2.0 * self.wavenumber[tpind] 
                                * self.bathysysorig / math.sinh(2.0 
                                * self.wavenumber[tpind] 
                                * self.bathysysorig)))) / 16.0                            
                            self.meanwavedrift[tpind] = [self.meandriftwaveload 
                                * math.sin(self.waveangattk[tpind] 
                                * math.pi / 180.0), self.meandriftwaveload 
                                * math.cos(self.waveangattk[tpind] 
                                * math.pi / 180.0), 0.0]
                            if self.wavenumber[tpind] * 0.5 * syswidth > 1.0:
                                module_logger.warn('WARNING: ka > 1.0. Horizontal mean drift load may be inaccurate')
                        elif systype in ("tidefloat","wavefloat"):
                             """ Approximation suitable for relatively long 
                                 wavelengths in deep water ka << 1 (Eatock 
                                 Taylor, Hu and Nielsen, 1990). This is correct 
                                 within 10% for ka < 0.4 """
                             self.meandriftwaveload = (5.0 
                                 * self._variables.seaden 
                                 * self._variables.gravity * math.pi ** 2.0 
                                 * (0.5 * self.hmax[tpind]) ** 2.0 
                                 * 0.5 * syswidth 
                                 * (self.wavenumber[tpind] 
                                 * 0.5 * syswidth) ** 3.0) / 16.0  
                        self.meanwavedrift[tpind] = [self.meandriftwaveload 
                            * math.sin(self.waveangattk[tpind] 
                            * math.pi / 180.0), self.meandriftwaveload 
                            * math.cos(self.waveangattk[tpind] * math.pi / 180.0)
                            , 0.0]
                        if self.wavenumber[tpind] * 0.5 * syswidth > 1.0:
                            module_logger.warn('WARNING: ka > 1.0. Horizontal mean drift load may be inaccurate')
                    elif sysprof == "rectangular":
                        """ Approximation of mean wave drift using reflection 
                            coefficients from Hermans and Remery 1970. Coefficients are 
                            for a barge with draft/water depth ratio of 0.175. """
                        module_logger.warn('WARNING: Drift coefficients based on a barge with draft/water depth ratio of 0.175')
                        self.refleccoefint = interpolate.interp1d(
                            self._variables.driftcoeffloatrect[:,0], 
                            self._variables.driftcoeffloatrect[:,1],
                            bounds_error=False) 
                        if (self.omegawave[tpind] 
                            >= min(self._variables.driftcoeffloatrect[:,0]) 
                            and self.omegawave[tpind]
                            <= max(self._variables.driftcoeffloatrect[:,0])):
                            self.refleccoef = self.refleccoefint(
                                self.omegawave[tpind])
                        elif (self.omegawave[tpind] 
                            < min(self._variables.driftcoeffloatrect[:,0])):
                            self.refleccoef = self._variables.driftcoeffloatrect[0,1]
                            module_logger.warn('WARNING: angular wave frequency out of data range')
                        elif (self.omegawave[tpind]
                            > max(self._variables.driftcoeffloatrect[:,0])):
                            self.refleccoef = self._variables.driftcoeffloatrect[-1,1] 
                            module_logger.warn('WARNING: angular wave frequency out of data range')                            
                        """ Mean wave drift load per unit length of side """
                        self.meandriftwaveload = (0.5 * self._variables.seaden 
                            * self._variables.gravity * (0.5 * self.hmax[tpind] 
                            * self.refleccoef) ** 2.0)                        
                        self.meanwavedrift[tpind] = [
                            self.meandriftwaveload * syslength 
                            * math.sin(self.waveangattk[tpind] 
                            * math.pi / 180.0),
                            self.meandriftwaveload 
                            * syswidth * math.cos(
                            self.waveangattk[tpind] * math.pi / 180.0), 0.0]                         
                else: self.meanwavedrift[tpind] = [0.0, 0.0, 0.0]                
                
                """ Determine if diffraction is important based on projected 
                    length of device """ 
                if sysprof == 'rectangular':
                    if (self.waveangattk[tpind] >= 0.0 and  self.waveangattk[tpind] < 180.0):
                        self.projlength = math.fabs((syswidth * math.cos(math.fabs(
                            self.waveangattk[tpind] * math.pi / 180.0)) 
                            + syslength * math.sin(math.fabs(
                            self.waveangattk[tpind] * math.pi / 180.0))))
                        self.projdepth = math.fabs((syswidth * math.sin(math.fabs(
                            self.waveangattk[tpind] * math.pi / 180.0)) 
                            + syslength * math.cos(math.fabs(
                            self.waveangattk[tpind] * math.pi / 180.0))))
                    elif (self.waveangattk[tpind] >= 180.0 and  self.waveangattk[tpind] < 360.0):
                        self.projlength = math.fabs((syswidth * math.cos(math.fabs(
                            (360.0 - self.waveangattk[tpind]) * math.pi / 180.0)) 
                            + syslength * math.sin(math.fabs(
                            (360.0 - self.waveangattk[tpind]) * math.pi / 180.0))))
                        self.projdepth = math.fabs((syswidth * math.sin(math.fabs(
                            (360.0 - self.waveangattk[tpind]) * math.pi / 180.0)) 
                            + syslength * math.cos(math.fabs(
                            (360.0 - self.waveangattk[tpind]) * math.pi / 180.0))))
                elif sysprof == 'cylindrical':
                    self.projlength = syswidth
                    self.projdepth = syswidth

                """ Combined steady wind, current and system loads (contribution of mean drift added later).
                    Note: if combined current and wave kinematics are used in the syswave calculations,
                    contribution of steadycurrent is removed from totsyssteadloads """                 
                self.totsyssteadloads[tpind] = map(sum,zip(self.rotorload,
                                    self.windload,
                                    self.steadycurrent))

    def syswave(self,deviceid,
                    systype,
                    syswidth,
                    syslength,
                    sysheight,
                    sysprof,
                    sysvol,
                    sysrough):        
        """ Calculate first-order wave loads on system """
        """ Note: this method will be invoked several times for Hs and Tp 
            values along the environmental condition contour line """
        if (systype == "tidefixed" and sysheight < self.bathysysorig):
            sysheight = sysheight - self._variables.Clen[0] / 2.0
        if not self._variables.hs:
            self.syswaveloadmax = [0.0, 0.0,  0.0] 
            self.diffractrig = 'False'
            pass
        else:
            wavefreqs = [0 for row in range(0,len(self._variables.tp))]  
            fexrao = [[0 for col in range(0,3)] for row in range(0,len(self._variables.tp))]
            fexraovaluesin = [0 for col in range(0,3)]
            fexraovalues = [0 for col in range(0,3)]
            self.syswaveload = [[0 for col in range(0,3)] for row in range(0,len(self._variables.tp))]
            self.syswaveloadsort = [0 for row in range(0,len(self._variables.tp))]
            self.ressyswaveload = [0 for row in range(0,len(self._variables.tp))]
            besselj0 = [0 for row in range(0,len(self._variables.tp))]  
            besselj1 = [0 for row in range(0,len(self._variables.tp))]  
            besselj2 = [0 for row in range(0,len(self._variables.tp))]
            bessely0 = [0 for row in range(0,len(self._variables.tp))]  
            bessely1 = [0 for row in range(0,len(self._variables.tp))]  
            besselj1p = [0 for row in range(0,len(self._variables.tp))]  
            bessely1p = [0 for row in range(0,len(self._variables.tp))]
            # hankel1 = [0 for row in range(0,len(self._variables.tp))]
            modbesseli0 = [0 for row in range(0,len(self._variables.tp))]
            modbesseli1 = [0 for row in range(0,len(self._variables.tp))]
            negdragflag = ['False' for row in range(0,len(self._variables.tp))]
            aka =  [0 for row in range(0,len(self._variables.tp))]
            phase =  [0 for row in range(0,len(self._variables.tp))]
            inlinecurrentvelstrp = [0 for row in range(0, self.numstrips)] 
            perpcurrentvelstrp = [0 for row in range(0, self.numstrips)] 
            fexraovaluesnew = [0 for row in range(0, 3)] 
            self.wavedragcoefsteady = [0.0, 0.0, 0.0]
            self.waveaddmasscoef = [0.0, 0.0, 0.0]
            self.wavedragcoef = [0.0, 0.0, 0.0]
            self.waveinertiacoef = [0.0, 0.0, 0.0]
            self.wakeampfactorpi = [0.0, 0.0, 0.0]
            self.wakeampfactor = [0.0, 0.0, 0.0]
            self.wavern = [0.0, 0.0, 0.0]
            self.wavekc = [0.0, 0.0, 0.0]
            fexfullset = 'False'
            wavefreqtol = 0.0001
            wavedirtol = 0.01
            
            """ Determine if externally calculated WAMIT or NEMOH excitation loads have been provided
                Note: Only one input file is used at the moment. Possibility to extend this for substations """
            if (self._variables.fex and systype in ('wavefloat','tidefloat','wavefixed','tidefixed')):
                module_logger.info('WAMIT/NEMOH parameters provided')
                """ Determine which wave frequencies, directions and translation modes have been analysed """
                """ If parameters for only one device have been supplied use these for all devices """      
                if isinstance(self._variables.fex,dict):
                    fexfreqs = np.array(self._variables.fex['wave_fr'])
                    fexdirs = np.array(self._variables.fex['wave_dir'])
                    fexmodes2 = np.array(map(int, self._variables.fex['mode_def']))
                    raoin = np.absolute(self._variables.fex['fex'][1])
                    fexraovalues[0] = [raoin.tolist()]
                    raoin = np.absolute(self._variables.fex['fex'][0])
                    fexraovalues[1] = [raoin.tolist()]
                    raoin = np.absolute(self._variables.fex['fex'][2])
                    fexraovalues[2] = [raoin.tolist()]
                else:
                    fexfreqs = np.array(self._variables.fex[0])
                    fexdirs = np.array(self._variables.fex[1])
                    fexmodes2 = np.array(map(int, self._variables.fex[2]))
                    fexraovaluesin = np.array(self._variables.fex[3]) 
                    fexraovalues = np.absolute([fexraovaluesin[1],fexraovaluesin[0],fexraovaluesin[2]])
                fexsym90flag = 'False'
                for i in range(0,len(self._variables.tp)):
                    if ((max(fexfreqs) < 1.0/self._variables.tp[i]) or (min(fexfreqs) > 1.0/self._variables.tp[i])):
                        errStr = ("Wave period outside of RAO frequency range")
                        raise RuntimeError(errStr)
                fexmodes = [fexmodes2[1],fexmodes2[0],fexmodes2[2]]
                
                """ If difference between maximum and minimum specified directions is 90 deg, assumed 
                    that the device has symmetry about the x and y axes """   
                
                if (math.fabs(90.0 - math.fabs(max(fexdirs) - min(fexdirs))) < wavedirtol
                    or math.fabs(180.0 - math.fabs(max(fexdirs) - min(fexdirs))) < wavedirtol):
                    if math.fabs(90.0 - math.fabs(max(fexdirs) - min(fexdirs))) < wavedirtol:
                        logmsg = [""]       
                        logmsg.append('Double body symmetry assumed')                        
                        module_logger.info("\n".join(logmsg))
                        fexsym90flag = 'True'
                        fexdirsnew = []
                        fexdirinds = range(0,len(fexdirs))
                        fexdirindsrev = sorted(fexdirinds, key=int, reverse=True)
                        for quadind in range(0,3):                        
                            for angind, ang in enumerate(fexdirs):                            
                                if fexdirs[0] == 0.0:
                                    fexdirsnew.append(quadind * max(fexdirs) + ang + quadind * fexdirs[1])
                                else:
                                    fexdirsnew.append(quadind * max(fexdirs) + ang + quadind * fexdirs[0])                                      
                        # del fexdirsnew[-1]
                        fexdirsnew = np.array(fexdirsnew)
                        for mode in range(0,3):                              
                            addfexvalues = []
                            for quadind in range(0,4): 
                                if quadind == 0:
                                    for dirind in fexdirinds:
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist()) 
                                if quadind == 1:
                                    for dirind in fexdirindsrev[1:]: 
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist()) 
                                if quadind == 2:
                                    for dirind in fexdirinds[1:]:               
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist()) 
                                if quadind == 3:
                                    for dirind in fexdirindsrev[1:]:
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist())    
                            fexraovaluesnew[mode] = addfexvalues
                        fexraovaluesnew = np.array(fexraovaluesnew)   
                    elif math.fabs(180.0 - math.fabs(max(fexdirs) - min(fexdirs))) < wavedirtol:
                        logmsg = [""]       
                        logmsg.append('Single body symmetry assumed') 
                        module_logger.info("\n".join(logmsg))
                        fexsym180flag = 'True'
                        fexdirsnew = []
                        fexdirinds = range(0,len(fexdirs))
                        fexdirindsrev = sorted(fexdirinds, key=int, reverse=True)
                        for halfind in range(0,2):
                            for angind, ang in enumerate(fexdirs):
                                if fexdirs[0] == 0.0:
                                    fexdirsnew.append(halfind * max(fexdirs) + ang + halfind * fexdirs[1])
                                else:
                                    fexdirsnew.append(halfind * max(fexdirs) + ang + halfind * fexdirs[0])
                        del fexdirsnew[-1:]
                        fexdirsnew = np.array(fexdirsnew)                              
                        for mode in range(0,3):     
                            addfexvalues = []
                            for halfind in range(0,2): 
                                if halfind == 0:
                                    for dirind in fexdirinds:
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist()) 
                                if halfind == 1:
                                    for dirind in fexdirindsrev[1:]:
                                        addfexvalues.append(fexraovalues[mode][dirind].tolist())                                
                            fexraovaluesnew[mode] = addfexvalues
                        fexraovaluesnew = np.array(fexraovaluesnew) 
                else:
                    fexraovaluesnew = fexraovalues
                    fexdirsnew = np.array(fexdirs)                     
                if all(i == 1 for i in fexmodes):
                    logmsg = [""]       
                    logmsg.append('Full set of first-order wave load RAOs provided')
                    fexfullset = 'True'
                for tpind in range(0, len(self._variables.tp)):
                    nearfreqinds = [0 for col in range(0,2)]
                    neardirinds = [0 for col in range(0,2)]
                    if len(fexdirsnew) ==  1:
                        self.waveangattk[tpind] = fexdirsnew[0]
                    wavefreqs[tpind] = self._variables.tp[tpind] ** -1.0             
                    nearfreqinds = [y for y, x in enumerate(fexfreqs) if math.fabs(x - wavefreqs[tpind]) < wavefreqtol]                    
                    neardirinds = [y for y, x in enumerate(fexdirsnew) if math.fabs(x - self.waveangattk[tpind]) < wavedirtol]
                    if neardirinds:
                        neardirinds = neardirinds
                    else:
                        """ Find nearest wave directions """
                        neardirinds = [[(x,y) for x,y in enumerate(fexdirsnew) if y == fexdirsnew[
                                        np.argsort(np.abs(fexdirsnew-self.waveangattk[tpind]))[0]]][0][0],
                                        [(x,y) for x,y in enumerate(fexdirsnew) if y == fexdirsnew[
                                        np.argsort(np.abs(fexdirsnew-self.waveangattk[tpind]))[1]]][0][0]]                                         
                    if nearfreqinds:
                        nearfreqinds = nearfreqinds
                    else:
                        """ Find nearest wave frequencies """
                        nearfreqinds = [[(x,y) for x,y in enumerate(fexfreqs) if y == fexfreqs[
                                        np.argsort(np.abs(fexfreqs-wavefreqs[tpind]))[0]]][0][0],
                                        [(x,y) for x,y in enumerate(fexfreqs) if y == fexfreqs[
                                        np.argsort(np.abs(fexfreqs-wavefreqs[tpind]))[1]]][0][0]]                        
                    for mode in range(0,3):                                       
                        if fexmodes[mode] == 1:
                            if mode in (0,1):
                                """ Default assumption that maximum wave loads are applied 
                                at crest height (surface piercing) or system height (subsea) """
                                if (sysheight >= self.bathysysorig + 0.5 
                                    * self.hmax[tpind]):
                                    self.eqhorzwaveloadloc = (self.bathysysorig + 0.5 
                                                                * self.hmax[tpind])
                                else:
                                    self.eqhorzwaveloadloc = sysheight
                                
                            if (len(nearfreqinds) > 1 and len(neardirinds) > 1): 
                                fexraoint = interpolate.interp2d(fexfreqs[nearfreqinds],
                                                                 fexdirsnew[neardirinds],
                                                                 [fexraovaluesnew[mode][neardirinds[0]][nearfreqinds],
                                                                  fexraovaluesnew[mode][neardirinds[1]][nearfreqinds]],
                                                                 bounds_error=False)
                                fexrao[tpind][mode] = fexraoint(wavefreqs[tpind], self.waveangattk[tpind])                                
                            else: 
                                if (len(nearfreqinds) == 1 and len(neardirinds) > 1):
                                    fexraoint = interpolate.interp1d(fexdirsnew[neardirinds],
                                                                     [fexraovaluesnew[mode][neardirinds[0]][nearfreqinds[0]],
                                                                      fexraovaluesnew[mode][neardirinds[1]][nearfreqinds[0]]],
                                                                     bounds_error=False)                                                                     
                                    fexrao[tpind][mode] = fexraoint(self.waveangattk[tpind])                                  
                                elif (len(nearfreqinds) > 1 and len(neardirinds) == 1):
                                    fexraoint = interpolate.interp1d(fexfreqs[nearfreqinds],
                                                                     [fexraovaluesnew[mode][neardirinds[0]][nearfreqinds[0]],
                                                                      fexraovaluesnew[mode][neardirinds[0]][nearfreqinds[1]]],
                                                                     bounds_error=False)                                                                     
                                    fexrao[tpind][mode] = fexraoint(wavefreqs[tpind])                          
                        elif fexmodes[mode] == 0:
                            logmsg.append('First-order wave load RAOs not provided for mode {}'.format(mode))  
                    """ X-Y values flipped to stick with WAMIT convention """      
                    if self.waveangattk[tpind] >= 0.0 and self.waveangattk[tpind] <= 90.0:
                        self.syswaveload[tpind][0] = 0.5 * self.hmax[tpind] * fexrao[tpind][1] 
                        self.syswaveload[tpind][1] = 0.5 * self.hmax[tpind] * fexrao[tpind][0]
                    elif self.waveangattk[tpind] > 90.0 and self.waveangattk[tpind] <= 180.0:
                        self.syswaveload[tpind][0] = 0.5 * self.hmax[tpind] * fexrao[tpind][1] 
                        self.syswaveload[tpind][1] = -0.5 * self.hmax[tpind] * fexrao[tpind][0] 
                    elif self.waveangattk[tpind] > 180.0 and self.waveangattk[tpind] < 270.0:
                        self.syswaveload[tpind][0] = -0.5 * self.hmax[tpind] * fexrao[tpind][1] 
                        self.syswaveload[tpind][1] = -0.5 * self.hmax[tpind] * fexrao[tpind][0] 
                    elif self.waveangattk[tpind] >= 270.0 and self.waveangattk[tpind] < 360.0:
                        self.syswaveload[tpind][0] = -0.5 * self.hmax[tpind] * fexrao[tpind][1] 
                        self.syswaveload[tpind][1] = 0.5 * self.hmax[tpind] * fexrao[tpind][0] 
                    self.syswaveload[tpind][2] = 0.5 * self.hmax[tpind] * fexrao[tpind][2] 
                    # logmsg.append('wavefreqs[tpind] {}'.format(wavefreqs[tpind]))
                    # logmsg.append('fexraoint(wavefreqs[tpind])  {}'.format(fexraoint(wavefreqs[tpind])))
                    # logmsg.append('self.syswaveload[tpind]  {}'.format(self.syswaveload))
                    # logmsg.append('raovalues  {}'.format(fexraovalues))
                    # logmsg.append('fexfreqs  {}'.format(fexfreqs))
                    module_logger.info("\n".join(logmsg)) 
                        
                    """ Find maximum resultant wave load due to wave conditions 
                        along upper contour """
                    self.ressyswaveload[tpind] = math.sqrt(
                        self.syswaveload[tpind][0] ** 2.0 
                        + self.syswaveload[tpind][1] ** 2.0 
                        + self.syswaveload[tpind][2] ** 2.0)
                            
            else:
                module_logger.info('WAMIT/NEMOH parameters not provided')
                fexmodes = [0, 0, 0]
            
            if fexfullset == 'True':
                pass
            else:            
                """ Timestep """
                deltat = 0.01
                """ Calculate wave length. using trial and error approach to solve 
                    w^2=gktanh(kd) """        
                for tpind in range(0, len(self._variables.tp)):
                    timevec = np.linspace(0.0, self._variables.tp[tpind], 
                                          self._variables.tp[tpind] / deltat)
                    horzpartvelmax = [[0 for col in range(0,len(timevec))] for row 
                        in range(0, self.numstrips)]
                    vertpartvelmax = [[0 for col in range(0,len(timevec))] for row 
                        in range(0, self.numstrips)]
                    horzpartaccmax = [[0 for col in range(0,len(timevec))] for row 
                        in range(0, self.numstrips)]
                    vertpartaccmax = [[0 for col in range(0,len(timevec))] for row 
                        in range(0, self.numstrips)]
                    combhorzvel = [[0 for col in range(0,len(timevec))] for row 
                        in range(0, self.numstrips)]
                
                    """ Bessel functions and their derivatives """
                    besselj0[tpind] = special.j0(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)
                    besselj1[tpind] = special.j1(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)
                    bessely0[tpind] = special.y0(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)
                    bessely1[tpind] = special.y1(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)
                    besselj1p[tpind] = special.jvp(1, self.wavenumber[tpind] * 0.5 
                                        * self.projlength, n=1)
                    bessely1p[tpind] = special.yvp(1, self.wavenumber[tpind] * 0.5 
                                        * self.projlength, n=1)
                    modbesseli0[tpind] = special.i0(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)
                    modbesseli1[tpind] = special.i1(self.wavenumber[tpind] * 0.5 
                                        * self.projlength)                
                    aka[tpind] = (besselj1p[tpind] ** 2.0 
                                        + bessely1p[tpind] ** 2.0) ** -0.5                
                    phase[tpind] = math.atan(besselj1p[tpind] / bessely1p[tpind])
                    self.steadycurrentremflag = ['False' for row 
                                                in range(0, len(self._variables.tp))] 
                    
                    """ Effective inertia coefficient """
                    self.effwaveinertiacoef = (4.0 * aka[tpind]) / (math.pi 
                        * (self.wavenumber[tpind] * 0.5 * self.projlength) ** 2.0)
                                

                    if systype in ("tidefixed","wavefixed","substation"): 
                        if sysheight >= self.bathysysorig:
                            self.wetheight = self.bathysysorig 
                        else:   
                            self.wetheight = sysheight 
                    elif systype in ("tidefloat", "wavefloat"):
                        self.wetheight = self._variables.sysdraft  
                    """ Vertical midpoint """
                    midpoint = int(self.numstrips/2)
                    for s in range(0, self.numstrips):
                        inlinecurrentvelstrp[s] = (self.currentvelstrp[s] 
                            * math.cos(math.pi * (self._variables.currentdir 
                            - self._variables.wavedir[tpind]) / 180.0))
                        perpcurrentvelstrp[s] = (self.currentvelstrp[s] 
                            * math.sin(math.pi * (self._variables.currentdir 
                            - self._variables.wavedir[tpind]) / 180.0))
                        if systype in ('tidefixed','wavefixed','substation'): 
                            z = self.wetheight * (s / float(self.numstrips))
                        elif systype in ("tidefloat", "wavefloat"):
                            z = (self.wetheight * (s / float(self.numstrips)) 
                                + self.bathysysorig - self._variables.sysdraft)
                        for tind, t in enumerate(timevec):
    #                            print t
                            if self.bathysysorig / self.wavelength[tpind] > 0.5:
                                # module_logger.info('Deep water') 
                                """ Deep water """
                                """ Without Wheeler stretching """
                                horzpartvelmax[s][tind] = ((self.omegawave[tpind] 
                                    * 0.5 * self.hmax[tpind]) 
                                    * math.exp(self.wavenumber[tpind] 
                                    * (z -self.bathysysorig)) * math.sin(
                                    self.omegawave[tpind] * t))
                                vertpartvelmax[s][tind] = ((self.omegawave[tpind] 
                                    * 0.5 * self.hmax[tpind]) 
                                    * math.exp(self.wavenumber[tpind] * (z 
                                    - self.bathysysorig)) * math.cos(
                                    self.omegawave[tpind] * t))
                                horzpartaccmax[s][tind] = ((self.omegawave[tpind] ** 2.0
                                    * 0.5 * self.hmax[tpind])
                                    * math.exp(self.wavenumber[tpind] * (z 
                                    - self.bathysysorig)) * math.cos(
                                    self.omegawave[tpind] * t))
                                vertpartaccmax[s][tind] = -((self.omegawave[tpind] ** 2.0
                                    * 0.5 * self.hmax[tpind])
                                    * math.exp(self.wavenumber[tpind] * (z 
                                    - self.bathysysorig)) * math.sin(
                                    self.omegawave[tpind] * t))
                                combhorzvel[s][tind] = (inlinecurrentvelstrp[s] 
                                    + horzpartvelmax[s][tind])
                            else: 
                                # module_logger.info('Finite water depth')
                                """ Finite depth """
                                """ Without Wheeler stretching """
                                horzpartvelmax[s][tind] = ((self.omegawave[tpind] 
                                    * 0.5 * self.hmax[tpind])  
                                    * (math.cosh(self.wavenumber[tpind] * z) 
                                    * math.sin(self.omegawave[tpind] * t)) 
                                    / math.sinh(self.wavenumber[tpind] 
                                    * self.bathysysorig))
                                vertpartvelmax[s][tind] = ((self.omegawave[tpind] 
                                    * 0.5 * self.hmax[tpind]) 
                                    * (math.sinh(self.wavenumber[tpind] * z) 
                                    * math.cos(self.omegawave[tpind] * t)) 
                                    / math.sinh(self.wavenumber[tpind] 
                                    * self.bathysysorig))
                                horzpartaccmax[s][tind] = ((self.omegawave[tpind] ** 2.0
                                    * 0.5 * self.hmax[tpind])
                                    * (math.cosh(self.wavenumber[tpind] * z) 
                                    * math.cos(self.omegawave[tpind] * t)) 
                                    / math.sinh(self.wavenumber[tpind] 
                                    * self.bathysysorig))
                                vertpartaccmax[s][tind] = -((self.omegawave[tpind] ** 2.0
                                    * 0.5 * self.hmax[tpind])
                                    * (math.sinh(self.wavenumber[tpind] * z) 
                                    * math.sin(self.omegawave[tpind] * t)) 
                                    / math.sinh(self.wavenumber[tpind] 
                                    * self.bathysysorig))
                                combhorzvel[s][tind] = (inlinecurrentvelstrp[s] 
                                    + horzpartvelmax[s][tind])
                            
                    if (math.pi * self.projlength / self.wavelength[tpind]) > 0.5:
                        self.diffractrig = 'True'
                        horzwaveload = [[0 for col in range(0,self.numstrips)] 
                                        for row in range(0, len(timevec))]
                        horzwaveloadloc = [0 for col in range(0, self.numstrips)]
                        horzwaveloadsum = [0 for row in range(0, len(timevec))] 
                        vertwaveload = [[0 for col in range(0,self.numstrips)] 
                                        for row in range(0, len(timevec))]
                        totalforce = [0 for row in range(0, len(timevec))]
                        
                        module_logger.info("Wave diffraction important")
                        """ F-K approximations in the absence of externally 
                            calculated hydrodynamic parameters """
                     
                        if self._variables.sysprof == "cylindrical":
                            self.waterlinearea = ((math.pi / 4.0) 
                                    * syswidth ** 2.0)
                            
                            if systype in ("wavefixed", "tidefixed", "substation"):                             
                                if (sysheight < 
                                        (self.bathysysorig)):
                                    """ Truncated cylinder """
                                    horzforcecoef = (1.0 + 0.75 
                                        * ((sysheight 
                                        / syswidth) ** (1.0 / 3.0)) 
                                        * (1.0 - 0.3 * (self.wavenumber[tpind] * 0.5
                                        * syswidth) ** 2.0))
                                    alpha = (1.48 * self.wavenumber[tpind] * 0.5                                    
                                        * sysheight)
                                    if alpha < 1.0:
                                        vertforcecoef = (1.0 + 0.74 
                                            * ((self.wavenumber[tpind] * 0.5                                    
                                            * syswidth) ** 2.0) 
                                            * (sysheight 
                                            / syswidth))
                                    elif alpha >= 1.0:
                                        vertforcecoef = (1.0 + 0.25 
                                            * self.wavenumber[tpind] 
                                            * syswidth)
                                    if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                        """ Approach by Hogben and Standing, 1975 which appears in Sarpkaya/Isaacson """
                                        # module_logger.warn('horzforcecoef {}'.format(horzforcecoef))
                                        # module_logger.warn('self._variables.seaden {}'.format(self._variables.seaden))
                                        # module_logger.warn('self.hmax[tpind] {}'.format(self.hmax[tpind]))
                                        # module_logger.warn('syswidth {}'.format(syswidth))
                                        # module_logger.warn('sysheight {}'.format(sysheight))
                                        # module_logger.warn('self.bathysysorig {}'.format(self.bathysysorig))
                                        # module_logger.warn('besselj1[tpind] {}'.format(besselj1[tpind]))
                                        # module_logger.warn('self.wavenumber[tpind] {}'.format(self.wavenumber[tpind]))
                                        self.eqhorzwaveload  = (horzforcecoef * math.pi 
                                            * self._variables.seaden 
                                            * self._variables.gravity
                                            * self.hmax[tpind] * 0.5
                                            * syswidth
                                            * self.bathysysorig
                                            * besselj1[tpind]
                                            * (math.sinh(self.wavenumber[tpind] 
                                            * sysheight) 
                                            / (self.wavenumber[tpind] 
                                            * self.bathysysorig 
                                            * math.cosh(self.wavenumber[tpind] 
                                            * self.bathysysorig))))
                                    if fexmodes[2] == 0:
                                        self.eqvertwaveload  = (vertforcecoef * 0.5 
                                            * math.pi * self._variables.seaden 
                                            * self._variables.gravity
                                            * self.hmax[tpind] * ((0.5
                                            * syswidth) ** 2.0)  
                                            * (besselj0[tpind] + besselj2[tpind])
                                            * ((math.cosh(self.wavenumber[tpind] 
                                            * sysheight) 
                                            / math.cosh(self.wavenumber[tpind] 
                                            * self.bathysysorig))))
                                else: 
                                    if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                        """ Surface piercing cylinder """
                                        """ Diffraction theory approximation to horizontal 
                                            force """
                                        for s in range(0, self.numstrips): 
                                            z = (self.wetheight * (s / float(self.numstrips))) - self.wetheight
                                            
                                            for tind, t in enumerate(timevec):
                                                horzwaveload[tind][s] = ((math.pi / 8.0) 
                                                    * self._variables.seaden 
                                                    * self._variables.gravity * self.hmax[tpind] 
                                                    * self.wavenumber[tpind] 
                                                    * syswidth ** 2.0 
                                                    * (math.cosh(self.wavenumber[tpind] * (z + self.bathysysorig)) 
                                                    / math.cosh(self.wavenumber[tpind] 
                                                    * self.bathysysorig)) * self.effwaveinertiacoef 
                                                    * math.cos(self.omegawave[tpind] * tind - phase[tpind]))
                                        """ Vertical wave load zero on seafloor fixed - 
                                        surface piercing structures """
                                        self.eqvertwaveload = 0.0
                                        
                                        for tind, t in enumerate(timevec):  
                                            totalforce[tind] = ((math.pi / 8.0) 
                                                    * self._variables.seaden 
                                                    * self._variables.gravity * self.hmax[tpind] 
                                                    * syswidth ** 2.0 
                                                    * math.tanh(self.wavenumber[tpind] 
                                                    * self.bathysysorig) * self.effwaveinertiacoef
                                                    * math.cos(self.omegawave[tpind] * tind - phase[tpind]))
                                            horzwaveloadsum[tind] = sum(horzwaveload[tind])
                                        horzwaveloadmaxindex, horzwaveloadmax = max(enumerate(horzwaveloadsum), key=operator.itemgetter(1))
                                        for s in range(0, self.numstrips):   
                                            horzwaveloadloc[s] = ((self.wetheight * (s 
                                                    / float(self.numstrips))) * horzwaveload[horzwaveloadmaxindex][s])
                                        self.eqhorzwaveload = horzwaveloadmax
                                    
                            elif systype in ("wavefloat", "tidefloat"):
                                for tind, t in enumerate(timevec):
                                    """ Diffraction theory approximation to horizontal 
                                        force McCormick and Cerquetti, 2004 """
                                    # qparamfunction = (1 - (self.wetheight / self.bathysysorig) 
                                                    # ** (10.0 / (self.wavenumber[tpind] * 0.5 
                                                    # * self.projlength)))
                                    if fexmodes[2] == 0:
                                        vertwaveload[tind] = (self._variables.seaden * self._variables.gravity 
                                                                * self.hmax[tpind].real * math.pi * (0.5 * self.projlength) ** 2.0
                                                                * (0.5 * self.projlength * math.sinh(self.wavenumber[tpind] 
                                                                * (self.bathysysorig - self.wetheight)) * modbesseli1[tpind]) 
                                                                / ((self.bathysysorig - self.wetheight) * math.cosh(self.wavenumber[tpind] 
                                                                * self.bathysysorig) * (self.wavenumber[tpind] * 0.5 * self.projlength) ** 2.0
                                                                * modbesseli0[tpind]))
                                    if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                        """ Diffraction theory approximation to horizontal 
                                            force MacCamy and Fuchs, 1954/ van Oortmessen 1971 """  
                                        horzwaveload[tind] = (2.0 * (self._variables.seaden * self._variables.gravity 
                                                                * self.hmax[tpind] / self.wavenumber[tpind] ** 2.0)
                                                                * math.tanh(self.wavenumber[tpind] * self.bathysysorig) 
                                                                * aka[tpind] * math.sin(self.omegawave[tpind] * tind - phase[tpind])
                                                                * ((math.sinh(self.wavenumber[tpind] * self.bathysysorig) 
                                                                - math.sinh(self.wavenumber[tpind] * (self.bathysysorig - self.wetheight))) 
                                                                / math.sinh(self.wavenumber[tpind] * self.bathysysorig)))
                                if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                    self.eqhorzwaveload = max(horzwaveload)
                                if fexmodes[2] == 0:
                                    self.eqvertwaveload = max(vertwaveload)
                            
                        elif sysprof == "rectangular":
                            """ F-K force coefficient approach from Chakrabarti 
                                (Table 4.7). Vertical acceleration at base of 
                                body for floating structures or top of 
                                truncated cylinder for fixed (non-surface 
                                piercing) structures """
                            if (systype in ("wavefixed", "tidefixed","substation") 
                                and sysheight 
                                < (self.bathysysorig 
                                + 0.5 * self.hmax[tpind])):
                                self.eqsysdraft = (self.bathysysorig - sysheight)
                                if fexmodes[2] == 0:
                                    self.eqvertwaveload = -(6.0 
                                        * self._variables.seaden 
                                        * sysvol * (math.sinh(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.eqsysdraft) * math.sin(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.projdepth)) * max(
                                        vertpartaccmax[-1]) / (
                                        self.wavenumber[tpind] * 0.5 
                                        * self.eqsysdraft * self.wavenumber[tpind] 
                                        * 0.5 * self.projdepth))
                                if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                    """ Note Table 4.7 in Chakrabarti states a load coefficient of 
                                        1.5, however this may be a typo """
                                    self.eqhorzwaveload = (6.0 
                                        * self._variables.seaden 
                                        * self._variables.sysvol * (math.sinh(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.eqsysdraft) * math.sin(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.projdepth)) * max(
                                        horzpartaccmax[-1]) / (
                                        self.wavenumber[tpind] * 0.5 
                                        * self.eqsysdraft * self.wavenumber[tpind] 
                                        * 0.5 * self.projdepth))                              
                            elif systype in ("wavefloat", "tidefloat"):   
                                if fexmodes[2] == 0:
                                    self.eqvertwaveload = (6.0 
                                        * self._variables.seaden 
                                        * sysvol * (math.sinh(
                                        self.wavenumber[tpind] * 0.5 
                                        * self._variables.sysdraft) * math.sin(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.projdepth)) * max(
                                        vertpartaccmax[-1]) 
                                        / (self.wavenumber[tpind] * 0.5 
                                        * self._variables.sysdraft 
                                        * self.wavenumber[tpind] 
                                        * 0.5 * self.projdepth))
                                if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                    self.eqhorzwaveload = (6.0 
                                        * self._variables.seaden 
                                        * sysvol * (math.sinh(
                                        self.wavenumber[tpind] * 0.5 
                                        * self._variables.sysdraft) * math.sin(
                                        self.wavenumber[tpind] * 0.5 
                                        * self.projdepth)) * max(
                                        horzpartaccmax[-1]) 
                                        / (self.wavenumber[tpind] * 0.5 
                                        * self._variables.sysdraft 
                                        * self.wavenumber[tpind] 
                                        * 0.5 * self.projdepth))
                        """ Default assumption that maximum wave loads are applied 
                            at crest height (surface piercing) or system height (subsea) """
                        if (sysheight >= self.bathysysorig + 0.5 
                            * self.hmax[tpind]):
                            self.eqhorzwaveloadloc = (self.bathysysorig + 0.5 
                                                        * self.hmax[tpind])
                        else:
                            self.eqhorzwaveloadloc = sysheight
                            
                        if (fexmodes[0] == 0 or fexmodes[1] == 0):   
                            self.syswaveload[tpind][0:2] = [self.eqhorzwaveload * math.sin(
                                self.waveangattk[tpind] * math.pi / 180.0), 
                                self.eqhorzwaveload * math.cos(self.waveangattk[tpind] 
                                * math.pi / 180.0)]
                        if fexmodes[2] == 0:
                            self.syswaveload[tpind][2] = self.eqvertwaveload                        
                    else:                
                        module_logger.info("Mass/viscous loads important, use Morison equation") 
                        self.diffractrig = 'False'
                        """ Horizontal load calculations use combined wave and current velocities, 
                            so steadycurrent contribution is removed from totsyssteadloads """                                       
                        
                        horzwaveloadsurge = [[0 for col in range(0,self.numstrips)] 
                                        for row in range(0, len(timevec))]
                        horzwaveloadsway = [[0 for col in range(0,self.numstrips)] 
                                        for row in range(0, len(timevec))] 
                        testvec = [0 for row in range(0,len(timevec))]
                        horzwaveloadsurgesum = [0 for row in range(0, len(timevec))] 
                        horzwaveloadswaysum = [0 for row in range(0, len(timevec))] 
                        horzwaveloadvec = [0 for row in range(0, len(timevec))]                         
                        vertwaveload = [[0 for col in range(0,self.numstrips)] 
                            for row in range(0,len(timevec))]       
                        vertwaveloadloc = [[0 for col in range(0,self.numstrips)] 
                            for row in range(0,len(timevec))]       
                        self.eqvertwaveload = [0 for col in range(0,len(timevec))]
                        self.eqhorzwaveload = [0 for col in range(0,len(timevec))]
                        horzwaveloadmax = [0 for col in range(0, self.numstrips)]
                        horzwaveloadloc = [0 for col in range(0, self.numstrips)]
                        
                        if (fexmodes[0] == 0 or fexmodes[1] == 0):
                            for s in range(0, self.numstrips):
                                """ Drag and added mass coefficient approximation """
                                """ Note: the influence of proximity to fixed 
                                    boundaries or the free surface, or indeed finite 
                                    length effects are not taken into account """
                                """ Kinematic viscosity at 10 degs (DNV-RP-C205) """
                                combhorzvelabs = [0.0 for row in range(0, len(timevec))]
                                for horzind, horzval in enumerate(combhorzvel[s]):
                                    combhorzvelabs[horzind] = math.fabs(horzval)
                                if sysprof == "cylindrical": 
                                    self.wavern = [math.fabs((syswidth 
                                                    * (max(combhorzvelabs) 
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)))/1.35e-6), 
                                                    math.fabs((syslength
                                                    * (max(combhorzvelabs) 
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)))/1.35e-6), 
                                                    math.fabs((syslength
                                                    * max(vertpartvelmax[s]))/1.35e-6)]
                                    self.wavekc = [math.fabs((max(combhorzvelabs) 
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)) * self._variables.tp[tpind] 
                                                    / syswidth), 
                                                    math.fabs((max(combhorzvelabs) 
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)) 
                                                    * self._variables.tp[tpind] 
                                                    / syslength)]
                                    """ Steady drag coefficient. Definition of large Re 
                                        from DNV-RP-C205. Mode 0 = surge, 
                                        mode 1 = sway """ 
                                    """ Non -dimensional surface roughness """
                                    ndsysrough = sysrough / syswidth                                
                                    for mode in range(0,2):
                                        if self.wavern[mode] > 1.0e6:
                                            """ For large Re and KC number flows 
                                                DNV-RPC205 """
                                            if ndsysrough < 1.0e-4:
                                                self.wavedragcoefsteady[mode] = 0.65
                                            elif (ndsysrough >= 1.0e-4 
                                                and ndsysrough < 1.0e-2):
                                                self.wavedragcoefsteady[mode] = ((29.0 
                                                    + 4.0 * math.log10(
                                                    ndsysrough)) / 20.0)                                                
                                            elif ndsysrough >= 1.0e-2:                                   
                                                self.wavedragcoefsteady[mode] = 1.05                                            
                                        elif self.wavern[mode] <= 1.0e6: 
                                            if (self.wavern[mode] 
                                                > self._variables.dragcoefcyl[-1,0]):
                                                self.wavedragcoefint = interpolate.interp1d(
                                                    self._variables.dragcoefcyl[0,1:len(
                                                    self._variables.dragcoefcyl[0,:])+1], 
                                                    self._variables.dragcoefcyl[-1,1:len(
                                                    self._variables.dragcoefcyl[0,:])+1],
                                                    bounds_error=False) 
                                                self.wavedragcoefsteady[mode] = self.wavedragcoefint(
                                                    ndsysrough)
                                                module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                                            elif (self.wavern[mode] 
                                                < self._variables.dragcoefcyl[1,0]):
                                                self.wavedragcoefint = interpolate.interp1d(
                                                    self._variables.dragcoefcyl[0,1:len(
                                                    self._variables.dragcoefcyl[0,:])+1], 
                                                    self._variables.dragcoefcyl[1,1:len(
                                                    self._variables.dragcoefcyl[0,:])+1],
                                                    bounds_error=False) 
                                                self.wavedragcoefsteady[mode] = self.wavedragcoefint(
                                                    ndsysrough)
                                                module_logger.warn('WARNING: Reynolds number out of range for drag coefficient data')
                                            else: 
                                                self.wavedragcoefsteady[mode] = self.dragcoefint(
                                                    self.wavern[mode], 
                                                    ndsysrough)
                                                self.wavedragcoefsteady[mode] = self.wavedragcoefsteady[mode][0]    
                                        
                                        """ Drag negligible when Hmax/principal dimension ratio is 
                                            less than 0.25 Chakrabarti, 1987 """
                                        if self.hmax[tpind]/self.projlength < 0.25:
                                            module_logger.warn('WARNING: Hmax / projected length < 0.25, drag negligible')
                                            self.wavedragcoefsteady[mode] = 0.0
                                            negdragflag[tpind] = 'True'
                                                                                        
                                        """ Added mass coefficient """  
                                        if self.wavekc[mode] <= 3.0:
                                            """ For low KC numbers, added mass 
                                                coefficient is equal to theoretical 
                                                value for smooth and rough 
                                                cylinders """
                                            self.waveaddmasscoef[mode] = 1.0
                                        elif self.wavekc[mode] > 3.0:  
                                            self.waveaddmasscoef[mode] = max(1.0 
                                                - 0.044 * (self.wavekc[mode] - 3.0), 
                                                0.6 - (self.wavedragcoefsteady[mode] 
                                                - 0.65))
                                    
                                elif sysprof == "rectangular": 
                                        self.wavekc = [math.fabs((max(combhorzvel[s]) 
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)) * self._variables.tp[tpind] 
                                                    / syswidth), 
                                                    math.fabs((max(combhorzvel[s]) 
                                                    * math.cos(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)
                                                    + max(perpcurrentvelstrp)
                                                    * math.sin(self.waveangattk[tpind] 
                                                    * math.pi / 180.0)) 
                                                    * self._variables.tp[tpind] 
                                                    / syslength)]                                        

                                        """ Steady drag coefficient """
                                        self.wavedragcoefint = interpolate.interp1d(
                                            self._variables.currentdragcoefrect[:,0], 
                                            self._variables.currentdragcoefrect[:,1],
                                            bounds_error=False)
                                        if (self.wlratio ** -1.0 <= self._variables.currentdragcoefrect[0,0]
                                            and self.wlratio ** -1.0 >= self._variables.currentdragcoefrect[-1,0]):                                        
                                                self.wavedragcoefsteady[0] = self.wavedragcoefint(
                                                                    self.wlratio ** -1.0)                                       
                                        elif self.wlratio ** -1.0 > self._variables.currentdragcoefrect[0,0]:
                                            self.wavedragcoefsteady[0] = self._variables.currentdragcoefrect[0,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                                        elif self.wlratio ** -1.0 < self._variables.currentdragcoefrect[-1,0]:
                                            self.wavedragcoefsteady[0] = self._variables.currentdragcoefrect[-1,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                                        if (self.wlratio <= self._variables.currentdragcoefrect[0,0]
                                            and self.wlratio >= self._variables.currentdragcoefrect[-1,0]):
                                                self.wavedragcoefsteady[1] = self.currentdragcoefint(
                                                                    self.wlratio)
                                        elif self.wlratio > self._variables.currentdragcoefrect[0,0]:
                                            self.wavedragcoefsteady[1] = self._variables.currentdragcoefrect[0,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                                        elif self.wlratio < self._variables.currentdragcoefrect[-1,0]:
                                            self.wavedragcoefsteady[1] = self._variables.currentdragcoefrect[-1,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for drag coefficient data')
                                            
                                        """ Drag negligible when Hmax/principal dimension ratio is 
                                            less than 0.25 Chakrabarti, 1987 """
                                        if self.hmax[tpind]/syswidth < 0.25:
                                            module_logger.warn('WARNING: Hmax / projected length < 0.25, drag negligible')
                                            self.wavedragcoefsteady[0] = 0.0   
                                            negdragflag[tpind] = 'True'
                                            
                                        self.wavedragcoef = self.wavedragcoefsteady
                                        
                                        """ Inertia coefficient """
                                        self.waveinertiacoefint = interpolate.interp1d(
                                            self._variables.waveinertiacoefrect[:,0], 
                                            self._variables.waveinertiacoefrect[:,1],
                                            bounds_error=False)
                                        if (self.wlratio ** -1.0 <= self._variables.waveinertiacoefrect[0,0]
                                            and self.wlratio ** -1.0 >= self._variables.waveinertiacoefrect[-1,0]):
                                                self.waveinertiacoef[0] = self.wavedragcoefint(
                                                                    self.wlratio ** -1.0)
                                        elif self.wlratio ** -1.0 > self._variables.waveinertiacoefrect[0,0]:
                                            self.waveinertiacoef[0] = 1.0
                                        elif self.wlratio ** -1.0 < self._variables.waveinertiacoefrect[-1,0]:
                                            self.waveinertiacoef[0] = self._variables.waveinertiacoefrect[-1,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for inertia coefficient data')
                                        if (self.wlratio <= self._variables.waveinertiacoefrect[0,0]
                                            and self.wlratio >= self._variables.waveinertiacoefrect[-1,0]):
                                                self.waveinertiacoef[1] = self.waveinertiacoefint(
                                                                    self.wlratio)
                                        elif self.wlratio > self._variables.waveinertiacoefrect[0,0]:
                                            self.waveinertiacoef[1] = 1.0
                                        elif self.wlratio < self._variables.waveinertiacoefrect[-1,0]:
                                            self.waveinertiacoef[1] = self._variables.waveinertiacoefrect[-1,1]
                                            module_logger.warn('WARNING: Width/length ratio out of range for inertia coefficient data')
                                        """ Mode 0 = surge, mode 1 = sway """
                                        for mode in range(0,2):                            
                                            """ Added mass coefficient """  
                                            if self.wavekc[mode] <= 3.0:
                                                """ For low KC numbers, added mass 
                                                    coefficient is equal to theoretical 
                                                    value for smooth and rough 
                                                    rectangular structures """
                                                self.waveaddmasscoef[mode] = 1.0 * self.waveinertiacoef[mode]
                                            elif self.wavekc[mode] > 3.0:  
                                                self.waveaddmasscoef[mode] = max(1.0 
                                                    - 0.044 * (self.wavekc[mode] 
                                                    - 3.0), 0.6 
                                                    - (self.wavedragcoefsteady[mode] 
                                                    - 0.65)) * self.waveinertiacoef[mode]
                                if sysprof == "cylindrical":
                                    self.waterlinearea = (math.pi / 4.0) * syswidth ** 2.0 
                                    """ Wake amplification factor """
                                    self.wakeampfactorint = interpolate.interp2d(
                                        self._variables.wakeampfactorcyl[0,1:],
                                        self._variables.wakeampfactorcyl[1:,0],
                                        self._variables.wakeampfactorcyl[1:,1:]) 
                                    """ Mode 0 = surge, mode 1 = sway """                                     
                                    for mode in range(0,2):
                                        if self.wavekc[mode] >= 12.0:  
                                            if self.wavedragcoefsteady[mode] > 0.0:
                                                self.wakeampfactorinterp = self.wakeampfactorint(
                                                    self.wavekc[mode] 
                                                    / self.wavedragcoefsteady[mode], 
                                                    self.wavedragcoefsteady[mode])
                                                self.wakeampfactor[mode] = self.wakeampfactorinterp[0]
                                            else:
                                                self.wakeampfactor[mode] = 0.0
                                        elif self.wavekc[mode] < 12.0:         
                                            if self.wavedragcoefsteady[mode] > 0.0:
                                                self.wakeampfactorpi[mode] = (1.5 - 0.024 
                                                    * (12.0 / self.wavedragcoefsteady[mode] 
                                                    - 10.0))
                                            else:
                                                self.wakeampfactorpi[mode] = 1.74
                                            if self.wavekc[mode] < 0.75:                                
                                                self.wakeampfactor[mode] = (
                                                    self.wakeampfactorpi[mode] - 1.0 - 2.0 
                                                    * (self.wavekc[mode] - 0.75))
                                            elif (self.wavekc[mode] >= 0.75 
                                                and self.wavekc[mode] < 2.0):
                                                self.wakeampfactor[mode] = (
                                                self.wakeampfactorpi[mode] - 1.0)
                                            elif self.wavekc[mode] < 12.0:
                                                self.wakeampfactor[mode] = (
                                                self.wakeampfactorpi[mode] + 0.1 
                                                * (self.wavekc[mode] - 12.0))
                                        self.wavedragcoef[mode] = (
                                            self.wavedragcoefsteady[mode] 
                                            * self.wakeampfactor[mode])
                                        """ Drag negligible when Hmax/principal dimension ratio is 
                                            less than 0.25 Chakrabarti, 1987 """
                                        if self.hmax[tpind]/self.projlength < 0.25:
                                            module_logger.warn('WARNING: Hmax / projected length < 0.25, drag negligible')
                                            self.wavedragcoefsteady[mode] = 0.0
                                            negdragflag[tpind] = 'True'    
                                    for tind, t in enumerate(timevec):
                                        """ Horizontal wave load includes drag, 
                                            hydrodynamic mass and Froude Krylov force """
                                        horzwaveloadsurge[tind][s] = (self._variables.seaden 
                                            * (1.0 + self.waveaddmasscoef[0])
                                            * self.waterlinearea * self.deltahwet
                                            * horzpartaccmax[s][tind] 
                                            * math.sin(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + 0.5 
                                            * self._variables.seaden 
                                            * self.wavedragcoef[0] 
                                            * syswidth * self.deltahwet
                                            * math.fabs(combhorzvel[s][tind] 
                                            * math.sin(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + max(perpcurrentvelstrp)
                                            * math.cos(self.waveangattk[tpind] 
                                            * math.pi / 180.0)) 
                                            * (combhorzvel[s][tind] 
                                            * math.sin(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + max(perpcurrentvelstrp)
                                            * math.cos(self.waveangattk[tpind] 
                                            * math.pi / 180.0)))
                                        horzwaveloadsway[tind][s] = (self._variables.seaden 
                                            * (1.0 + self.waveaddmasscoef[1])
                                            * self.waterlinearea * self.deltahwet
                                            * horzpartaccmax[s][tind] 
                                            * math.cos(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + 0.5 
                                            * self._variables.seaden 
                                            * self.wavedragcoef[1] 
                                            * syslength * self.deltahwet
                                            * math.fabs(combhorzvel[s][tind] 
                                            * math.cos(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + max(perpcurrentvelstrp) 
                                            * math.sin(self.waveangattk[tpind] 
                                            * math.pi / 180.0))
                                            * (combhorzvel[s][tind] 
                                            * math.cos(self.waveangattk[tpind] 
                                            * math.pi / 180.0)
                                            + max(perpcurrentvelstrp) 
                                            * math.sin(self.waveangattk[tpind] 
                                            * math.pi / 180.0)))
                                elif sysprof == "rectangular":
                                    """ Approach adapted from Venugopal, V. (2002) """
                                    self.waterlinearea = syswidth * syslength
                                    for tind, t in enumerate(timevec): 
                                        horzwaveloadsurge[tind][s] = (syslength * (self._variables.seaden * (1.0 
                                                                    + self.waveaddmasscoef[0]) * self.deltahwet * syswidth
                                                                    * (horzpartaccmax[s][tind] * math.sin(self.waveangattk[tpind] * math.pi / 180.0)) 
                                                                    + 0.5 * self._variables.seaden 
                                                                    * self.wavedragcoef[0]
                                                                    * self.deltahwet
                                                                    * (combhorzvel[s][tind] * math.sin(self.waveangattk[tpind] * math.pi / 180.0)
                                                                    + max(perpcurrentvelstrp) 
                                                                    * math.cos(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0))
                                                                    * math.sqrt(math.fabs(combhorzvel[s][tind] 
                                                                    * math.sin(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0)
                                                                    + max(perpcurrentvelstrp) 
                                                                    * math.cos(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0)) ** 2.0
                                                                    + vertpartvelmax[s][tind] ** 2.0)))
                                        horzwaveloadsway[tind][s] = (syswidth * (self._variables.seaden * (1.0 
                                                                    + self.waveaddmasscoef[1]) * self.deltahwet * syslength
                                                                    * (horzpartaccmax[s][tind] * math.cos(self.waveangattk[tpind] * math.pi / 180.0)) 
                                                                    + 0.5 * self._variables.seaden 
                                                                    * self.wavedragcoef[1]
                                                                    * self.deltahwet
                                                                    * (combhorzvel[s][tind] * math.cos(self.waveangattk[tpind] * math.pi / 180.0)
                                                                    + max(perpcurrentvelstrp) 
                                                                    * math.sin(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0))
                                                                    * math.sqrt(math.fabs(math.fabs(combhorzvel[s][tind] 
                                                                    * math.cos(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0)
                                                                    + max(perpcurrentvelstrp) 
                                                                    * math.sin(self.waveangattk[tpind] 
                                                                    * math.pi / 180.0))) ** 2.0
                                                                    + vertpartvelmax[s][tind] ** 2.0)))                                   
                            for tind, t in enumerate(timevec):    
                                horzwaveloadsurgesum[tind] = sum(horzwaveloadsurge[tind])
                                horzwaveloadswaysum[tind] = sum(horzwaveloadsway[tind])
                                horzwaveloadvec[tind] = math.sqrt(horzwaveloadsurgesum[tind] ** 2.0 + horzwaveloadswaysum[tind] ** 2.0)
                            horzwaveloadmaxindex, horzwaveloadmax = max(enumerate(horzwaveloadvec), key=operator.itemgetter(1))
                            horzwaveloadminindex, horzwaveloadmin = min(enumerate(horzwaveloadvec), key=operator.itemgetter(1))
                            
                            
                            if math.fabs(horzwaveloadmin) > math.fabs(horzwaveloadmax):
                                horzwaveloadmax = horzwaveloadmin
                                horzwaveloadmaxindex = horzwaveloadminindex
                            for s in range(0, self.numstrips):   
                                horzwaveloadloc[s] = ((self.wetheight * (s 
                                        / float(self.numstrips))) * math.sqrt(horzwaveloadsurge[horzwaveloadmaxindex][s] ** 2.0 + horzwaveloadsway[horzwaveloadmaxindex][s] ** 2.0))   
                                        
                            self.eqhorzwaveload = horzwaveloadmax           
                            
                            if math.fabs(self.eqhorzwaveload) > 0.0:
                                self.eqhorzwaveloadloc = sum(horzwaveloadloc) / self.eqhorzwaveload 
                            else: 
                                self.eqhorzwaveloadloc = 0.5 * sysheight
                        if fexmodes[2] == 0:
                            if (systype in ("wavefixed", "tidefixed","substation") 
                                and sysheight > (self.bathysysorig 
                                + 0.5 * self.hmax[tpind])):
                                """ Vertical wave load zero on seafloor fixed - surface 
                                    piercing structures """
                                self.eqvertwaveload = 0.0
                            else:   
                                if sysprof == "cylindrical":
                                    self.waterlinearea = ((math.pi / 4.0) 
                                                    * syswidth ** 2.0)
                                    if (systype in ("wavefixed", "tidefixed","substation") 
                                        and sysheight 
                                        < (self.bathysysorig + 0.5 
                                        * self.hmax[tpind])):
                                        for tind, t in enumerate(timevec):
                                            """ Small body approximation to vertical 
                                                Froude-Krylov force acting on top of 
                                                fixed cylinder """
                                            self.eqvertwaveload[tind] = (
                                                self.waterlinearea 
                                                * (self._variables.seaden 
                                                * self._variables.gravity * 0.5 
                                                * self.hmax[tpind] * math.exp(
                                                -self.wavenumber[tpind] 
                                                * sysheight)))
                                    elif systype in ("wavefloat", "tidefloat"):
                                        for tind, t in enumerate(timevec):
                                            """ Small body approximation to vertical 
                                                Froude-Krylov force acting on base of 
                                                floating cylinder """
                                            self.eqvertwaveload[tind] = (
                                                self.waterlinearea 
                                                * (self._variables.seaden 
                                                * self._variables.gravity * 0.5 
                                                * self.hmax[tpind] * math.exp(
                                                -self.wavenumber[tpind] 
                                                * self._variables.sysdraft)))
                                elif sysprof == "rectangular":                            
                                    """ F-K force coefficient approach from Chakrabarti 
                                        (Table 4.7). Vertical acceleration at base of 
                                        body for floating structures or top of 
                                        truncated cylinder for fixed (non-surface 
                                        piercing) structures """
                                    if (systype in ("wavefixed", "tidefixed","substation") 
                                        and sysheight < (
                                        self.bathysysorig + 0.5 * self.hmax[tpind])):
                                        self.eqsysdraft = (self.bathysysorig + 0.5 
                                            * self.hmax[tpind] 
                                            - sysheight)
                                        for tind, t in enumerate(timevec):
                                            self.eqvertwaveload[tind] = (-6.0 
                                                * self._variables.seaden 
                                                * sysvol 
                                                * (math.sinh(self.wavenumber[tpind] 
                                                * 0.5 * self.eqsysdraft * math.sin(
                                                self.wavenumber[tpind] * 0.5 
                                                * self.projdepth)) 
                                                * vertpartaccmax[-1][tind] 
                                                / (self.wavenumber[tpind] * 0.5 
                                                * self.eqsysdraft 
                                                * self.wavenumber[tpind] * 0.5 
                                                * self.projdepth)))                                
                                    elif systype in ("wavefloat", "tidefloat"):
                                        for tind, t in enumerate(timevec):
                                            self.eqvertwaveload[tind] = (6.0 
                                                * self._variables.seaden 
                                                * sysvol 
                                                * (math.sinh(self.wavenumber[tpind] 
                                                * 0.5 * self._variables.sysdraft) 
                                                * math.sin(
                                                self.wavenumber[tpind] * 0.5 
                                                * self.projdepth)) 
                                                * vertpartaccmax[0][tind] 
                                                / (self.wavenumber[tpind] * 0.5 
                                                * self._variables.sysdraft 
                                                * self.wavenumber[tpind] * 0.5 
                                                * self.projdepth))   
                                                
                        if type(self.eqvertwaveload) is list:
                            if (fexmodes[0] == 0 or fexmodes[1] == 0):                                
                                self.syswaveload[tpind][0:2] = [horzwaveloadsurgesum[horzwaveloadmaxindex], 
                                                                horzwaveloadswaysum[horzwaveloadmaxindex]]
                            if fexmodes[2] == 0:            
                                self.syswaveload[tpind][2] = max(self.eqvertwaveload)
                        else: 
                            if (fexmodes[0] == 0 or fexmodes[1] == 0):
                                self.syswaveload[tpind][0:2] = [horzwaveloadsurgesum[horzwaveloadmaxindex], 
                                                                horzwaveloadswaysum[horzwaveloadmaxindex]]
                            if fexmodes[2] == 0:            
                                self.syswaveload[tpind][2] = self.eqvertwaveload       
                        if negdragflag[tpind] == 'False':
                            self.totsyssteadloads[tpind][0] = self.totsyssteadloads[tpind][0] - self.steadycurrent[0]
                            self.totsyssteadloads[tpind][1] = self.totsyssteadloads[tpind][1] - self.steadycurrent[1]
                            self.totsyssteadloads[tpind][2] = self.totsyssteadloads[tpind][2] - self.steadycurrent[2]   
                    """ Find maximum resultant wave load due to wave conditions 
                        along upper contour """
                    self.ressyswaveload[tpind] = math.sqrt(
                        self.syswaveload[tpind][0] ** 2.0 
                        + self.syswaveload[tpind][1] ** 2.0 
                        + self.syswaveload[tpind][2] ** 2.0)
                     
                    # module_logger.warn('fexmodes {}'.format(fexmodes))
            """ Sort wave loads by ascending magnitude """
            self.ressyswaveloadsortind = [i[0] for i in sorted(enumerate(self.ressyswaveload), key=lambda x:x[1])]
            """ Reverse list (descending magnitude) """
            self.ressyswaveloadsortind = self.ressyswaveloadsortind[::-1]
            for wcind, wc in enumerate(self.ressyswaveloadsortind):
                self.syswaveloadsort[wcind] = self.syswaveload[wc]
                self.meanwavedriftsort[wcind] = self.meanwavedrift[wc]  
            
            self.syswaveloadmaxind = self.ressyswaveload.index(max(
                                    self.ressyswaveload))
            self.syswaveloadmax = self.syswaveload[self.syswaveloadmaxind] 
            # module_logger.warn('self.syswaveloadsort[wcind] {}'.format(self.syswaveloadsort[wcind]))
    
    def _set_fairloc(self, quanfound=None):
        
        if quanfound is not None:
            assert len(self._variables.fairloc) == quanfound
        
        if not self.foundradnew: self.fairloc = self._variables.fairloc
        self.fairlocglob = self.fairloc[:]
            
        return

    def _get_foundation_quantity(self, systype, foundloc):
        
        if not systype in ('wavefloat',
                           'tidefloat',
                           'wavefixed',
                           'tidefixed',
                           'substation'):
            
            errStr = "System type '{}' not ".format(systype)
            raise ValueError(errStr)
        
        if self.foundradnew and systype in ('wavefloat', 'tidefloat'):
            quanfound = self.numlines                 
        else:
            quanfound = len(foundloc)
            
        return quanfound
    
    def _get_minimum_distance(self, sysorig):
        
        # Determine minimum distance between neighbouring devices
        devdist = []
        
        # Check if more than one device
        if len(self._variables.sysorig) > 1:
            
            for dev in self._variables.sysorig:     
                
                local_orig = self._variables.sysorig[dev]
                x_dist = local_orig[0] - sysorig[0]
                y_dist = local_orig[1] - sysorig[1]
                
                dist =  math.sqrt(x_dist ** 2 + y_dist ** 2)    
                devdist.append(dist)
            
            # First sorted entry will be zero as distance to self
            devlistsort = sorted(devdist)
            mindevdist = devlistsort[1]
            
        else:
            
            mindevdist = 10000.0
            
        return mindevdist
    
    def _get_maximum_displacement(self):
            
        if self._variables.maxdisp is not None:
            maxdisp = self._variables.maxdisp[:]
            maxdisp[2] += self._variables.sysdraft
        else:
            # Set very high displacement limits if not specified by the user
            maxdisp = [1000.0, 1000.0, 1000.0]
            
        return maxdisp
    
    def _get_foundation_locations(self, lineangs,
                                        quanfound=None,
                                        depth_multiplier=8.0):
            
        if (self._variables.foundloc is not None and
            len(self._variables.foundloc) > 0): return self._variables.foundloc
              
        if quanfound is not None: assert len(lineangs) == quanfound
                                            
        loc_list = []
            
        if self._variables.prefootrad:
            
            for angle in lineangs:
                
                x = self._variables.prefootrad * math.sin(angle)
                y = self._variables.prefootrad * math.cos(angle)
                z = 0.0
                
                loc_list.append([x, y, z])
            
        else:
            
            # Set initial anchor radius based on mean water depth. Note the 
            # suitability of this factor as an initial guess needs to be 
            # determined
        
            sum_depth = sum(self._variables.bathygrid[:,2])
            n_bathy = self._variables.bathygrid.shape[0]
            meanwatdep = math.fabs(sum_depth / n_bathy) 

            for angle in lineangs:
                
                x = meanwatdep * depth_multiplier * math.sin(angle)
                y = meanwatdep * depth_multiplier * math.cos(angle)
                z = 0.0
                
                loc_list.append([x, y, z])
                                                
        foundloc = np.array(loc_list)
        
        return foundloc
    
    def _get_closest_grid_point(self, sysorig,
                                      local_point=None):
        
        """ Find nearest grid point(s) to given point. Transformation
        required from local (foundloc) to global coordinates.
        """ 
        
        # Bathymetry grid tolerances
        bathygridxtol = 0.5 * self._variables.bathygriddeltax
        bathygridytol = 0.5 * self._variables.bathygriddeltay

        griddist = []
        xglobal = sysorig[0]
        yglobal = sysorig[1]
        
        if local_point is not None:
            xglobal += local_point[0]
            yglobal += local_point[1]
        
        for i, grid_point in enumerate(self._variables.bathygrid):
                        
            xdist = xglobal - grid_point[0]
            ydist = yglobal - grid_point[1]
            point_dist = math.sqrt(xdist ** 2 + ydist ** 2)
            
            # Don't include points further than the grid tolerence
            if (math.fabs(xdist) > bathygridxtol or
                math.fabs(ydist) > bathygridytol): continue
            
            point_dist_list = [i] + list(grid_point) + [point_dist]
            griddist.append(point_dist_list)
            
        # Error if foundation location is outside of supplied grid.
        if not griddist:
            errStr = ("No suitable grid points found for placing foundation "
                      "at ({}, {})").format(xglobal, yglobal)
            raise RuntimeError(errStr)
        
        griddist = np.array(griddist)
        sorted_points = griddist[np.argsort(griddist[:, 4])]
        closest_point = sorted_points[0]
        
        return int(closest_point[0]), closest_point[1:4]
    
    def _get_neighbours(self, sysorig,
                              closest_point,
                              local_point=None):
        
        """ If the foundation/system location happens to coincide with a grid 
        line or grid point the nearest grid points at one grid spacing in
        either direction are used """
        
        # Bathymetry grid tolerances
        bathygridxtol = 0.5 * self._variables.bathygriddeltax
        bathygridytol = 0.5 * self._variables.bathygriddeltay

        xglobal = sysorig[0]
        yglobal = sysorig[1]
        
        if local_point is not None:
            xglobal += local_point[0]
            yglobal += local_point[1]
                
        # Check for coincidence with grid
        if (np.isclose(xglobal, closest_point[0]) or
            np.isclose(yglobal, closest_point[1])):
                
            module_logger.info("Foundation point coincides with grid line "
                               "or grid point")
            
            # Get diagonals from closest point
            x0 = closest_point[0] - self._variables.bathygriddeltax
            x1 = closest_point[0] + self._variables.bathygriddeltax

            y0 = closest_point[1] - self._variables.bathygriddeltay
            y1 = closest_point[1] + self._variables.bathygriddeltay
          
        else:
            
            # Surrounding x points
            if xglobal > closest_point[0]:
                x0 = closest_point[0]
                x1 = closest_point[0] + self._variables.bathygriddeltax
            else:
                x0 = closest_point[0] - self._variables.bathygriddeltax
                x1 = closest_point[0]
              
            # Surrounding y points
            if yglobal > closest_point[1]:
                y0 = closest_point[0]
                y1 = closest_point[0] + self._variables.bathygriddeltay
            else:
                y0 = closest_point[0] - self._variables.bathygriddeltay
                y1 = closest_point[0]
                
        gpsearch = [[x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1]]
        
        # Collect indices of neighbours in grid
        gpnearinds = []
        
        for search_point in gpsearch:
            
            find_inds = []
            
            for idx, grid_point in enumerate(self._variables.bathygrid):
                
                x = grid_point[0] - search_point[0]
                y = grid_point[1] - search_point[1]
            
                if (math.fabs(x) < bathygridxtol and
                    math.fabs(y) < bathygridytol):
                    
                    find_inds.append(idx)
                                                    
            if not find_inds:

                errStr = ("Could not find suitable grid points near "
                          "({}, {})").format(search_point[0],
                                             search_point[1])
                raise RuntimeError(errStr)
                
            elif len(find_inds) > 1:
                
                errStr = ("Multiple grid points found near "
                          "({}, {})").format(search_point[0],
                                             search_point[1])
                raise RuntimeError(errStr)
                                                    
            gpnearinds.append(find_inds[0])
            
        # Collect points
        gpnear = [self._variables.bathygrid[idx, :] for idx in gpnearinds]

        return gpnearinds, gpnear

    def _get_depth(self, point,
                         gpnearinds):
        
        """Determine bathymetry depth at the given point.
        """
        
        locbathyvalues = np.array([self._variables.bathygrid[gpnearinds, 0],
                                   self._variables.bathygrid[gpnearinds, 1],
                                   self._variables.bathygrid[gpnearinds, 2]]).T
        
        bathyint = interpolate.interp2d(locbathyvalues[:, 0],
                                        locbathyvalues[:, 1],
                                        locbathyvalues[:, 2],
                                        kind='linear')
        
        depth = bathyint(point[0], point[1]) - self._variables.wlevmax
                        
        return depth

    def _get_soil_type_depth(self, point_idx):
        
        """Determine soil type and depth at a given grid point index.
        """
                
        # Check that soil table is correctly formatted
        list_depth = [x[-1] for x in self._variables.soiltypgrid]
        
        if not np.isinf(list_depth).all():
            errStr = "Final sediment layer must have infinite depth."
            raise ValueError(errStr)
        
        soil_point = self._variables.soiltypgrid[point_idx]
        
        # Check for multiple layers
        if len(soil_point) > 4:
            soiltype, soildepth = self._get_soil_type_depth_multi(soil_point)
        else:
            soiltype = soil_point[-2]
            soildepth = np.inf
            
        return soiltype, soildepth
    
    @classmethod
    def _get_soil_type_depth_multi(cls, soil_point):
        
        soillayerdep = []
        totsoillayerdep = 0.0
        
        soft_sediments = ('ls', 'ms', 'ds', 'vsc', 'sc', 'fc', 'stc')
        
        # Iterate through layers pairwise ignoring last layer
        layer_iter = iter(soil_point[2:-2])
        layer_pairs = izip(layer_iter, layer_iter)
        
        # Determine significance of top layers above bedrock
        for soil_type, layer_depth in layer_pairs:
            
            if soil_type not in soft_sediments: continue

            soillayerdep.append((soil_type, layer_depth))
            totsoillayerdep = totsoillayerdep + layer_depth
            
        if totsoillayerdep <= 6.0:
            
            # Soil layer(s) covering bedrock classed as a skim (i.e. less than
            # 6.0m deep), bedrock is used for subsequent foundation
            # calculations
            soiltype = soil_point[-2]
            soildepth = np.inf
            
        elif totsoillayerdep > 6.0:
            
            # For deep sediments over bedrock assume soil is homogeneous and
            # based on the soil type with the deepest layer
            layermax = max(soillayerdep, key=operator.itemgetter(1))
            
            soiltype = layermax[0]
            soildepth = totsoillayerdep
            
        return soiltype, soildepth
    
    @classmethod
    def _get_soil_group(cls, soiltype):
        
        if soiltype in ('ls', 'ms', 'ds'): 
            soilgroup = 'cohesionless'
        elif soiltype in ('vsc', 'sc', 'fc', 'stc'): 
            soilgroup = 'cohesive'
        elif soiltype in ('hgt', 'cm', 'src', 'hr', 'gc'): 
            soilgroup = 'other'
        else:
            errStr = "Soil type '{}' is not recognised".format(soiltype)
            raise ValueError(errStr)
            
        return soilgroup
