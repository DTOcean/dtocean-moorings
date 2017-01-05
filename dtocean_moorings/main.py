"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter
"""
# Start logging
import logging
module_logger = logging.getLogger(__name__)

# Built in modulesrad
import os
import csv
import copy
import pandas as pd
from .core import Found, Moor, Subst
from scipy import interpolate
    
this_dir = os.path.dirname(os.path.realpath(__file__))


class Variables(object):
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Variables class
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    Input of variables into this class 

    Functions:        

    Args:        
        devices (list) [-]: list of device identification numbers
        gravity (float) [m/s2]: acceleration due to gravity
        seaden (float) [kg/m3]: sea water density
        airden (float) [kg/m3]: air density
        steelden (float) [kg/m3]: steel density
        conden (float) [kg/m3]: concrete density
        groutden (float) [kg/m3]: grout density
        compdict (dict) [various]: component dictionary
        soiltypgrid (numpy.ndarray): soil type grid: X coordinate (float) [m], 
                                    Y coordinate (float) [m], 
                                    soil type options (str) [-]: 'ls': loose sand,
                                                            'ms': medium sand,
                                                            'ds': dense sand,
                                                            'vsc': very soft clay,
                                                            'sc': soft clay,
                                                            'fc': firm clay,
                                                            'stc': stiff clay,
                                                            'hgt': hard glacial till,
                                                            'cm': cemented,
                                                            'src': soft rock coral,
                                                            'hr': hard rock,
                                                            'gc': gravel cobble,
                                    depth (float) [m]    
        seaflrfriccoef (float) [-]: optional soil friction coefficient
        bathygrid (numpy.ndarray): bathymetry grid: X coordinate (float) [m], 
                                                    Y coordinate (float) [m], 
                                                    Z coordinate (float) [m]
        bathygriddeltax (float) [m]: bathymetry grid X axis grid spacing
        bathygriddeltay (float) [m]: bathymetry grid Y axis grid spacing
        wlevmax (float) [m]: maximum water level above mean sea level
                             (50 year return period)
        wlevmin (float) [m]: minimum water level below mean sea level
                             (50 year return period)
        currentvel (float) [m/s]: maximum current velocity magnitude
                                  (10 year return period)
        currentdir (float) [deg]: current direction at maximum velocity 
        currentprof (str) [-]: current profile options: 'uniform', 
                                                        '1/7 power law'
        wavedir (float) [deg]: predominant wave direction(s)
        hs (list) [m]: maximum significant wave height (100 year return period)
        tp (list) [s]: maximum peak period (100 year return period)
        gamma (float) [-]: jonswap shape parameter(s)
        windvel (float) [m/s]: mean wind velocity magnitude
                               (100 year return period)
        winddir (float) [deg]: predominant wind direction 
        windgustvel (float) [m/s]: wind gust velocity magnitude
                                   (100 year return period)
        windgustdir (float) [deg]: wind gust direction
        soilprops (pandas) [-]: default soil properties table:  drained soil friction angle [deg],
                                                                relative soil density [%],
                                                                buoyant unit weight of soil [N/m^3],
                                                                undrained soil shear strengths [N/m^2, N/m^2] or rock shear strengths [N/m^2],
                                                                effective drained cohesion [N/m^2],
                                                                Seafloor friction coefficient [-],
                                                                Soil sensitivity [-],
                                                                rock compressive strength [N/m^2]        
        linebcf (numpy.ndarray): buried mooring line bearing capacity factor:   soil friction angle [deg],
                                                                                bearing capacity factor[-]
        k1coef (numpy.ndarray): coefficient of subgrade reaction (cohesive soils):  allowable deflection/diameter[-],
                                                                                    soft clay coefficient [-],
                                                                                    stiff clay coefficient [-]
        soilsen (float) [-]: soil sensitivity
        subgradereaccoef (numpy.ndarray): coefficient of subgrade reaction (cohesionless soils): allowable deflection/diameter[-],
                                                                                                35% relative density coefficient [-],
                                                                                                50% relative density coefficient [-],
                                                                                                65% relative density coefficient [-],
                                                                                                85% relative density coefficient [-]
                                           Note that the first row of the array contains the percentiles themselves using 0.0 as the
                                           value for the first column
        piledefcoef (numpy.ndarray): pile deflection coefficients:  depth coefficient[-],
                                                                    coefficient ay [-],
                                                                    coefficient by [-]
                                                                    
        pilemomcoefam (numpy.ndarray): pile moment coefficient am:  depth coefficient[-],
                                                                    pile length/relative soil-pile stiffness = 10 [-],
                                                                    pile length/relative soil-pile stiffness = 5 [-],	
                                                                    pile length/relative soil-pile stiffness = 4 [-],	
                                                                    pile length/relative soil-pile stiffness = 3 [-],	
                                                                    pile length/relative soil-pile stiffness = 2 [-]
                                       Note that the first row of the array contains the stiffnesses themselves using 
                                       0.0 as the value for the first column
        pilemomcoefbm (numpy.ndarray): pile moment coefficient bm:  depth coefficient[-],
                                                                    pile length/relative soil-pile stiffness = 10 [-],
                                                                    pile length/relative soil-pile stiffness = 5 [-],	
                                                                    pile length/relative soil-pile stiffness = 4 [-],	
                                                                    pile length/relative soil-pile stiffness = 3 [-],	
                                                                    pile length/relative soil-pile stiffness = 2 [-]
                                       Note that the first row of the array contains the stiffnesses themselves using 
                                       0.0 as the value for the first column
        pilefricresnoncal (numpy.ndarray): pile skin friction and end bearing capacity: soil friction angle [deg],
                                                                                        friction angle sand-pile [deg],	
                                                                                        max bearing capacity factor [-],
                                                                                        max unit skin friction [N/m2],
                                                                                        max end bearing capacity [N/m2]
        hcfdrsoil (numpy.ndarray): holding capacity factor for drained soil condition: relative embedment depth [-],
                                                                                        drained friction angle = 20deg [-],
                                                                                        drained friction angle = 25deg [-],	
                                                                                        drained friction angle = 30deg [-],	
                                                                                        drained friction angle = 35deg [-],	
                                                                                        drained friction angle = 40deg [-]
                                   Note that the first row of the array contains the drained friction angle using 
                                   0.0 as the value for the first column        
        systype (str) [-]: system type: options:    'tidefloat', 
                                                    'tidefixed', 
                                                    'wavefloat', 
                                                    'wavefixed'                            
        depvar (bool) [-]: depth variation permitted: options:   True,
                                                                False                                                            
        sysprof (str) [-]: system profile: options:    'cylindrical', 
                                                        'rectangular'
        sysmass (float) [kg]: system mass
        syscog (numpy.ndarray): system centre of gravity in local coordinates:  X coordinate (float) [m], 
                                                                                Y coordinate (float) [m], 
                                                                                Z coordinate (float) [m]
        sysvol (float) [m3]: system submerged volume
        sysheight (float) [m]: system height
        syswidth (float) [m]: system width
        syslength (float) [m]: system length        
        sysrough (float) [m]: system roughness
        sysorig (dict): system origin (UTM):   {deviceid: (X coordinate (float) [m], 
                                                           Y coordinate (float) [m], 
                                                           Z coordinate (float) [m])}
        fairloc (numpy.ndarray): fairlead locations in local coordinates for N lines:   X coordinate (float) [m], 
                                                                                        Y coordinate (float) [m], 
                                                                                        Z coordinate (float) [m]
        foundloc (numpy.ndarray): foundation locations in local coordinates for N foundations:      X coordinate (float) [m], 
                                                                                                    Y coordinate (float) [m], 
                                                                                                    Z coordinate (float) [m]
        umbconpt (numpy.ndarray): umbilical connection point:   X coordinate (float) [m], 
                                                                Y coordinate (float) [m], 
                                                                Z coordinate (float) [m] 
        sysdryfa (float) [m2]: system dry frontal area
        sysdryba (float) [m2]: system dry beam area
        dragcoefcyl (numpy.ndarray): cylinder drag coefficients:    reynolds number [-],
                                                                    smooth coefficient [-],	
                                                                    roughness = 1e-5	coefficient [-],
                                                                    roughness = 1e-2 coefficient [-]
        wakeampfactorcyl (numpy.ndarray): cylinder wake amplificiation factors: kc/steady drag coefficient [-],
                                                                                smooth cylinder amplification factor [-],
                                                                                rough cylinder amplification factor [-]
        winddragcoefrect (numpy.ndarray): rectangular wind drag coefficients:   width/length [-],
                                                                                0<height/breadth<1 [-],
                                                                                height/breadth = 1 [-],
                                                                                height/breadth = 2 [-],
                                                                                height/breadth = 4	 [-],
                                                                                height/breadth = 6	 [-],
                                                                                height/breadth = 10 [-],
                                                                                height/breadth = 20 [-]
        currentdragcoefrect (numpy.ndarray): rectangular current drag coefficients: width/length [-],
                                                                                    thickness/width = 0 [-]
        driftcoeffloatrect (numpy.ndarray): rectangular drift coefficients: wavenumber*draft [m],
                                                                            reflection coefficient [-]        
        Clen (tuple): rotor parameters: rotor diameter [m],
                                        distance from centreline [m]
        thrustcurv (numpy.ndarray): thrust curve:   inflow velocity magnitude [m/s],
                                                    thrust coefficient [-]
        hubheight (float) [m]: rotor hub height 
        sysorienang (float) [deg]: system orientation angle
        fex (numpy.ndarray): first-order wave excitation forces: analysed frequencies (list) [Hz],
                                                                 complex force amplitudes (nxm list) for n directions and m degrees of freedom                                                                 
        premoor (str) [-]: predefined mooring system type: options:     'catenary', 
                                                                        'taut' 
        maxdisp (numpy.ndarray): optional maximum device displacements:  surge (float) [m], 
                                                                         sway (float) [m], 
                                                                         heave (float) [m]
        prefound (str) [-]: predefined foundation type: options:    'shallowfoundation', 
                                                                    'gravity',
                                                                    'pile',
                                                                    'suctioncaisson',
                                                                    'directembedment',
                                                                    'drag'                                                                    
        coststeel (float) [euros/kg]: cost of steel
        costgrout (float) [euros/kg]: cost of grout
        costcon (float) [euros/kg]: cost of concrete
        groutstr (float) [N/mm2]: grout compressive strength
        preumb (str) [-]: predefined umbilical type
        umbsf (float) [-]: umbilical safety factor
        foundsf (float) [-]: foundation safety factor
        prefootrad (float) [m]: predefined foundation radius
        subcabconpt (dict) [-]: subsea cable connection point for each device:     X coordinate (float) [m], 
                                                                                        Y coordinate (float) [m], 
                                                                                        Z coordinate (float) [m] 
        presubstfound (str) [-]: predefined foundation type: options:   'gravity',
                                                                        'pile'                                                                         
        suborig (numpy.ndarray): substation origin(s) (UTM): 'array'   X coordinate (float) [m], 
                                                                    Y coordinate (float) [m], 
                                                                    Z coordinate (float) [m]
                                                             'subhubXXX'   X coordinate (float) [m], 
                                                                           Y coordinate (float) [m], 
                                                                           Z coordinate (float) [m]
        submass (float) [kg]: substation mass
        subvol (float) [m3]: substation submerged volume
        subcog (numpy.ndarray): substation centre of gravity in local coordinates:  X coordinate (float) [m], 
                                                                                Y coordinate (float) [m], 
                                                                                Z coordinate (float) [m]
        subwetfa (float) [m]: substation wet frontal area
        subdryfa (float) [m]: substation dry frontal area
        subwetba (float) [m]: substation wet beam area
        subdryba (float) [m]: substation dry beam area
        substloc (numpy.ndarray): substation foundation locations in local coordinates for N foundations:   X coordinate (float) [m], 
                                                                                                            Y coordinate (float) [m], 
                                                                                                            Z coordinate (float) [m]    
        moorsfuls (float) [-]: mooring ultimate limit state safety factor
        moorsfals (float) [-]: mooring accident limit state safety factor
        groutsf (float) [-]: grout strength safety factor
        syswetfa (float) [m2]: system wet frontal area
        syswetba (float) [m2]: system wet beam area
        sysdraft (float) [m]: system draft
        sublength (float) [m]: substation length
        subwidth (float) [m]: substation width
        subheight (float) [m]: substation height
        subprof (str) [-]: substation profile: options:    'cylindrical', 
                                                            'rectangular'
        subrough (float) [m]: substation roughness
        suborienang (float) [deg]: substation orientation angle
        waveinertiacoefrect (numpy.ndarray): rectangular wave inertia coefficients: width/length [-],
                                                                                    inertia coefficients [-] 
        preline (list) [-]: predefined mooring component list
        fabcost (float) [-]: optional fabrication cost factor
        maxlines (int, optional) [-]: maximum number of lines per device.
                                      Defaults to 12.
    
    Attributes: 
        None
    
    """

    def __init__(self,   devices,
                         gravity,
                         seaden,
                         airden,
                         steelden,
                         conden,
                         groutden,
                         compdict,
                         soiltypgrid,
                         seaflrfriccoef,
                         bathygrid,
                         bathygriddeltax,
                         bathygriddeltay,
                         wlevmax,
                         wlevmin,
                         currentvel,
                         currentdir,
                         currentprof,
                         wavedir,
                         hs,
                         tp,
                         gamma,
                         windvel,
                         winddir,
                         windgustvel,
                         windgustdir,
                         soilprops,
                         linebcf,
                         k1coef,
                         soilsen,
                         subgradereaccoef,
                         piledefcoef,
                         pilemomcoefam,
                         pilemomcoefbm,
                         pilefricresnoncal,
                         hcfdrsoil,
                         systype,
                         depvar,
                         sysprof,
                         sysmass,
                         syscog,
                         sysvol,
                         sysheight,
                         syswidth,
                         syslength,
                         sysrough,
                         sysorig,
                         fairloc,
                         foundloc,
                         umbconpt,
                         sysdryfa,
                         sysdryba,
                         dragcoefcyl,
                         wakeampfactorcyl,
                         winddragcoefrect,
                         currentdragcoefrect,
                         driftcoeffloatrect,
                         Clen,
                         thrustcurv,
                         hubheight,
                         sysorienang,
                         fex,
                         premoor,
                         maxdisp,
                         prefound,
                         coststeel,
                         costgrout,
                         costcon,
                         groutstr,
                         preumb,
                         umbsf,
                         foundsf,
                         prefootrad,
                         subcabconpt,
                         substparams,
                         moorsfuls,
                         moorsfals,
                         groutsf,
                         syswetfa,
                         syswetba,
                         sysdraft,
                         waveinertiacoefrect,
                         preline,
                         fabcost,
                         maxlines=None):
                             
        self.devices = devices
        self.gravity = gravity
        self.seaden = seaden
        self.airden = airden
        self.steelden = steelden
        self.conden = conden
        self.groutden = groutden
        self.compdict = compdict
        self.soiltypgrid = soiltypgrid
        self.seaflrfriccoef = seaflrfriccoef
        self.bathygrid = bathygrid
        self.bathygriddeltax = bathygriddeltax
        self.bathygriddeltay = bathygriddeltay
        self.wlevmax = wlevmax
        self.wlevmin = wlevmin
        self.currentvel = currentvel
        self.currentdir = currentdir
        self.currentprof = currentprof
        self.wavedir = wavedir
        self.hs = hs
        self.tp = tp
        self.gamma = gamma
        self.windvel = windvel
        self.winddir = winddir
        self.windgustvel = windgustvel
        self.windgustdir = windgustdir
        self.soilprops = soilprops
        self.linebcf = linebcf
        self.k1coef = k1coef
        self.soilsen = soilsen
        self.subgradereaccoef = subgradereaccoef
        self.piledefcoef = piledefcoef
        self.pilemomcoefam = pilemomcoefam
        self.pilemomcoefbm = pilemomcoefbm
        self.pilefricresnoncal = pilefricresnoncal
        self.hcfdrsoil = hcfdrsoil
        self.systype = systype
        self.depvar = depvar
        self.sysprof = sysprof
        self.sysmass = sysmass
        self.syscog = syscog
        self.sysvol = sysvol
        self.sysheight = sysheight
        self.syswidth = syswidth
        self.syslength = syslength
        self.sysrough = sysrough
        self.sysorig = sysorig
        self.fairloc = fairloc
        self.foundloc = foundloc
        self.umbconpt = umbconpt
        self.sysdryfa = sysdryfa
        self.sysdryba = sysdryba
        self.dragcoefcyl = dragcoefcyl
        self.wakeampfactorcyl = wakeampfactorcyl
        self.winddragcoefrect = winddragcoefrect
        self.currentdragcoefrect = currentdragcoefrect
        self.driftcoeffloatrect = driftcoeffloatrect
        self.Clen = Clen
        self.thrustcurv = thrustcurv
        self.hubheight = hubheight
        self.sysorienang = sysorienang
        self.fex = fex
        self.premoor = premoor
        self.maxdisp = maxdisp
        self.prefound = prefound
        self.coststeel = coststeel
        self.costgrout = costgrout
        self.costcon = costcon
        self.groutstr = groutstr
        self.preumb = preumb
        self.umbsf = umbsf
        self.foundsf = foundsf
        self.prefootrad = prefootrad
        self.subcabconpt = subcabconpt
        self.substparams = substparams
        self.moorsfuls = moorsfuls
        self.moorsfals = moorsfals
        self.groutsf = groutsf
        self.syswetfa = syswetfa
        self.syswetba = syswetba
        self.sysdraft = sysdraft        
        self.waveinertiacoefrect = waveinertiacoefrect
        self.preline = preline
        self.fabcost = fabcost
        
        # Set default maximum lines to 12
        if maxlines is None:
            self.maxlines = 12
        else:
            self.maxlines = abs(int(maxlines))
        
        return
        
class Main(Found,Moor,Subst):
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Main class
    #--------------------------------------------------------------------------
    #-------------------------------------------------------------------------- 
    This is the main class of the WP4 module which inherits from the 
    Foundation, Mooring and Substation Foundation sub-modules 

    Functions:
        moorsub: top level mooring module
        moorsel: selects mooring system type
        moordes: designs mooring system
        moorinst: calculates installation parameters
        moorcost: calculates mooring system capital cost
        self.moorbomdict: creates of mooring system bill of materials
        moorhier: creates of mooring system hierarchy
        umbdes: specifies umbilical geometry
        umbinst: calculates installation parameters
        umbcost: calculates of umbilical capital cost
        umbbom: creates umbilical bill of materials
        umbhier: creates umbilical hierarchy
        substsub: top level substation foundation module
        substsel: selects substation foundation type
        substdes: designs substation foundation
        substinst: calculates installation parameters
        substcost: calculates of substation foundation capital cost
        substbom: creates of substation foundation bill of materialss
        substhier: creates of substation foundation hierarchy
        foundsub: top level foundation module
        foundsel: selects foundation type
        founddes: designs foundation
        foundinst: calculates installation parameters
        foundcost: calculates of foundation capital cost
        foundbom: creates of foundation bill of materials
        foundhier: creates of foundation hierarchy

    Args:
        variables: input variables
    
    Attributes:
        deviceid (str) [-]: device identification number
        syslabel: device label
        
    Returns:
        sysmoorinsttab(pandas) [-]: array-level mooring system installation requirements table for all devices:     device number (int) [-], 
                                                                                                                    line number (int) [-],
                                                                                                                    component list (list) [-],
                                                                                                                    mooring system type (str) [-],
                                                                                                                    line length (float) [m],
                                                                                                                    dry mass (float) [kg] 
        sysfoundinsttab(pandas) [-]: array-level foundation installation requirements table for all devices:    device number (int) [-], 
                                                                                                                line number (int) [-],
                                                                                                                component list (list) [-],
                                                                                                                mooring system type (str) [-],
                                                                                                                line length (float) [m],
                                                                                                                dry mass (float) [kg] 
        sysumbinsttab(pandas) [-]: array-level umbilical installation requirements table for all devices:    device number (int) [-],
                                                                                                              component list (list) [-],
                                                                                                              subsea connection x coord [m],
                                                                                                              subsea connection y coord [m], 
                                                                                                              subsea connection z coord [m],
                                                                                                              length [m],
                                                                                                              dry mass [kg],
                                                                                                              required flotation [N/m],
                                                                                                              flotation length [m] 
        sysbom (dict) [-]: array-level bill of materials: keys: device identification number (str) [-]: umbilical: cost: (float) [euros],
                                                                                                                   diameter: (float) [m],
                                                                                                                   length: (float) [m],
                                                                                                                   total weight: (float) [kg],
                                                                                                                   umbilical type: (str) [-]
                                                                                                        mooring system: mooring system type (str) [-],
                                                                                                                        component quantities (list) [-],
                                                                                                                        total cost (float [euros]),
                                                                                                                        line lengths (list) [m],
                                                                                                                        component id numbers (list) [-]
                                                                                                        device foundation for N foundation points: foundation type (str) [-],
                                                                                                                                                   foundation subtype (str) [-],
                                                                                                                                                   dimensions (list): width (float) [m],
                                                                                                                                                                      length (float) [m],
                                                                                                                                                                      height (float) [m]
                                                                                                                                                   cost (float) [euros],
                                                                                                                                                   total weight (float) [kg], 
                                                                                                                                                   quantity (int) [-]                                                                                                                                            
                                                                                                                                                   grout type (str) [-],
                                                                                                                                                   grout volume (float) [m3],
                                                                                                                                                   component identification numbers (list) [-]
                                                                array (str) [-]: substation foundation: foundation type (str) [-],
                                                                                                        foundation subtype (str) [-],
                                                                                                        dimensions (list): width (float) [m],
                                                                                                                           length (float) [m],
                                                                                                                           height (float) [m]
                                                                                                        cost (float) [euros],
                                                                                                        dry weight (float) [kg], 
                                                                                                        quantity (int) [-],
                                                                                                        grout type (str) [-],
                                                                                                        grout volume (float) [m3]
        syshier (str) [-]: array-level hierarchy: keys: device identification number: umbilical: umbilical type (str) [-],
                                                                                      foundation: foundation components (list) [-],
                                                                                      mooring system: mooring components (list) [-],
                                              array (str) [-]: substation foundation: foundation components (list) [-]
        devsyscat (str) [-]: device-level system category
        
        
    
    """
    def __init__(self, variables):        
        super(Main, self).__init__(variables)

    def __call__(self):
        
        self.sysrambom = {}
        self.sysecobom = {}
        self.syshier = {}
        self.syslabel = {}
        self.syscat = {}
        self.sysenv = {}
        self.uniqcompind = 1
        self.netuniqcompind = 0
        self.projectyear = 0
        self.linenum = 0
        self.foundnum = 0
        
        # Devices must be parsed in numerical order
        for deviceid in sorted(self._variables.devices):
            module_logger.info('===================================================================')
            module_logger.info("Device ID: {}".format(deviceid))    
            module_logger.info('===================================================================')
            self.foundradnew = []
            self.gpnearloc(deviceid,
                           self._variables.systype,
                           self._variables.foundloc,
                           self._variables.sysorig[deviceid],
                           self._variables.sysorienang)
            if self._variables.systype in ("wavefloat","tidefloat"):
                self.moorsub()
                self.moorsel()            
                self.moordes(deviceid)                
                self.moorcost()
                self.moorinst(deviceid)
                self.moorbom(deviceid)
                self.moorhierarchy()   
                if self._variables.preumb:
                    self.umbcost()   
                    self.umbbom(deviceid)    
                    self.umbhierarchy()
                    self.umbinst(deviceid)
            self.foundsub(deviceid,
                          self._variables.systype,
                          self._variables.foundloc,
                          self._variables.syscog)                                     
            self.foundsel(self._variables.systype)
            self.founddes(self._variables.systype)        
            self.foundcost()   
            self.foundbom(deviceid)
            self.foundhierarchy()
            self.foundinst(deviceid,
                           self._variables.systype,
                           self._variables.sysorig,
                           self._variables.foundloc,
                           self._variables.sysorienang)
            """Sub system bill of materials and hierarchy """
            self.syslabel['deviceid'] = deviceid   
            self.sysenv[self.syslabel['deviceid']] = {'Configuration footprint area [m2]':self.devconfigfootarea, 
                                                'Configuration volume [m3]':self.devconfigvol}
            if self._variables.systype in ("wavefloat","tidefloat"):                
                if self._variables.preumb:
                    self.sysrambom[self.syslabel['deviceid']] = {'Umbilical':self.umbrambomdict, 
                                                    'Mooring system':self.moorrambomdict, 
                                                    'Foundation':self.foundrambomdict}
                    self.syshier[self.syslabel['deviceid']] = {'Umbilical':self.umbhier, 
                                                    'Mooring system':self.moorhier, 
                                                    'Foundation':self.foundhier}
                    self.syscat[self.syslabel['deviceid']] = (deviceid + " :" 
                        + self.selmoortyp 
                        + " moored system comprising " + str(self.numlines) 
                        + " lines and " + str([seq[0] for seq in self.selfoundtyp]) 
                        + " anchors.")  
                else:
                    self.sysrambom[self.syslabel['deviceid']] = {'Umbilical':[], 
                                'Mooring system':self.moorrambomdict, 
                                'Foundation':self.foundrambomdict}
                    self.syshier[self.syslabel['deviceid']] = {'Umbilical':[], 
                                                    'Mooring system':self.moorhier, 
                                                    'Foundation':self.foundhier}
                    self.syscat[self.syslabel['deviceid']] = (deviceid + " :" 
                        + self.selmoortyp 
                        + " moored system comprising " + str(self.numlines) 
                        + " lines and " + str([seq[0] for seq in self.selfoundtyp]) 
                        + " anchors.")  
            elif self._variables.systype in ("wavefixed","tidefixed"): 
                self.sysrambom[self.syslabel['deviceid']] = {'Umbilical':[], 
                                                'Mooring system':[], 
                                                'Foundation':self.foundrambomdict}
                self.sysecobom = self.foundecobomtab
                self.syshier[self.syslabel['deviceid']] = {'Umbilical':[], 
                                                'Mooring system':[], 
                                                'Foundation':self.foundhier}
                self.syscat[self.syslabel['deviceid']] = (deviceid + ": " 
                    + " foundation system comprising " 
                    + str([seq[0] for seq in self.selfoundtyp]) + " foundation(s).")
            if deviceid == 'device001': 
                self.sysfoundinsttab = copy.deepcopy(self.foundinsttab)
                self.sysfoundecobomtab = copy.deepcopy(self.foundecobomtab)
                if self._variables.systype in ("wavefloat","tidefloat"): 
                    if self._variables.preumb:
                        self.sysumbinsttab = copy.deepcopy(self.umbinsttab)
                        self.sysumbecobomtab = copy.deepcopy(self.umbecobomtab)
                    self.sysmoorinsttab = copy.deepcopy(self.moorinsttab)
                    self.sysmoorecobomtab = copy.deepcopy(self.moorecobomtab)
            elif deviceid != 'device001':
                self.sysfoundinsttab = pd.concat([self.sysfoundinsttab,
                                                  self.foundinsttab], axis=0, ignore_index=True)
                self.sysfoundecobomtab = pd.concat([self.sysfoundecobomtab,
                                                    self.foundecobomtab], axis=0, ignore_index=True)               
                if self._variables.systype in ("wavefloat","tidefloat"):
                    if self._variables.preumb:
                        self.sysumbinsttab = self.umbinsttab    
                        self.sysumbecobomtab = pd.concat([self.sysumbecobomtab,
                                                            self.umbecobomtab], axis=0, ignore_index=True)
                    self.sysmoorinsttab = pd.concat([self.sysmoorinsttab,
                                                     self.moorinsttab], axis=0, ignore_index=True)
                    self.sysmoorecobomtab = pd.concat([self.sysmoorecobomtab,
                                                     self.moorecobomtab], axis=0, ignore_index=True)
        if self._variables.systype in ("wavefloat","tidefloat"):  
            if self._variables.preumb:
                self.sysecobom = pd.concat([self.sysumbecobomtab,
                                            self.sysmoorecobomtab,
                                            self.sysfoundecobomtab], axis=0, ignore_index=True)
            else:
                self.sysecobom = pd.concat([self.sysmoorecobomtab,
                            self.sysfoundecobomtab], axis=0, ignore_index=True)
        elif self._variables.systype in ("wavefixed","tidefixed"):    
            self.sysecobom = self.sysfoundecobomtab
                
        if self._variables.prefound and self._variables.prefound[0:6] == 'uniary':
            """ Uniform foundations across array. Determine largest weight """
            fwarray = self.sysfoundinsttab['dry mass [kg]'].tolist()
            fwmaxindaryind = fwarray.index(max(fwarray))
            fwmaxdev = self.sysfoundinsttab.ix[fwmaxindaryind, 'devices [-]']
            
            """ Update installation parameters, RAM hierarchy, RAM bill of materials and economics tables"""
            for rowind in self.sysfoundinsttab.index:
                self.sysfoundinsttab.ix[rowind,'length [m]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'length [m]']
                self.sysfoundinsttab.ix[rowind,'width [m]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'width [m]']
                self.sysfoundinsttab.ix[rowind,'height [m]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'height [m]']
                self.sysfoundinsttab.ix[rowind,'installation depth [m]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'installation depth [m]']
                self.sysfoundinsttab.ix[rowind,'dry mass [kg]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'dry mass [kg]']  
                self.sysfoundinsttab.ix[rowind,'grout type [-]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'grout type [-]']  
                self.sysfoundinsttab.ix[rowind,'grout volume [m3]'] = self.sysfoundinsttab.ix[fwmaxindaryind,'grout volume [m3]']  
            for deviceid in self._variables.devices:
                self.sysrambom[deviceid]['Foundation']['quantity'] = self.sysrambom[fwmaxdev]['Foundation']['quantity']
                self.syshier[deviceid]['Foundation'] = self.syshier[fwmaxdev]['Foundation']
            for rowind in self.sysecobom.index:
                """ Search for compid and cost """
                if (self.sysecobom.ix[rowind,'devices [-]'] == fwmaxdev
                    and self.sysecobom.ix[rowind,'logistics id [-]'][:-4] == self._variables.prefound[6:]):
                    # logmsg = [""]
                    # logmsg.append('rowind {}'.format(rowind ))
                    # module_logger.info("\n".join(logmsg)) 
                    fwmaxcompid = self.sysecobom.ix[rowind,'compid [-]']
                    fwmaxcost = self.sysecobom.ix[rowind,'component cost [euros]']
            for rowind in self.sysecobom.index:
                """ Update compid and cost """
                if (self.sysecobom.ix[rowind,'devices [-]'][0:6] == 'device'
                    and self.sysecobom.ix[rowind,'logistics id [-]'][:-4] == self._variables.prefound[6:]):
                    self.sysecobom.ix[rowind,'compid [-]'] = fwmaxcompid
                    self.sysecobom.ix[rowind,'component cost [euros]'] = fwmaxcost
        
        # Check for None
        if self._variables.substparams is None:
            substlist = []
        else:
            substlist = self._variables.substparams.index.tolist()

        for substid in substlist:
            module_logger.info('===================================================================')
            module_logger.info("Substation ID: {}".format(substid))
            module_logger.info('===================================================================')
            self.substsub(substid)
            self.substsel(substid)
            self.substdes(substid)
            self.substcost()               
            self.substbom(substid) 
            self.substhierarchy()  
            self.substinst(substid)
            self.sysenv[substid] = {'Configuration footprint area [m2]':self.substconfigfootarea, 
                                                'Configuration volume [m3]':self.substconfigvol}
            if self.substfoundrambomdict:
                self.sysrambom[substid] = {'Substation foundation':self.substfoundrambomdict}       
                self.sysecobom = pd.concat([self.sysecobom,
                                            self.substfoundecobomtab], axis=0, ignore_index=True)
                self.syscat[substid] = ('Selected substation foundation(s): ' + str(self.selfoundtyp))
            if self.substhier != 'n/a':
                self.syshier[substid] = {'Substation foundation':self.substhier}            
            self.sysfoundinsttab = pd.concat([self.sysfoundinsttab,
                                                  self.foundinsttab], axis=0, ignore_index=True)
        
        self.sysecobom.index = range(len(self.sysecobom))
        
        logmsg = [""]
        if self._variables.systype in ("wavefloat","tidefloat"):   
            logmsg.append('_________________________________________________________________________')
            logmsg.append('Array level mooring installation table') 
            logmsg.append('{}'.format(self.sysmoorinsttab))
        logmsg.append('_________________________________________________________________________')            
        logmsg.append('Array level foundation installation table')                
        logmsg.append('{}'.format(self.sysfoundinsttab))  
        logmsg.append('_________________________________________________________________________')
        if self._variables.systype in ("wavefloat","tidefloat") and self._variables.preumb: 
            logmsg.append('Array level umbilical installation table')
            logmsg.append('{}'.format(self.sysumbinsttab))        
            logmsg.append('_________________________________________________________________________')            
        logmsg.append('Array level economics bill of materials (self.sysecobom)')                
        logmsg.append('{}'.format(self.sysecobom))
        logmsg.append('_________________________________________________________________________')
        logmsg.append('Array level RAM bill of materials (self.sysrambom)')
        logmsg.append('{}'.format(self.sysrambom))
        logmsg.append('_________________________________________________________________________')
        logmsg.append('Array level RAM hierarchy (self.syshier)')
        logmsg.append('{}'.format(self.syshier))
        module_logger.info("\n".join(logmsg)) 
        
        # if self._variables.systype in ("wavefloat","tidefloat"):            
            # result = {"category": self.syscat, 
                      # "bill of materials": self.sysrambom,          
                      # "hierarchy": self.syshier,    
                      # "mooring system installation requirements": self.moorinsttab,
                      # "foundation installation requirements": self.foundinsttab,
                      # "umbilical installation requirements": self.umbinsttab,
                      # }
        # elif self._variables.systype in ("wavefixed","tidefixed"):    
            # result = {"category": self.syscat, 
                      # "bill of materials": self.sysrambom,          
                      # "hierarchy": self.syshier,    
                      # "foundation installation requirements": self.foundinsttab,
                      # "umbilical installation requirements": self.umbinsttab,
                      # }
        # return result
    
        
   
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))
