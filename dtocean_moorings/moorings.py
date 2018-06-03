"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter
"""

# Built in modulesrad
import math
import copy
import logging
import operator
from collections import Counter

# External module import
import numpy as np
import pandas as pd
from scipy import interpolate

# Local import 
from .loads import Loads
from .umbilical import Umb

# Start logging
module_logger = logging.getLogger(__name__)


class Moor(Umb, Loads):    
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Moor class
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    Mooring system submodule
    
        Args:
            fairloc
            systype
            premoor
            depvar
            soiltyp
            fairloc
            foundloc
            gravity
            hs
            tp
            sysprof
            syswidth
            sysleng
            totsysstatloads
            totsyssteadloads
            syswaveload
            seaden
            sysmass
            compdict
            sysorig
            deviceid
            maxdisp
            fex 
            diffractrig
            seaflrfriccoef
        
        Attributes:
            numlines (int) [-]: number of mooring lines
            shareanc (bool) [-]: shared anchor points: options:   True,
                                                                False
            lineangs (list) [rad]: line angles in X-Y plane
            selmoortyp (str) [-]: selected mooring system type
            j, l, wc, p, dof, comps, blockind, jind, caseind, casevals, lineind, waveint, linevals (int, float): temporary integers and values
            klim, llim, mlim, wclim (int) [-]: iteration loop limits
            numnodes (int) [-]: number of line nodes
            lineleng (list) [m]: line lengths for N lines
            linegeo (numpy.ndarray): line node coordinates for N lines: X direction (float) [m],
                                                                        Y direction (float) [m],
                                                                        Z direction (float) [m]
            linelengbed (list) [m]: length of line resting on seafloor for N lines
            fairten (list) [m]: fairlead tension for N lines
            ancten (numpy.ndarray) [N]: anchor tension for N lines and for all analysed wave conditions
            lineten (numpy.ndarray) [N]: line tension for N lines and for all analysed wave conditions
            Hfline (list) [N]: horizontal fairlead tension for N lines
            HflineX (list) [N]: X component of horizontal fairlead tension for N lines
            HflineY (list) [N]: Y component of horizontal fairlead tension for N lines
            Vfline (list) [N]: vertical fairlead tension for N lines
            Haline (list) [N]: horizontal anchor tension for N lines
            Valine (list) [N]: horizontal anchor tension for N lines
            linexf (list) [m]: horizontal catenary distance for N lines
            linezf (list) [m]: vertical catenary distance for N lines
            lineangdisp (list) [rad]: displaced line angle from Y axis for N lines
            tol (float) [-]: convergence tolerance
            Hf (list) [N]: calculated horizontal fairlead tensions
            Vf (list) [N]: calculated vertical fairlead tensions
            Ha (list) [N]: calculated horizontal anchor tensions
            Va (list) [N]: calculated vertical anchor tensions
            Hfsys (list) [N]: calculated horizontal system loads
            Vfsys (list) [N]: calculated vertical system loads
            Hsysloadang (list) [rad]: vector of calculated angles of horizontal reaction load
            limitstate (str) [-]: limit state
            linexfref (list) [m]: reference horizontal catenary distance for N lines
            linezfref (list) [m]: reference vertical catenary distance for N lines
            Hflineref (list) [N]: reference horizontal fairlead tension for N lines
            Vflineref (list) [N]: reference vertical fairlead tension for N lines
            analines (list) [-]: analysed lines
            hangweight (float) [N]: cumulative hanging (or resting) line weight
            hangleng (float) [m]: cumulative hanging (or resting) line length
            eaav (float) [Nm]: cumulative axial stiffness
            friccoef (float) [-]: soil-chain friction coefficient
            lambdacat (float) [-]: dimensionless catenary parameter
            compblocks (list) [-]: mooring component list
            linelenghead (str) [-]: line length header
            moorcomptab (pandas) [-]: mooring component pandas table
            ropeea (float) [N]: rope axial stiffness
            omega (float) [N/m]: line weight per unit length
            ea (float) [N]: line axial stiffness
            syspos (list): system position:  X coordinate (float) [m],
                                             Y coordinate (float) [m]
            sysposrefnoenv (list): reference system position with no environmental loading:  X coordinate (float) [m],
                                                                                             Y coordinate (float) [m]
            initcond (list): initial conditions for N lines: linexfref (float) [m],
                                                             linezfref (float) [m],
                                                             Hflineref (float) [N],
                                                             Vflineref (float) [N]
            xf (float) [m]: horizontal catenary distance
            zf (float) [m]: vertical catenary distance
            HsysloadX (float) [N]: X component of horizontal system load
            HsysloadY (float) [N]: Y component of horizontal system load
            syswpa (float) [m2]: system water plane area
            dHf (list) [N]: calculated horizontal fairlead tension increments
            dVf (list) [N]: calculated vertical fairlead tension increments
            errxf (list) [m]: horizontal catenary distance errors 
            errzf (list) [m]: vertical catenary distance errors
            xfdHf (list) [m/N]: horizontal catenary distance - horizontal fairlead load differential
            zfdHf (list) [m/N]: vertical catenary distance - horizontal fairlead load differential
            xfdVf (list) [m/N]: horizontal catenary distance - vertical fairlead load differential
            zfdVf (list) [m/N]: vertical catenary distance - vertical fairlead load differential
            VfovHf (list) [-]: horizontal - vertical fairlead load ratio
            VfminlomegaovHf (list) [N]: catenary parameter
            linelengbedk (list) [m]: length of line resting on seafloor
            mu (float) [m]: seafloor friction parameter
            jac (numpy.ndarray): jacobian load differential matrix
            invjac (numpy.ndarray): inverse jacobian matrix
            update (list): iteration update:    horizontal load increment [N]
                                                vertical load increment [N]
            det (float) [-]: jacobian matrix determinant
            errang (float) [rad]: load error angle from the Y axis
            loaddiffmag (float) [N]: load difference magnitude
            sysdisp (float) [m]: proportional system displacement increment
            rn (int) [-]: simulation run number
            nodes (float) [-]: line nodes
            complist (list) [-]: component list
            block (str) [-]: component name
            complim (float) [m]: component size limit
            complistsort (list) [-]: list of suitable components from database sorted by cost
            compsminsize (tuple) [-]: minimum size of suitable component list
            selcomp (str) [-]: id of selected component
            compprops (pandas): component property pandas table
            ulscheck (bool) [-]: ultimate limit state flag
            alscheck (bool) [-]: accident limit state flag
            selcomplist (list) [-]: selected component list
            compid (str) [-]: id of selected component
            linesuls (list) [-]: ultimate limit state analysed lines
            anctenuls (list) [N]: ultimate limit state calculated anchor tensions
            amendcomp (str) [-]: amended component id
            amendcompprops (list) [-]: amended component properties
            maxlinetenval (float) [N]: maximum line tension magnitude
            maxlinetenind (int) [-]: line with maximum tension from ULS analysis
            linesals (list) [-]: accident limit state analysed lines
            anctenals (list) [N]: accident limit state calculated anchor tensions
            statpos (list): system static position:   X coordinate (float) [m], 
                                                      Y coordinate (float) [m], 
                                                      Z coordinate (float) [m]
            connleng (float) [m]: cumulative connected component length
            steadpos (list): system steady position:   X coordinate (float) [m], 
                                                       Y coordinate (float) [m], 
                                                       Z coordinate (float) [m]
            totmoorsteadloads (list) [N]: total mooring system steady loads
            qsdisps (list) [m]: quasi-static displacements
            moorstiff (list): quasi-static mooring system stiffness:     X component (float) [N/m],
                                                                         Y component (float) [N/m],
                                                                         Z component (float) [N/m]
            fowdispnd (list): first-order wave response:    X component (float) [-],
                                                            Y component (float) [-],
                                                            Z component (float) [-]
            omegawavediff (list) [rad/s]: angular wave frequency for all diffraction analysed wave conditions
            wavecontour (numpy.ndarray): wave contour:  tp (float) [s]
                                                        hs (float) [m]
            totancsteadloads (numpy.ndarray): steady anchor loads for N lines: X component (float) [N],
                                                                               Y component (float) [N],
                                                                               Z component (float) [N]
            anctentabindsuls (list) [-]: ultimate limit state analysis labels
            anctentabindsals (list) [-]: accident limit state analysis labels
            anctentabcols (list) [-]: anchor tension column headers
            anctentabuls (pandas) [-]: ultimate limit state anchor tension pandas table
            anctentabals (pandas) [-]: accident limit state anchor tension pandas table
            frames (numpy.ndarray): anchor tension table frames
            anctentab (pandas) [-]: concatenated anchor tension table
            anctentabinds (list) [-]: concatenated anchor tension labels
            maxten (list) [N]: maximum tension magnitude for N lines
            maxtenind (list) [int]: maximum tension integer for N lines
            moorconnsize (float) [m]: anchor-mooring line connecting size
            moorinstparams (list) [-]: mooring system installation parameters for N foundation points: device number (int) [-], 
                                                                                                       line number (int) [-],
                                                                                                       component list (list) [-],
                                                                                                       mooring system type (str) [-],
                                                                                                       line length (float) [m],
                                                                                                       dry mass (float) [kg]           
            listmoorcomp (list) [-]: mooring component list
            tabind (list) [-]: table indices
            moorinstsubtab (pandas) [-]: mooring system sub-table 
            linedrymasshead (str) [-]: line dry mass header
            quanmoorcomp (list) [-]: mooring component quantities
            moorinsttab (pandas) [-]: mooring system installation requirements pandas table:   device number (int) [-], 
                                                                                               line number (int) [-],
                                                                                               component list (list) [-],
                                                                                               mooring system type (str) [-],
                                                                                               line length (float) [m],
                                                                                               dry mass (float) [kg] 
            moorcompcosts (list) [euros]: mooring component costs
            totmoorcost (float) [euros]: total mooring system cost
            moorbom (dict) [-]: mooring system bill of materials: mooring system type (str) [-],
                                                                    component quantities (list) [-],
                                                                    total cost (float [euros]),
                                                                    line lengths (list) [m],
                                                                    component id numbers (list) [-]
            linehier (list) [-]: line hierarchy 
            moorhier (list) [-]: mooring system hierarchy 
            
        Functions:
            moorsub: retrieves umbilical details and shared anchor flag
            moorsel: uses premoor or specifies taut or catenary mooring system
            moordes: mooring system design and analysis:    
                                                            mooreqav: mooring system analysis
                                                            moorcompret: component retrieval
            moorinst: calculates installation parameters
            moorcost: calculates mooring system capital cost
            moorbom: creates mooring system bill of materials
            moorhierarchy: creates mooring system hierarchy   
                                                                
    
    """    
    
    def __init__(self, variables):        
        super(Moor, self).__init__(variables)   
        self._variables = variables
        
    def moorsub(self):
        if len(self._variables.foundloc) > 0:
            self.numlines = len(self._variables.foundloc)
        elif self._variables.systype in ("wavefloat","tidefloat"): 
            self.numlines = len(self._variables.fairloc)  
            
    def moorsel(self):         
        """ Selection of possible mooring systems. In Version Beta only taut 
            and catenary systems will be considered """
        self.lineangs = [0 for row in range(self.numlines)]
        
        if self._variables.premoor: 
            # print 'Mooring system already selected'
            self.selmoortyp = self._variables.premoor          
        else:    
            if self._variables.depvar == True:
                self.selmoortyp = 'taut'
                #print 'Taut moored system'
            elif self._variables.depvar == False:
                self.selmoortyp = 'catenary'
                #print 'Catenary mooring system'   
        for j in range(0,self.numlines):                                       
            self.lineangs[j] = ((j * math.pi * 2.0 / self.numlines) 
                                    + (self._variables.sysorienang * math.pi / 180.0))

    def moordes(self, deviceid):       
        super(Moor, self).sysstat(self._variables.sysvol,
                                    self._variables.sysmass)
        super(Moor, self).sysstead(self._variables.systype,
                                    self._variables.syswidth,
                                    self._variables.syslength,
                                    self._variables.sysheight,
                                    self._variables.sysorienang,
                                    self._variables.sysprof,
                                    self._variables.sysdryfa,
                                    self._variables.sysdryba,
                                    self._variables.syswetfa,
                                    self._variables.syswetba,
                                    self._variables.sysvol,
                                    self._variables.sysrough)
        super(Moor, self).syswave(deviceid,
                                    self._variables.systype,
                                    self._variables.syswidth,
                                    self._variables.syslength,
                                    self._variables.sysheight,
                                    self._variables.sysprof,
                                    self._variables.sysvol,
                                    self._variables.sysrough)
        
        self.deviceid = deviceid
        klim = 100
           
        """ Definition of static mooring geometry. First design is chain 
            only """
        """ Number of nodes per chain or rope """
        numnodes = 20  
        self.lineleng = [0 for row in range(self.numlines)]        
        self.tautleng = [0 for row in range(self.numlines)] 
        linegeo = []        
        linexfref = [[0 for col in range(0, len(self.foundloc))] for row in range(0, 4)]
        linezfref = [[0 for col in range(0, len(self.foundloc))] for row in range(0, 4)]
        Hflineref = [[0 for col in range(0, len(self.foundloc))] for row in range(0, 4)]
        Vflineref = [[0 for col in range(0, len(self.foundloc))] for row in range(0, 4)] 
        
        """ Quasi-static mooring tension approximation """
        def mooreqav(selmoortyp, numlines, fairloc, foundloc, lineleng, 
                     initcond, analines, llim, limitstate):
            
            logmsg = [""]
            logmsg.append('**************************************************')
            logmsg.append('{} analysis'.format(limitstate))
            logmsg.append('**************************************************')
            logmsg.append("{}".format(self.moorcomptab))
            module_logger.info("\n".join(logmsg))
        
            fairten = [0 for row in range(0, len(self.foundloc))]
            if self._variables.hs:
                if llim in ([0, 1, 2], [1, 2]):
                    ancten = [[0 for col in range(0, 
                        len(self.foundloc))] for row 
                        in range(0, 2 + len(self._variables.hs))]
                    lineten = [[0 for col in range(0, 
                        len(self.foundloc))] for row 
                        in range(0, 2 + len(self._variables.hs))]                    
                elif llim in ([3], [3,4]):
                    ancten = self.anctenuls
                    lineten = self.linetenuls
                    lineten[0] = [0 for col in range(0, 
                                    len(self.foundloc))]                     
                self.umbtenmax = [0 for row 
                    in range(0, 2 + len(self._variables.hs))]
                self.umbradmin = [0 for row 
                    in range(0, 2 + len(self._variables.hs))]
                finalsyspos = [[0 for col in range(0,3)] for row 
                            in range(0, 5 + len(self._variables.hs) - 1)]
            else:  
                if llim in ([0, 1, 2], [1, 2]):
                    ancten = [[0 for col in range(0, 
                                len(self.foundloc))] for row 
                                in range(0, max(llim) + 1)]
                    lineten = [[0 for col in range(0, 
                                len(self.foundloc))] for row 
                                in range(0, max(llim) + 1)]
                elif llim in ([3], [3,4]):
                    ancten = self.anctenuls
                    lineten = self.linetenuls
                    lineten[0] = [0 for col in range(0, 
                                    len(self.foundloc))]                     
                self.umbtenmax = [0.0]
                self.umbradmin = [0.0]
                finalsyspos = [[0 for col in range(0,3)] for row 
                            in range(0, 5)]
            self.anctenhigh = copy.deepcopy(ancten)
            self.linetenhigh = copy.deepcopy(lineten)
            if llim == [3,4]:
                self.anctenlow = copy.deepcopy(ancten)
                self.linetenlow = copy.deepcopy(lineten)
            Hfline = [0 for row in range(0, len(self.foundloc))]
            HflineX = [0 for row in range(0, len(self.foundloc))]
            HflineY = [0 for row in range(0, len(self.foundloc))]
            Vfline = [0 for row in range(0, len(self.foundloc))]
            Haline = [0 for row in range(0, len(self.foundloc))]
            Valine = [0 for row in range(0, len(self.foundloc))]
            linexf = [0 for row in range(0, len(self.foundloc))]
            linezf = [0 for row in range(0, len(self.foundloc))]
            self.linelengdiff = [0 for row in range(0, len(self.foundloc))]
            lineangdisp = [0 for row in range(0, len(self.foundloc))]
            lineconvfailflag = ['False' for row in range(0, len(self.foundloc))]
            linelengchangeflag = ['False' for row in range(0, len(self.foundloc))]
            
            tol = 0.01       
            klim = 500            
            Hf = [0 for row in range(klim)]
            Vf = [0 for row in range(klim)] 
            Ha = [0 for row in range(klim)]
            Va = [0 for row in range(klim)] 
            Hfsys = [0 for row in range(5)] 
            Vfsys = [0 for row in range(5)]   
            
            mlim = 40000
            compexceedflag = 'False' 
            self.dispexceedflag = 'False'
            umbcheck = 'False'
            
            if 'rope' in self.moorcomptab.index:
                """ Look-up axial stiffness based on applied load and update average value """
                ropeeavals = np.transpose(self.moorcomptab.ix[self.ropeeaind,'ea'])
                ropeeavalsint = interpolate.interp1d(ropeeavals[1],ropeeavals[0],'cubic') 
                # ropedummyloads = range(0, 100)
                # ropedummystrains = [0 for row in range(0, len(ropedummyloads))]
                # for ind,vals in enumerate(ropedummyloads):
                    # ropedummystrains[ind] = ropeeavalsint(vals)
            
            """ System QS loop """                
            for l in llim:                 
                """ Design stages
                l = 0: Static equilibrium of lines only (high water)
                l = 1: Static equilibrium with device (ULS and ALS at high water)
                l = 2: Equilibrium with environmental loading (ULS and ALS at high water) 
                l = 3: Equilibrium with environmental loading with umbilical (ULS at high water) 
                l = 4: Low water level check with umbilical (ULS)
                """  
                if not(self._variables.preumb) and l >= 3:
                    umbcheck = 'True'
                    self.wlevlowflag = 'True'
                    break
                if l == 4:
                    """ Recalculate system loads at lower water level  """
                    self.bathysysorig = self.bathysysorig - (self._variables.wlevmax + self._variables.wlevmin)
                    super(Moor, self).sysstat(self._variables.sysvol,
                                                self._variables.sysmass)
                    super(Moor, self).sysstead(self._variables.systype,
                                                self._variables.syswidth,
                                                self._variables.syslength,
                                                self._variables.sysheight,
                                                self._variables.sysorienang,
                                                self._variables.sysprof,
                                                self._variables.sysdryfa,
                                                self._variables.sysdryba,
                                                self._variables.syswetfa,
                                                self._variables.syswetba,
                                                self._variables.sysvol,
                                                self._variables.sysrough)
                    super(Moor, self).syswave(deviceid,
                                                self._variables.systype,
                                                self._variables.syswidth,
                                                self._variables.syslength,
                                                self._variables.sysheight,
                                                self._variables.sysprof,
                                                self._variables.sysvol,
                                                self._variables.sysrough)                    
                for j in analines: 
                    if l == 4:
                        foundloc[j][2] = foundloc[j][2] + self._variables.wlevmin
                    if (l in (0,1,2,3,4)):
                        hangweightwet = []
                        hangweightdry = []
                        hangleng = []
                        eaav = []
                        
                        """ Determine soil type at anchor point. Note: Soil type may 
                            vary along ground chain and this is currently not accounted 
                            for """
                        """ Friction coefficient between chain and soil """
                        if self._variables.seaflrfriccoef:
                            friccoef = self._variables.seaflrfriccoef
                        else:
                            friccoef = float(self._variables.soilprops.ix[self.soiltyp[j],'seaflrfriccoef'])
                        for blockind in range(0, len(compblocks)):
                            """ Line length header """
                            linelenghead = 'line ' + str(j) + ' length' 
                            """ If necessary update line length in mooring component table by decreasing chain or rope length """
                            if linelengchangeflag[j] == 'True':
                                if 'chain' in compblocks:
                                    if compblocks[blockind] == 'chain':
                                        self.moorcomptab.ix[blockind, linelenghead] = self.moorcomptab.ix[blockind, linelenghead] - self.linelengdiff[j]
                                else:
                                    if compblocks[blockind] == 'rope':
                                        self.moorcomptab.ix[blockind, linelenghead] = self.moorcomptab.ix[blockind, linelenghead] - self.linelengdiff[j]
                                        
                            """ Cumulative hanging (or resting) weight """
                            if compblocks[blockind] in ('chain', 'forerunner assembly', 'rope'):
                                hangweightwet.append(self.moorcomptab.ix[blockind,'wet mass'] 
                                    * self._variables.gravity * self.moorcomptab.ix[blockind,
                                    linelenghead]) 
                                hangweightdry.append(self.moorcomptab.ix[blockind,'dry mass'] 
                                    * self._variables.gravity * self.moorcomptab.ix[blockind,
                                    linelenghead]) 
                            else:
                                hangweightwet.append(self.moorcomptab.ix[blockind,'wet mass'] 
                                        * self._variables.gravity)
                                hangweightdry.append(self.moorcomptab.ix[blockind,'dry mass'] 
                                        * self._variables.gravity)
                            eaav.append(self.moorcomptab.ix[blockind,'ea'] 
                                * self.moorcomptab.ix[blockind,linelenghead]) 
                            """ Cumulative hanging length """
                            hangleng.append(self.moorcomptab.ix[blockind,linelenghead]) 
                        omega = sum(hangweightwet) / sum(hangleng)
                        omegadry = sum(hangweightdry) / sum(hangleng)
                        if 'rope' in self.moorcomptab.index:                              
                            """ In first instance use breaking load and strain to 
                                estimate axial stiffness """
                            ropeea = (self.moorcomptab.ix['rope','mbl'] 
                                * (self.moorcomptab.ix['rope','ea'][-1][1] 
                                - self.moorcomptab.ix['rope','ea'][0][1]) 
                                / (self.moorcomptab.ix['rope','ea'][-1][0] 
                                - self.moorcomptab.ix['rope','ea'][0][0]))                                   
                            ea = ropeea
                        else:
                            ea = sum(eaav) / sum(hangleng)                                               
                if (compexceedflag == 'True' or self.dispexceedflag == 'True'):
                    break
                    
                if (l == 0 and limitstate == 'ALS'):
                        continue
                            
                if l >= 2:
                    if self._variables.hs:
                        """ Cycle through wave contour values of steady loading """
                        wclim = len(self._variables.hs)
                    else:
                        wclim = 1
                else: wclim = 1
                
                for wc in range(0, wclim):  
                    if (compexceedflag == 'True' or self.dispexceedflag == 'True'):
                        break                      
                    logmsg = [""]
                    logmsg.append('_________________________________________________')
                    logmsg.append('Run number {}'.format(l))
                    if self._variables.hs:
                        logmsg.append('Wave condition number {}'.format(wc))
                    else:
                        logmsg.append('Condition number {}'.format(wc))
                    module_logger.info("\n".join(logmsg))

    #                print 'Position number ' + str(l)
                    if l < 2:
                        syspos = [0.0, 0.0, 0.0]
                    if l == 2:                         
                        if limitstate == 'ALS':
                            if self.sysposfail:
                                if self.sysposfail[0] == 'ALS':
                                    syspos = self.sysposfail[1]
                                else:
                                    syspos = [0.0, 0.0, 0.0]
                            else:
                                syspos = copy.deepcopy(sysposrefnoenv)  
                        elif limitstate == 'ULS':
                            if self.sysposfail:
                                if self.sysposfail[0] == 'ULS':
                                    syspos = self.sysposfail[1]
                                else:
                                    syspos = [0.0, 0.0, 0.0]
                            else:
                                syspos = [0.0, 0.0, 0.0]
                                
                    if l in (3,4):
                        syspos = self.sysposuls[l-1+wc]
                        self.umbconpt = copy.deepcopy(self._variables.umbconpt)
                        if l == 3:
                            self.umbconpt[2] = self.umbconpt[2] - self._variables.wlevmax
                        elif l == 4:
                            self.umbconpt[2] = self.umbconpt[2] + self._variables.wlevmin
                        
                        (HumbloadX,
                         HumbloadY,
                         Vumbload,
                         self.umbleng,
                         umbcheck) = self.umbdes(self.deviceid,
                                                 syspos,
                                                 wc,
                                                 self.umbconpt,
                                                 l) 
                        
                    if l < 3:
                        HumbloadX = 0.0
                        HumbloadY = 0.0
                        Vumbload = 0.0   
                        
                    module_logger.info("Analysed lines: {}".format(analines))
                    
                    sysposrec = []
                    sysposrecdiff = []
                    loaddiffmagrec = []
                    errangrec = [0.0]      
                    dispangrec = [0.0]
                    Hloaddiffmag = [0.0]
                    Vloaddiffmag = [0.0]
                    hsysdisp = 0.0
                    horzoscflag = 'False'
                    hosccount = 1
                    vosccount = 1
                    sysposX = []
                    sysposY = []
                    for m in range(1,mlim):
                        Hloadcheck = 'False'
                        Vloadcheck = 'False'
                        errang = 0.0
                        vsysdisp = 0.0
                        lineconvfailflag = ['False' for row in range(0, len(self.foundloc))]
                        
                        for jind, j in enumerate(analines):
                            linelenghead = 'line ' + str(j) + ' length' 
                            xf = [0 for row in range(klim)]
                            zf = [0 for row in range(klim)] 
                            
                            if m == 1:                    
                                """ Horizontal and vertical distance between fairlead and 
                                    anchors at rest """
                                if (fairloc[j][0] - foundloc[j][0] == 0) and (fairloc[j][1] - foundloc[j][1] == 0):
                                    foundloc[j][0] = 0.01
                                linexf[j] = math.sqrt((fairloc[j][0] + syspos[0] - foundloc[j][0]) ** 2.0 
                                    + (fairloc[j][1] + syspos[1] - foundloc[j][1]) ** 2.0)
                                linezf[j] = fairloc[j][2] - self._variables.sysdraft - foundloc[j][2]
                            
                            if l == 0:
                                """ Equilibrium of lines only """
                                """ Initial guess of horizontal and vertical 
                                    tension at fairlead. Use chain 
                                    throughout """
                                nlim = 50
                                """ Dimensionless catenary parameter """
                                if linexf[j] == 0:            
                                    lambdacat = 1e6
                                elif math.sqrt(linexf[j] ** 2.0 + linezf[j] ** 2.0) >= lineleng[j]:
                                    lambdacat = 0.2
                                else: 
                                    lambdacat = math.sqrt(3.0 * (((lineleng[j] ** 2.0 
                                                - linezf[j] ** 2.0) / linexf[j] ** 2.0) - 1.0))
                                
                                Hf[0] = max(math.fabs(0.5 * omega * linexf[j] 
                                    / lambdacat),tol)  
                                Vf[0] = 0.5 * omega * ((linezf[j] 
                                    / math.tanh(lambdacat)) + lineleng[j]) 
                                xf[0] = linexf[j]
                                zf[0] = linezf[j]
                                HsysloadX = 0.0
                                HsysloadY = 0.0                                

                            elif l >= 1:
                                """ Only allow line lengths to be adjusted during ULS analysis """
                                if limitstate == 'ULS':
                                    nlim = 50
                                elif limitstate == 'ALS':
                                    nlim = 2
                                if jind == 0: 
                                    """ Include static loading """
                                    """ Equivalent water plane area """
                                    self.syswpa = (self._variables.sysvol 
                                                / self._variables.sysdraft)                                    
                                    
                                    """ System reaction loads """   
                                    if l == 1:
                                        """ Static loading only """
                                        Hfsys[l] = math.sqrt(
                                            self.totsysstatloads[0] ** 2.0 
                                            + self.totsysstatloads[1] ** 2.0)
                                        
                                        HsysloadX = -self.totsysstatloads[0]
                                        HsysloadY = -self.totsysstatloads[1]

                                    elif l >= 2:  
                                        """ System reaction loads due to 
                                            static, steady and wave loading """
                                        if self._variables.hs:    
                                            # module_logger.warn('self.totsysstatloads[0] {}'.format(self.totsysstatloads[0]))
                                            # module_logger.warn('self.totsyssteadloads[wc][0] {}'.format(self.totsyssteadloads[wc][0]))
                                            # module_logger.warn('self.meanwavedriftsort[wc][0] {}'.format(self.meanwavedriftsort[wc][0]))
                                            # module_logger.warn('self.totsysstatloads[1] {}'.format(self.totsysstatloads[1]))
                                            # module_logger.warn('self.totsyssteadloads[wc][1]  {}'.format(self.totsyssteadloads[wc][1] ))
                                            # module_logger.warn('self.meanwavedriftsort[wc][1] {}'.format(self.meanwavedriftsort[wc][1]))
                                            # module_logger.warn('self.syswaveloadsort[wc][1] {}'.format(self.syswaveloadsort[wc][1]))
                                            Hfsys[l] = math.sqrt((
                                                self.totsysstatloads[0] 
                                                + self.totsyssteadloads[wc][0]
                                                + self.meanwavedriftsort[wc][0]
                                                + self.syswaveloadsort[wc][0]) ** 2.0 
                                                + (self.totsysstatloads[1] 
                                                + self.totsyssteadloads[wc][1] 
                                                + self.meanwavedriftsort[wc][1]
                                                + self.syswaveloadsort[wc][1]) 
                                                ** 2.0)
                                            HsysloadX = -(self.totsysstatloads[0] 
                                                + self.totsyssteadloads[wc][0]
                                                + self.meanwavedriftsort[wc][0]                                                
                                                + self.syswaveloadsort[wc][0]
                                                + HumbloadX)
                                            HsysloadY = -(self.totsysstatloads[1] 
                                                + self.totsyssteadloads[wc][1] 
                                                + self.meanwavedriftsort[wc][1]
                                                + self.syswaveloadsort[wc][1]
                                                + HumbloadY)                                            
                                        else:
                                            Hfsys[l] = math.sqrt((
                                                self.totsysstatloads[0] 
                                                + self.totsyssteadloads[0]) 
                                                ** 2.0 + (self.totsysstatloads[1] 
                                                + self.totsyssteadloads[1]) 
                                                ** 2.0)
                                            HsysloadX = -(self.totsysstatloads[0] 
                                                + self.totsyssteadloads[0]
                                                + HumbloadX)
                                            HsysloadY = -(self.totsysstatloads[1] 
                                                + self.totsyssteadloads[1]
                                                + HumbloadY)

                                if math.fabs(HsysloadX) < 1e-10:
                                    HsysloadX = 0.0
                                if math.fabs(HsysloadY) < 1e-10:
                                    HsysloadY = 0.0
        
                            dHf = [0 for row in range(klim)]
                            dVf = [0 for row in range(klim)]
                            errxf = [0 for row in range(klim)]
                            errzf = [0 for row in range(klim)] 
                            xfdHf = [0 for row in range(klim)] 
                            zfdHf = [0 for row in range(klim)] 
                            xfdVf = [0 for row in range(klim)] 
                            zfdVf = [0 for row in range(klim)] 
                            VfovHf = [0 for row in range(klim)]
                            VfminlomegaovHf = [0 for row in range(klim)] 
                            linelengbedk = [0 for row in range(klim)] 
                            
                            if (l == 0 and m == 1):
                                
                                sysdraft = self._variables.sysdraft
                                subvol = self._variables.sysvol
                                
                            if l >= 1:
                                
                                if m == 1:
                                    
                                    devdraft = self._variables.sysmass / \
                                        (self._variables.seaden * self.syswpa)
                                        
                                    # Check the given equilibrium draft
                                    draft_diff = abs(self._variables.sysdraft -
                                                                     devdraft)
                                    draft_delta = draft_diff / \
                                                    self._variables.sysdraft
                                    
                                    if draft_delta > 0.05:
                                        
                                        errMsg = ("Calculated draft of {}m "
                                                  "exceeds the given "
                                                  "equilibrium  draft {}m by "
                                                  "{}%. Consider changing the "
                                                  "system mass or submerged "
                                                  "volume").format(
                                                      devdraft,
                                                      self._variables.sysdraft,
                                                      int(draft_delta * 100))
                                        
                                        raise RuntimeError(errMsg)
                                    
                                    umbload = sum(Vflineref[l - 1]) + Vumbload
                                    umbmass = umbload / self._variables.gravity
                                    umbdraft = umbmass / \
                                        (self._variables.seaden * self.syswpa)
                                    
                                    sysdraft = devdraft + umbdraft
                                    
                                    linezf[j] = fairloc[j][2] - sysdraft - \
                                                                 foundloc[j][2]
                                
                                if sysdraft <= 0.0:
                                    subvol = 0.0
                                elif sysdraft > self._variables.sysheight:
                                    subvol = self.syswpa * \
                                                    self._variables.sysheight
                                else: 
                                    subvol = self.syswpa * sysdraft
                                
                                if l >= 2: 
                                        if sysdraft > 0.0:      
                                            """ Note: Vertical steady and wave forces
                                                are expressed as a percentage of device immersion """
                                            if self._variables.hs:
                                                Vfsys[l] = (self._variables.gravity 
                                                            * (self._variables.seaden 
                                                            * subvol
                                                            - self._variables.sysmass)
                                                            + ((self.totsyssteadloads[0][2]
                                                            + self.meanwavedriftsort[wc][2]
                                                            + self.syswaveloadsort[wc][2])
                                                            * sysdraft / self._variables.sysdraft)
                                                            - Vumbload)
                                            else:
                                                Vfsys[l] = (self._variables.gravity 
                                                            * (self._variables.seaden 
                                                            * subvol
                                                            - self._variables.sysmass)
                                                            + (self.totsyssteadloads[2]
                                                            * sysdraft / self._variables.sysdraft)
                                                            - Vumbload)                                                    
                                        else:
                                            Vfsys[l] = (self._variables.gravity 
                                                        * - self._variables.sysmass
                                                        - Vumbload)
                                else:
                                    Vfsys[l] = (self._variables.gravity 
                                                * (self._variables.seaden 
                                                * subvol
                                                - self._variables.sysmass)
                                                - Vumbload) 
                                
                                if (m == 1 and limitstate == 'ULS'): 
                                    xf[0] = linexfref[l-1][j]
                                    zf[0] = linezfref[l-1][j]
                                    Hf[0] = Hflineref[l-1][j]
                                    Vf[0] = Vflineref[l-1][j]
                                
                                if (m == 1 and limitstate == 'ALS'):
                                    xf[0] = initcond[0][j]
                                    zf[0] = initcond[1][j]
                                    Hf[0] = initcond[2][j]
                                    Vf[0] = initcond[3][j]                                 
                                
                                """ If reference values are missing for a particular line 
                                    because the last run was ALS use previous ULS values """
                                if Hfline[j] == 0.0:
                                    Hfline[j] = Hflineref[l-1][j]
                                    Hf[0] = Hfline[j] 
                                if Vfline[j] == 0.0:
                                    Vfline[j] = Vflineref[l-1][j]
                                    if Hloadcheck == 'False':
                                        Vf[0] = Vfline[j]   
                                    elif Hloadcheck == 'True':
                                        Vf[0] = Vfsys[l] / self.numlines 
                                if m > 1:
                                    xf[0] = linexf[j]
                                    zf[0] = linezf[j]
                                    Hf[0] = Hfline[j] 
                                    if Hloadcheck == 'False':
                                        Vf[0] = Vfline[j]   
                                    elif Hloadcheck == 'True':
                                        Vf[0] = Vfsys[l] / self.numlines
                                                                    
                            lineconvfailflag[j] = 'False'
                            
                            if m > 1:               
                                if 'rope' in self.moorcomptab.index: 
                                    """ Applied load as % of MBL """
                                    ropeapploadpc = Hfline[j] * 100.0 / self.moorcomptab.ix[self.ropeeaind,'mbl']
                                    if ropeapploadpc > ropeeavals[1][-1]:
                                        ropeapploadpc = ropeeavals[1][-1]
                                    elif ropeapploadpc < ropeeavals[1][0]:
                                        ropeapploadpc  = ropeeavals[1][0]
                                    
                                    if ropeapploadpc - ropeeavals[1][0] <= 1.0:
                                        """ Forward difference scheme using +2% MBL of target load """
                                        ropestrains = [0.0, 
                                                       ropeeavalsint(ropeapploadpc), 
                                                       ropeeavalsint(ropeapploadpc + 1.0)]
                                        ropeea = math.fabs((((ropeapploadpc + 1.0) - ropeapploadpc) 
                                                            * self.moorcomptab.ix[self.ropeeaind,'mbl']
                                                            / (ropestrains[2] - ropestrains[1])))
                                    elif (ropeapploadpc - ropeeavals[1][0] > 1.0
                                          and ropeeavals[1][-1] - ropeapploadpc > 1.0):
                                        """ Central difference scheme uses values +/- 1% MBL of target load """
                                        ropestrains = [ropeeavalsint(ropeapploadpc - 1.0), 
                                                       ropeeavalsint(ropeapploadpc), 
                                                       ropeeavalsint(ropeapploadpc + 1.0)]
                                        ropeea = math.fabs(((2.0 * self.moorcomptab.ix[self.ropeeaind,'mbl'])
                                                            / (ropestrains[2] - ropestrains[0])))
                                    elif ropeeavals[1][-1] - ropeapploadpc <= 1.0:
                                        """ Backward difference scheme using -2% MBL of target load """
                                        ropestrains = [ropeeavalsint(ropeapploadpc - 1.0),
                                                       ropeeavalsint(ropeapploadpc), 
                                                       0.0]
                                        ropeea = math.fabs(((ropeapploadpc - (ropeapploadpc - 2.0)) 
                                                            * self.moorcomptab.ix[self.ropeeaind,'mbl']
                                                            / (ropestrains[1] - ropestrains[0])))
                                    """ Update average line axial stiffness """
                                    ea = ropeea
                            
                            for n in range(1, nlim):    
                                if lineconvfailflag[j] == 'True':
                                    """ If an individual line fails to converge, reduce line length 
                                    by 0.5%. For catenary lines, the lower limit of line length is 
                                    set as taut configuration """
                                    self.linelengdiff[j] = 0.005 * self.lineleng[j]
                                    self.lineleng[j] = self.lineleng[j] - self.linelengdiff[j]                                    
                                    if self.lineleng[j] < self.tautleng[j]:
                                        self.lineleng[j] = self.tautleng[j]
                                   
                                """ Line QS loop """
                                for k in range(1,klim):  
                                    """ Newton - Raphson scheme using average line 
                                        values """                                      
                                    VfovHf[k] = Vf[k-1] / Hf[k-1]
                                    VfminlomegaovHf[k] = ((Vf[k-1] - lineleng[j] 
                                                        * omega) / Hf[k-1])
                                    if Vf[k-1] < omega * lineleng[j]:                                    
                                        linelengbedk[k] = (lineleng[j] - (Vf[k-1] 
                                                        / omega))
                                        if (linelengbedk[k] - (Hf[k-1] / (friccoef 
                                            * omega))  > 0):
                                            mu = (linelengbedk[k] - (Hf[k-1] 
                                                / (friccoef * omega)))
                                            xf[k] = (linelengbedk[k] + (Hf[k-1] 
                                                / omega) * math.log(VfovHf[k] 
                                                + math.sqrt(1.0 + VfovHf[k] ** 2.0)) 
                                                + (Hf[k-1] * lineleng[j] / ea) 
                                                + (0.5 * friccoef * omega / ea) 
                                                * (mu * (lineleng[j] - (Vf[k-1] 
                                                / omega) - (Hf[k-1] / (friccoef 
                                                * omega))) - (lineleng[j] 
                                                - (Vf[k-1] / omega)) ** 2.0))
                                            xfdHf[k] = ((math.log(math.sqrt(1 
                                                + VfovHf[k] ** 2.0) + VfovHf[k]) 
                                                + ((-VfovHf[k] - VfovHf[k] ** 2.0 
                                                * (1.0 + VfovHf[k] ** 2.0) ** -0.5) 
                                                / (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                + VfovHf[k]))) / omega 
                                                + (lineleng[j] / ea) 
                                                + (friccoef * (Vf[k-1] 
                                                - lineleng[j] * omega) 
                                                + Hf[k-1]) 
                                                / (ea * friccoef * omega))
                                            xfdVf[k] = ((((VfovHf[k] 
                                                * (1.0 + VfovHf[k] ** 2.0) ** -0.5 + 1) 
                                                / (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                + VfovHf[k])) - 1) / omega 
                                                + Hf[k-1] / (ea *omega))
                                            
                                            zf[k] = ((Hf[k-1] / omega) 
                                                * (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                - 1) + (0.5 * Vf[k-1] ** 2.0 
                                                / (ea * omega)) )
                                            zfdHf[k] = (((math.sqrt(1 
                                                + VfovHf[k] ** 2.0) - 1) 
                                                - (VfovHf[k] ** 2.0 * (1 
                                                + VfovHf[k] ** 2.0) ** -0.5)) 
                                                / omega)
                                            zfdVf[k] = (((Vf[k-1] / ea) 
                                                + (VfovHf[k] * (1.0 + VfovHf[k] ** 2.0) 
                                                ** - 0.5)) / omega)
                                        else: 
                                            mu = 0.0
                                            xfdHf[k] = ((math.log(math.sqrt(1 
                                                + VfovHf[k] ** 2.0) + VfovHf[k]) 
                                                + ((-VfovHf[k] - VfovHf[k] ** 2.0 
                                                * (1.0 + VfovHf[k] ** 2.0) ** -0.5) 
                                                / (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                + VfovHf[k]))) / omega 
                                                + (lineleng[j] / ea))
                                            xfdVf[k] = ((((VfovHf[k] * (1 
                                                + VfovHf[k] ** 2.0) ** -0.5 + 1) 
                                                / (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                + VfovHf[k])) - 1) / omega 
                                                + ((friccoef / ea) * (lineleng[j] 
                                                - (Vf[k-1] / omega))))
                                            xf[k] = (linelengbedk[k] + (Hf[k-1] 
                                                / omega) * math.log(VfovHf[k] 
                                                + math.sqrt(1.0 + VfovHf[k] ** 2.0)) 
                                                + (Hf[k-1] * lineleng[j] / ea) 
                                                + (0.5 * friccoef * omega / ea) 
                                                * (mu * (lineleng[j] - (Vf[k-1] 
                                                / omega) - (Hf[k-1] / (friccoef 
                                                * omega))) - (lineleng[j] 
                                                - (Vf[k-1] / omega)) ** 2.0))
                                            zf[k] = ((Hf[k-1] / omega) 
                                                * (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                                - 1) + (0.5 * Vf[k-1] ** 2.0 
                                                / (ea * omega)))
                                            zfdHf[k] = (((math.sqrt(1 
                                                + VfovHf[k] ** 2.0) - 1) 
                                                - (VfovHf[k] ** 2.0 * (1 
                                                + VfovHf[k] ** 2.0) ** -0.5)) 
                                                / omega)
                                            zfdVf[k] = (((Vf[k-1] / ea) 
                                                + (VfovHf[k] * (1.0 + VfovHf[k] ** 2.0) 
                                                ** -0.5)) / omega)
                                            
                                    elif Vf[k-1] >= omega * lineleng[j]:
                                        xf[k] = ((Hf[k-1] / omega) * (math.log(
                                            VfovHf[k] + math.sqrt(1 
                                            + VfovHf[k] ** 2.0)) - math.log(
                                            ((Vf[k-1] - omega * lineleng[j]) 
                                            / Hf[k-1]) + math.sqrt(1.0 + ((Vf[k-1] 
                                            - omega * lineleng[j]) 
                                            / Hf[k-1]) ** 2.0))) + (Hf[k-1] 
                                            * lineleng[j] / ea))
                                        xfdHf[k] = ((math.log(math.sqrt(1 
                                            + VfovHf[k] ** 2.0) + VfovHf[k]) 
                                            - math.log(math.sqrt(1 
                                            + VfminlomegaovHf[k] ** 2.0) 
                                            + VfminlomegaovHf[k]) + ((-VfovHf[k] 
                                            - VfovHf[k] ** 2.0 * (1.0 + VfovHf[k] ** 2.0) 
                                            ** -0.5) / (math.sqrt(1 
                                            + VfovHf[k] ** 2.0) + VfovHf[k])) 
                                            - ((-VfminlomegaovHf[k] 
                                            - VfminlomegaovHf[k] ** 2.0 * (1 
                                            + VfminlomegaovHf[k] ** 2.0) ** -0.5) 
                                            / (math.sqrt(1 
                                            + VfminlomegaovHf[k] ** 2.0) 
                                            + VfminlomegaovHf[k]))) / omega 
                                            + (lineleng[j] / ea))
                                        xfdVf[k] = ((((VfovHf[k] * (1 
                                            + VfovHf[k] ** 2.0) ** -0.5 + 1) 
                                            / (math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                            + VfovHf[k])) - ((VfminlomegaovHf[k] 
                                            * (1.0 + VfminlomegaovHf[k] ** 2.0) ** -0.5 
                                            + 1) / (math.sqrt(1 
                                            + VfminlomegaovHf[k] ** 2.0) 
                                            + VfminlomegaovHf[k]))) / omega)
                                        zf[k] = ((Hf[k-1] / omega) * (math.sqrt(1 
                                            + VfovHf[k] ** 2.0) - math.sqrt(1 
                                            + VfminlomegaovHf[k] ** 2.0)) + (1 / ea) 
                                            * (Vf[k-1] * lineleng[j] - 0.5 * omega 
                                            * lineleng[j] ** 2.0))
                                        zfdHf[k] = ((math.sqrt(1.0 + VfovHf[k] ** 2.0) 
                                            - math.sqrt(1 
                                            + VfminlomegaovHf[k] ** 2.0) 
                                            + (VfminlomegaovHf[k] ** 2.0 * (1 
                                            + VfminlomegaovHf[k] ** 2.0) ** -0.5 
                                            - VfovHf[k] ** 2.0 * (1.0 + VfovHf[k] ** 2.0) 
                                            ** -0.5)) / omega)
                                        zfdVf[k] = ((VfovHf[k] * (1
                                            + VfovHf[k] ** 2.0) ** -0.5 
                                            - (VfminlomegaovHf[k] * (1 
                                            + VfminlomegaovHf[k] ** 2.0) ** -0.5)) 
                                            / omega + (lineleng[j] / ea))
                    
                                    errxf[k] = xf[k] - linexf[j]
                                    errzf[k] = zf[k] - linezf[j]
                                    
                                    """ Jacobian matrix """
                                    jac = np.array([[xfdHf[k], xfdVf[k]], 
                                                    [zfdHf[k], zfdVf[k]]])
                                    det = np.linalg.det(jac)
                                    
                                    if abs(det) < 1e-8:
                                        update = np.array([[0.0], [0.0]])
                                    else:
                                        x = np.array([[-xf[k]], [-zf[k]]])
                                        update = np.linalg.solve(jac, x)
                                    
                                    dHf[k] = update.item(0)
                                    dVf[k] = update.item(1)          
                                    
                                    """ Avoid negative horizontal fairlead loads """
                                    if Hf[k-1] < 0.0:
                                        Hf[k-1] = 0.0
                                                                         
                                    dHf[k] = (-zfdVf[k] * errxf[k] 
                                                + xfdVf[k] * errzf[k])/det   
                                    dVf[k] = (zfdHf[k] * errxf[k] 
                                                - xfdHf[k] * errzf[k])/det                                             
                                    
                                    Hf[k] = dHf[k] + Hf[k-1]
                                    Vf[k] = dVf[k] + Vf[k-1]    
                                    
                                    if Vf[k] < omega * lineleng[j]:
                                        """ Part of line is resting on seafloor """
                                        Ha[k] = max(Hf[k] - friccoef * omega 
                                            * linelengbedk[k], 0.0)
                                        Va[k] = 0.0                                        
                                    elif Vf[k] >= omega * lineleng[j]:
                                        """ Line fully suspended """ 
                                        Ha[k] = Hf[k]
                                        Va[k] = Vf[k] - omega * lineleng[j]                                         
                                        
                                    """ Break condition once convergence has been achieved """                        
                                    if (math.fabs((Hf[k] - Hf[k-1]) / Hf[k]) < tol 
                                        and math.fabs((Vf[k] - Vf[k-1]) / Vf[k]) 
                                        < tol):
                                        """ Displaced line angle from y-axis """
                                        if (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] < 0.0 
                                            and fairloc[j][1] + syspos[1] 
                                            - foundloc[j][1] < 0.0):
                                            lineangdisp[j] = math.atan(
                                                (fairloc[j][0] + syspos[0] 
                                                - foundloc[j][0]) / (fairloc[j][1] 
                                                + syspos[1] - foundloc[j][1])) 
                                                                           
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] < 0.0 and fairloc[j][1] 
                                            + syspos[1] - foundloc[j][1] > 0.0):
                                            lineangdisp[j] = (math.pi / 2.0 
                                                - math.atan((fairloc[j][1] 
                                                + syspos[1] - foundloc[j][1]) 
                                                / (fairloc[j][0] + syspos[0] 
                                                - foundloc[j][0])))
                                                
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] > 0.0 
                                            and fairloc[j][1] + syspos[1]   
                                            - foundloc[j][1] > 0.0):
                                            lineangdisp[j] = (math.pi + math.atan(
                                                (fairloc[j][0] + syspos[0] 
                                                - foundloc[j][0]) / (fairloc[j][1] 
                                                + syspos[1] - foundloc[j][1])))
                                            
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] > 0.0 
                                            and fairloc[j][1] + syspos[1] 
                                            - foundloc[j][1] < 0.0):
                                            lineangdisp[j] = (1.5 * math.pi 
                                                - math.atan((fairloc[j][1] 
                                                + syspos[1] - foundloc[j][1]) 
                                                / (fairloc[j][0] + syspos[0] 
                                                - foundloc[j][0])))       
                                            
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] == 0.0 
                                            and fairloc[j][1] + syspos[1] 
                                            - foundloc[j][1] < 0.0):
                                            lineangdisp[j] = 0.0
                                            
                                            
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] < 0.0 
                                            and fairloc[j][1] + syspos[1] 
                                            - foundloc[j][1] == 0.0):
                                            lineangdisp[j] = math.pi / 2.0
                                            
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] == 0.0 and fairloc[j][1] 
                                            + syspos[1] - foundloc[j][1] > 0.0):
                                            lineangdisp[j] = math.pi
                                            
                                        elif (fairloc[j][0] + syspos[0] 
                                            - foundloc[j][0] > 0.0 
                                            and fairloc[j][1] + syspos[1] 
                                            - foundloc[j][1] == 0.0):
                                            lineangdisp[j] = 1.5 * math.pi
                                        
                                        HflineX[j] = (Hf[k] * math.sin(
                                                    lineangdisp[j]))
                                        HflineY[j] = (Hf[k] * math.cos(
                                                    lineangdisp[j]))
                                        Vfline[j] = Vf[k]
                                        Hfline[j] = Hf[k]
                                        Valine[j] = Va[k]
                                        Haline[j] = Ha[k]                                                                                   
                                        lineconvfailflag[j] = 'False'
                                        
                                        if (l in (0,1) and limitstate == 'ULS' and linelengbedk[k] != self.linelengbed[j]):
                                            self.linelengbed[j] = linelengbedk[k]   
                                        break
                                    
                                    if k == klim - 1:   
                                        if l == 0:
                                            """ Line-only system failed to converge """
                                            lineconvfailflag[j] = 'True'
                                            linelengchangeflag[j] = 'True'
                                        continue  
                                if lineconvfailflag[j] == 'False':
                                    self.lineleng[j] = lineleng[j]                                    
                                    break
                            
                            fairten[j] = [Hfline[j], Vfline[j]]

                            """ Anchor tensions in x- and y-axis and line tensions """
                            if l <= 2:
                                ancten[l+wc][j] = [-Haline[j] * math.sin(
                                    lineangdisp[j]), -Haline[j] * math.cos(
                                    lineangdisp[j]), Valine[j]]
                                lineten[l+wc][j] = math.sqrt(Hfline[j] ** 2.0 
                                                            + Vfline[j] ** 2.0) 
                                linetenind = l + wc
                            elif l in (3,4):
                                if l == 3:
                                    self.wlevlowflag = 'False'
                                    self.linetenhigh[2+wc][j] = math.sqrt(Hfline[j] ** 2.0 
                                                                + Vfline[j] ** 2.0)
                                    self.anctenhigh[2+wc][j] = [-Haline[j] * math.sin(
                                                                lineangdisp[j]), -Haline[j] * math.cos(
                                                                lineangdisp[j]), Valine[j]]
                                elif l == 4:
                                    self.wlevlowflag = 'True'
                                    self.linetenlow[2+wc][j] = math.sqrt(Hfline[j] ** 2.0 
                                                                + Vfline[j] ** 2.0)
                                    self.anctenlow[2+wc][j] = [-Haline[j] * math.sin(
                                                                lineangdisp[j]), -Haline[j] * math.cos(
                                                                lineangdisp[j]), Valine[j]]
                                """ Replace line and anchor tensions if higher with umbilical """
                                if (math.sqrt((-Haline[j] * math.sin(
                                    lineangdisp[j])) ** 2.0 + (-Haline[j] * math.cos(
                                    lineangdisp[j])) ** 2.0 + Valine[j] ** 2.0) > ancten[2+wc][j]):   
                                    
                                    ancten[2+wc][j] = [-Haline[j] * math.sin(
                                        lineangdisp[j]), -Haline[j] * math.cos(
                                        lineangdisp[j]), Valine[j]]
                                else:
                                    ancten[2+wc][j] = ancten[2+wc][j] 
                                if (math.sqrt(Hfline[j] ** 2.0 + Vfline[j] ** 2.0
                                    > lineten[2+wc][j])):
                                    lineten[2+wc][j] = math.sqrt(Hfline[j] ** 2.0 
                                                                + Vfline[j] ** 2.0) 
                                else:
                                    lineten[2+wc][j] = lineten[2+wc][j]
                                linetenind = 2 + wc 
                        
                        # """ If calculated  line tensions are higher than the capacity of 
                        # the components abort run """                        
                        if (l >= 1 and  m > 100 and min(self.moorcomptab['mbl'].tolist()) < self.moorsf 
                            * max(lineten[linetenind])):
                                logmsg = [""]
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                logmsg.append('Component MBL exceeded!')
                                logmsg.append('Maximum line tension (including FoS) = {}'.format(self.moorsf * max(lineten[linetenind])))
                                logmsg.append('Minimum component MBL = {}'.format(min(self.moorcomptab['mbl'].tolist())))
                                compexceedflag = 'True'
                                self.sysposfail = (limitstate, [syspos[0], syspos[1], sysdraft])
                                self.initcondfail = [linexf, linezf, Hfline, Vfline]
                                logmsg.append('System position at failure = {}'.format(self.sysposfail))
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')    
                                module_logger.info("\n".join(logmsg))
                                    
                                break
                            
                        if l >= 1 and m > 10:
                            
                            sysdis = math.sqrt(syspos[0] ** 2.0 +
                                                           syspos[1] ** 2.0)
                            maxdis = max(self.maxdisp[:2])
                            
                            alstest = max(self.mindevdist - maxdis,
                                          0.5 * self.mindevdist)
                            
                            # A check is carried out to determine if the device
                            # vertical position exceeds the user-specified 
                            # displacement limits for ULS and ALS tests
                            if syspos[2] > self.maxdisp[2]:

                                logmsg = [""]
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!')
                                logmsg.append('Device vertical displacement '
                                              'limit {} exceeded!'.format(
                                                            self.maxdisp[2]))
                                self.dispexceedflag = 'True'
                                self.sysposfail = (limitstate, [syspos[0],
                                                                syspos[1],
                                                                sysdraft])
                                self.initcondfail = [linexf,
                                                     linezf,
                                                     Hfline,
                                                     Vfline]
                                logmsg.append('System position at failure = '
                                              '{}'.format(self.sysposfail))
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!!')    
                                module_logger.info("\n".join(logmsg))
                                self.moordesfail = 'True'
                                break
                            
                            # For ULS tests, ensure the device does not exceed
                            # the given horizontal displacements
                            if (limitstate == 'ULS' and
                                (syspos[0] > self.maxdisp[0] or 
                                 syspos[1] > self.maxdisp[1])):
                                

                                logmsg = [""]
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!')
                                logmsg.append('Device horizontal displacement '
                                              'limit exceeded!')
                                self.dispexceedflag = 'True'
                                self.sysposfail = (limitstate, [syspos[0],
                                                                syspos[1],
                                                                sysdraft])
                                self.initcondfail = [linexf,
                                                     linezf,
                                                     Hfline,
                                                     Vfline]
                                logmsg.append('System position at failure = '
                                              '{}'.format(self.sysposfail))
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!!')    
                                module_logger.info("\n".join(logmsg))
                                self.moordesfail = 'True'
                                break
                            
                            # For the ALS test ensure that the device can not
                            # foul the closest device if it were at its maximum
                            # extension
                            if (limitstate == 'ALS' and
                                sysdis > alstest):
                                
                                logmsg = [""]
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!!')
                                logmsg.append('Device displacement exceeds '
                                              'minimum separation of '
                                              '{}m!'.format(alstest))
                                
                                self.dispexceedflag = 'True'
                                self.sysposfail = (limitstate, [syspos[0],
                                                                syspos[1],
                                                                sysdraft])
                                self.initcondfail = [linexf,
                                                     linezf,
                                                     Hfline,
                                                     Vfline]
                                
                                logmsg.append('System position at failure = '
                                              '{}'.format(self.sysposfail))
                                logmsg.append('Minimum separation = '
                                              '{}'.format(self.mindevdist))
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                                              '!!!!!!!!!!!!!!!!!!')    
                                module_logger.info("\n".join(logmsg))
                                self.moordesfail = 'True'
                                break

                        """ Equilibrium load tolerance """
                        if self.selmoortyp == 'catenary':
                            if limitstate == 'ULS':
                                if l in (0, 1):
                                    loadtol = 1.0e3
                                else:
                                    """ Load tolerance is 2% of pretension at equilibrium """
                                    loadtol = max(0.02 * np.mean(lineten[1]), 1.0e3) 
                            elif limitstate == 'ALS': 
                                if l in (0, 1):
                                    loadtol = 1.0e3
                                else:
                                    """ 2% of maximum line tension from ULS """
                                    loadtol = max(0.02 * np.mean(lineten[1]), 1.0e3)
                        elif self.selmoortyp == 'taut':
                            if limitstate == 'ULS':
                                if l in (0, 1):
                                    loadtol = 1.0e3
                                else:                                         
                                    """ Load tolerance is 2% of pretension at equilibrium """
                                    loadtol = max(0.02 * np.mean(lineten[1]), 1.0e3)
                            elif limitstate == 'ALS': 
                                """ 5% of maximum line tension from ULS """
                                loadtol = max(0.02 * np.mean(lineten[1]), 1.0e3)
                                                    
                        """ Horizontal and vertical load difference magnitudes """
                        Hloaddiffmag.append(math.sqrt((HsysloadX - sum(HflineX)) ** 2.0 
                                + (HsysloadY - sum(HflineY) - HumbloadY) ** 2.0))
                        Vloaddiffmag.append(math.fabs(Vfsys[l] - sum(Vfline)))
                        """ Angle from vertical axis """  
                        if (HsysloadY - sum(HflineY)) == 0.0:
                            errang = 90.0 * math.pi / 180.0
                        else:
                            errang = math.atan((HsysloadX - sum(HflineX)) 
                                            / (HsysloadY - sum(HflineY)))  
                        sysposrec.append([syspos[0], syspos[1]])
                        
                        if m == 1:
                            dispang = 0.0
                        elif m >= 2:                       
                            sysposrecdiff.append(math.sqrt((sysposrec[m-1][0]-sysposrec[m-2][0]) ** 2.0 
                                                    + (sysposrec[m-1][1]-sysposrec[m-2][1]) ** 2.0))
                            if (sysposrec[m-1][0] - sysposrec[m-2][0] > 0.0 
                                and sysposrec[m-1][1] - sysposrec[m-2][1] > 0.0):
                                dispang = math.atan(math.fabs(sysposrec[m-1][0] - sysposrec[m-2][0]) 
                                                    / math.fabs(sysposrec[m-1][1] - sysposrec[m-2][1]))
                            elif (sysposrec[m-1][0] - sysposrec[m-2][0] > 0.0 
                                and sysposrec[m-1][1] - sysposrec[m-2][1] < 0.0):
                                dispang = (math.atan(math.fabs(sysposrec[m-1][1] - sysposrec[m-2][1])
                                                    / math.fabs(sysposrec[m-1][0] - sysposrec[m-2][0]))
                                                    + math.pi / 2.0)
                            elif (sysposrec[m-1][0] - sysposrec[m-2][0] < 0.0 
                                and sysposrec[m-1][1] - sysposrec[m-2][1] < 0.0):
                                dispang = (math.atan(math.fabs(sysposrec[m-1][0] - sysposrec[m-2][0]) 
                                                    / math.fabs(sysposrec[m-1][1] - sysposrec[m-2][1])) 
                                                    + math.pi)
                            elif (sysposrec[m-1][0] - sysposrec[m-2][0] < 0.0 
                                and sysposrec[m-1][1] - sysposrec[m-2][1] > 0.0):
                                dispang = (math.atan(math.fabs(sysposrec[m-1][1] - sysposrec[m-2][1])
                                                    / math.fabs(sysposrec[m-1][0] - sysposrec[m-2][0]))
                                                    + 1.5 * math.pi) 
                        dispangrec.append(dispang)
                        errangrec.append(errang)
                        """ For very small load differences """
                        if math.fabs(errang) < 1.5e-3:
                            errang = 0.0 
                        sysposX.append(syspos[0])
                        sysposY.append(syspos[1])                     
                        if Hloaddiffmag[m] > loadtol:                                 
                            if horzoscflag == 'False':
                                if self.selmoortyp == 'catenary':
                                    if limitstate == 'ULS':
                                        hsysdisp = math.fabs(1.0 - math.exp(1.0e-6 * Hloaddiffmag[m]))
                                    elif limitstate == 'ALS':
                                        hsysdisp = math.fabs(1.0 - math.exp(1.0e-5 * Hloaddiffmag[m]))
                                elif self.selmoortyp == 'taut':
                                    if limitstate == 'ULS':
                                        if l in (0, 1):
                                            hsysdisp = math.fabs(1.0 - math.exp(1.0e-6 * Hloaddiffmag[m]))
                                        else:
                                            hsysdisp = math.fabs(1.0 - math.exp(1.0e-5 * Hloaddiffmag[m]))
                                    elif limitstate == 'ALS':
                                        hsysdisp = math.fabs(1.0 - math.exp(1.0e-5 * Hloaddiffmag[m]))
                            
                            if math.fabs(hsysdisp) > 1.0:
                                hsysdisp = 1.0
                                
                            """ If horizontal position is oscillating or increasing """
                            if (m > 5 and ((Hloaddiffmag[m] - Hloaddiffmag[m-1] < 0.0 
                                and Hloaddiffmag[m-1] - Hloaddiffmag[m-2] > 0.0
                                and Hloaddiffmag[m-2] - Hloaddiffmag[m-3] < 0.0) 
                                or (Hloaddiffmag[m] - Hloaddiffmag[m-1] > 0.0 
                                and Hloaddiffmag[m-1] - Hloaddiffmag[m-2] < 0.0
                                and Hloaddiffmag[m-2] - Hloaddiffmag[m-3] > 0.0)
                                or (Hloaddiffmag[m] - Hloaddiffmag[m-1] > 0.0 
                                and Hloaddiffmag[m-1] - Hloaddiffmag[m-2] > 0.0
                                and Hloaddiffmag[m-2] - Hloaddiffmag[m-3] < 0.0
                                and Hloaddiffmag[m-3] - Hloaddiffmag[m-4] < 0.0)
                                or (Hloaddiffmag[m] - Hloaddiffmag[m-1] < 0.0 
                                and Hloaddiffmag[m-1] - Hloaddiffmag[m-2] < 0.0
                                and Hloaddiffmag[m-2] - Hloaddiffmag[m-3] > 0.0
                                and Hloaddiffmag[m-3] - Hloaddiffmag[m-4] > 0.0)
                                or Hloaddiffmag[m] - Hloaddiffmag[m-1] > 0.0
                                or math.fabs(errangrec[m] - errangrec[m-1]) > math.pi / 2.0
                                or math.fabs(dispangrec[m]-dispangrec[m-1]) > math.pi / 2.0
                                or sum(sysposrecdiff[m-5:]) / len(sysposrecdiff[m-5:]) < 1.0e-3)):                                
                                
                                # if math.fabs(dispangrec[m]-dispangrec[m-1]) > math.pi / 2.0:
                                horzoscflag = 'True'
                                hsysdisp = hsysdisp / 2.0

                                hosccount = hosccount + 1
                                
                                if hosccount > 500:
                                    Hloadcheck = 'True' 
                                
                            if self.selmoortyp == 'catenary':
                                if math.fabs(hsysdisp) < 1.0e-3:
                                    hsysdisp = 1.0e-3 
                            elif self.selmoortyp == 'taut':
                                if math.fabs(hsysdisp) < 1.0e-3:
                                    hsysdisp = 1.0e-3 
                            
                            if round(HsysloadX - sum(HflineX), 3) > 0.0:  
                                syspos[0] = syspos[0] - hsysdisp * math.sin(
                                    math.fabs(errang))
                                # print ['move -x', sysdisp]                                
                            if round(HsysloadY - sum(HflineY), 3) > 0.0:
                                syspos[1] = syspos[1] - hsysdisp * math.cos(
                                    math.fabs(errang))
                                # print ['move -y', sysdisp]
                            if round(HsysloadX - sum(HflineX), 3) < 0.0:                                
                                syspos[0] = syspos[0] + hsysdisp * math.sin(
                                    math.fabs(errang))
                                # print ['move +x', sysdisp]
                            if round(HsysloadY - sum(HflineY), 3) < 0.0:
                                syspos[1] = syspos[1] + hsysdisp * math.cos(
                                    math.fabs(errang))
                            syspos[2] = sysdraft

                        else:
                            Hloadcheck = 'True'                           
                                            
                        if l == 0:
                            Vloadcheck = 'True'
                        elif l >= 1:
                            if Hloadcheck == 'True':
                                if Vloaddiffmag[m] > loadtol:
                                    if Vloaddiffmag[m] > 1.0e6:
                                        vsysdisp  = Vloaddiffmag[m] / 10e6
                                    elif (Vloaddiffmag[m] > 100.0e3 and Vloaddiffmag[m] <= 1.0e6):
                                        vsysdisp  = Vloaddiffmag[m] / 20e6
                                    elif (Vloaddiffmag[m] > 10.0e3 and Vloaddiffmag[m] <= 100.0e3):
                                        vsysdisp  = Vloaddiffmag[m] / 50e6
                                    elif Vloaddiffmag[m] <= 10.0e3:
                                        vsysdisp  = Vloaddiffmag[m] / 100e6                                    
                                    if round(Vfsys[l] - sum(Vfline), 3) > 0.0:                              
                                        sysdraft = sysdraft - vsysdisp 
                                        # print ['move +z', sysdisp]                                
                                    if round(Vfsys[l] - sum(Vfline), 3) < 0.0:
                                        sysdraft = sysdraft + vsysdisp 
                                        # print ['move -z', sysdisp]  
                                   
                                    if (m > 50 and (Vloaddiffmag[m] - Vloaddiffmag[m-1] < 0.0 
                                        and Vloaddiffmag[m-1] - Vloaddiffmag[m-2] > 0.0
                                        and Vloaddiffmag[m-2] - Vloaddiffmag[m-3] < 0.0) 
                                        or (Vloaddiffmag[m] - Vloaddiffmag[m-1] > 0.0 
                                        and Vloaddiffmag[m-1] - Vloaddiffmag[m-2] < 0.0
                                        and Vloaddiffmag[m-2] - Vloaddiffmag[m-3] > 0.0)
                                        or (Vloaddiffmag[m] - Vloaddiffmag[m-1] > 0.0 
                                        and Vloaddiffmag[m-1] - Vloaddiffmag[m-2] > 0.0
                                        and Vloaddiffmag[m-2] - Vloaddiffmag[m-3] < 0.0
                                        and Vloaddiffmag[m-3] - Vloaddiffmag[m-4] < 0.0)
                                        or (Vloaddiffmag[m] - Vloaddiffmag[m-1] < 0.0 
                                        and Vloaddiffmag[m-1] - Vloaddiffmag[m-2] < 0.0
                                        and Vloaddiffmag[m-2] - Vloaddiffmag[m-3] > 0.0
                                        and Vloaddiffmag[m-3] - Vloaddiffmag[m-4] > 0.0)):
                                        vertoscflag = 'True'
                                        vosccount = vosccount + 1
                                        
                                        if vosccount > 10:
                                            Vloadcheck = 'True' 
                                        else:
                                            vertoscflag = 'False'
                                        if vertoscflag == 'False':
                                            if self.selmoortyp == 'catenary':
                                                if math.fabs(vsysdisp) < 1.0e-4:
                                                    vsysdisp = 1.0e-4
                                            elif self.selmoortyp == 'taut':
                                                if math.fabs(vsysdisp) < 0.1e-5:
                                                    vsysdisp = 0.1e-5
                                        elif vertoscflag == 'True':
                                            if self.selmoortyp == 'catenary':
                                                if math.fabs(vsysdisp) < 1.0e-5:
                                                    vsysdisp = 1.0e-5
                                            elif self.selmoortyp == 'taut':
                                                if math.fabs(vsysdisp) < 0.1e-6:
                                                    vsysdisp = 0.1e-6

                                else:
                                    Vloadcheck = 'True'    

                        loaddiffmagrec.append([m, copy.deepcopy(syspos), Hloaddiffmag[m], Vloaddiffmag[m], round(Vfsys[l] - sum(Vfline), 3), sysdraft])
                        
                        
                        if (Hloadcheck == 'True' and Vloadcheck == 'True'): 

                            
                            if l in (2,3):
                                for j in analines:    
                                    logmsg = [""]                       
                                    logmsg.append('------------------------------------------------------------')                                
                                    logmsg.append('Mooring line ' + str(j) + ' average weight per unit length [dry[N/m], wet[N/m]] and axial stiffness [N] {}'.format([omegadry, omega, ea]))                                
                                    logmsg.append('------------------------------------------------------------')                                
                                    module_logger.info("\n".join(logmsg))
                           
                                
                            # self.sysposfail = []
                            """ Global system position with any rotation applied """
                            sysposglobrot = [(syspos[0] * math.cos(-self._variables.sysorienang 
                                            * math.pi / 180.0) - syspos[1] 
                                            * math.sin(-self._variables.sysorienang * math.pi 
                                            / 180.0)) + self._variables.sysorig[deviceid][0], 
                                            (syspos[0] * math.sin(-self._variables.sysorienang 
                                            * math.pi / 180.0) + syspos[1] 
                                            * math.cos(-self._variables.sysorienang * math.pi 
                                            / 180.0)) + self._variables.sysorig[deviceid][1],
                                            -syspos[2]]
                            logmsg = [""]
                            logmsg.append(('Draft equilibrium at {} reached '
                                           'after {} run(s)').format(sysdraft,
                                                                     m))
                            logmsg.append(('New local system position (device frame of reference) at '
                                           '{}').format(syspos))
                            logmsg.append(('New global system position (with rotation) at '
                                           '{}').format(sysposglobrot))
                            module_logger.info("\n".join(logmsg))
                                
                            if (l <= 3 and limitstate == 'ULS'):                                
                                linexfref[l] = copy.deepcopy(linexf)
                                linezfref[l] = copy.deepcopy(linezf)
                                Hflineref[l] = copy.deepcopy(Hfline)
                                Vflineref[l] = copy.deepcopy(Vfline)                            
                                            
                            if (l == 1 and limitstate == 'ULS'):
                                self.loadlim = max(lineten[1])                                 
                                initcond = [linexfref[l], 
                                            linezfref[l], 
                                            Hflineref[l], 
                                            Vflineref[l]]  
                            break
                        
                            
                        """ Update linexf and linezf for new device positions """
                        for j in analines:                            
                            linexf[j] = math.sqrt((fairloc[j][0] 
                                + syspos[0] - foundloc[j][0]) ** 2.0
                                + (fairloc[j][1] + syspos[1] 
                                - foundloc[j][1]) ** 2.0)
                            linezf[j] = ((fairloc[j][2]  - (sysdraft 
                                    - self._variables.sysdraft)) 
                                    - foundloc[j][2]) 
                                    
                        if m == mlim:
                            logmsg = ('Position not converged: [Hloadcheck, '
                                      'Vloadcheck] {}').format([Hloadcheck,
                                                                Vloadcheck])
                            module_logger.info(logmsg)
                            continue
                            
                        if limitstate == 'ALS':                            
                            ancten[0] = [[0.0, 0.0, 0.0] for row in range(0, self.numlines)]   
                    if l == 1:
                        sysposrefnoenv = copy.deepcopy(syspos)    
                        # if (self.rn == 0 and limitstate == 'ULS'):
                            # self.linelengbedref = copy.deepcopy(self.linelengbed)  
                    """ Final system positions relative to local device origin """        
                    finalsyspos[l+wc][0:2] = syspos[0:2]
                    finalsyspos[l+wc][2] = sysdraft
                    if l >= 2:
                        logmsg = [""]
                        logmsg.append('_________________________________________________________________________') 
                        logmsg.append('System applied loads [HsysloadX, HsysloadY, Vfsys] {}'.format([float(-(HsysloadX + HumbloadX)), 
                                                                                                      float(-(HsysloadY + HumbloadY)), 
                                                                                                      Vfsys[l] - (self._variables.gravity 
                                                                                                        * (self._variables.seaden 
                                                                                                        * subvol
                                                                                                        - self._variables.sysmass)) + Vumbload]))
                        logmsg.append('_________________________________________________________________________')            
                        module_logger.info("\n".join(logmsg))
            
                                
            return lineten, fairten, ancten, initcond, finalsyspos, umbcheck
                
    
        """ There will be two design runs in the Release Candidate -
            0) Chain only system, 
            1) Chain-rope system or user defined line configuration """        
        for self.rn in range(0,2):
        
            logmsg = [""]
            logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logmsg.append('Configuration {}'.format(self.rn))
            logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            module_logger.info("\n".join(logmsg))
                       
            self.quanfound = len(self.foundlocglob) - 1            
            self.moordesfail = 'False'
            self.sysposfail = []
            self.numlines = len(self.foundlocglob) - 1
            self.fairloc = [[0 for col in range(3)] for row 
                                    in range(0, self.numlines)]  
            if self.rn == 0:            
                compblocks = ['shackle001',
                              'chain',
                              'shackle002']
                continue              
            elif self.rn == 1: 
                if self._variables.preline:
                    compblocks = self._variables.preline
                else:
                    if self.selmoortyp == 'catenary': 
                        compblocks = ['forerunner assembly', 
                                      'shackle001', 
                                      'chain', 
                                      'shackle002', 
                                      'rope', 
                                      'shackle003', 
                                      'swivel', 
                                      'shackle004']
                    elif self.selmoortyp == 'taut':
                        compblocks = ['shackle001', 
                                      'rope', 
                                      'shackle002', 
                                      'swivel', 
                                      'shackle003']
            
            def moorcompret(block,complim):          
                complist = []
                if k == 0:
                    if l == 0:
                        for comps in self._variables.compdict:                             
                            if self._variables.compdict[comps]['item1'] == 'mooring system':
                                """ Start with component at anchor end """
                                if (self._variables.compdict[comps]['item2'] 
                                    == block 
                                    and 
                                    self._variables.compdict[comps]['item6'][0] 
                                    >= complim[0]):
                                    complist.append((comps,
                                        self._variables.compdict[comps]['item6'][0],
                                        self._variables.compdict[comps]['item11']))                                    
                        """ Sort to find lowest cost """
                        complistsort = sorted(complist, 
                                                 key=operator.itemgetter(1))                                                 
                        """ complim['size','mbl'] """  
                        complim[0] = complistsort[0][1]                        
                        compsminsize = [x for x in complistsort 
                                        if x[1] == complim[0]]
                        selcomp = min(compsminsize, key=operator.itemgetter(2))
                        """ If the same size component cannot be found look for 
                            larger size components """                
                        if not complist:                            
                            for comps in self._variables.compdict:
                                if (self._variables.compdict[comps]['item1'] 
                                    == 'mooring system'):
                                    if (self._variables.compdict[comps]['item2'] 
                                        == block 
                                        and 
                                        self._variables.compdict[comps]['item5'][0] 
                                        > complim[1] 
                                        and 
                                        self._variables.compdict[comps]['item6'][0] 
                                        > complim[0]):
                                        complist.append((comps,
                                            self._variables.compdict[comps]['item6'][0],
                                            self._variables.compdict[comps]['item11']))
                            if complist:                        
                                """ Sort to find lowest cost and then (because multiple 
                                    grades may exist), the smallest diameter """
                                complistsort = sorted(complist, key=operator.itemgetter(2,1)) 
                                selcomp = complistsort[0][0]
                            else: 
                                selcomp = 0
                    else:  
                        """ Find components with the same connecting size """
                        for comps in self._variables.compdict:                             
                            if self._variables.compdict[comps]['item1'] == 'mooring system':
                                if (self._variables.compdict[comps]['item2'] 
                                    == block 
                                    and 
                                    self._variables.compdict[comps]['item6'][0] 
                                    == complim[0]):
                                                                        
                                    if block == 'rope':
                                        """ If the component in question is a rope, sort by lowest 
                                            axial stiffness instead of cost """                                        
                                        ropeea = (self._variables.compdict[comps]['item5'][0] 
                                                * (self._variables.compdict[comps]['item5'][1][-1][1] 
                                                - self._variables.compdict[comps]['item5'][1][0][1]) 
                                                / (self._variables.compdict[comps]['item5'][1][-1][0] 
                                                - self._variables.compdict[comps]['item5'][1][0][0])) 
                                        complist.append((comps,
                                                    self._variables.compdict[comps]['item6'][0],
                                                   ropeea))
                                    else:
                                        complist.append((comps,
                                                    self._variables.compdict[comps]['item6'][0],
                                                    self._variables.compdict[comps]['item11']))
                        """ Sort to find lowest cost (or axial stiffness in the case of ropes) 
                        and then (because multiple grades may exist), smallest diameter """
                        complistsort = sorted(complist, 
                                          key=operator.itemgetter(2,1))                          
                        
                        """ If the same size component cannot be found look for 
                            larger size components """                
                        if not complist:                            
                            for comps in self._variables.compdict:
                                if (self._variables.compdict[comps]['item1'] 
                                    == 'mooring system'):
                                    if (self._variables.compdict[comps]['item2'] 
                                        == block 
                                        and 
                                        self._variables.compdict[comps]['item5'][0] 
                                        > complim[1] 
                                        and 
                                        self._variables.compdict[comps]['item6'][0] 
                                        > complim[0]):
                                        if block == 'rope':
                                            """ If the component in question is a rope, sort by lowest 
                                                axial stiffness instead of cost """                                        
                                            ropeea = (self._variables.compdict[comps]['item5'][0] 
                                                    * (self._variables.compdict[comps]['item5'][1][-1][1] 
                                                    - self._variables.compdict[comps]['item5'][1][0][1]) 
                                                    / (self._variables.compdict[comps]['item5'][1][-1][0] 
                                                    - self._variables.compdict[comps]['item5'][1][0][0])) 
                                            complist.append((comps,
                                                        self._variables.compdict[comps]['item6'][0],
                                                       ropeea))
                                        else:
                                            complist.append((comps,
                                                        self._variables.compdict[comps]['item6'][0],
                                                        self._variables.compdict[comps]['item11']))
                            if complist:                        
                                """ Sort to find lowest cost and then (because multiple 
                                    grades may exist), the smallest diameter """
                                complistsort = sorted(complist, key=operator.itemgetter(2,1)) 
                                selcomp = complistsort[0][0]
                            else: 
                                selcomp = 0
                        if complist:                        
                            """ Sort to find lowest cost and then (because multiple 
                                grades may exist), the smallest diameter """
                            complistsort = sorted(complist, key=operator.itemgetter(2,1)) 
                            selcomp = complistsort[0][0]
                        else: 
                            selcomp = 0
                elif k > 0:                    
                    if not complist:
                        """ If component MBL is exceeded, first search for a 
                            higher capacity component with the same connecting 
                            size """
                        for comps in self._variables.compdict:
                            if (self._variables.compdict[comps]['item1'] 
                                == 'mooring system'):
                                if (self._variables.compdict[comps]['item2'] 
                                    == block 
                                    and self._variables.compdict[comps]['item5'][0] 
                                    > complim[1] 
                                    and self._variables.compdict[comps]['item6'][0] 
                                    == complim[0]):
                                    if block == 'rope':
                                        """ If the component in question is a rope, sort by lowest 
                                            axial stiffness instead of cost """                                        
                                        ropeea = (self._variables.compdict[comps]['item5'][0] 
                                                * (self._variables.compdict[comps]['item5'][1][-1][1] 
                                                - self._variables.compdict[comps]['item5'][1][0][1]) 
                                                / (self._variables.compdict[comps]['item5'][1][-1][0] 
                                                - self._variables.compdict[comps]['item5'][1][0][0])) 
                                        complist.append((comps,
                                                    self._variables.compdict[comps]['item6'][0],
                                                   ropeea))
                                    else:
                                        complist.append((comps,
                                                    self._variables.compdict[comps]['item6'][0],
                                                    self._variables.compdict[comps]['item11']))
                        
                        """ If the same size component cannot be found look for 
                            larger size components """                
                        if not complist:                            
                            for comps in self._variables.compdict:
                                if (self._variables.compdict[comps]['item1'] 
                                    == 'mooring system'):
                                    if (self._variables.compdict[comps]['item2'] 
                                        == block 
                                        and 
                                        self._variables.compdict[comps]['item5'][0] 
                                        > complim[1] 
                                        and 
                                        self._variables.compdict[comps]['item6'][0] 
                                        > complim[0]):
                                        # print 'Larger size component'
                                        """ If suitable size cannot be found increase component 
                                        size. Note: it is possible that there will be no 
                                        components in the database with sufficient loading capacity """ 
                                        if block == 'rope':
                                            """ If the component in question is a rope, sort by lowest 
                                                axial stiffness instead of cost """                                        
                                            ropeea = (self._variables.compdict[comps]['item5'][0] 
                                                    * (self._variables.compdict[comps]['item5'][1][-1][1] 
                                                    - self._variables.compdict[comps]['item5'][1][0][1]) 
                                                    / (self._variables.compdict[comps]['item5'][1][-1][0] 
                                                    - self._variables.compdict[comps]['item5'][1][0][0])) 
                                            complist.append((comps,
                                                        self._variables.compdict[comps]['item6'][0],
                                                       ropeea))
                                        else:
                                            complist.append((comps,
                                                        self._variables.compdict[comps]['item6'][0],
                                                        self._variables.compdict[comps]['item11']))
                                        if not complist: 
                                            pass
                                            """ If a component still cannot be 
                                            found (which may be the case in a 
                                            'death spiral') and it is a chain-only 
                                            system  design loop and move onto 
                                            rope-chain system """                                         
                    if complist:                        
                        """ Sort to find lowest cost and then (because multiple 
                            grades may exist), the smallest diameter """
                        complistsort = sorted(complist, key=operator.itemgetter(2,1))                    
                        selcomp = complistsort[0][0]                                                    
                    else: 
                        selcomp = 0                      
                return selcomp 
                
            compprops = [[0 for col in range(2)] for row 
                            in range(len(compblocks))]
            complim = [0.0, 0.0]
            
            ulscheck = 'False' 
            alscheck = 'False'
            umbcheck = 'False'     
            umblengchangeflag = 'False'            
            
            """ Design/Redesign loop """
            for k in range(0,klim): 
                selcomplist = []   
                if self.moordesfail == 'True':     
                    module_logger.debug('Break K loop')
                    break
                if k == 0:
                    if deviceid == 'device001':
                        """ Design loop starts with the specification of components for the first device """
                        for l in range(0,len(compblocks)):
                            if l == 0:
                                """ Start with minimum fairlead shackle size in 
                                    database """
                                complim[0] = 0.0
                                if compblocks[l][0:7] == 'shackle':
                                    compid = moorcompret(compblocks[l][0:7], complim)
                                else:
                                    compid = moorcompret(compblocks[l], complim)                                    
                                selcomplist.append(compid[0])
                            else: 
                                complim[0] = compid[1]
                                if compblocks[l][0:7] == 'shackle':
                                    """ Find matching size components in block list """
                                    selcomplist.append(moorcompret(compblocks[l][0:7], 
                                                                   complim)) 
                                else:
                                    """ Find matching size components in block list """
                                    selcomplist.append(moorcompret(compblocks[l], 
                                                                   complim))
                    elif deviceid != 'device001':     
                        """ For all devices apart from the first one, the same mooring configuration 
                        (i.e. components) are used as a starting point """
                        selcomplist = self.moorcomptab['compid'].tolist()
                        logmsg = 'Starting configuration {}'.format(selcomplist)
                        module_logger.info(logmsg)
                        
                    for l in range(0,len(compblocks)):

                        # Test for rope and add fake length variable
                        if self._variables.compdict[
                                selcomplist[l]]['item2'] in ["rope", "cable"]:
                            length = 0.
                        else:
                            length = self._variables.compdict[
                                                    selcomplist[l]]['item6'][1]
                        
                        compprops[l] = [selcomplist[l], 
                            self._variables.compdict[selcomplist[l]]['item6'][0], 
                            length, 
                            self._variables.compdict[selcomplist[l]]['item7'][0], 
                            self._variables.compdict[selcomplist[l]]['item7'][1], 
                            self._variables.compdict[selcomplist[l]]['item5'][0], 
                            self._variables.compdict[selcomplist[l]]['item5'][1]]
                            
                    # Set up table of components      
                    colheads = ['compid', 'size', 'length', 'dry mass', 
                                'wet mass', 'mbl', 'ea']
                    self.moorcomptab = pd.DataFrame(compprops, 
                                                    index=compblocks, 
                                                    columns=colheads)

                elif k > 0: 
                    if (ulscheck == 'True' and alscheck == 'True'):
                        pass
                    else:
                        """ ULS design loop """
                        self.limitstate = 'ULS'   
                        self.moorsf = self._variables.moorsfuls
                        """ Lines analysed """
                        self.linesuls =  range(0, len(self.foundloc))
                        self.initcond = [[0 for col in range(self.numlines)] 
                                        for row in range(0,5)]
                        self.llim = [0, 1, 2]
           
                        self.numlines = len(self.foundloc)                        
                        "*******************************************************"    
                        (self.linetenuls,
                         self.fairten,
                         self.anctenuls,
                         self.initcond,
                         self.sysposuls,
                         self.umbcheck) = mooreqav(self.selmoortyp,
                                                   self.numlines, 
                                                   self.fairloc, 
                                                   self.foundloc, 
                                                   self.lineleng,  
                                                   self.initcond,
                                                   self.linesuls,
                                                   self.llim,
                                                   self.limitstate)
                        "*******************************************************"  
                        logmsg = [""]
                        logmsg.append('ULS line tensions {}'.format(self.linetenuls))
                        logmsg.append('ULS anchor tensions WC {}'.format(self.anctenuls[-1]))
                        module_logger.info("\n".join(logmsg))
                        
                        
                        
                        if (min(self.moorcomptab['mbl'].tolist()) < self.moorsf 
                            * max(max(p) for p in self.linetenuls)
                            or self.dispexceedflag == 'True'):           
                            maxcompsize = max(self.moorcomptab['size'].tolist())
                            for blockind in range(0, len(compblocks)): 
                                colheads = ['compid', 'size', 'length', 'dry mass', 
                                            'wet mass', 'mbl', 'ea']
                                if self.dispexceedflag == 'False':
                                    """ Initiate search for higher capacity 
                                        component """
                                    complim[1] = (self.moorsf * max(max(p) for p 
                                                    in self.linetenuls))
                                elif self.dispexceedflag == 'True':
                                    """ Initiate search for larger size component to 
                                        increase stiffness of mooring system """
                                    complim[0] =  maxcompsize + 1.0e-3
                                if compblocks[blockind][0:7] == 'shackle':
                                    amendcomp = moorcompret(compblocks[blockind][0:7],
                                                        complim)
                                else: 
                                    amendcomp = moorcompret(compblocks[blockind],
                                                        complim)
                                if amendcomp == 0:                                    
                                    self.moordesfail = 'True'
                                else:
                                    amendcompprops = [amendcomp, 
                                        self._variables.compdict[amendcomp]['item6'][0], 
                                        self.moorcomptab.ix[blockind,'length'], 
                                        self._variables.compdict[amendcomp]['item7'][0], 
                                        self._variables.compdict[amendcomp]['item7'][1], 
                                        self._variables.compdict[amendcomp]['item5'][0], 
                                        self._variables.compdict[amendcomp]['item5'][1]]
                                    for j in range(0,self.numlines): 
                                        linelenghead = 'line ' + str(j) + ' length' 
                                        amendcompprops.append(self.moorcomptab.ix[blockind,linelenghead])
                                    if compblocks[blockind] == 'rope':
                                        self.moorcomptab.loc['rope','compid'] = amendcompprops[0]
                                        self.moorcomptab.loc['rope','size'] = amendcompprops[1]
                                        self.moorcomptab.loc['rope','length'] = amendcompprops[2]
                                        self.moorcomptab.loc['rope','dry mass'] = amendcompprops[3]
                                        self.moorcomptab.loc['rope','wet mass'] = amendcompprops[4]
                                        self.moorcomptab.loc['rope','mbl'] = amendcompprops[5]
                                        self.moorcomptab.set_value(compblocks[blockind],'ea',amendcompprops[6])
                                    else:
                                        self.moorcomptab.ix[blockind] = amendcompprops
                        else: 
                            if (self.dispexceedflag == 'True' and self.selmoortyp == 'catenary'):
                                """ If an individual line fails to converge, reduce line length 
                                by 0.5%. For catenary lines, the lower limit of line length is 
                                set as taut configuration """                                
                                self.linelengdiff[j] = 0.01 * self.lineleng[j]
                                self.lineleng[j] = self.lineleng[j] - self.linelengdiff[j]
                                if self.lineleng[j] < self.tautleng[j]:
                                    self.lineleng[j] = self.tautleng[j]
                            else:
                                ulscheck = 'True'
                                self.dispexceedflag = 'False'
          
                        """ Solution cannot be found for current configuration """
                        if self.moordesfail == 'True':     
                            # print 'Break K loop'
                            break
                        elif self.moordesfail == 'False':
                            if ulscheck == 'True':                            
                                if (self.numlines >= 2 and l != 3):
                                    """ ALS check for mooring systems with multiple mooring lines """
                                    self.limitstate = 'ALS'
                                    self.moorsf = self._variables.moorsfals
                                    maxlinetenval = 0.0
                                    for caseind, casevals in enumerate(self.linetenuls):
                                        for lineind, lineval in enumerate(casevals):
                                            if lineval > maxlinetenval:
                                                maxlinetenval = lineval
                                                maxlinetenind = lineind
                                                          
                                    """ Remove line with maximum ULS tension """  
                                    self.linesals = np.delete(self.linesuls, 
                                                              (maxlinetenind), axis=0)
                                    self.llim = [1, 2]
                                    # ********************************************** 
                                    self.linetenals, self.fairten, self.anctenals, self.initcond, self.sysposals, self.umbcheck = mooreqav(self.selmoortyp, 
                                                             self.numlines - 1, 
                                                             self.fairloc, 
                                                             self.foundloc, 
                                                             self.lineleng, 
                                                             self.initcond, 
                                                             self.linesals,self.llim, self.limitstate)
                                    # **********************************************
                                    self.anctenmagals = [0 for row in range(self.quanfound)]
                                    for j in self.linesals:
                                        self.anctenmagals[j] = math.sqrt(self.anctenals[2][j][0] ** 2.0 + self.anctenals[2][j][1] ** 2.0 + self.anctenals[2][j][2] ** 2.0)
                                    
                                    logmsg = [""]
                                    logmsg.append(('ALS line tensions {}').format(self.linetenals))
                                    logmsg.append('ALS anchor tensions WC {}'.format(self.anctenals[-1]))
                                    logmsg.append('ALS anchor tension magnitudes {}'.format(self.anctenmagals))                                   
                                    module_logger.info("\n".join(logmsg))
                                    
                                    if (min(self.moorcomptab['mbl'].tolist())  
                                        < self.moorsf * max(max(p) for p 
                                        in self.linetenals)
                                        or self.dispexceedflag == 'True'):           
                                        maxcompsize = max(self.moorcomptab['size'].tolist())
                                        # print 'Component MBL exceeded'                                    
                                        for blockind in range(0, len(compblocks)):   
                                            if self.dispexceedflag == 'False':
                                                """ Initiate search for higher capacity 
                                                    component """
                                                complim[1] = (self.moorsf * max(max(p) for p 
                                                                in self.linetenals))
                                            elif self.dispexceedflag == 'True':
                                                """ Initiate search for larger size 
                                                    component """
                                                complim[0] =  maxcompsize + 1.0e-3
                                            if compblocks[blockind][0:7] == 'shackle':
                                                amendcomp = moorcompret(compblocks[blockind][0:7],
                                                        complim)
                                            else: 
                                                amendcomp = moorcompret(compblocks[blockind],
                                                        complim)
                                            if amendcomp == 0:
                                                self.moordesfail = 'True'

                                            else:                                            
                                                amendcompprops = [amendcomp, 
                                                    self._variables.compdict[amendcomp]['item6'][0], 
                                                    self.moorcomptab.ix[blockind,'length'], 
                                                    self._variables.compdict[amendcomp]['item7'][0], 
                                                    self._variables.compdict[amendcomp]['item7'][1], 
                                                    self._variables.compdict[amendcomp]['item5'][0], 
                                                    self._variables.compdict[amendcomp]['item5'][1]] 
                                                for j in range(0,self.numlines): 
                                                    linelenghead = 'line ' + str(j) + ' length' 
                                                    amendcompprops.append(self.moorcomptab.ix[blockind,linelenghead])
                                                #amendcompprops.append(self.moorcomptab.ix[blockind,'line ' + str(maxlinetenind) + ' length'])
                                                if compblocks[blockind] == 'rope':
                                                    self.moorcomptab.loc['rope','compid'] = amendcompprops[0]
                                                    self.moorcomptab.loc['rope','size'] = amendcompprops[1]
                                                    self.moorcomptab.loc['rope','length'] = amendcompprops[2]
                                                    self.moorcomptab.loc['rope','dry mass'] = amendcompprops[3]
                                                    self.moorcomptab.loc['rope','wet mass'] = amendcompprops[4]
                                                    self.moorcomptab.loc['rope','mbl'] = amendcompprops[5]
                                                    self.moorcomptab.set_value(compblocks[blockind],'ea',amendcompprops[6])
                                                else:                                            
                                                    self.moorcomptab.ix[blockind] = amendcompprops                                               
                                    else:      
                                        alscheck = 'True'
                                        self.dispexceedflag = 'False'
                                else:
                                    alscheck = 'True'   
                                    self.dispexceedflag = 'False'
                                    maxlinetenind = self.linetenuls[0]
                if (ulscheck == 'True' and alscheck == 'True'): 
                    logmsg = [""]
                    logmsg.append('++++++++++++++++++++++++++++++++++++++++++++++++++')
                    logmsg.append('ULS and ALS checks passed')
                    logmsg.append('++++++++++++++++++++++++++++++++++++++++++++++++++')
                    module_logger.info("\n".join(logmsg))
                    
                    if self.rn == 1:
                        """ ULS design loop with umbilical"""
                        self.limitstate = 'ULS'   
                        """ Only carry out ULS check with umbilical """
                        ulscheck = 'False'  
                        self.moorsf = self._variables.moorsfuls
                        self.linesuls =  range(0, len(self.foundloc))
                        self.initcond = [[0 for col in range(self.numlines)] 
                                        for row in range(0,5)]
                                        
                        if math.fabs(self._variables.wlevmin) > 0.0:  
                            self.llim = [3,4]
                        else:
                            self.llim = [3]
                                                                
                        "*******************************************************"    
                        (self.linetenuls,
                         self.fairten,
                         self.anctenuls,
                         self.initcond,
                         self.sysposuls,
                         self.umbcheck) = mooreqav(self.selmoortyp,
                                                   self.numlines, 
                                                   self.fairloc, 
                                                   self.foundloc, 
                                                   self.lineleng,
                                                   self.initcond,
                                                   self.linesuls,
                                                   self.llim,
                                                   self.limitstate)
                        "*******************************************************" 
                        self.anctenmaguls = [0 for row in range(self.quanfound)]
                        for j in range(0,self.quanfound):
                            self.anctenmaguls[j] = math.sqrt(self.anctenuls[2][j][0] ** 2.0 + self.anctenuls[2][j][1] ** 2.0 + self.anctenuls[2][j][2] ** 2.0)
                                    
                        logmsg = [""]
                        logmsg.append('ULS line tensions with umbilical: high water {}'.format(self.linetenhigh))                        
                        logmsg.append('ULS anchor tensions with umbilical WC: high water {}'.format(self.anctenhigh[-1]))
                        if self.wlevlowflag == 'True':
                            logmsg.append('ULS line tensions with umbilical: low water {}'.format(self.linetenlow))
                            logmsg.append('ULS anchor tensions with umbilical WC: low water {}'.format(self.anctenlow[-1]))
                        logmsg.append('ULS anchor tension magnitudes {}'.format(self.anctenmaguls))
                        module_logger.info("\n".join(logmsg))
                        
                        if (min(self.moorcomptab['mbl'].tolist()) < self.moorsf 
                            * max(max(p) for p in self.linetenuls)
                            or self.dispexceedflag == 'True'):           
                            maxcompsize = max(self.moorcomptab['size'].tolist())
                            for blockind in range(0, len(compblocks)): 
                                colheads = ['compid', 'size', 'length', 'dry mass', 
                                            'wet mass', 'mbl', 'ea']
                                if self.dispexceedflag == 'False':
                                    """ Initiate search for higher capacity 
                                        component """
                                    complim[1] = (self.moorsf * max(max(p) for p 
                                                    in self.linetenuls))
                                elif self.dispexceedflag == 'True':
                                    """ Initiate search for larger size 
                                        component """
                                    complim[0] =  maxcompsize + 1.0e-3
                                if compblocks[blockind][0:7] == 'shackle':
                                    amendcomp = moorcompret(compblocks[blockind][0:7],
                                                        complim)
                                else: 
                                    amendcomp = moorcompret(compblocks[blockind],
                                                        complim)
                                if amendcomp == 0:
                                    self.moordesfail = 'True'
                                else:
                                    amendcompprops = [amendcomp, 
                                        self._variables.compdict[amendcomp]['item6'][0], 
                                        self.moorcomptab.ix[blockind,'length'], 
                                        self._variables.compdict[amendcomp]['item7'][0], 
                                        self._variables.compdict[amendcomp]['item7'][1], 
                                        self._variables.compdict[amendcomp]['item5'][0], 
                                        self._variables.compdict[amendcomp]['item5'][1]]
                                    for j in range(0,self.numlines): 
                                        linelenghead = 'line ' + str(j) + ' length' 
                                        amendcompprops.append(self.moorcomptab.ix[blockind,linelenghead])
                                    if compblocks[blockind] == 'rope':
                                        self.moorcomptab.loc['rope','compid'] = amendcompprops[0]
                                        self.moorcomptab.loc['rope','size'] = amendcompprops[1]
                                        self.moorcomptab.loc['rope','length'] = amendcompprops[2]
                                        self.moorcomptab.loc['rope','dry mass'] = amendcompprops[3]
                                        self.moorcomptab.loc['rope','wet mass'] = amendcompprops[4]
                                        self.moorcomptab.loc['rope','mbl'] = amendcompprops[5]
                                        self.moorcomptab.set_value(compblocks[blockind],'ea',amendcompprops[6])
                                    else:
                                        self.moorcomptab.ix[blockind] = amendcompprops
                        else:
                            ulscheck = 'True'
                            self.dispexceedflag = 'False'
                            if self.umbcheck == 'True':
                                logmsg = [""]
                                logmsg.append('++++++++++++++++++++++++++++++++++++++++++++++++++')
                                logmsg.append('Umbilical tension and minimum bend radius checks passed')
                                logmsg.append('Umbilical maximum tension and minimum bend radius {}'.format([self.umbtenmax, self.umbradmin]))                                
                                logmsg.append('++++++++++++++++++++++++++++++++++++++++++++++++++')                                
                                module_logger.info("\n".join(logmsg))
                                break
                
                """ Connecting lengths of components """ 
                """ Studlink chain = 4*DN
                    D-type connecting shackle = 3.4*DN <----LHR Marine
                    D-type anchor shackle = 4.5*DN <----LHR Marine
                    Bow eye swivel = 7.4*DN <----LHR Marine 
                    Forerunner assembly = 28.75*DN """              
                                          
                if self.selmoortyp == 'taut':
                    for j in range(0,self.numlines):
                        """ Local fairlead and foundation locations """
                        self.fairloc[j][0] = self.fairlocglob[j][0]
                        self.fairloc[j][1] = self.fairlocglob[j][1]
                        if self.foundradnew:
                            """ Single value selected based on original fairlead position of line 1 """
                            self.fairloc[j][2] = self._variables.fairloc[0][2] 
                        else:
                            self.fairloc[j][2] = self._variables.fairloc[j][2] 
                         
                        self.foundloc[j][0] = self.foundlocglob[j][0]
                        self.foundloc[j][1] = self.foundlocglob[j][1]
                        self.foundloc[j][2] = self.foundlocglob[j][2]
                        
                        self.tautleng[j] = math.sqrt((self.fairloc[j][0] 
                                            - self.foundloc[j][0]) ** 2.0 
                                            + (self.fairloc[j][1] 
                                            - self.foundloc[j][1]) ** 2.0 
                                            + (self.fairloc[j][2] 
                                            - self.foundloc[j][2] - self._variables.sysdraft) ** 2.0)
                                               
                        self.lineleng[j] = copy.deepcopy(self.tautleng[j])
                elif self.selmoortyp == 'catenary':      
                    for j in range(0,self.numlines):  
                        """ Local fairlead and foundation locations """
                        self.fairloc[j][0] = self.fairlocglob[j][0]
                        self.fairloc[j][1] = self.fairlocglob[j][1]
                        if self.foundradnew:
                            """ Single value selected based on original fairlead position of line 1 """
                            self.fairloc[j][2] = self._variables.fairloc[0][2] 
                        else:
                            self.fairloc[j][2] = self._variables.fairloc[j][2] 
                        
                        self.foundloc[j][0] = self.foundlocglob[j][0]
                        self.foundloc[j][1] = self.foundlocglob[j][1]                        
                        self.foundloc[j][2] = self.foundlocglob[j][2]
                        
                        self.tautleng[j] = math.sqrt((self.fairloc[j][0] 
                                            - self.foundloc[j][0]) ** 2.0 
                                            + (self.fairloc[j][1] 
                                            - self.foundloc[j][1]) ** 2.0 
                                            + (self.fairloc[j][2] 
                                            - self.foundloc[j][2] - self._variables.sysdraft) ** 2.0)

                        """ Set line length as 125% taut length """                        
                        self.lineleng[j] = 1.25 * self.tautleng[j]                                            
                    
                linexf = [0 for row in range(0, self.numlines)]
                linezf = [0 for row in range(0, self.numlines)]
                slope = [0 for row in range(0, self.numlines)]
                # self.linelengbed = [0 for row in range(0, self.numlines)]
                self.linelengbed = [0 for row in range(0, 
                                               len(self.foundloc))]
                
                """ Catenary tolerance """
                tol = 0.01 
                for j in range(0,self.numlines): 
                    connleng = []    
                    """ Line length header """
                    linelenghead = 'line ' + str(j) + ' length' 
                    for blockind in range(0, len(compblocks)):                            
                        if self.selmoortyp == 'catenary':                             
                            if (compblocks[blockind] == 'rope' and 'chain' in compblocks):
                                """ Approximate catenary profile used in to approximate length of line resting on seafloor at equilibrium """
                                linexf[j] = math.sqrt((self.fairloc[j][0] - self.foundloc[j][0]) ** 2.0 
                                            + (self.fairloc[j][1] - self.foundloc[j][1]) ** 2.0)                                                
                                linezf[j] = self.fairloc[j][2] - self.foundloc[j][2]          
                                
                                """ Weight per unit length of chain """
                                omega = self.moorcomptab.ix['chain','wet mass'] * self._variables.gravity 
                                lambdacat = math.sqrt(3.0 * (((self.lineleng[j] ** 2.0 
                                                - linezf[j] ** 2.0) / linexf[j] ** 2.0) - 1.0))  
                                """ Approximate fairlead horizontal and vertical loads """
                                Hf = max(math.fabs(0.5 * omega * linexf[j] 
                                    / lambdacat),tol)  
                                Vf = 0.5 * omega * ((linezf[j] 
                                    / math.tanh(lambdacat)) + self.lineleng[j]) 
                                """ Approximate fairlead tension """
                                Tf = math.sqrt(Hf ** 2.0 + Vf ** 2.0)
                                afact = Tf / omega
                                """ Approximate seafloor slope at device equilibrium position """
                                slope[j] = math.atan((self.foundloc[j][2] + self.bathysysorig) / linexf[j])
                                self.linelengbed[j] = max(self.lineleng[j] - math.sqrt((linezf[j] + 2.0 * linezf[j] * afact)), 0.0)
                                
                                if compblocks[blockind] == 'rope':
                                    self.ropeeaind = blockind 
                                    """ Rope length is calculated based on 
                                        touchdown point of chain-only system 
                                        minus a fixed amount """
                                    self.moorcomptab.ix[blockind,
                                        linelenghead] = (self.lineleng[j] 
                                        - self.linelengbed[j] - 20.0)                                     
                                    connleng.append(self.moorcomptab.ix[
                                        blockind,linelenghead])
                            if (compblocks[blockind] 
                                in ('forerunner assembly', 'swivel') 
                                or compblocks[blockind][0:7] == 'shackle'):                        
                                self.moorcomptab.ix[blockind,
                                    linelenghead] = self.moorcomptab.ix[
                                    blockind,'length']
                                """ Calculate connected length of 
                                    components """ 
                                connleng.append(self.moorcomptab.ix[
                                    blockind,linelenghead])
                            if (len(compblocks) == 1 and compblocks[blockind] in ('chain', 'rope')):
                                self.moorcomptab.ix[blockind,
                                    linelenghead] = self.lineleng[j]            
                        elif self.selmoortyp == 'taut': 
                            """ Rope length is calculated based on length 
                                of other components """
                            if (compblocks[blockind] 
                                in ('forerunner assembly', 
                                'swivel') or compblocks[blockind][0:7] == 'shackle'):  
                                self.moorcomptab.ix[blockind,
                                    linelenghead] = self.moorcomptab.ix[
                                    blockind,'length']
                                """ Calculate connected length of 
                                    components """ 
                                connleng.append(self.moorcomptab.ix[
                                    blockind,linelenghead])    
                            if compblocks[blockind] == 'rope':
                                self.ropeeaind = blockind 
                            if (len(compblocks) == 1 and compblocks[blockind] in ('chain', 'rope')):
                                self.moorcomptab.ix[blockind,
                                    linelenghead] = self.lineleng[j]

                    if self.selmoortyp == 'catenary':
                        if 'chain' in compblocks: 
                            self.moorcomptab.ix['chain',
                                linelenghead] = (self.lineleng[j] 
                                - sum(connleng))
                    if self.selmoortyp == 'taut':
                        """ Rope length is equal to remaining line length """
                        self.moorcomptab.ix['rope',
                            linelenghead] = self.lineleng[j] - sum(connleng)    
                            
                    if (self._variables.preline and compblocks == ['shackle001', 'rope','shackle002']):
                        if self.rn == 1:                                
                            self.moorcomptab.ix['rope',
                                linelenghead] = (self.lineleng[j] 
                                - sum(connleng))
                        
            if self.moordesfail == 'False':
                """ Construct table of anchor tensions """
                self.totancsteadloads = [[0 for col in range(3)] for row in range(0, 
                                          len(self.foundloc))]
                anctentabindsuls = ['lines only (uls)','static (uls)']
                
                
                    
                if self._variables.hs:
                    for wc in range(0, len(self._variables.hs)):
                        anctentabindsuls.append('wc ' + str(wc) + ' (uls)')
                else:
                    anctentabindsuls.append('c ' + str(1) + ' (uls)')
                
                anctentabcols = []
                linetentabcols = []
                
                if self.numlines > 1:
                    anctentabindsals = ['lines only (als)','static (als)']
                    if self._variables.hs:        
                        for wc in range(0, len(self._variables.hs)):    
                            anctentabindsals.append('wc ' + str(wc) + ' (als)')
                    else:
                        anctentabindsals.append('c ' + str(1) + ' (als)')
                    """ Remaining lines analysed """
                    self.linesals = copy.deepcopy(self.linesuls)
                    if maxlinetenind in self.linesals:
                        self.linesals.remove(maxlinetenind)
        
                for j in range(0, len(self.foundloc)):
                    anctentabcols.append('line ' + str(j))
                    linetentabcols.append('line ' + str(j))
                linetentabcols.append('system position x [m]')
                linetentabcols.append('system position y [m]')
                linetentabcols.append('system draft [m]')
                                
                anctentabuls = pd.DataFrame(self.anctenuls, index=anctentabindsuls, 
                                            columns=anctentabcols)
                if self.numlines > 1:             
                    anctentabals = pd.DataFrame(self.anctenals, index=anctentabindsals, 
                                                columns=anctentabcols) 
                    for ind,vals in enumerate(self.linetenals):                    
                        for ind2,val2 in enumerate(self.sysposals[ind]):
                            self.linetenals[ind].append(self.sysposals[ind][ind2]) 
                            
                for ind,vals in enumerate(self.linetenuls):
                    for ind2,val2 in enumerate(self.sysposuls[ind]):
                        self.linetenuls[ind].append(self.sysposuls[ind][ind2])
                   
                linetentabuls = pd.DataFrame(self.linetenuls, index=anctentabindsuls, 
                                            columns=linetentabcols)
                if self.numlines > 1:
                    linetentabals = pd.DataFrame(self.linetenals, index=anctentabindsals, 
                                                columns=linetentabcols)
                    ancframes = [anctentabuls, anctentabals]
                    lineframes = [linetentabuls, linetentabals]
                    anctentabinds = anctentabindsuls + anctentabindsals
                else:
                    ancframes = [anctentabuls]
                    lineframes = [linetentabuls]
                    anctentabinds = anctentabindsuls
                self.anctentab = pd.concat(ancframes)
                self.linetentab = pd.concat(lineframes)
                 
                
                
                """ Compare calculated loads for all cases and identify maximum loads 
                    for each line """
                maxten = [0 for row in range(0, len(self.foundloc))]
                maxtenind = [0 for row in range(0, len(self.foundloc))]
                for j in range(0, len(self.foundloc)):
                    for caseind, casevals in enumerate(self.anctentab['line ' 
                                                        + str(j)]):
                        if type(casevals) is not list:
                            casevals = [0.0, 0.0, 0.0]
                            self.anctentab['line ' + str(j)][caseind] = casevals
                        if caseind == 0:
                            maxten[j] = (math.sqrt(casevals[0] ** 2.0 
                                + casevals[1] ** 2.0 + casevals[2] ** 2.0))
                            maxtenind[j] = caseind
                        else:
                            if (math.sqrt(casevals[0] ** 2.0 + casevals[1] ** 2.0 
                                + casevals[2] ** 2.0) > maxten[j]):
                                maxten[j] = math.sqrt(casevals[0] ** 2.0 
                                    + casevals[1] ** 2.0 + casevals[2] ** 2.0)
                                maxtenind[j] = caseind
                
                for j in range(0, len(self.foundloc)):
                    self.totancsteadloads[j]  = self.anctentab.ix[anctentabinds[
                                                maxtenind[j]], 'line ' + str(j)]
                logmsg = [""]                       
                logmsg.append('Anchor tension table_________________________________________')
                logmsg.append('self.anctentab {}'.format(self.anctentab))
                logmsg.append('self.totancsteadloads {}'.format(self.totancsteadloads))
                module_logger.info("\n".join(logmsg)) 

                """ Connecting size with anchor """
                self.moorconnsize = self.moorcomptab.ix[compblocks[0],'size'] 
                self.compblocks = compblocks  
        """ If a suitable configuration cannot be found add a mooring line """        
        if (self.rn == 1 and self.moordesfail == 'True'):

            """ Abort if numlimes exceeds maximum number of mooring lines """
            if self.numlines == self._variables.maxlines + 1:
                
                errStr = ("Maximum number of lines ({}) "
                          "exceeded").format(self._variables.maxlines)
                raise RuntimeError(errStr)
                
            self.quanfound = len(self._variables.foundloc)
            foundrad = [0 for row in range(0, self.quanfound)] 
            fairrad = [0 for row in range(0, self.quanfound)]
            foundloc = []
            fairloc = []
            if self.foundradnew:
                self.numlines = self.numlines + 1
            else:
                self.numlines = self.quanfound + 1
                """ Approximate fairlead and foundation radius (may not be positioned on a pitch circle diameter) """
                for j in range(0,self.quanfound): 
                    foundrad[j] = math.sqrt(self.foundloc[j][0] ** 2.0 + self.foundloc[j][1] ** 2.0)
                    fairrad[j] = math.sqrt(self._variables.fairloc[j][0] ** 2.0 + self._variables.fairloc[j][1] ** 2.0)
                    if foundrad[j] < 1.0:
                        meanwatdep = math.fabs(sum(self._variables.bathygrid[:,2]) 
                        / len(self._variables.bathygrid[:,2]))
                        foundrad[j] = meanwatdep * 3.0
                self.foundradnew = sum(foundrad) / len(foundrad)
                self.fairradnew = sum(fairrad) / len(fairrad)
                if self.fairradnew == 0.0:
                    self.fairradnew = 0.01
                
            self.lineangs = [0 for row in range(0, self.numlines)] 

            for j in range(0,self.numlines):                 
                self.lineangs[j] = ((j * math.pi * 2.0 / self.numlines) 
                                    + (self._variables.sysorienang * math.pi / 180.0))
                foundloc.append([self.foundradnew
                    * math.sin(self.lineangs[j]), self.foundradnew 
                    * math.cos(self.lineangs[j]), self.bathysysorig])
                """ Fairlead draft same as original """
                fairloc.append([self.fairradnew
                    * math.sin(self.lineangs[j]), self.fairradnew 
                    * math.cos(self.lineangs[j]), self._variables.fairloc[0][2]])
            self.foundloc  = np.array(foundloc)
            self.fairloc  = np.array(fairloc)              
            """ Re-run mooring design code with additional mooring line """
            self.gpnearloc(self.deviceid,
                           self._variables.systype,
                           self.foundloc,
                           self._variables.sysorig[self.deviceid],
                           self._variables.sysorienang)
            self.moordes(self.deviceid)
        
    def moorcost(self):
        """ Mooring system capital cost calculations """
        self.moorcompcosts = []
        self.uniqmoorcomp = []
        self.netlistuniqmoorcomp = [0 for row in range(0, self.quanfound)]
         
        """ The number of values in this dictionary will be increased """
        self.ropecostdict = {'lengthcost': {'polyester': np.array([[500.0e3, 9.37], 
                                                      [4000.0e3, 8.61], 
                                                      [8000.0e3, 8.23]]),
                                        'nylon': np.array([[500.0e3, 9.24], 
                                                  [4000.0e3, 7.79], 
                                                  [8000.0e3, 7.73]]),
                                        'hmpe': np.array([[500.0e3, 82.32], 
                                                 [4000.0e3, 88.65], 
                                                 [8000.0e3, 94.98]])},
                            'splicecost': {'polyester': np.array([[500.0e3, 443.25],
                                                     [2000.0e3, 443.25], 
                                                     [2500.0e3, 664.88], 
                                                     [5000.0e3, 664.88],
                                                     [5500.0e3, 994.15],
                                                     [9500.0e3, 994.15],
                                                     [10000.0e3, 1488.06],
                                                     [12000.0e3, 1488.06]]),
                                        'nylon': np.array([[500.0e3, 443.25],
                                                 [2000.0e3, 443.25], 
                                                 [2500.0e3, 664.88], 
                                                 [5000.0e3, 664.88],
                                                 [5500.0e3, 994.15],
                                                 [9500.0e3, 994.15],
                                                 [10000.0e3, 1488.06],
                                                 [12000.0e3, 1488.06]]),
                                        'hmpe':  np.array([[500.0e3, 664.88],
                                                 [2000.0e3, 664.88], 
                                                 [2500.0e3, 994.15], 
                                                 [5000.0e3, 994.15],
                                                 [5500.0e3, 1488.06],
                                                 [9500.0e3, 1488.06],
                                                 [10000.0e3, 2235.25],
                                                 [12000.0e3, 2235.25]])}}
                                             
        for j in range(0,self.numlines):             
            """ Line length header """
            linelistuniqmoorcomp = []
            linelenghead = 'line ' + str(j) + ' length'         
            for blockind in range(0, len(self.compblocks)):                
                complabel = self._variables.compdict[self.moorcomptab.ix[blockind,'compid']]['item2']
                self.uniqmoorcomp = '{0:04}'.format(self.netuniqcompind)
                linelistuniqmoorcomp.append(self.uniqmoorcomp)
                self.netuniqcompind = self.netuniqcompind + 1
                if self.moorcomptab.index[blockind]  == 'chain':             
                    self.moorcompcosts.append((self.uniqmoorcomp, 
                                                self.moorcomptab.ix[blockind,'compid'],
                                                self.moorcomptab.ix[blockind, linelenghead]
                                              * self._variables.compdict[
                                              self.moorcomptab.ix[blockind,'compid']]['item11']))
                elif self.moorcomptab.index[blockind]  == 'rope':
                    if (self._variables.compdict[
                        self.moorcomptab.ix[blockind,'compid']]['item4'][0] in ('polyester','nylon','hmpe')):
                            ropemat = (self._variables.compdict[
                                        self.moorcomptab.ix[blockind,'compid']]['item4'][0])
                            ropembl = (self._variables.compdict[
                                        self.moorcomptab.ix[blockind,'compid']]['item5'][0])
                            """ Length surcharge and splice costs calculated """
                            ropelengthcostint = interpolate.interp1d(self.ropecostdict['lengthcost'][ropemat][:,0],
                                                                     self.ropecostdict['lengthcost'][ropemat][:,1])
                            ropesplicecostint = interpolate.interp1d(self.ropecostdict['splicecost'][ropemat][:,0],
                                                                     self.ropecostdict['splicecost'][ropemat][:,1])
                            if ropembl < min(self.ropecostdict['lengthcost'][ropemat][:,0]):
                                self.moorcompcosts.append((self.uniqmoorcomp, 
                                                        self.moorcomptab.ix[blockind,'compid'],
                                                        self.moorcomptab.ix[blockind, linelenghead] 
                                                        * self.ropecostdict['lengthcost'][ropemat][0,1]
                                                        + 2.0 * self.ropecostdict['splicecost'][ropemat][0,1]))
                            elif (ropembl >= min(self.ropecostdict['lengthcost'][ropemat][:,0]) 
                                and ropembl <= max(self.ropecostdict['lengthcost'][ropemat][:,0])):
                                self.moorcompcosts.append((self.uniqmoorcomp, 
                                                            self.moorcomptab.ix[blockind,'compid'],
                                                            self.moorcomptab.ix[blockind, linelenghead]
                                                            * ropelengthcostint(ropembl) 
                                                            + 2.0 * ropesplicecostint(ropembl)))
                            elif ropembl > max(self.ropecostdict['lengthcost'][ropemat][:,0]):
                                self.moorcompcosts.append((self.uniqmoorcomp, 
                                                        self.moorcomptab.ix[blockind,'compid'],
                                                        self.moorcomptab.ix[blockind, linelenghead] 
                                                        * self.ropecostdict['lengthcost'][ropemat][-1,1]
                                                        + 2.0 * self.ropecostdict['splicecost'][ropemat][-1,1]))
                            
                    else: 
                        """ For other rope materials the database cost per unit length value is used """
                        self.moorcompcosts.append((self.uniqmoorcomp, 
                                                   self.moorcomptab.ix[blockind,'compid'],
                                                   self.moorcomptab.ix[blockind, linelenghead] 
                                                  * self._variables.compdict[
                                                  self.moorcomptab.ix[blockind,'compid']]['item11']))
                else:
                    self.moorcompcosts.append((self.uniqmoorcomp, 
                                                self.moorcomptab.ix[blockind,'compid'], 
                                                self._variables.compdict[
                                                self.moorcomptab.ix[blockind,'compid']]['item11']))   
            self.netlistuniqmoorcomp[j] = linelistuniqmoorcomp
            
        self.totmoorcost = sum(x[2] for x in self.moorcompcosts)

    def moorinst(self, deviceid):
        """ Mooring system installation calculations """
        self.deviceid = deviceid
        self.quanfound = len(self.foundloc)
        self.moorinstparams = [0 for row in range(0, self.quanfound)]        
        tabind = []        
        self.compblocks = self.moorcomptab.index
        """ Set up installation table """    
        self.moorinstsubtab = (self.moorcomptab[['compid', 'size', 'dry mass']]
                                .copy(deep=True))
       
        self.listmoorcomp = []
        for j in range(0,self.quanfound):             
            self.listuniqmoorcomp = []  
            """ Line header """  
            """ Line length and mass header """
            linelenghead = 'line ' + str(j) + ' length'             
            linedrymasshead = 'line ' + str(j) + ' total dry mass'   
            for blockind in range(0, len(self.compblocks)):   
                self.moorinstsubtab.ix[blockind, linelenghead] = copy.deepcopy(
                                self.moorcomptab.ix[blockind, linelenghead])
                if (self.moorinstsubtab.index[blockind] 
                                in ('chain','rope','forerunner assembly')):
                    """ Dry weight calculation based on mass per unit length """                    
                    self.moorinstsubtab.ix[blockind, 
                            linedrymasshead] = (self.moorinstsubtab.ix[
                                blockind,linelenghead] 
                                * self.moorinstsubtab.ix[blockind, 'dry mass'])
                else:
                    """ Dry weight calculation based on unit weight """
                    self.moorinstsubtab.ix[blockind, 
                            linedrymasshead] = self.moorinstsubtab.ix[blockind, 
                            'dry mass']  
                for compind, comp in enumerate(self.moorcompcosts):
                    """ All lines have the same number of components """
                    if ((compind >= (j * len(self.compblocks)) 
                        and compind <= ((j + 1) * len(self.compblocks) - 1))): 
                        self.listuniqmoorcomp.append((comp[0],comp[1]))
                self.listmoorcomp.append(self.moorinstsubtab.ix[blockind, 'compid'])
                
            
            self.moorinstparams[j] = [self.deviceid, 
                                    'line'+ '{0:03}'.format(self.linenum), 
                                    self.netlistuniqmoorcomp[j],
                                    self.selmoortyp, 
                                    self.lineleng[j], 
                                    sum(self.moorinstsubtab.ix[:, 
                                                             linedrymasshead])]
            self.linenum = self.linenum + 1
            """ Pandas table index """
            devind = int(float(self.deviceid[-3:]))
            tabind.append(j + (devind - 1) * self.quanfound)
        self.quanmoorcomp = Counter(self.listmoorcomp)        
        self.moorinsttab = pd.DataFrame(self.moorinstparams,
                                        index=tabind, 
                                        columns=['devices [-]', 
                                        'lines [-]', 
                                        'marker [-]',
                                        'type [-]', 
                                        'length [m]', 
                                        'dry mass [kg]'])   
        
                                        
    def moorbom(self, deviceid):       
        """ Create mooring system bill of materials for the RAM and logistic functions """ 
        tabind = []
        self.moorecoparams = [0 for row in range(0,len(self.moorcompcosts))]        
        moormarkerlist = [0 for row in range(0, self.quanfound)] 
        devind = int(float(deviceid[-3:]))
        tabind = range(0,len(self.moorcompcosts))
        for compind, comp in enumerate(self.moorcompcosts):
            self.moorecoparams[compind] = [comp[1], 
                                           1.0, 
                                           comp[2],
                                           self.projectyear] 
        """ Create economics BOM """
        self.moorecobomtab = pd.DataFrame(self.moorecoparams, 
                                         index=tabind, 
                                         columns=['compid [-]', 
                                                  'quantity [-]',
                                                  'component cost [euros] [-]',
                                                  'project year'])                                           
        """ Create RAM BOM """
        self.moorrambomdict = {}          
        self.moorrambomdict['quantity'] = self.quanmoorcomp
        for j in range(0,self.quanfound):
            linemarkerlist = []
            for comp in self.netlistuniqmoorcomp[j]:
                linemarkerlist.append(int(comp[-4:]))
            moormarkerlist[j] = linemarkerlist
        self.moorrambomdict['marker'] = moormarkerlist
        
    def moorhierarchy(self):
        """ Create mooring system hierarchy """
        self.linehier = []
        self.moorhier = []        
        for blockind in range(0, len(self.compblocks)):    
            self.linehier.append(self.moorinstsubtab.ix[blockind,'compid'])
        """ Generate parallel mooring hierarchy """
        for lines in range(0,self.numlines):    
            self.moorhier.append(self.linehier) 
        