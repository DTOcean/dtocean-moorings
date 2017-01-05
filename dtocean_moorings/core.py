"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter
"""
# Start logging
import logging
module_logger = logging.getLogger(__name__)

# Built in modulesrad
import operator
from collections import Counter
import math
import copy
from scipy import interpolate, optimize, special
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# External module import
import numpy as np
import pandas as pd

class Umb(object):    
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Umbilical class
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    Umbilical geometry specification submodule 
    
        Args:
            preumb: (str) [-]: predefined umbilical type
            systype (str) [-]: system type: options:    'tidefloat', 
                                                        'tidefixed', 
                                                        'wavefloat', 
                                                        'wavefixed'
            compdict (dict) [various]: component dictionary
            
        Attributes:
            selumbtyp: (str) [-]: selected umbilical type
            umbgeolw (numpy.ndarray): not yet fully defined
            umbleng (float) [m]: umbilical length
            totumbcost (float) [euros]: umbilical total cost 
            umbbom (dict): umbilical bill of materials: 'umbilical type' [str],
                                                        'length' [m],
                                                        'cost' [euros],
                                                        'total weight' [kg],
                                                        'diameter' [m]
            umbhier (list) [-]: umbilical hierarchy
                                                        
        Functions:
            umbdes: specifies umbilical geometry
            umbinst: calculates installation parameters
            umbcost: calculates umbilical capital cost
            umbbom: creates umbilical bill of materials
            umbhier: creates umbilical hierarchy        

    """
    
    def __init__(self, variables):        
        self._variables = variables
        
    def umbdes(self, deviceid, syspos, wc, umbconpt, l):
        """ This method will be used to look-up umbilical properties """
        self.selumbtyp = self._variables.preumb        
        """ Define geometry """        
        """ Umbilical defined by WP3 """
        if  self._variables.compdict[self._variables.preumb]['item2'] != 'cable':
            errStr = ("Selected umbilical component is not of type: 'cable'")
            raise RuntimeError(errStr)
        self.selumbtyp = self._variables.compdict[self._variables.preumb]['item3']
        if self._variables.systype in ("wavefloat","tidefloat"):
            """ Lazy-wave geometry comprises three sections; hang-off, buoyancy and decline """
            compblocks = ['hang off', 'buoyancy', 'decline']
        elif self._variables.systype in ("wavefixed","tidefixed"):
            syspos = [0.0, 0.0, 0.0]
            compblocks = ['hang off']
        umbprops = [0 for row in range(len(compblocks))]
        umbwetmass = [0 for row in range(len(compblocks))]
        umbtopconn = [0 for row in range(0, 3)]
        umbbotconnloc = [0 for row in range(0, 3)]
        umbbotconnrot = [0 for row in range(0, 3)]
        """ Global position of umbilical top connection """
        umbtopconn[0] = self._variables.umbconpt[0] 
        umbtopconn[1] = self._variables.umbconpt[1]
                                    
        umbbotconnloc[0] = self._variables.subcabconpt[deviceid][0] - (self._variables.sysorig[deviceid][0] + syspos[0])
        umbbotconnloc[1] = self._variables.subcabconpt[deviceid][1] - (self._variables.sysorig[deviceid][1] + syspos[1])
        umbbotconnrot[0] = round(umbbotconnloc[0] * math.cos(
                                    - self._variables.sysorienang * math.pi / 180.0)
                                    - umbbotconnloc[1]
                                    * math.sin(-self._variables.sysorienang 
                                    * math.pi / 180.0), 3)
        umbbotconnrot[1] = round(umbbotconnloc[0] * math.sin(
                                    - self._variables.sysorienang * math.pi / 180.0)
                                    + umbbotconnloc[1]
                                    * math.cos(-self._variables.sysorienang 
                                    * math.pi / 180.0), 3)                       
        if self._variables.systype in ("wavefloat","tidefloat"):  
            umbtopconn[2] = -(self.umbconpt[2] - (syspos[2] - self._variables.sysdraft))
            klim = 100
        elif self._variables.systype in ("wavefixed","tidefixed"):
            umbtopconn[2] = -self.umbconpt[2]
            klim = 500
        for i in range(0,len(compblocks)):
            if compblocks[i] == 'buoyancy':
                umbwetmass[i] = -2.0 * self._variables.compdict[self._variables.preumb]['item7'][1]
            else:
                umbwetmass[i] = self._variables.compdict[self._variables.preumb]['item7'][1]
                
        umblengcheck = 'False'
        umbtencheck = 'False'
        umbcheck = 'False'
        umblengchangeflag = 'False'
        umbbuoychangeflag = 'False'
        umbtenlog = [0 for row in range(klim)]
        
        for k in range(0,klim):
            if k == 0:
                """ Umbilical length set initially as 10.0% greater than the shortest distance between 
                the upper and lower connection points """
                if self._variables.systype in ("wavefloat","tidefloat"):
                    umbleng = 1.2 * math.sqrt((umbtopconn[0] - umbbotconnrot[0]) ** 2.0 
                             + (umbtopconn[1] - umbbotconnrot[1]) ** 2.0
                             + (umbtopconn[2] - self._variables.subcabconpt[deviceid][2]) ** 2.0)
                elif self._variables.systype in ("wavefixed","tidefixed"):
                    umbleng = 1.02 * math.sqrt((umbtopconn[0] - umbbotconnrot[0]) ** 2.0 
                                 + (umbtopconn[1] - umbbotconnrot[1]) ** 2.0
                                 + (umbtopconn[2] - self.bathysysorig - self._variables.subcabconpt[deviceid][2]) ** 2.0)
            elif k > 0:    
                if l == 3:
                    """ If the umbilical tension is exceeded increase length by 0.5% """
                    if (self._variables.umbsf * umbtenmax > self._variables.compdict[self._variables.preumb]['item5'][0]
                        and  errumbxf > 0.0): 
                        umbleng = 1.005 * umbleng
                        umblengchangeflag = 'True'                     
                    if (umbradmin < self._variables.compdict[self._variables.preumb]['item5'][1]):
                        umbleng = 0.995 * umbleng    
                        umblengchangeflag = 'True'
                    if (self._variables.umbsf * umbtenmax < self._variables.compdict[self._variables.preumb]['item5'][0] and 
                        umbradmin > self._variables.compdict[self._variables.preumb]['item5'][1]):
                        umblengchangeflag = 'False'  
                        umbbuoychangeflag = 'False'                                       
            """ If any z-coordinate is below the global subsea cable connection point reduce umbilical length by 0.5% """ 
            if (self._variables.systype in ("wavefixed","tidefixed") and k > 0 and umblengcheck == 'False'):
                umbleng = 0.999 * umbleng              
            if self._variables.systype in ("wavefloat","tidefloat"):
                """ If umbilical tension is increasing then increase length of buoyancy section by 0.5% """
                if k <= 1:
                    """ Initial lengths of hang-off, buoyancy and decline sections (40%, 20% and 40%) """
                    cablesecthoff = 0.4 * umbleng
                    cablesectdec = 0.4 * umbleng
                    buoysect = 0.2 * umbleng
                if (l == 3 and k > 1):
                    if (self._variables.umbsf * umbtenmax > self._variables.compdict[self._variables.preumb]['item5'][0]
                        and  errumbxf < 0.0): 
                        # umbleng = 1.005 * umbleng
                        # buoysect = buoysect
                        # cablesect = (umbleng - buoysect) / 2.0
                        cablesecthoff = 0.995 * cablesecthoff
                        cablesectdec = 1.005 * cablesectdec
                        umbbuoychangeflag = 'True'
                    # if (umbtenlog[k-1] - umbtenlog[k-2] > 0.0 or errumbxf < 0.0):                        
                        # umbleng = 1.005 * umbleng                        
                        
                umbsecleng = [cablesecthoff, buoysect, cablesectdec]                
            elif self._variables.systype in ("wavefixed","tidefixed"):
                umbsecleng = [umbleng]
            
            for i in range(0,len(compblocks)):
                umbprops[i] = [self._variables.preumb, 
                            self._variables.compdict[self._variables.preumb]['item6'][0], 
                            umbsecleng[i], 
                            self._variables.compdict[self._variables.preumb]['item7'][0], 
                            umbwetmass[i], 
                            self._variables.compdict[self._variables.preumb]['item5'][0], 
                            self._variables.compdict[self._variables.preumb]['item5'][1]]                    
            """ Set up umbilical table """      
            colheads = ['compid', 'size', 'length', 'dry mass', 
                        'wet mass', 'mbl', 'mbr']     
            self.umbcomptab = pd.DataFrame(umbprops, 
                                            index=compblocks, 
                                            columns=colheads)                   
            mlim = 1000
            """ Catenary tolerance """
            tol = 0.01 
            """ Distance tolerance """
            disttol = 0.001    
            """ Number of segements along cable """
            numseg = 50
            flipzumb = [0 for row in range(0,numseg)]
            """ Segment length """
            ds = umbleng / numseg 
            umbxf = math.sqrt((umbtopconn[0] - umbbotconnrot[0]) ** 2.0 
                            + (umbtopconn[1] - umbbotconnrot[1]) ** 2.0)
            if self._variables.systype in ("wavefloat","tidefloat"): 
                umbzf = umbtopconn[2] - self._variables.subcabconpt[deviceid][2]
            elif self._variables.systype in ("wavefixed","tidefixed"): 
                if self.bathysysorig > math.fabs(self._variables.subcabconpt[deviceid][2]):
                    umbzf = -(umbtopconn[2] - (self.bathysysorig - math.fabs(self._variables.subcabconpt[deviceid][2])))
                elif self.bathysysorig < math.fabs(self._variables.subcabconpt[deviceid][2]):
                    umbzf = -(umbtopconn[2] - (math.fabs(self._variables.subcabconpt[deviceid][2]) - self.bathysysorig))
                else: umbzf = -umbtopconn[2]
            
            Humb = [0 for row in range(numseg)]
            Vumb = [0 for row in range(numseg)] 
            Tumb = [0 for row in range(numseg)] 
            thetaumb = [0 for row in range(numseg)] 
            xumb = [0 for row in range(numseg)] 
            zumb = [0 for row in range(numseg)] 
            leng = [0 for row in range(numseg)]
            """ Approximate catenary profile used in first instance to estimate top end loads """
            if umbxf == 0:            
                lambdacat = 1e6
            elif math.sqrt(umbxf ** 2.0 + umbzf ** 2.0) >= umbleng:
                lambdacat = 0.2
            else:     
                lambdacat = math.sqrt(3.0 * (((umbleng ** 2.0 
                            - umbzf ** 2.0) / umbxf ** 2.0) - 1.0))                             
            Hf = max(math.fabs(0.5 * self.umbcomptab.ix['hang off','wet mass'] * self._variables.gravity * umbxf
                / lambdacat),tol)  
            Vf = 0.5 * self.umbcomptab.ix['hang off','wet mass'] * self._variables.gravity * ((umbzf
                / math.tanh(lambdacat)) + umbleng)     
            Tf = math.sqrt(Hf ** 2.0 + Vf ** 2.0)
            theta_0 = math.atan(Vf / Hf)
            Tumb[0] = Tf
            Humb[0] = Hf
            Vumb[0] = Vf
            thetaumb[0] = theta_0    
            xumb[0] = 0.0
            zumb[0] = 0.0
            for m in range(0,mlim):           
                if m >= 1:
                    if math.fabs(errumbzf) > disttol * umbzf:                          
                        if (np.diff(zumb[k-2:k+1:2]) == 0.0 and np.diff(zumb[k-3:k:2]) == 0.0):
                            Tffactor = 0.0001
                        else:
                            Tffactor = 0.001                        
                        if errumbzf > 0.0:
                            Tumb[0] = Tumb[0] + Tffactor * Tf
                        elif errumbzf < 0.0:
                            Tumb[0] = Tumb[0] - Tffactor * Tf                        
                        Vumb[0] = math.sqrt(Tumb[0] ** 2.0 - Humb[0] ** 2.0)
                        thetaumb[0] = math.atan(Vumb[0] / Humb[0])
                
                    if math.fabs(errumbxf) > disttol * umbxf:
                        if (np.diff(xumb[k-2:k+1:2]) == 0.0 and np.diff(xumb[k-3:k:2]) == 0.0):
                            Hffactor = 0.005
                        else:
                            Hffactor = 0.01
                        if errumbxf > 0.0:
                            Humb[0] = Humb[0] + 0.001 * Hf
                        elif errumbxf < 0.0:
                            Humb[0] = Humb[0] - 0.001 * Hf
                        
                        Vumb[0] = Vumb[0]
                        thetaumb[0] = math.atan(Vumb[0] / Humb[0])
                        if Humb[0] < 0.0:
                            thetaumb[0] = math.pi / 2.0
                            Humb[0] = 0.0           
                for n in range(1,numseg):
                    leng[n] = ds * n 
                    if leng[n] <= umbsecleng[0]:
                        """ Hang-off section """
                        Vumb[n] = Vumb[n-1] - self.umbcomptab.ix['hang off','wet mass'] * self._variables.gravity  * ds
                        Humb[n] = Humb[n-1] 
                        thetaumb[n] = math.atan(Vumb[n] / Humb[n])
                        xumb[n] = xumb[n-1] + ds * math.cos(thetaumb[n-1])
                        zumb[n] = zumb[n-1] + ds * math.sin(thetaumb[n-1])
                    elif (leng[n] > umbsecleng[0] and leng[n] <= sum(umbsecleng[0:2])): 
                        """ Buoyancy section """
                        Vumb[n] = Vumb[n-1] - self.umbcomptab.ix['buoyancy','wet mass'] * self._variables.gravity * ds
                        Humb[n] = Humb[n-1]
                        thetaumb[n] = math.atan(Vumb[n] / Humb[n])
                        xumb[n] = xumb[n-1] + ds * math.cos(thetaumb[n-1])
                        zumb[n] = zumb[n-1] + ds * math.sin(thetaumb[n-1])
                    elif (leng[n] > sum(umbsecleng[0:2]) and leng[n] <= umbleng):
                        """ Decline section """
                        Vumb[n] = Vumb[n-1] - self.umbcomptab.ix['decline','wet mass'] * self._variables.gravity * ds
                        Humb[n] = Humb[n-1]                
                        thetaumb[n] = math.atan(Vumb[n] / Humb[n])                
                        if Vumb[n] < 0.0:
                            thetaumb[n] = 0.0 
                            Vumb[n] = 0.0
                        xumb[n] = xumb[n-1] + ds * math.cos(thetaumb[n-1])                
                        zumb[n] = zumb[n-1] + ds * math.sin(thetaumb[n-1]) 
                    Tumb[n] = math.sqrt(Humb[n] ** 2.0 + Vumb[n] ** 2.0)     
                    """ Allow cable to embed by up to 0.5m """
                    if (self._variables.systype in ("wavefixed","tidefixed") 
                        and zumb[n] > umbzf + 0.5):
                        umblengcheck = 'False'                        
                        break
                    else:
                        umblengcheck = 'True'                    
                errumbxf = umbxf - xumb[-1]
                errumbzf = umbzf - zumb[-1]
                                
            
                if umblengcheck == 'False':
                    break
                
                if (math.fabs(errumbxf) < disttol * umbxf  and math.fabs(errumbzf) < disttol * umbzf): 
                    logmsg = [""]
                    logmsg.append('Umbilical converged, max tension  {}'.format([max(Tumb), umbleng]))
                    module_logger.info(logmsg)
                    break  
            """ Maximum tension """
            umbtenmax = max(Tumb)
            umbtenlog[k] = umbtenmax
            
            """ Radius of curvature along umbilical (starting at device end) """
            dzdx = np.diff(zumb) / np.diff(xumb) 
            d2zdx2 = np.diff(dzdx) / np.diff(xumb[:-1])
            umbradcurv = [abs(number) for number in (((1.0 + dzdx[0:-1] ** 2.0) ** 1.5) / d2zdx2)]
            umbradmin = min(umbradcurv)
                
            """ Umbilical load angle from y-axis """
            deltaumbx = umbbotconnrot[0] - umbtopconn[0]
            deltaumby = umbbotconnrot[1] - umbtopconn[1]
            if (deltaumbx > 0.0 and deltaumby > 0.0):
                Humbloadang = math.atan(deltaumbx / deltaumby) 
            elif (deltaumbx > 0.0 and deltaumby < 0.0):
                Humbloadang = math.atan(deltaumby / deltaumbx) + 90.0 * math.pi / 180.0
            elif (deltaumbx < 0.0 and deltaumby < 0.0):
                Humbloadang = math.atan(deltaumbx / deltaumby) + 180.0 * math.pi / 180.0
            elif (deltaumbx < 0.0 and deltaumby > 0.0):
                Humbloadang = math.atan(deltaumby / deltaumbx) + 270.0 * math.pi / 180.0
            elif (deltaumbx == 0.0 and deltaumby > 0.0):
                Humbloadang = 0.0
            elif (deltaumbx > 0.0 and deltaumby == 0.0):
                Humbloadang = 90.0 * math.pi / 180.0
            elif (deltaumbx == 0.0 and deltaumby < 0.0):
                Humbloadang = 180.0 * math.pi / 180.0
            elif (deltaumbx < 0.0 and deltaumby == 0.0):
                Humbloadang = 270.0 * math.pi / 180.0                                  
            
            HumbloadX = Humb[0] * math.sin(Humbloadang)
            HumbloadY = Humb[0] * math.cos(Humbloadang)
            Vumbload = Vumb[0]    
                
            if (self._variables.umbsf * umbtenmax >  self._variables.compdict[self._variables.preumb]['item5'][0] 
                or umbradmin < self._variables.compdict[self._variables.preumb]['item5'][1]):
                umbtencheck = 'False'
            else:
                umbtencheck = 'True'            
         
                    
            if k == klim - 1:
                for zind, zval in enumerate(zumb):
                    flipzumb[zind] = umbtopconn[2] - zval
            if (umblengcheck == 'True' and umbtencheck == 'True'): 
                umbcheck = 'True'
                self.umbtenmax[wc] = umbtenmax
                self.umbradmin[wc] = umbradmin
                self.umbleng = umbleng
                logmsg = [""]
                logmsg.append('Umbilical upper end loads and horizontal angle {}'.format([HumbloadX,HumbloadY,Vumbload, Humbloadang]))
                module_logger.info(logmsg)
                if self._variables.systype in ('wavefloat', 'tidefloat'):
                    for zind, zval in enumerate(zumb):
                        flipzumb[zind] = -(zval - umbtopconn[2])
#                    plt.plot(xumb,flipzumb)    
#                    plt.scatter(umbxf,self._variables.subcabconpt[deviceid][2])
#                    plt.ylabel('Distance from MSL [m]', fontsize=10)
                elif self._variables.systype in ('wavefixed', 'tidefixed', 'substation'):
                    for zind, zval in enumerate(zumb):
                        flipzumb[zind] = - zval - umbtopconn[2]
                # plt.plot(xumb,flipzumb)    
                # if self.bathysysorig > math.fabs(self._variables.subcabconpt[deviceid][2]):
                    # plt.scatter(umbxf,(math.fabs(self._variables.subcabconpt[deviceid][2]) - self.bathysysorig))
                # elif self.bathysysorig < math.fabs(self._variables.subcabconpt[deviceid][2]):
                    # plt.scatter(umbxf,(self.bathysysorig - math.fabs(self._variables.subcabconpt[deviceid][2])))
                # plt.ylabel('Distance from seafloor [m]', fontsize=10)
                # plt.xlabel('X coord [m]', fontsize=10)     
                # plt.show()
                break 
        
        return HumbloadX, HumbloadY, Vumbload, umbleng, umbcheck
        
    def umbcost(self):
        """ Umbilical cost calculations. Note: flotation cost not included! """           
        self.totumbcost = self.umbleng * self._variables.compdict[self._variables.preumb]['item11']
        
    def umbbom(self, deviceid):        
        """ Create umbilical bill of materials """                     
        self.umbrambomdict = {}         
        
        complabel = self._variables.compdict[self._variables.preumb]['item2']                
        self.uniqumbcomp = '{0:04}'.format(self.netuniqcompind)
        self.netlistuniqumbcomp = self.uniqumbcomp
        umbquantity = Counter({self._variables.preumb: 1})
        self.netuniqcompind = self.netuniqcompind + 1 
                      
        devind = int(float(deviceid[-3:]))   
        self.umbecoparams = [self._variables.preumb,
                                  1.0, 
                                  self.totumbcost,
                                  self.projectyear]
                                   
        """ Create RAM BOM """                         
        self.umbrambomdict['quantity'] = Counter({self._variables.preumb: 1})
        self.umbrambomdict['marker'] = [[int(self.netlistuniqumbcomp[-4:])]]
        
        """ Create economics BOM """
        self.umbecobomtab = pd.DataFrame(columns=['compid [-]', 
                                                  'quantity [-]',
                                                  'component cost [euros] [-]',
                                                  'project year'])  
        self.umbecobomtab.loc[0] = self.umbecoparams
        
    def umbhierarchy(self):
        """ Create umbilical hierarchy """         
        self.umbhier = [[self._variables.preumb]]
    
    def umbinst(self, deviceid):
        """ Umbilical installation calculations """
        self.umbweight = self.umbleng * self._variables.compdict[self._variables.preumb]['item7'][1]
        if self._variables.systype in ("wavefloat","tidefloat"):
            """ Required floation per unit length """
            self.umbfloat = 1.4 * self._variables.compdict[self._variables.preumb]['item7'][1]  * self._variables.gravity 
            self.umbfloatleng = 0.2 * self.umbleng
        elif self._variables.systype in ("wavefixed","tidefixed"):
            self.umbfloat = 0.0
            self.umbfloatleng = 0.0
        
        self.umbinstparams = [deviceid, 
                        self._variables.preumb,
                        self.netlistuniqumbcomp,
                        self._variables.subcabconpt[deviceid][0],
                        self._variables.subcabconpt[deviceid][1],
                        self._variables.subcabconpt[deviceid][2],
                        self.umbleng,
                        self.umbweight, 
                        self.umbfloat, 
                        self.umbfloatleng]                        
        """ Pandas table index """
        devind = int(float(deviceid[-3:]) - 1)
        if devind == 0:
            self.umbinsttab = pd.DataFrame(columns=['devices [-]', 
                                        'component id [-]',
                                        'marker [-]',
                                        'subsea connection x coord [m]', 
                                        'subsea connection y coord [m]', 
                                        'subsea connection z coord [m]', 
                                        'length [m]', 
                                        'dry mass [kg]', 
                                        'required flotation [N/m]',
                                        'flotation length [m]'])
        self.umbinsttab.loc[devind] = self.umbinstparams

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
        
    def gpnearloc(self,deviceid,systype,foundloc,sysorig,sysorienang):          
        """ Bathymetry grid tolerances """  
        bathygridxtol = 0.5 * self._variables.bathygriddeltax
        bathygridytol = 0.5 * self._variables.bathygriddeltay
        if systype in ('wavefloat', 'tidefloat'):
            if self.foundradnew:
                quanfound = self.numlines                 
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # for j in range(0,quanfound):
                    # ax.scatter(self.foundloc[j][0]+sysorig[0],self.foundloc[j][1]+sysorig[1],self.foundloc[j][2], c='r')  
                # ax.scatter(self._variables.bathygrid[:,0],self._variables.bathygrid[:,1],self._variables.bathygrid[:,2])
                # ax.view_init(azim=0, elev=90)
                # plt.show()
            else:
                self.fairloc = self._variables.fairloc
                quanfound = len(foundloc)
        elif systype in ('wavefixed', 'tidefixed', 'substation'):
            quanfound = len(foundloc)
        self.quanfound = quanfound
        if systype in ('wavefloat', 'tidefloat'): 
            self.fairlocglob = [[0 for col in range(2)] for row 
                                in range(0, quanfound)] 
            for j in range(0,quanfound):
                self.fairlocglob[j] = self.fairloc[j]             
                   
        self.griddist = np.array([[0 for col in range(4)] for row 
            in range(len(self._variables.bathygrid))], dtype=float)
        self.gpnear = [[0 for col in range(2)] for row in range(4)] 
        self.gpnearlocs = [0 for row in range(0, quanfound + 1)]         
        self.gpnearestinds = [0 for row in range(0, quanfound + 1)]
        self.soiltyp = [0 for row in range(0, quanfound)] 
        self.soildep = [0 for row in range(0, quanfound)] 
        self.soilgroup = [0 for row in range(0, quanfound)]
        
        self.foundlocglob = [[0 for col in range(3)] for row 
                                in range(0, quanfound + 1)] 
                                
        if systype in ('wavefloat', 'tidefloat'):
            """ Determine minimum distance between neighbouring devices """
            devdist = [0 for row in range(0, len(self._variables.sysorig))]
            if len(self._variables.sysorig) > 1:
                for devposind, dev in enumerate(self._variables.sysorig):                
                    devdist[devposind] = math.sqrt((self._variables.sysorig[dev][0] - sysorig[0]) ** 2.0 
                        + (self._variables.sysorig[dev][1] - sysorig[1]) ** 2.0)    
                devlistsort = sorted(devdist)
                self.mindevdist = devlistsort[1]
            else:   
                self.mindevdist = 10000.0       
            if self._variables.maxdisp is not None:
                self.maxdisp = self._variables.maxdisp 
                self.maxdisp[2] = self.maxdisp[2] + self._variables.sysdraft
            else:
                """ Set very high displacement limits if not specified by the user """
                self.maxdisp = [1000.0, 1000.0, 1000.0]
            
            if self.foundradnew:
                self.foundloc = foundloc
            else:
                if len(self._variables.foundloc) > 0:                
                    self.foundloc = self._variables.foundloc
                elif not self._variables.foundloc and self._variables.prefootrad:
                    for j in range(0,quanfound):              
                        self.foundloc.append([self._variables.prefootrad 
                            * math.sin(self.lineangs[j]), self._variables.prefootrad 
                            * math.cos(self.lineangs[j]), 0.0])
                    self.foundloc = np.array(self.foundloc)
                elif not self._variables.foundloc and not self._variables.prefootrad:
                    """ Set initial anchor radius based on mean water depth. Note the 
                    suitability of this factor as an initial guess needs to be 
                    determined """
                    meanwatdep = math.fabs(sum(self._variables.bathygrid[:,2]) 
                                    / len(self._variables.bathygrid[:,2]))
                    for j in range(0,quanfound):           
                        self.foundloc.append([meanwatdep * 8.0 * math.sin(
                            self.lineangs[j]),meanwatdep * 8.0 * math.cos(
                            self.lineangs[j]), -meanwatdep])
                    self.foundloc = np.array(self.foundloc) 
        elif systype in ('wavefixed', 'tidefixed','substation'):
            self.foundloc = np.array(foundloc)      
        
        for j in range(0,quanfound + 1):
            """ Find nearest grid point(s) to foundation locations. Run through foundations and then
            find bathymetry at system origin.
            Transformation required from local (foundloc) to global coordinates """
            if j < quanfound:  
                self.foundlocglob[j] = np.array([round(self.foundloc[j][0],3),
                                                round(self.foundloc[j][1],3), 
                                                0.0])
            self.gpnearind = [0, 0, 0, 0]            
            for gpind, gp in enumerate(self._variables.bathygrid):
                self.griddist[gpind][:3] = gp[:3]                
                self.griddist[gpind][3] = math.sqrt(((
                    self.foundlocglob[j][0] 
                    + sysorig[0])
                    -gp[0]) ** 2.0 
                    + ((self.foundlocglob[j][1] 
                    + sysorig[1]) 
                    - gp[1]) ** 2.0)    
            self.gpsort = self.griddist[np.argsort(self.griddist[:, 3])] 
            self.gpnearind[0] = [y for y, x in enumerate(self._variables.bathygrid)
                if math.fabs(x[0] - self.gpsort[0][0]) < bathygridxtol and math.fabs(x[1] - self.gpsort[0][1]) < bathygridytol][0]
            self.gpnearestinds[j] = copy.deepcopy(self.gpnearind[0])
            self.gpnear[0][0] = self.gpsort[0][0]    
            self.gpnear[0][1] = self.gpsort[0][1]
            
            """ Warning if foundation location is outside of supplied grid. Note: this relies on a 
                rectangular grid """
            if (self.foundlocglob[j][0] + sysorig[0] > max(item[0] for item in self._variables.bathygrid)
                or sysorig[0] - self.foundlocglob[j][0] < min(item[0] for item in self._variables.bathygrid)
                or self.foundlocglob[j][1] + sysorig[1] > max(item[1] for item in self._variables.bathygrid)
                or sysorig[1] - self.foundlocglob[j][1] < min(item[1] for item in self._variables.bathygrid)):  
                module_logger.warn('!!!!!!!!!!!!!!!!! WARNING: Foundation location off bathymetry grid !!!!!!!!!!!!!!!!!')
            
            """ If the foundation/system location happens to coincide with a grid line 
            or grid point the nearest grid points at one grid spacing in either 
            direction are used """
            conincident_point = self.gpnear[0]
            
            if (self.foundlocglob[j][0] + sysorig[0]  == self.gpnear[0][0] 
                or self.foundlocglob[j][1] + sysorig[1] == self.gpnear[0][1]):
                    
                module_logger.info("Foundation point coincides with grid line "
                                   "or grid point")
                                  
                self.gpnear[0] = [
                    self.gpnear[0][0] - self._variables.bathygriddeltax, 
                    self.gpnear[0][1] - self._variables.bathygriddeltay,
                    0.0]
                self.gpnear[1] = [
                    self.gpnear[0][0] + 2 * self._variables.bathygriddeltax, 
                    self.gpnear[0][1],
                    0.0]
                self.gpnear[2] = [
                    self.gpnear[0][0],
                    self.gpnear[0][1] + 2 * self._variables.bathygriddeltay,
                    0.0]
                self.gpnear[3] = [
                    self.gpnear[0][0] + 2 * self._variables.bathygriddeltax, 
                    self.gpnear[0][1] + 2 * self._variables.bathygriddeltay,
                    0.0]
                
                for k in range(0,4):
                    
                    # Break this down to report error
                    gpnearinds = [y for y, x
                                    in enumerate(self._variables.bathygrid)
                            if math.fabs(x[0] - self.gpnear[k][0])
                                                            < bathygridxtol
                            and math.fabs(x[1] - self.gpnear[k][1])
                                                            < bathygridytol]
                                                            
                    if not gpnearinds:

                        errStr = ("Could not find suitable grid points near "
                                  "({}, {})").format(conincident_point[0],
                                                     conincident_point[1])
                        raise RuntimeError(errStr)
                                                            
                    self.gpnearind[k] = gpnearinds[0]  
                    self.gpnear[k][2] = copy.deepcopy(
                            self._variables.bathygrid[self.gpnearind[k]][2])
              
            elif (self.foundlocglob[j][0] 
                + sysorig[0] 
                != self.gpnear[0][0] and self.foundlocglob[j][1] 
                + sysorig[1] 
                != self.gpnear[0][1]):                
                """ Locate neighbouring grid points 2-4 """            
                if (self.foundlocglob[j][0] 
                    + sysorig[0] 
                    > self.gpnear[0][0]):
                    self.gpnearind[1] = [y for y, x in 
                    enumerate(self._variables.bathygrid) 
                    if (math.fabs(x[0] - (self.gpnear[0][0] 
                    + self._variables.bathygriddeltax)) < bathygridxtol 
                    and math.fabs(x[1] - self.gpnear[0][1]) < bathygridytol)][0]
                elif (self.foundlocglob[j][0] 
                    + sysorig[0] 
                    < self.gpnear[0][0]):
                    self.gpnearind[1] = [y for y, x in 
                    enumerate(self._variables.bathygrid) 
                    if (math.fabs(x[0] - (self.gpnear[0][0] 
                    - self._variables.bathygriddeltax)) < bathygridxtol 
                    and math.fabs(x[1] - self.gpnear[0][1]) < bathygridytol)][0]                 
                if (self.foundlocglob[j][1] 
                    + sysorig[1] 
                    > self.gpnear[0][1]):
                    self.gpnearind[2] = [y for y, x in 
                    enumerate(self._variables.bathygrid) 
                    if (math.fabs(x[1] - (self.gpnear[0][1] 
                    + self._variables.bathygriddeltay)) < bathygridytol 
                    and math.fabs(x[0] - self.gpnear[0][0]) < bathygridxtol)][0]                   
                elif (self.foundlocglob[j][1] 
                    + sysorig[1] 
                    < self.gpnear[0][1]):
                    self.gpnearind[2] = [y for y, x in 
                    enumerate(self._variables.bathygrid)
                    if (math.fabs(x[1] - (self.gpnear[0][1] 
                    - self._variables.bathygriddeltay)) < bathygridytol 
                    and math.fabs(x[0] - self.gpnear[0][0]) < bathygridxtol)][0]
                    
                self.gpnearind[3] = [y for y, x in enumerate(self._variables.bathygrid) 
                    if (math.fabs(x[0] - self._variables.bathygrid[self.gpnearind[1]][0]) < bathygridxtol
                    and math.fabs(x[1] - self._variables.bathygrid[self.gpnearind[2]][1]) < bathygridytol)][0]
                for k in range(0,4):
                    self.gpnear[k] = copy.deepcopy(self._variables.bathygrid[self.gpnearind[k]])
            self.gpnearlocs[j] = self.gpnear
            
            """ Determine bathymetry depth at each foundation point and system origin """
            locbathyvalues = np.transpose(np.array([self._variables.bathygrid[self.gpnearind,0],
                                        self._variables.bathygrid[self.gpnearind,1],
                                        self._variables.bathygrid[self.gpnearind,2]]))            
            bathyint = interpolate.interp2d(locbathyvalues[:,0],
                                            locbathyvalues[:,1],
                                            locbathyvalues[:,2],
                                            kind='linear')
            if j < quanfound:
                self.foundlocglob[j][2] = (bathyint(self.foundlocglob[j][0] + sysorig[0],
                                                   self.foundlocglob[j][1] + sysorig[1])
                                                   - self._variables.wlevmax)  
            elif j == quanfound:                
                self.foundlocglob[j][2] = (bathyint(sysorig[0],
                                                   sysorig[1])
                                                   - self._variables.wlevmax)               
                self.bathysysorig = -self.foundlocglob[j][2][0]   
            
            if j < quanfound: 
                """ Determine soil type at nearest grid point """
                if len(self._variables.soiltypgrid[self.gpnearestinds[j]]) > 4:
                    soillayerdep = []
                    totsoillayerdep = 0.0
                    """ Multiple layers exist, determine significance of top layers above bedrock """
                    soillayerlist = list(self._variables.soiltypgrid[self.gpnearestinds[j]])
                    soillayersind = [y for y, x in enumerate(soillayerlist) if x in ('ls', 'ms', 'ds', 'vsc', 'sc', 'fc', 'stc')
                                     and math.isinf(float(soillayerlist[-1]))]
                    for ind in soillayersind:
                        soillayerdep.append((ind, soillayerlist[ind+1]))
                        totsoillayerdep = totsoillayerdep + soillayerlist[ind+1]
                    if totsoillayerdep <= 6.0:
                        """ Soil layer(s) covering bedrock classed as a skim (i.e. less than 6.0m deep), 
                            bedrock is used for subsequent foundation calculations """
                        self.soiltyp[j] = (self._variables.soiltypgrid[
                                            self.gpnearestinds[j]][-2])
                        self.soildep[j] = float(self._variables.soiltypgrid[
                                            self.gpnearestinds[j]][-1])
                    elif totsoillayerdep > 6.0:
                        """ For deep sediments over bedrock assume soil is homogeneous and based on the 
                            soil type with the deepest layer """
                        layermax = max(soillayerdep, key=operator.itemgetter(1))
                        self.soiltyp[j] = (self._variables.soiltypgrid[
                                            self.gpnearestinds[j]][layermax[0]])
                        self.soildep[j] = totsoillayerdep                    
                else:    
                    if math.isinf(float(self._variables.soiltypgrid[self.gpnearestinds[j]][3])):
                        """ Use first layer as it has infinite depth """
                        self.soiltyp[j] = (self._variables.soiltypgrid[
                                            self.gpnearestinds[j]][2])
                        self.soildep[j] = float(self._variables.soiltypgrid[
                                            self.gpnearestinds[j]][3])
                    else:
                        self.soiltyp[j] = (self._variables.soiltypgrid[
                                        self.gpnearestinds[j]][2])
                        self.soildep[j] = float(self._variables.soiltypgrid[
                                        self.gpnearestinds[j]][3])
                                    
                if self.soiltyp[j] in ('ls', 'ms', 'ds'): 
                    self.soilgroup[j] = 'cohesionless'
                if self.soiltyp[j] in ('vsc', 'sc', 'fc', 'stc'): 
                    self.soilgroup[j] = 'cohesive'
                if self.soiltyp[j] in ('hgt', 'cm', 'src', 'hr', 'gc'): 
                    self.soilgroup[j] = 'other'    
            
        
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
                logmsg = [""]       
                logmsg.append('WAMIT/NEMOH parameters provided') 
                module_logger.info("\n".join(logmsg))
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
                logmsg = [""]       
                logmsg.append('WAMIT/NEMOH parameters not provided') 
                module_logger.info("\n".join(logmsg))
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
class Moor(Umb,Loads):    
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
                        HumbloadX, HumbloadY, Vumbload,  self.umbleng, umbcheck = self.umbdes(self.deviceid, syspos, wc, self.umbconpt,l) 
                        
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
                                linezf[j] = fairloc[j][2] - foundloc[j][2]  
                            
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
                                    sysdraft = ((1 / (self._variables.seaden 
                                            * self.syswpa)) * ((sum(Vflineref[l-1]) + Vumbload)
                                            / self._variables.gravity 
                                            + self._variables.sysmass))
                                    
                                    linezf[j] = ((fairloc[j][2]  - (sysdraft 
                                            - self._variables.sysdraft)) 
                                            - foundloc[j][2]) 
                               
                                if sysdraft > 0.0:
                                    subvol = self.syswpa * sysdraft
                                else: 
                                    subvol = 0.0
                                """ Float fully submerged """
                                if sysdraft > self._variables.sysheight:
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
                                    jac = np.matrix([[xfdHf[k], xfdVf[k]], 
                                                     [zfdHf[k], zfdVf[k]]])
                                    
                                    if np.linalg.det(jac) == 0.0:
                                        update = np.matrix([[0.0],[0.0]])
                                    else:
                                        """ Inverse of Jacobian matrix """                
                                        invjac = np.linalg.inv(jac)
                                                    
                                        """ Update Hf and Vf """
                                        update =  invjac * np.matrix([[-xf[k]],
                                                                      [-zf[k]]])
                                    det = np.linalg.det(jac)
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
                        if (l >= 1 and m > 10 and (math.sqrt(syspos[0] ** 2.0 + syspos[1] ** 2.0) 
                                        > math.sqrt(self.maxdisp[0] ** 2.0 
                                        + self.maxdisp[1] ** 2.0)
                                        or math.sqrt(syspos[0] ** 2.0 + syspos[1] ** 2.0)
                                        > 0.5 * self.mindevdist
                                        or syspos[2] > self.maxdisp[2])):
                                """ A check is carried out to determine if the device position 
                                    exceeds either i) the user-specified displacement limits or ii)
                                    half the minimum distance between adjacent device positions """
                                logmsg = [""]
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                logmsg.append('Device displacement limit exceeded!')
                                self.dispexceedflag = 'True'
                                self.sysposfail = (limitstate, [syspos[0], syspos[1], sysdraft])
                                self.initcondfail = [linexf, linezf, Hfline, Vfline]
                                logmsg.append('System position at failure = {}'.format(self.sysposfail))
                                logmsg.append('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')    
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
                                            syspos[2]]
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
                            logmsg = [""]                                
                            logmsg.append('Position not converged: [Hloadcheck, Vloadcheck] {}'.format([Hloadcheck, Vloadcheck]))
                            module_logger.info("\n".join(logmsg)) 
                            continue  
                            
                        if limitstate == 'ALS':                            
                            ancten[0] = [[0.0, 0.0, 0.0] for row in range(0, self.numlines)]   
                    if l == 1:
                        sysposrefnoenv = copy.deepcopy(syspos)    
                        # if (self.rn == 0 and limitstate == 'ULS'):
                            # self.linelengbedref = copy.deepcopy(self.linelengbed)  
                    """ Final system positions relative to local device origin """        
                    finalsyspos[l+wc][0:2] = syspos[0:2]
                    finalsyspos[l+wc][0] = sysdraft
                    if l >= 2:
                        logmsg = [""]
                        logmsg.append('_________________________________________________________________________') 
                        logmsg.append('System applied loads [HsysloadX, HsysloadX, Vfsys] {}'.format([-(HsysloadX + HumbloadX), 
                                                                                                      -(HsysloadY + HumbloadY), 
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
                        self.linetenuls, self.fairten, self.anctenuls, self.initcond, self.sysposuls, self.umbcheck = mooreqav(self.selmoortyp, self.numlines, 
                                                 self.fairloc, 
                                                 self.foundloc, 
                                                 self.lineleng,  
                                                 self.initcond, self.linesuls, self.llim, self.limitstate)
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
                        self.linetenuls, self.fairten, self.anctenuls, self.initcond, self.sysposuls, self.umbcheck = mooreqav(self.selmoortyp, self.numlines, 
                                             self.fairloc, 
                                             self.foundloc, 
                                             self.lineleng,
                                             self.initcond, self.linesuls, self.llim, self.limitstate)
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

class Subst(Loads):
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Substation Foundation class
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    Substation foundation submodule 
    
        Args:
            substparams 
            
        Attributes:
            possfoundtyp (tuple): selected foundation type (str) [-],
                                  integer (int) [-]  
            substbom (dict) [-]: substation foundation dictionary : foundation type (str) [-],
                                                                       foundation subtype (str) [-],
                                                                       dimensions (list): width (float) [m],
                                                                                          length (float) [m],
                                                                                          height (float) [m]
                                                                                          cost (float) [euros],
                                                                       total weight (float) [kg], 
                                                                       quantity (int) [-]                                                                                                                                            
                                                                       grout type (str) [-],
                                                                       grout volume (float) [m3],
                                                                       component identification number (str) [-]
            substhier (list) [-]: foundation type and grout (if applicable)
                                                        
        Functions:
            substsub: calls Loads.gpnearloc, Loads.sysstat, Loads.sysstat and Loads.syswave
            substsel: identifies suitable foundation type
            substdes: foundation system design and analysis using Found.foundsub and Found.founddes                                                          
            substcost: calculates foundation system capital cost using Found.foundcost
            substinst: calculates installation parameters using Found.foundinst
            substbom: creates foundation system bill of materials
            substhierarchy: creates foundation system hierarchy     

    """

    def __init__(self, variables):        
        super(Subst, self).__init__(variables)
        
    def substsub(self, substid):
        """ Pile foundations are considered for 
        the above-surface substations and gravity foundations for 
        sub-sea substations """        
        self.gpnearloc('array',
                       'substation',
                       eval(self._variables.substparams.ix[substid,'substloc']),
                       eval(self._variables.substparams.ix[substid,'suborig']),
                       float(self._variables.substparams.ix[substid,'suborienang']))
        self.sysstat(float(self._variables.substparams.ix[substid,'subvol']),
                     float(self._variables.substparams.ix[substid,'submass']))
        self.sysstead('substation',
                     float(self._variables.substparams.ix[substid,'subwidth']),
                      float(self._variables.substparams.ix[substid,'sublength']),
                      float(self._variables.substparams.ix[substid,'subheight']), 
                      float(self._variables.substparams.ix[substid,'suborienang']),
                      str(self._variables.substparams.ix[substid,'subprof']),
                      float(self._variables.substparams.ix[substid,'subdryfa']),
                      float(self._variables.substparams.ix[substid,'subdryba']),
                      float(self._variables.substparams.ix[substid,'subwetfa']),
                      float(self._variables.substparams.ix[substid,'subwetba']),
                      float(self._variables.substparams.ix[substid,'subvol']),
                      float(self._variables.substparams.ix[substid,'subrough']))
        self.syswave(substid,
                    'substation',
                    float(self._variables.substparams.ix[substid,'subwidth']),
                    float(self._variables.substparams.ix[substid,'sublength']),
                    float(self._variables.substparams.ix[substid,'subheight']),
                    str(self._variables.substparams.ix[substid,'subprof']),
                    float(self._variables.substparams.ix[substid,'subvol']),
                    float(self._variables.substparams.ix[substid,'subrough']))
        
    def substsel(self, substid):      
        """ Determine if a monopile (surface substation) or pin (subsea 
            substation) foundation are required """
        self.selfoundtyp = [0 for row in range(self.quanfound)]    
        for j in range(0,self.quanfound):    
            if eval(self._variables.substparams.ix[substid,'suborig'])[2] < 0.0:
                self.selfoundtyp[j] = [('gravity',0)]
            else:
                self.selfoundtyp[j] = [('pile',0)]    
        
    def substdes(self, substid): 
        """ Substation foundation design """ 
        self.foundsub(substid,
                      'substation',
                      eval(self._variables.substparams.ix[substid,'substloc']),
                      eval(self._variables.substparams.ix[substid,'subcog']))
        self.possfoundtyp = self.selfoundtyp
        self.founddes('substation')
 
    def substcost(self):        
        """ Substation foundation capital cost calculations """
        """ Fabrication costs will have to be added here """ 
        self.foundcost()
              
    def substbom(self, substid):        
        """ Create substation foundation bill of materials """      
        tabind = range(0,self.quanfound)
        self.substfoundecoparams = [0 for row in range(0,self.quanfound)]
        self.uniqfoundcomp = [0 for row in range(0,self.quanfound)]
        self.substfoundrambomdict = {} 
        self.netlistuniqsubstfoundcomp = [0 for row in range(0, self.quanfound)]
        foundmarkerlist = [0 for row in range(0, self.quanfound)]
        foundremind = []
        for j in range(0, self.quanfound):   
            self.listfoundcomp = []
            listuniqsubstfoundcomp = []
            if (self.selfoundtyp[j][0] not in ('Foundation not required', 'Foundation solution not found')):     
                if self.selfoundtyp[j][0] == 'pile':
                    self.listfoundcomp.append(self.piledim[j][4])
                    if self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]] == 'n/a':
                        pass 
                    else:
                        self.listfoundcomp.append(self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]])
                else: self.listfoundcomp.append(self.selfoundtyp[j][0])                
                complabel = self.selfoundtyp[j][0]                
                self.uniqfoundcomp[j] = '{0:04}'.format(self.netuniqcompind)
                self.netlistuniqfoundcomp[j] = self.uniqfoundcomp[j]
                self.netuniqcompind = self.netuniqcompind + 1 
                
                if self.selfoundtyp[j][0] == 'drag': 
                    compid = self.seldraganc[j]
                elif self.selfoundtyp[j][0] == 'pile':
                    compid = self.piledim[j][4]
                elif self.selfoundtyp[j][0] == 'suctioncaisson':
                    compid = self.caissdim[j][3]    
                else:
                    compid = 'n/a'
                self.substfoundecoparams[j] = [compid, 
                                          1.0, 
                                          self.sorttotfoundcost[j][1],
                                          self.projectyear]
            else:
                self.netlistuniqfoundcomp[j] = []
                self.substfoundecoparams[j] = ['n/a', 
                                          'n/a', 
                                          0.0,
                                          'n/a']
            self.quanfoundcomp = Counter(self.listfoundcomp)
            
            if (self.selfoundtyp[j][0] not in ('Foundation not required', 'Foundation solution not found')):
                """ Create RAM BOM """                         
                self.substfoundrambomdict['quantity'] = self.quanfoundcomp                     
            
                if self.selfoundtyp[j][0] == 'pile' and self.piledim[j][5] > 0.0:
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:]),int(self.netlistuniqfoundcomp[j][-4:])+1]
                    self.netuniqcompind = self.netuniqcompind + 1      
                else:
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:])]
            else:
                foundmarkerlist[j] = []
        self.substfoundrambomdict['marker'] = foundmarkerlist       
        if self.possfoundtyp:        
            """ Create economics BOM """
            self.substfoundecobomtab = pd.DataFrame(self.substfoundecoparams, 
                                             index=tabind, 
                                             columns=['compid [-]', 
                                                      'quantity [-]',
                                                      'component cost [euros] [-]',
                                                      'project year'])       
    def substhierarchy(self):
        """ Create substation foundation hierarchy """  
        self.substhier = [0 for row in range(0, self.quanfound)]
        for j in range(0, self.quanfound):
            if (self.selfoundtyp[0][0] == 'Foundation not required' or self.selfoundtyp[0][0] == 'Foundation solution not found'):
                self.substhier[j] = 'n/a'
            else:
                if self.selfoundtyp[0][0] == 'pile':                    
                    if self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]] == 'n/a':
                        self.substhier[j] = self.piledim[j][4]
                    else:    
                        self.substhier[j] = [self.piledim[j][4], self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]]]
                else: self.substhier[j] = [self.selfoundtyp[0][0]]   
    
    def substinst(self, substid):
        """ Substation foundation installation calculations """ 
        self.foundinst(substid,
                       'substation',
                       eval(self._variables.substparams.ix[substid,'suborig']),
                       self.foundlocglob,
                       float(self._variables.substparams.ix[substid,'suborienang']))

        
class Found(Moor,Loads):
    """
    #-------------------------------------------------------------------------- 
    #--------------------------------------------------------------------------
    #------------------ WP4 Found class
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    Foundation (or anchoring) system submodule
    
        Args:
            foundloc
            quanfound
            moorconnsize
            linebcf
            systype
            totancsteadloads
            foundsf
            totsysstatloads
            hs
            totsyssteadloads
            rotorload
            hubheight
            steadycurrent
            eqcurrentloadloc
            windload
            bathysysorig
            dryheight
            syscog
            syswaveloadmax
            eqhorzwaveloadloc
            meanwavedrift
            syswaveloadmaxind
            gpnearestinds
            gpnear
            bathygriddeltax
            bathygriddeltay
            prefound
            soiltyp
            soildep
            shareanc            
            dsfang
            compdict
            unshstr
            soilweight
            soilgroup
            draincoh
            conden
            gravity
            steelden
            coststeel
            costcon
            k1coef
            subgradereaccoef
            piledefcoef
            pilefricresnoncal
            groutsf
            groutstr
            hcfdrsoil
            costgrout            
            soilsen
            
        
        Attributes:
            foundrad (float) [m]: foundation radius
            foundradadd (float) [m]: foundation radius additional offset
            totfoundsteadloads (numpy.ndarray): total foundation steady loads for N foundation points:  X component (float) [N],
                                                                                                        Y component (float) [N],
                                                                                                        Z component (float) [N]
            totfoundloads (numpy.ndarray): total foundation loads for N foundation points:  X component (float) [N],
                                                                                            Y component (float) [N],
                                                                                            Z component (float) [N]
            horzfoundloads (numpy.ndarray): horizontal foundation loads for N foundation points:  maximum (absolute) load direction (float) [N],
                                                                                                  minimum (absolute) load direction (float) [N]
            vertfoundloads (list) [N]: vertical foundation loads for N foundation points
            sysbasemom (numpy.ndarray): system base moments: maximum (absolute) load direction (float) [Nm],
                                                             minimum (absolute) load direction (float) [Nm]
            moorsf (float) [-]: anchor safety factor
            totfoundstatloadmag (float) [N]: total foundation static load magnitude
            totfoundstatloaddirs (numpy.ndarray): total foundation static load angles:   X direction (float) [rad],
                                                                                         Y direction (float) [rad],
                                                                                         Z direction (float) [rad]
            totfoundsteadloadmag (float) [N]: total foundation steady load magnitude               
            totfoundsteadloaddirs (numpy.ndarray): total foundation steady load angles:   X direction (float) [rad],
                                                                                          Y direction (float) [rad],
                                                                                          Z direction (float) [rad]
            tothorzfoundload (list): horizontal shear loads at seafloor: maximum (absolute) load direction (float) [N],
                                                                         minimum (absolute) load direction (float) [N]
            a (numpy.ndarray): load matrix for least squares analysis:  list of 1's for N foundation points [-],
                                                                        foundation Y coordinate for N foundation points (float) [m],
                                                                        foundation X coordinate for N foundation points (float) [m]
            b (numpy.ndarray): load and moment matrix for least squares analysis:   list of vertical total static loads (float) [N],
                                                                                    system base moments maximum direction (float) [N],
                                                                                    system base moments minimum direction (float) [N]
            reactfoundloads (list) [N]: reaction loads for N foundation points
            i, j, k, q, m, n, p, loads, stkey, slkey, sdkey, ldkey, lmkey, posftyps, unsuitftyps, akey, comps (int, float): temporary integers and values
            seabedslpk (string) [-]: seabed slope key: options:     'moderate', 
                                                                    'steep'
            seabedslp (numpy.ndarray): seabed slope for N foundation points: maximum (absolute) load direction (float) [rad],
                                                                             minimum (absolute) load direction (float) [rad]  
            totfoundslopeloads (numpy.ndarray): foundation slope loads for N foundation points: maximum (absolute) load direction (float) [N],
                                                                                                minimum (absolute) load direction (float) [N] 
            deltabathyy (float) [m]: seafloor vertical height difference Y direction
            deltabathyx (float) [m]: seafloor vertical height difference X direction
            maxloadindex (list) [-]: maximum load index for N foundation points
            maxloadvalue (list) [N]: maximum load value for N foundation points
            minloadindex (list) [-]: minimum load index for N foundation points
            minloadvalue (list) [N]: minimum load value for N foundation points
            totfoundslopeloadsabs (numpy.ndarray): absolute foundation slope loads for N foundation points: maximum (absolute) load direction (float) [N],
                                                                                                            minimum (absolute) load direction (float) [N] 
            loaddir (list) [rad]: load direction for N foundation points
            loadmag (list) [rad]: load magnitude for N foundation points
            possfoundtyp (list) [str]: identified possible foundation types for N foundation points
            foundtyps (list) [-]: possible foundation type list
            selfoundtyp (list): selected foundation type for N foundation points: type (str) [-],
                                                                                  cost (float) [euros]
            foundsoiltypdict (dict): foundation soil type dictionary with scores for each foundation type: keys: 'vsc' (dict) [-]: very soft clay  
                                                                                                                 'sc' (dict) [-]: soft clay 
                                                                                                                 'fc' (dict) [-]: firm clay 
                                                                                                                 'stc' (dict) [-]: stiff clay 
                                                                                                                 'ls' (dict) [-]: loose sand 
                                                                                                                 'ms' (dict) [-]: medium sand 
                                                                                                                 'ds' (dict) [-]: dense sand 
                                                                                                                 'hgt' (dict) [-]: hard glacial till
                                                                                                                 'cm' (dict) [-]: cemented 
                                                                                                                 'gc' (dict) [-]: gravel cobble 
                                                                                                                 'src' (dict) [-]: soft rock coral 
                                                                                                                 'hr' (dict) [-]: hard rock 
            foundslopedict (dict): foundation slope dictionary with scores for each foundation type: keys: 'moderate' (dict) [-]: moderate slope 
                                                                                                           'steep' (dict) [-]: steep slope 
            foundsoildepdict (dict): foundation soil depth dictionary with scores for each foundation type: keys: 'none' (dict) [-]: zero depth 
                                                                                                                  'veryshallow' (dict) [-]: very shallow
                                                                                                                  'shallow' (dict) [-]: shallow
                                                                                                                  'moderate' (dict) [-]: moderate
                                                                                                                  'deep' (dict) [-]: deep
            foundloaddirdict (dict): foundation load direction dictionary with scores for each foundation type: keys: 'down' (dict) [-]: downward 
                                                                                                                      'omni' (dict) [-]: omnidirectional
                                                                                                                      'uni' (dict) [-]: unidirectional
                                                                                                                      'up' (dict) [-]: upward
            foundloadmagdict (dict): foundation load magnitude dictionary with scores for each foundation type: keys: 'low' (dict) [-]: low
                                                                                                                      'moderate' (dict) [-]: moderate
                                                                                                                      'high' (dict) [-]: high
            foundsoiltyp (str) [-]: local foundation soil type
            foundslp (str) [-]: local foundation slope
            founddep (str) [-]: local foundation depth
            foundloaddir (str) [-]: local foundation load direction
            foundloadmag (str) [-]: local foundation load magnitude 
            foundmatrixsum (list) [-]: sum of foundation matrix for each foundation type
            foundmatsumdict (dict): foundation matrix score dictionary: keys: 'shallowfoundation' (int) [-]: shallow foundation/anchor
                                                                              'gravity' (int) [-]: gravity foundation/anchor
                                                                              'pile' (int) [-]: pile foundation/anchor
                                                                              'suctioncaisson' (int) [-]: suctioncaisson anchor
                                                                              'directembedment' (int) [-]: directembedment anchor
                                                                              'drag' (int) [-]: drag embedment anchor
            unsuitftyps (list) [-]: unsuitable foundation type list
            ancs (list) [-]: anchor-only type list
            dsfangrad (float) [rad]: drained soil friction angle in radians
            foundvolsteeldict (dict): steel volume dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (float) [m3]: shallow foundation/anchor
                                                                                                                                    'gravity' (float) [m3]: gravity foundation/anchor
                                                                                                                                    'pile' (float) [m3]: pile foundation/anchor
                                                                                                                                    'suctioncaisson' (float) [m3]: suctioncaisson anchor
                                                                                                                                    'directembedment' (float) [m3]: directembedment anchor
                                                                                                                                    'drag' (float) [m3]: drag embedment anchor
            foundvolcondict (dict): concrete volume dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (float) [m3]: shallow foundation/anchor
                                                                                                                                     'gravity' (float) [m3]: gravity foundation/anchor
                                                                                                                                     'pile' (float) [m3]: pile foundation/anchor
                                                                                                                                     'suctioncaisson' (float) [m3]: suctioncaisson anchor
                                                                                                                                     'directembedment' (float) [m3]: directembedment anchor
                                                                                                                                     'drag' (float) [m3]: drag embedment anchor
            foundvolgroutdict (dict): grout volume dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (float) [m3]: shallow foundation/anchor
                                                                                                                                    'gravity' (float) [m3]: gravity foundation/anchor
                                                                                                                                    'pile' (float) [m3]: pile foundation/anchor
                                                                                                                                    'suctioncaisson' (float) [m3]: suctioncaisson anchor
                                                                                                                                    'directembedment' (float) [m3]: directembedment anchor
                                                                                                                                    'drag' (float) [m3]: drag embedment anchor
            selfoundgrouttypdict (dict): selected grout type dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (str) [-]: shallow foundation/anchor
                                                                                                                                              'gravity' (str) [-]: gravity foundation/anchor
                                                                                                                                              'pile' (str) [-]: pile foundation/anchor
                                                                                                                                              'suctioncaisson' (str) [-]: suctioncaisson anchor
                                                                                                                                              'directembedment' (str) [-]: directembedment anchor
                                                                                                                                              'drag' (str) [-]: drag embedment anchor
            foundweightdict (dict): weight dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (float) [kg]: shallow foundation/anchor
                                                                                                                            'gravity' (float) [kg]: gravity foundation/anchor
                                                                                                                            'pile' (float) [kg]: pile foundation/anchor
                                                                                                                            'suctioncaisson' (float) [kg]: suctioncaisson anchor
                                                                                                                            'directembedment' (float) [kg]: directembedment anchor
                                                                                                                            'drag' (float) [kg]: drag embedment anchor                                                                                                  
            founddimdict (dict): dimension dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (list): shallow foundation/anchor: base width (float) [m]
                                                                                                                                              base length (float) [m]
                                                                                                                                              base height (float) [m]
                                                                                                                                              skirt height (float) [m]
                                                                                                                                              shear key height (float) [m]
                                                                                                                                              shear key number (int) [-]
                                                                                                                                              shear key spacing (float) [m]
                                                                                                                                              required concrete volume (float) [m3]
                                                                                                                                              required steel volume (float) [m3]
                                                                                                                            'gravity' (float) (list): gravity foundation/anchor: base width (float) [m]
                                                                                                                                                                                 base length (float) [m]
                                                                                                                                                                                 base height (float) [m]
                                                                                                                                                                                 skirt height (float) [m]
                                                                                                                                                                                 shear key height (float) [m]
                                                                                                                                                                                 shear key number (int) [-]
                                                                                                                                                                                 shear key spacing (float) [m]
                                                                                                                                                                                 required concrete volume (float) [m3]
                                                                                                                                                                                 required steel volume (float) [m3]
                                                                                                                            'pile' (float) (list): pile foundation/anchor: pile diameter (float) [m]
                                                                                                                                                                           pile thickness (float) [m]
                                                                                                                                                                           pile length (float) [m]
                                                                                                                                                                           pile closed end (str) [-]
                                                                                                                                                                           pile component index (str) [-]
                                                                                                                                                                           pile grout diameter (float) [m]
                                                                                                                            'suctioncaisson' (list): suctioncaisson anchor: caisson diameter (float) [m]
                                                                                                                                                                              caisson thickness (float) [m]
                                                                                                                                                                              caisson length (float) [m]
                                                                                                                                                                              caisson component index (str) [-]
                                                                                                                            'directembedment' (list): directembedment anchor: plate width (float) [m]
                                                                                                                                                                               plate length (float) [m]
                                                                                                                                                                               plate thickness (float) [m]
                                                                                                                                                                               embedment depth (float) [m]
                                                                                                                            'drag' (list): drag embedment anchor: anchor width (float) [m]
                                                                                                                                                                  anchor depth (float) [m]
                                                                                                                                                                  anchor height (float) [m]
                                                                                                                                                                  anchor connecting size (float) [m]
            foundcostdict (dict): cost dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (list): shallow foundation/anchor unit cost of steel (float) [euros/m3]
                                                                                                                                           shallow foundation/anchor unit cost of concrete (float) [euros/m3]
                                                                                                                        'gravity' (float) (list): gravity foundation/anchor unit cost of steel (float) [euros/m3]
                                                                                                                                                  gravity foundation/anchor unit cost of concrete (float) [euros/m3]
                                                                                                                        'pile' (float) (list): pile foundation/anchor unit cost (float) [euros/m]
                                                                                                                        'suctioncaisson' (list): suctioncaisson anchor unit cost (float) [euros/m]
                                                                                                                        'directembedment' (list): directembedment anchor unit cost of steel (float) [euros/m3]
                                                                                                                        'drag' (list): drag embedment anchor unit cost (float) [euros]
            klim (int) [-]: iteration loop limits 
            complim (float) [m]: component limiting size
            piledia (float) [m]: possible pile diameter
            piledias (list) [m]: possible pile diameters
            pilemindia (float) [m]: pile minimum diameter
            pilethk (float) [m]: pile thickness
            pileyld (float) [N/m2]: pile yield strength
            pilemod (float) [N/m2]: pile modulus of elasticity
            ancweights (list) [kg]: possible anchor weights
            ancminweight (float) [kg]: anchor minimum weight
            anccoef (dict): anchor performance coefficients: keys: 'soft': anchor holding coefficient m (float) [-]
                                                                           anchor holding coefficient b (float) [-]
                                                                           anchor penetration coefficient n (float) [m/kg]
                                                                           anchor pentration coefficient c (float) [m]
                                                                   'sand': anchor holding coefficient m (float) [-]
                                                                           anchor holding coefficient b (float) [-]
                                                                           anchor penetration coefficient n (float) [m/kg]
                                                                           anchor pentration coefficient c (float) [m]
            embeddepth (float) [m]: anchor/foundation embedment depth
            resdesload (float) [N]: resultant design load at dip down point
            resdesloadang (float) [rad]: resultant design load angle at dip down point
            omega (float) [N/m]: buried line wet weight per unit length 
            unshstrav (float) [N/m2]: average undrained soil shear strength
            linebcfnc (float) [-]: line bearing capacity factor Nc
            bearres (float) [N]: buried line soil bearing capacity
            friccoef (float) [-]: buried line-soil friction coefficient
            linebcfnqint (float) [-]: interpolation parameter
            linebcfnq (float) [-]: line bearing capacity factor Nq
            theta_a (float) [rad]: padeye line angle with vertical axis
            x (float) [m]: distance between anchor and dip down point
            le (float) [m]: buried line length
            ten_a (float) [N]: padeye tension         
            latloadcheck (str) [-]: lateral load check pass flag: options: 'True'
                                                                           'False'
            comploadcheck (str) [-]: compressive load check pass flag: options: 'True'
                                                                                'False'
            upliftcheck (str) [-]: uplift load check pass flag: options: 'True'
                                                                         'False'
            momcheck (str) [-]: uplift load check pass flag: options: 'True'
                                                                      'False'
            latresst (numpy.ndarray): short-term lateral load resistance for N foundation points: maximum (absolute) load direction (float) [N],
                                                                                                  minimum (absolute) load direction (float) [N] 
            reqfoundweight (float) [N]: required shallow/gravity foundation/anchor weight
            normload (numpy.ndarray): normal load for N foundation points: maximum (absolute) load direction (float) [N],
                                                                           minimum (absolute) load direction (float) [N] 
            momsbase (numpy.ndarray): base moments for N foundation points: maximum (absolute) load direction (float) [Nm],
                                                                            minimum (absolute) load direction (float) [Nm]             
            eccen (numpy.ndarray): base eccentricity for N foundation points: maximum (absolute) load direction (float) [m],
                                                                              minimum (absolute) load direction (float) [m] 
            totvertfoundloads (list) [N]: total vertical loads for N foundation points
            reqsteelvol (list) [m3]: required base steel volume for N foundation points
            reqconvol (list) [m3]: required base concrete volume for N foundation points
            horzdesload (list) [N]: horizontal design load for N foundation points
            vertdesload (list) [N]: vertical design load for N foundation points
            loadattpnt (list) [N]: load attachment point for N foundation points
            padeyedesload (list) [N]: padyeye design load for N foundation points
            padeyetheta  (list) [rad]: padeye load angle from vertical axis for N foundation points
            padeyehorzload (list) [N]: padeye horizontal load for N foundation points
            padeyevertload (list) [N]: padeye vertical load for N foundation points
            gravbasedim (list): gravity base dimensions for N foundation points: base width (float) [m]
                                                                                 base length (float) [m]
                                                                                 base height (float) [m]
                                                                                 skirt height (float) [m]
                                                                                 shear key height (float) [m]
                                                                                 shear key number (int) [-]
                                                                                 shear key spacing (float) [m]
                                                                                 required concrete volume (float) [m3]
                                                                                 required steel volume (float) [m3]
            shallbasedim (list): shallow base dimensions for N foundation points: base width (float) [m]
                                                                                  base length (float) [m]
                                                                                  base height (float) [m]
                                                                                  skirt height (float) [m]
                                                                                  shear key height (float) [m]
                                                                                  shear key number (int) [-]
                                                                                  shear key spacing (float) [m]
                                                                                  required concrete volume (float) [m3]
                                                                                  required steel volume (float) [m3]
            piledim (list): pile dimensions for N foundation points:   pile diameter (float) [m],
                                                                       pile thickness (float) [m],
                                                                       pile length (float) [m],
                                                                       pile closed end (str) [-],
                                                                       pile component index (str) [-],
                                                                       pile grout diameter (float) [m]                                                           
            caissdim (list): suctioncaisson dimensions for N foundation points:  caisson diameter (float) [m]
                                                                                  caisson thickness (float) [m]
                                                                                  caisson length (float) [m]
                                                                                  caisson component index (str) [-] 
            platedim (list): directembedment anchor dimensions for N foundation points: plate width (float) [m]
                                                                                         plate length (float) [m]
                                                                                         plate thickness (float) [m]
                                                                                         embedment depth (float) [m]
            ancdim (list): drag embedment anchor dimensions for N foundation points:  anchor width (float) [m]
                                                                                      anchor depth (float) [m]
                                                                                      anchor height (float) [m]
                                                                                      anchor connecting size (float) [m]
            seldraganc (list) [-]: selected drag anchor type for N foundation points
            ancpendep (list) [m]: drag embedment anchor penetration depth for N foundation points
            selfoundcost (list) [m]: selected foundation cost for N foundation points
            pileitemcost (list) [euros/m]: pile foundation/anchor unit cost for N foundation points
            ancitemcost (list) [euros]: drag embedment anchor unit cost for N foundation points
            pilevolsteel (float) [m3]: pile steel volume for N foundation points
            shallvolsteel (float) [m3]: shallow foundation/anchor steel volume for N foundation points
            caissvolsteel (float) [m3]: suctioncaisson anchor steel volume for N foundation points
            gravvolsteel (float) [m3]: gravity foundation/anchor steel volume for N foundation points
            platevolsteel (list) [euros]: drag embedment anchor steel volume for N foundation points
            pilevolcon (float) [m3]: pile concrete volume for N foundation points
            shallvolcon (float) [m3]: shallow foundation/anchor concrete volume for N foundation points
            caissvolcon (float) [m3]: suctioncaisson anchor concrete volume for N foundation points
            gravvolcon (float) [m3]: gravity foundation/anchor concrete volume for N foundation points
            platevolcon (float) [m3]: drag embedment concrete volume for N foundation points
            pilevolgrout (float) [m3]: pile grout volume for N foundation points
            seldragancsubtyp (str) [-]: selected drag embedment anchor subtype for N foundation points
            seldeancsubtyp (str) [-]: selected directembedment anchor subtype for N foundation points
            selpilesubtyp (str) [-]: selected pile foundation/anchor subtype for N foundation points
            selshallsubtyp (str) [-]: selected shallow foundation/anchor subtype for N foundation points
            selfoundsubtypdict (dict): subtype dictionary for each potential foundation type and for N foundation points: keys: 'shallowfoundation' (str) [-],
                                                                                                                                'gravity' (str) [-],
                                                                                                                                'suctioncaisson' (str) [-],
                                                                                                                                'pile' (str) [-],
                                                                                                                                'directembedment' (str) [-],
                                                                                                                                'drag' (str) [-]
            soilfric (float) [-]: soil friction coefficient
            uniformfound (str) [-]: uniform foundation base dimensions (width = length): options: 'True'
                                                                                                  'False'
            shearkeys (str) [-]: shear key provision: options:  'True'
                                                                'False'
            basewidth (float) [m]: base width
            baseleng (float) [m]: base length
            mcoef (list) [-]: correction factor for cohesion, overburden and density: maximum (absolute) load direction (float) [-],
                                                                                      minimum (absolute) load direction (float) [-] 
            loadincc (list): load inclination factor ic: maximum (absolute) load direction (float) [-],
                                                         minimum (absolute) load direction (float) [-] 
            loadincq (list): load inclination factor iq: maximum (absolute) load direction (float) [-],
                                                         minimum (absolute) load direction (float) [-]
            loadincgamma (list): load inclination factor igamma: maximum (absolute) load direction (float) [-],
                                                                 minimum (absolute) load direction (float) [-] 
            kcorrfacc (list): cohesion correction factor kc: maximum (absolute) load direction (float) [-],
                                                             minimum (absolute) load direction (float) [-] 
            kcorrfacq (list): overburden correction factor kq: maximum (absolute) load direction (float) [-],
                                                               minimum (absolute) load direction (float) [-]
            kcorrfacgamma (list): density correction factor kgamma: maximum (absolute) load direction (float) [-],
                                                                    minimum (absolute) load direction (float) [-] 
            bearcapst (list): short-term bearing capacity: maximum (absolute) load direction (float) [N],
                                                           minimum (absolute) load direction (float) [N]                                                                     
            maxbearstress (list): maximum bearing stress: maximum (absolute) load direction (float) [N/m2],
                                                           minimum (absolute) load direction (float) [N/m2]
            transdepth (list): transition depth related to onset of grain crushing behaviour: maximum (absolute) load direction (float) [m],
                                                                                              minimum (absolute) load direction (float) [m]                                                           
            depthattfac (list): depth attenuation factor: maximum (absolute) load direction (float) [-],
                                                          minimum (absolute) load direction (float) [-]
            totnormloads (list): total normal loads: maximum (absolute) load direction (float) [N],
                                                     minimum (absolute) load direction (float) [N]                                                           
            shkeyres (list): shear key resistance: maximum (absolute) load direction (float) [N],
                                                   minimum (absolute) load direction (float) [N]                                                           
            nshkeys (list): number of shear keys: maximum (absolute) load direction (float) [-],
                                                  minimum (absolute) load direction (float) [-] 
            shkeyspc (list): shear key spacing: maximum (absolute) load direction (float) [m],
                                                minimum (absolute) load direction (float) [m]                                                   
            latcheck (str) [-]: lateral load check pass flag: options: 'True'
                                                                       'False'  
            ecccheck (str) [-]: eccentricity check pass flag: options: 'True'
                                                                       'False'
            bcapcheck (str) [-]: bearing capacity check pass flag: options: 'True'
                                                                            'False'    
            shkcheck (str) [-]: shear key check pass flag: options: 'True'
                                                                    'False' 
            basearea (float) [m2]: base surface area
            skirtheight (float) [m]: skirt height
            unshstrz (float) [N/m2]: undrained soil shear strength at specified depth
            trapsoilweight (float) [N]: trapped soil weight inside skirt
            passlatpresscoef (float) [-]: passive lateral resistance coefficient
            baseheight(float) [m]: base height
            reqvol (float) [m3]: required base volume
            reqbaseden (float) [m3]: required base density
            reqsteelvol (float) [m3]: required base steel volume
            reqconvol (float) [m3]: required base concrete volume
            effbaseleng (float) [m]: effective base length
            effbasewidth (float) [m]: effective base width
            effbasearea (float) [m2]: effective base area
            bcfntheta (float) [-]: bearing capacity factor ntheta
            bcfnq (float) [-]: bearing capacity factor nq
            bcfngamma (float) [-]: bearing capacity factor ngamma
            depthcoefc (float) [-]: embedment depth coefficient dc
            depthcoefq (float) [-]: embedment depth coefficient dq
            depthcoefgamma (float) [-]: embedment depth coefficient dgamma
            shpcoefc (float) [-]: foundation shape coefficient sc
            shpcoefq (float) [-]: foundation shape coefficient sq
            shpcoefgamma (float) [-]: foundation shape coefficient sgamma
            basecoefc (float) [-]: base inclination coefficient bc
            basecoefq (float) [-]: base inclination coefficient bq
            basecoefgamma (float) [-]: base inclination coefficient bgamma
            groundcoefc (float) [-]: ground inclination coefficient gc
            groundcoefq (float) [-]: ground inclination coefficient gq
            groundcoefgamma (float) [-]: ground inclination coefficient ggamma
            soilsen (float) [-]: cohesive soil sensitivity
            fracrelden (float) [-]: fractional relative density
            critconpres (float) [N/m2]: critical confining pressure
            effshstr (float) [N/m2]: effective shear strength at critical confining pressure
            efffricang (float) [rad]: effective friction angle alongside footing
            minspc (float) [m]: minimum shear key spacing
            basedim (list) [-]: base dimensions: basewidth (float) [m],
                                                 baseleng (float) [m],
                                                 baseheight (float) [m],
                                                 skirtheight (float) [m],
                                                 keyheight (float) [m],
                                                 nshkeys (int) [-]
                                                 shkeyspc (float) [m],
                                                 reqconvol (float) [m3],
                                                 reqsteelvol (float) [m3]
            configfootarea (float) [m2]: configuration footprint area
            configvol (float) [m3]: configuration volume
            pilecloseend (str) [-]: pile closed end: options:  'True'
                                                               'False'
            deflmax (float) [m]: pile maximum horizontal deflection
            k1coefcol (int) [-]: k1 coefficient column
            k1coefint (float) [-]: interpolation parameter
            k1coef (float) [-]: non-dimensional clay coefficient
            sgrcoefint (float) [-]: interpolation parameter
            sgrcoef (float) [N/m3]: subgrade soil reaction coefficient
            pilesoilstiff (float) [m]: relative pile-soil stiffness 
            maxdepthcoef (float) [-]: maximum depth coefficient
            piledefcoefay (float) [-]: pile deflection coefficient ay
            piledefcoefby (float) [-]: pile deflection coefficient by
            piledefcoefayint (float) [-]: interpolation parameter
            piledefcoefbyint (float) [-]: interpolation parameter
            latloadcap (float) [N]: lateral load capacity
            rockcomstr (float) [N/m2]: rock compressive strength
            effoverbdpressav (float) [N/m2]: effective average overburden pressure
            skinfricreslimint (float) [-]: interpolation parameter
            skinfricreslim (float) [N/m2]: maximum skin friction resistance limit
            skinfricresavup (float) [N/m2]: upward skin friction resistance
            pilesurfarea (float) [m2]: pile surface area
            upliftcap (float) [N]: uplift capacity
            effoverbdpresstip (float) [N/m2]: effective overburden pressure at pile tip
            unshstrtip (float) [N/m2]: undrained soil shear strength at pile tip
            soilbearcaptip (float) [N/m2]: unit soil bearing capacity at pile tip
            skinfricresavcomp (float) [N/m2]: compression skin friction resistance
            bcflimint (float) [-]: interpolation parameter
            bcflim (float) [-] bearing capacity factor limit
            soilbearcaplimint (float) [-]: interpolation parameter
            soilbearcaplim (float) [N/m2]: unit soil bearing capacity limit
            pilecaparea (float) [m2]: pile cap surface area
            bearcap (float) [N]: pile bearing capacity
            bearcaptip (float) [N]: pile tip bearing capacity
            totbearcap (float) [N]: total bearing capacity
            modratio (float) [-]: grout modular ratio          
            groutdiathkratio (float) [-]: grout diameter/thickness ratio
            pilegroutdia (float) [m]: grout outer diameter
            glpdratios (numpy.ndarray): grout coefficients: grouted length to pile diameter ratio (float) [-]
                                                            coefficent of grouted length to pile diameter ratio (float) [-]
            glpdratiosint (float) [-]: interpolation parameter
            groutlengpilediaratio (float) [-]: grout length to pile diameter ratio
            surfcondfactr (float) [-]: surface condition factor
            stifffact (float) [-]: stiffness factor
            groutpilebondstr (float) [-]: grout-pile bond strength
            pileperim (float) [m]: pile perimeter
            pilemomcoefamint (float) [-]: interpolation parameter
            pilemomcoefbmint (float) [-]: interpolation parameter
            pilecsa (float) [m2]: pile cross sectional area
            pilesecmod (float) [m3]: pile section modulus
            pilelengs (list) [m]: list of possible pile lengths
            pilemomcoefam (list) [-]: pile moment coefficients am
            pilemomcoefbm (list) [-]: pile moment coefficients bm
            depthcoef (float) [-]: pile depth coefficient
            maxmom (float) [Nm]: pile maximum moment
            pileloadten (float) [N]: pile tensile load
            pileloadcomp (float) [N]: pile compressive load
            maxstressten (float) [N/m2]: maxmium tensile stress
            maxstresscomp (float) [N/m2]: maxmium compressive stress
            pilecompind (str) [-]: pile component index
            pilegrouttyp (list) [-]: pile grout type for N foundation points
            caissmindia (float) [m]: caisson minimum diameter
            caissthk (float) [m]: caisson thickness
            caissyld (float) [N/m2]: caisson yield strength
            caissmod (float) [N/m2]: caisson modulus of elasticity
            caissdia (float) [m]: caisson diameter
            caisslengdiaratio (float) [-]: caisson length to diameter ratio
            caissleng (float) [m]: caisson length
            caissarea (float) [m2]: caisson major cross sectional area
            upliftcapfacnp (float) [-]: uplift capacity factor
            zetace (float) [-]: vertical embedment factor
            zetas (float) [-]: caisson shape factor
            latloadcapfacnh (float) [-]: lateral capacity factor nh
            envcoefa (float) [-]: failure envelope factor a
            envcoefb (float) [-]: failure envelope factor b
            failenv (float) [-]: failure envelope
            caisscompind (str) [-]: caisson component index
            platewidth (float) [m]: plate width
            plateleng (float) [m]: plate length
            pendepth (float) [m]: anchor penetration depth
            hcfncs (float) [-]: short-term holding capacity factors
            hcfnc (float) [-]: long-term holding capacity factors
            hcfncsint (float) [-]: interpolation parameter
            strredfactor (float) [-]: strength reduction factor
            platearea (float) [m2]: plate area
            hcfnqint (float) [-]: interpolation parameter
            hcfnq (float) [-]: holding capacity factors for drained soil condition
            stancholdcap (float) [N]: short-term anchor holding capacity 
            ltancholdcap (float) [N]: long-term anchor holding capacity 
            platethk (float) [m]: plate thickness
            stltancholdcap (float) [N]: short-term anchor holding capacity 
            reqmoorconnsize (float) [m]: required mooring connection size
            ancholdcap (float) [N]: anchor holding capacity
            ancpen (float) [m]: anchor penetration depth
            ancconnsize (float) [m]: anchor connection size
            totfoundcostdict (list): total foundation cost dictionary: keys: 'shallowfoundation' (float) [euros]: shallow foundation/anchor
                                                                             'gravity' (float) [euros]: gravity foundation/anchor
                                                                             'pile' (float) [euros]: pile foundation/anchor
                                                                             'suctioncaisson' (float) [euros]: suctioncaisson anchor
                                                                             'directembedment' (float) [euros]: directembedment anchor
                                                                             'drag' (float) [euros]: drag embedment anchor
            pilecost (list) [euros]: pile cost for N foundation points
            anccost (list) [euros]: drag anchor cost for N foundation points
            platecost (list) [euros]: directembedment cost for N foundation points
            gravcost (list) [euros]: gravity foundation/anchor cost for N foundation points
            caisscost (list) [euros]: suctio caisson cost for N foundation points
            shallcost (list) [euros]: shallow foundation/anchor cost for N foundation points
            sorttotfoundcost (list) [euros]: lowest foundation cost for N foundation points
            foundinstdep (list) [m]: foundation installation depth for N foundation points
            foundweight (list) [kg]: foundation dry mass for N foundation points
            tabind (list) [-]: table indices
            devind (int) [-]: device index
            foundinstparams (list) [-]: foundation installation parameters for N foundation points: device number (int) [-], 
                                                                                                    foundation number (int) [-],
                                                                                                    foundation type (str) [-]', 
                                                                                                    foundation subtype (str) [-], 
                                                                                                    x coord (float) [-], 
                                                                                                    y coord (float) [-], 
                                                                                                    length (float) [m], 
                                                                                                    width (float) [m], 
                                                                                                    height (float) [m], 
                                                                                                    installation depth (float) [m], 
                                                                                                    dry mass (float) [kg], 
                                                                                                    grout type [-], 
                                                                                                    grout volume (float) [m3]
            foundinsttab (pandas) [-]: foundation installation requirements pandas table:   device number (int) [-], 
                                                                                            foundation number (int) [-],
                                                                                            foundation type (str) [-]', 
                                                                                            foundation subtype (str) [-], 
                                                                                            x coord (float) [-], 
                                                                                            y coord (float) [-], 
                                                                                            length (float) [m], 
                                                                                            width (float) [m], 
                                                                                            height (float) [m], 
                                                                                            installation depth (float) [m], 
                                                                                            dry mass (float) [kg], 
                                                                                            grout type [-], 
                                                                                            grout volume (float) [m3]            
                                                                                            
            foundbom (dict) [-]: device foundation dictionary for N foundation points: foundation type (str) [-],
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
            foundhier (list) [-]: foundation type and grout (if applicable)
        
        Functions:
            foundsub: retrieves umbilical details (if relevant)
            foundsel: uses prefound or identifies potential foundation types
            founddes: foundation system design and analysis:    
                                                            foundcompret: component retrieval
                                                            buriedline: buried mooring line analysis
                                                                    buriedfunc: buried mooring line sub-function
                                                            shallowdes: shallow and gravity foundation/anchor design
                                                            piledes: pile foundation/anchor design
                                                                    coefssr: coefficient of subgrade soil reaction calculation
                                                            caissdes: suctioncaisson anchor design
                                                            directembedanc: directembedment anchor design
                                                            dragembedanc: drag embedment anchor design                                                            
            foundcost: calculates foundation/anchor system capital cost
            foundinst: calculates installation parameters
            foundbom: creates foundation/anchor system bill of materials
            foundhierarchy: creates foundation/anchor system hierarchy 
    
    """ 
    
    def __init__(self, variables):        
        super(Found, self).__init__(variables)
        
    def foundsub(self,deviceid,systype,foundloc,syscog):
        if systype in ('wavefloat','tidefloat'):
            super(Found, self).moorsub()
        elif systype in ('wavefixed','tidefixed'):  
            combloadmag = [0 for row in range(0,2)]
            rotorloadlist = [0 for row in range(0,2)]
            steadycurrentlist = [0 for row in range(0,2)]
            syswaveloadmaxlist = [0 for row in range(0,2)]
            meanwavedriftlist = [0 for row in range(0,2)]
            windloadlist = [0 for row in range(0,2)]
            totsysstatloadslist = [0 for row in range(0,2)]
            for l in range(0,2):
                if l == 1:
                    self.bathysysorig = (self.bathysysorig - (self._variables.wlevmax 
                                                            + self._variables.wlevmin)) 
                super(Found, self).sysstat(self._variables.sysvol,
                                            self._variables.sysmass)
                super(Found, self).sysstead(systype,
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
                super(Found, self).syswave(deviceid,
                                            systype,
                                            self._variables.syswidth,
                                            self._variables.syslength,
                                            self._variables.sysheight,
                                            self._variables.sysprof,
                                            self._variables.sysvol,
                                            self._variables.sysrough) 
                if self._variables.hs:
                    combloadmag[l] = math.sqrt((self.rotorload[0] + self.steadycurrent[0]
                                + self.syswaveloadmax[0] + self.meanwavedrift[self.syswaveloadmaxind][0]
                                + self.windload[0] + self.totsysstatloads[0]) ** 2.0 
                                + (self.rotorload[1] + self.steadycurrent[1]
                                + self.syswaveloadmax[1] + self.meanwavedrift[self.syswaveloadmaxind][1]
                                + self.windload[1] + self.totsysstatloads[1]) ** 2.0 
                                + (self.rotorload[2] + self.steadycurrent[2]
                                + self.syswaveloadmax[2] + self.meanwavedrift[self.syswaveloadmaxind][2]
                                + self.windload[2] + self.totsysstatloads[2]) ** 2.0)
                else:
                    combloadmag[l] = math.sqrt((self.rotorload[0] + self.steadycurrent[0]
                                + self.windload[0] + self.totsysstatloads[0]) ** 2.0 
                                + (self.rotorload[1] + self.steadycurrent[1]
                                + self.windload[1] + self.totsysstatloads[1]) ** 2.0 
                                + (self.rotorload[2] + self.steadycurrent[2]
                                + self.windload[2] + self.totsysstatloads[2]) ** 2.0)
                rotorloadlist[l] = self.rotorload
                steadycurrentlist[l] = self.steadycurrent
                syswaveloadmaxlist[l] = self.syswaveloadmax
                meanwavedriftlist[l] = self.meanwavedrift
                windloadlist[l] = self.windload
                totsysstatloadslist[l] = self.totsysstatloads
                
                if (l == 1 and combloadmag[l] < combloadmag[l-1]):
                    self.rotorload = rotorloadlist[0]
                    self.steadycurrent = steadycurrentlist[0]
                    self.syswaveloadmax = syswaveloadmaxlist[0]
                    self.meanwavedrift = meanwavedriftlist[0]
                    self.windload = windloadlist[0]
                    self.totsysstatload = totsysstatloadslist[0]                   
                
        elif systype == 'substation':
            combloadmag = [0 for row in range(0,2)]
            steadycurrentlist = [0 for row in range(0,2)]
            syswaveloadmaxlist = [0 for row in range(0,2)]
            meanwavedriftlist = [0 for row in range(0,2)]
            windloadlist = [0 for row in range(0,2)]
            totsysstatloadslist = [0 for row in range(0,2)]
            for l in range(0,2):            
                if l == 1:
                    self.bathysysorig = (self.bathysysorig - (self._variables.wlevmax 
                                                            + self._variables.wlevmin)) 
                super(Found, self).sysstat(self._variables.substparams.ix[deviceid,'subvol'],
                                            self._variables.substparams.ix[deviceid,'submass'])
                super(Found, self).sysstead(systype,
                                            self._variables.substparams.ix[deviceid,'subwidth'],
                                            self._variables.substparams.ix[deviceid,'sublength'],
                                            self._variables.substparams.ix[deviceid,'subheight'],
                                            self._variables.substparams.ix[deviceid,'suborienang'],
                                            self._variables.substparams.ix[deviceid,'subprof'],
                                            self._variables.substparams.ix[deviceid,'subdryfa'],
                                            self._variables.substparams.ix[deviceid,'subdryba'],
                                            self._variables.substparams.ix[deviceid,'subwetfa'],
                                            self._variables.substparams.ix[deviceid,'subwetba'],
                                            self._variables.substparams.ix[deviceid,'subvol'],
                                            self._variables.substparams.ix[deviceid,'subrough'])
                super(Found, self).syswave(deviceid,
                                            systype,
                                            self._variables.substparams.ix[deviceid,'subwidth'],
                                            self._variables.substparams.ix[deviceid,'sublength'],
                                            self._variables.substparams.ix[deviceid,'subheight'],
                                            self._variables.substparams.ix[deviceid,'subprof'],
                                            self._variables.substparams.ix[deviceid,'subvol'],
                                            self._variables.substparams.ix[deviceid,'subrough']) 
                
                if self.steadycurrentremflag[self.syswaveloadmaxind] == 'True':
                    self.steadycurrent = [0.0, 0.0, 0.0]
                if self._variables.hs:
                    combloadmag[l] = math.sqrt((self.steadycurrent[0]
                                    + self.syswaveloadmax[0] + self.meanwavedrift[self.syswaveloadmaxind][0]
                                    + self.windload[0] + self.totsysstatloads[0]) ** 2.0 
                                    + (self.steadycurrent[1]
                                    + self.syswaveloadmax[1] + self.meanwavedrift[self.syswaveloadmaxind][1]
                                    + self.windload[1] + self.totsysstatloads[1]) ** 2.0 
                                    + (self.steadycurrent[2]
                                    + self.syswaveloadmax[2] + self.meanwavedrift[self.syswaveloadmaxind][2]
                                    + self.windload[2] + self.totsysstatloads[2]) ** 2.0)
                else:
                    combloadmag[l] = math.sqrt((self.steadycurrent[0]
                                    + self.windload[0] + self.totsysstatloads[0]) ** 2.0 
                                    + (self.steadycurrent[1]
                                    + self.windload[1] + self.totsysstatloads[1]) ** 2.0 
                                    + (self.steadycurrent[2]
                                    + self.windload[2] + self.totsysstatloads[2]) ** 2.0)
                steadycurrentlist[l] = self.steadycurrent
                syswaveloadmaxlist[l] = self.syswaveloadmax
                meanwavedriftlist[l] = self.meanwavedrift
                windloadlist[l] = self.windload
                totsysstatloadslist[l] = self.totsysstatloads
        self.totfoundslopeloads = [0.0, 0.0]
        if systype in ('wavefloat', 'tidefloat'):
            self.quanfound = len(self.fairloc)
        elif systype in ('wavefixed', 'tidefixed','substation'):
            self.quanfound = len(self.foundloc)
            
        self.foundrad = [0 for row in range(self.quanfound)]
        self.foundradadd = [0 for row in range(self.quanfound)]
        """ Calculate load components and apply safety factor """
        if systype in ("wavefloat","tidefloat"):          
            self.totfoundsteadloads = self.totancsteadloads
            
            self.totfoundloads = [[0 for col in range(3)] for row 
                                    in range(self.quanfound)]
            self.horzfoundloads = [[0 for col in range(3)] for row 
                                    in range(self.quanfound)]
            self.vertfoundloads = [[0 for col in range(3)] for row 
                                    in range(self.quanfound)]
            for j in range(0,self.quanfound): 
                """ Steady anchor load calculation includes static loads """                
                self.totfoundloads[j] = self.totfoundsteadloads[j]
                self.horzfoundloads[j] = self.totfoundloads[j][0:2]
                self.vertfoundloads[j] = self.totfoundloads[j][2]
            self.sysbasemom = [0.0, 0.0]
            self.moorsf = self._variables.foundsf                          
            
        elif systype in ("wavefixed","tidefixed",'substation'):
            """ Vector load magnitudes and directions. Angles are specified 
                w.r.t to horizontal and vertical planes from the x axis and 
                x-y plane """                                
            self.totfoundstatloadmag = math.sqrt(self.totsysstatloads[0] ** 2.0 
                + self.totsysstatloads[1] ** 2.0 + self.totsysstatloads[2] ** 2.0)
            self.totfoundstatloaddirs = np.nan_to_num([math.atan(np.divide(
                self.totsysstatloads[1],self.totsysstatloads[0])), 
                math.acos(np.divide(self.totsysstatloads[2],
                self.totfoundstatloadmag))])            
    
            self.sysbasemom = [0.0, 0.0] 
            self.tothorzfoundload = [0.0, 0.0] 
            self.totfoundloads = np.array([[0 for col in range(2)] for row 
                                            in range(self.quanfound)])
            for i in range(0,2):
                """ System base overturning moment. Note Mx = 1, My = 0 """   
                if not self._variables.hs:
                    self.sysbasemom[i] = (self.rotorload[i] 
                        * self._variables.hubheight 
                        + self.steadycurrent[i] * self.eqcurrentloadloc 
                        + self.windload[i] * (self.bathysysorig + 0.5 
                        * self.dryheight) + syscog[i] 
                        * math.fabs(self.totsysstatloads[2]))
                    """ Horizontal shear loads at base """
                    self.tothorzfoundload[i] = (self.rotorload[i] 
                        + self.steadycurrent[i] + self.windload[i])
                else:
                    self.sysbasemom[i] = (self.rotorload[i] 
                        * self._variables.hubheight 
                        + self.steadycurrent[i] * self.eqcurrentloadloc 
                        + self.syswaveloadmax[i] * self.eqhorzwaveloadloc 
                        + self.meanwavedrift[self.syswaveloadmaxind][i] 
                        * self.bathysysorig + self.windload[i] 
                        * (self.bathysysorig + 0.5 * self.dryheight) 
                        + syscog[i] * math.fabs(
                        self.totsysstatloads[2]))
                    """ Horizontal shear loads at base """
                    self.tothorzfoundload[i] = (self.rotorload[i] 
                        + self.steadycurrent[i] + self.syswaveloadmax[i] 
                        + self.meanwavedrift[self.syswaveloadmaxind][i] 
                        + self.windload[i])                   
            """ Least squares solution to linear system of equations of 
                vertical reaction forces """
            for j in range(0,self.quanfound):
                self.foundloc[j][0] = self.foundlocglob[j][0]
                self.foundloc[j][1] = self.foundlocglob[j][1] 
                self.foundloc[j][2] = self.foundlocglob[j][2]
                        
            a = np.array([np.array([1.0 for row in range(self.quanfound)]), 
                          self.foundloc[:,1], 
                          self.foundloc[:,0]])
            b = np.array([math.fabs(self.totsysstatloads[2]),
                          self.sysbasemom[1],self.sysbasemom[0]])
            self.reactfoundloads = np.linalg.lstsq(a, b)            
            """ To keep with convention a negative sign is added to the 
                foundation reaction loads """
            self.vertfoundloads = []     
            for loads in self.reactfoundloads[0]:
                self.vertfoundloads.append(-loads)
                
            self.horzfoundloads = [[self.tothorzfoundload[0] / self.quanfound, 
                                    self.tothorzfoundload[1] / self.quanfound] 
                                    for row in range(self.quanfound)] 

        """ Default is moderate seabed slope """        
        self.seabedslpk = 'moderate'
        self.seabedslp = [[0 for col in range(2)] for row 
                            in range(self.quanfound)]
        self.totfoundslopeloads = [[0 for col in range(2)] for row 
                            in range(self.quanfound)]      
        
        for j in range(0,self.quanfound):
            """ Bathymetry at each foundation location """            
            self.foundloc[j][2] = self.foundlocglob[j][2]
            """ Maximum apparent radius to outer foundation position """
            self.foundrad[j] = math.sqrt(self.foundloc[j][0] ** 2.0 
                + self.foundloc[j][1] ** 2.0)
         
            """ X-Y seabed slope estimated from bathymetry grid points nearest 
                to each foundation location """             
            self.deltabathyy = (np.mean([self.gpnear[2][2],self.gpnear[3][2]])  
                - np.mean([self.gpnear[0][2],self.gpnear[1][2]]))
            self.deltabathyx = (np.mean([self.gpnear[1][2],self.gpnear[3][2]]) 
                - np.mean([self.gpnear[0][2],self.gpnear[2][2]]))
            self.seabedslp[j][0] = math.atan(self.deltabathyx 
                                / self._variables.bathygriddeltax)
            self.seabedslp[j][1] = math.atan(self.deltabathyy 
                                / self._variables.bathygriddeltay)
            if (math.fabs(self.seabedslp[j][0]) > 10.0 * math.pi / 180.0
                or math.fabs(self.seabedslp[j][1]) > 10.0 * math.pi / 180.0):                
                self.seabedslpk = 'steep'  
                
            """ Load component parallel to downslope """
            for i in range(0,2):
                if self.seabedslp[j][i] != 0:
                    self.totfoundslopeloads[j][i] = (self.horzfoundloads[j][i] 
                        * math.cos(self.seabedslp[j][i]) 
                        - self.vertfoundloads[j] 
                        * math.sin(self.seabedslp[j][i]))
                    if self.seabedslp[j][i] < 0:
                        self.totfoundslopeloads[j][i] = -self.totfoundslopeloads[j][i]  
                elif self.seabedslp[j][i] == 0:
                    self.totfoundslopeloads[j][i] = self.horzfoundloads[j][i]

        self.maxloadindex = [[0 for col in range(2)] for row 
            in range(self.quanfound)]
        self.maxloadvalue = [[0 for col in range(2)] for row 
            in range(self.quanfound)]
        self.minloadindex = [[0 for col in range(2)] for row 
            in range(self.quanfound)]
        self.minloadvalue = [[0 for col in range(2)] for row 
            in range(self.quanfound)]
        self.totfoundslopeloadsabs = [[0 for col in range(2)] for row 
            in range(self.quanfound)]
        self.loaddir = [0 for row in range(self.quanfound)]
        self.loadmag = [0 for row in range(self.quanfound)]
        for j in range(0,self.quanfound):            
            for i in range(0,2):
                self.totfoundslopeloadsabs[j][i] = math.fabs(
                    self.totfoundslopeloads[j][i])
                self.maxloadindex[j], self.maxloadvalue[j] = max(enumerate(
                    self.totfoundslopeloadsabs[j]), key=operator.itemgetter(1))
                self.minloadindex[j], self.minloadvalue[j] = min(enumerate(
                    self.totfoundslopeloadsabs[j]), key=operator.itemgetter(1))
                self.seabedslp[j][0] = math.fabs(self.seabedslp[j][0])
                self.seabedslp[j][1] = math.fabs(self.seabedslp[j][1])        
            
            self.loaddir[j] = np.sign(self.vertfoundloads[j])
            self.loadmag[j] = math.sqrt(self.horzfoundloads[j][0] ** 2.0 
                + self.horzfoundloads[j][1] ** 2.0 
                + self.vertfoundloads[j] ** 2.0)        
        
    def foundsel(self,systype): 
        """ Select suitable foundation types """ 
        self.possfoundtyp = [0 for row in range(self.quanfound)]
        self.foundtyps = {'shallowfoundation', 
                          'gravity', 
                          'pile', 
                          'suctioncaisson', 
                          'directembedment', 
                          'drag'}
        
        """ Selection of possible foundations """
        if self._variables.prefound: 
            module_logger.info('Foundation or anchor type already selected')
            if self._variables.prefound[0:3] == 'uni':
                self.founduniformflag = 'True'
                self.prefound = self._variables.prefound[6:]
            else:
                self.founduniformflag = 'False'
                self.prefound = self._variables.prefound

            self.selfoundtyp = [[self.prefound] for row 
                                    in range(self.quanfound)]
            self.possfoundtyp = [([(self.prefound, 0)]) for row 
                                    in range(self.quanfound)]           
        else:    
            self.founduniformflag = 'False'
            self.foundsoiltypdict = {'vsc': {'shallowfoundation': 1, 
                                             'gravity': 1, 
                                             'pile': 2, 
                                             'suctioncaisson': 1, 
                                             'directembedment': 1, 
                                             'drag': 1},
                    'sc': {'shallowfoundation': 1, 
                           'gravity': 1, 
                           'pile': 2, 
                           'suctioncaisson': 1, 
                           'directembedment': 1, 
                           'drag': 1},
                    'fc': {'shallowfoundation': 1, 
                           'gravity': 1, 
                           'pile': 1, 
                           'suctioncaisson': 88, 
                           'directembedment': 1, 
                           'drag': 1},
                    'stc': {'shallowfoundation': 1, 
                            'gravity': 1, 
                            'pile': 1, 
                            'suctioncaisson': 88, 
                            'directembedment': 1, 
                            'drag': 1},
                    'ls': {'shallowfoundation': 1, 
                           'gravity': 1, 
                           'pile': 1, 
                           'suctioncaisson': 1, 
                           'directembedment': 1, 
                           'drag': 1},
                    'ms': {'shallowfoundation': 1, 
                           'gravity': 1, 
                           'pile': 1, 
                           'suctioncaisson': 1, 
                           'directembedment': 1, 
                           'drag': 1},                               
                    'ds': {'shallowfoundation': 1, 
                           'gravity': 1, 
                           'pile': 1, 
                           'suctioncaisson': 1, 
                           'directembedment': 1, 
                           'drag': 1},
                    'hgt': {'shallowfoundation': 2, 
                            'gravity': 1, 
                            'pile': 1, 
                            'suctioncaisson': 88, 
                            'directembedment': 88, 
                            'drag': 88},
                    'cm': {'shallowfoundation': 2, 
                           'gravity': 1, 
                           'pile': 1, 
                           'suctioncaisson': 88, 
                           'directembedment': 88, 
                           'drag': 88},                    
                    'gc': {'shallowfoundation': 88, 
                           'gravity': 1, 
                           'pile': 88, 
                           'suctioncaisson': 88, 
                           'directembedment': 88, 
                           'drag': 88},
                    'src': {'shallowfoundation': 88, 
                            'gravity': 1, 
                            'pile': 1, 
                            'suctioncaisson': 88, 
                            'directembedment': 88, 
                            'drag': 88},
                    'hr': {'shallowfoundation': 88, 
                           'gravity': 1, 
                           'pile': 2, 
                           'suctioncaisson': 88, 
                           'directembedment': 88, 
                           'drag': 88}}
            self.foundslopedict = {'moderate': {'shallowfoundation': 1, 
                                                'gravity': 1, 
                                                'pile': 1, 
                                                'suctioncaisson': 1, 
                                                'directembedment': 1, 
                                                'drag': 1},
                     'steep': {'shallowfoundation': 88, 
                               'gravity': 88, 
                               'pile': 1, 
                               'suctioncaisson': 2, 
                               'directembedment': 2, 
                               'drag': 88}}
            self.foundsoildepdict = {'none': {'shallowfoundation': 88, 
                                              'gravity': 1, 
                                              'pile': 1, 
                                              'suctioncaisson': 88, 
                                              'directembedment': 88, 
                                              'drag': 88},
                     'veryshallow': {'shallowfoundation': 1, 
                                     'gravity': 1, 
                                     'pile': 1, 
                                     'suctioncaisson': 88, 
                                     'directembedment': 88, 
                                     'drag': 88},
                     'shallow': {'shallowfoundation': 1, 
                                 'gravity': 1, 
                                 'pile': 1, 
                                 'suctioncaisson': 2, 
                                 'directembedment': 88, 
                                 'drag': 2},
                     'moderate': {'shallowfoundation': 1, 
                                  'gravity': 1, 
                                  'pile': 1, 
                                  'suctioncaisson': 1, 
                                  'directembedment': 1, 
                                  'drag': 1},
                     'deep': {'shallowfoundation': 1, 
                              'gravity': 1, 
                              'pile': 1, 
                              'suctioncaisson': 1, 
                              'directembedment': 1, 
                              'drag': 1}}
            self.foundloaddirdict = {'down': {'shallowfoundation': 1, 
                                              'gravity': 1, 
                                              'pile': 1, 
                                              'suctioncaisson': 1, 
                                              'directembedment': 88, 
                                              'drag': 2},
                     'uni': {'shallowfoundation': 1, 
                             'gravity': 1, 
                             'pile': 1, 
                             'suctioncaisson': 1, 
                             'directembedment': 1, 
                             'drag': 1},
                     'up': {'shallowfoundation': 1, 
                            'gravity': 1, 
                            'pile': 1, 
                            'suctioncaisson': 1, 
                            'directembedment': 1, 
                            'drag': 2}}
            self.foundloadmagdict = {'low': {'shallowfoundation': 1, 
                                             'gravity': 1, 
                                             'pile': 2, 
                                             'suctioncaisson': 2, 
                                             'directembedment': 1, 
                                             'drag': 1},
                     'moderate': {'shallowfoundation': 2, 
                                  'gravity': 2, 
                                  'pile': 1, 
                                  'suctioncaisson': 1, 
                                  'directembedment': 2, 
                                  'drag': 1},
                     'high': {'shallowfoundation': 88, 
                              'gravity': 2, #This is to be discussed. HMGE says that gravity foundations don't work well for large loads
                              'pile': 1, 
                              'suctioncaisson': 1, 
                              'directembedment': 88, 
                              'drag': 88}}  
            
            for j in range(0,self.quanfound): 
                
                if self.soildep[j] == 0.0:
                    self.soildepk = 'none'
                elif self.soildep[j] > 0.0 and self.soildep[j] <= 1.0:
                    self.soildepk = 'veryshallow'
                elif self.soildep[j] > 1.0 and self.soildep[j] <= 6.0:
                    self.soildepk = 'shallow'
                elif self.soildep[j] > 6.0 and self.soildep[j] <= 25.0:
                    self.soildepk = 'moderate'
                elif self.soildep[j] > 25.0 or math.isinf(self.soildep[j]):
                    self.soildepk = 'deep'
                if (self.soildep[j] == 'shallow' 
                    and self.soiltyp[j] in ('vsc','sc') 
                    and systype in ("wavefloat","tidefloat")):
                    self.foundsoildepdict.get('shallow')['drag'] = 88
                elif (self.soildep[j] == 'moderate' 
                    and self.soiltyp[j] in ('vsc','sc') 
                    and systype in ("wavefloat","tidefloat")):
                    self.foundsoildepdict.get('shallow')['drag'] = 2    
                if self.loadmag[j] < 444.8e3:
                    self.loadmagk = 'low'
                elif self.loadmag[j] >= 444.8e3 and self.loadmag[j] <= 4448.2e3:
                    self.loadmagk = 'moderate'
                elif self.loadmag[j] > 4448.2e3 :
                    self.loadmagk = 'high' 
                if self.loaddir[j] == 1.0:
                    self.loaddirk = 'up'
                elif self.loaddir[j] == -1.0:
                    self.loaddirk = 'down'
                elif self.loaddir[j] == 0.0:
                    self.loaddirk = 'uni'
                
                for stkey in self.foundsoiltypdict:
                    self.foundsoiltyp = self.foundsoiltypdict.get(
                                                            self.soiltyp[j])          
                for slkey in self.foundslopedict:
                    self.foundslp = self.foundslopedict.get(self.seabedslpk)  
                for sdkey in self.foundsoildepdict:
                    self.founddep = self.foundsoildepdict.get(self.soildepk)   
                for ldkey in self.foundloaddirdict:
                    self.foundloaddir = self.foundloaddirdict.get(
                                                        self.loaddirk) 
                for lmkey in self.foundloadmagdict:
                    self.foundloadmag = self.foundloadmagdict.get(
                                                        self.loadmagk)  
                
                self.foundmatrixsum = map(sum, zip(self.foundsoiltyp.values(),
                    self.foundslp.values(), self.founddep.values(),
                    self.foundloaddir.values(),self.foundloadmag.values()))
                self.foundmatsumdict = dict(zip(self.foundtyps, 
                                                self.foundmatrixsum))  
                
                self.unsuitftyps = []
                for posftyps in self.foundmatsumdict: 
                    """ Remove unsuitable foundations with scores above maximum 
                        permissible limit of 10 """
                    if self.foundmatsumdict[posftyps] > 10.0:
                        self.unsuitftyps.append(posftyps)
                for unsuitftyps in self.unsuitftyps:      
                    del self.foundmatsumdict[unsuitftyps]
                self.ancs = {'drag', 'directembedment', 'suctioncaisson'}   
                
                """ Use mooring system loads if system is floating and exclude 
                    anchors if system is fixed """            
                if systype in ("wavefixed","tidefixed"):
                    for akey in self.ancs:  
                        if akey in self.foundmatsumdict:
                            del self.foundmatsumdict[akey]   
                """ Sort selected solutions if required. Note: ranking is 
                    currently not used """
                self.possfoundtyp[j] = sorted(self.foundmatsumdict.items(), 
                                                key=operator.itemgetter(1))             
    def founddes(self,systype):      
        """Foundation design """                
        self.foundvolsteeldict = [{} for row in range(self.quanfound)]
        self.foundvolcondict = [{} for row in range(self.quanfound)]
        self.foundvolgroutdict = [{} for row in range(self.quanfound)]
        self.selfoundgrouttypdict = [{} for row in range(self.quanfound)]     
        self.founddimdict = [{} for row in range(self.quanfound)]    
        self.foundcostdict = [{} for row in range(self.quanfound)]
        self.founddesfail = ['False' for row in range(self.quanfound)]
        self.pileyld = 0.0
        klim = 50
        
        def foundcompret(complim):
                if posftyps[0] in ('pile', 'suctioncaisson'):                    
                    piledias = []                    
                    for comps in self._variables.compdict:                
                        if self._variables.compdict[comps]['item2'] == 'pile':
                            if (self._variables.compdict[comps]['item6'][0] 
                                > complim):
                                piledias.append((comps,
                                    self._variables.compdict[comps]['item6'][0]))
                    if piledias:
                        """ Find smallest diameter pile first """                        
                        pilemindia = min(piledias, key=operator.itemgetter(1))
                        pilethk = self._variables.compdict[pilemindia[0]]['item6'][1]
                        pileyld = self._variables.compdict[pilemindia[0]]['item5'][0]
                        pilemod = self._variables.compdict[pilemindia[0]]['item5'][1]
                    else:
                        pilemindia = 0.0
                        pilethk = 0.0
                        pileyld = 0.0
                        pilemod = 0.0
                    self.pileyld = pileyld
                    return pilemindia, pilethk, pileyld, pilemod
                    
                elif posftyps[0] == 'drag':
                    ancweights = []
                    for comps in self._variables.compdict:                         
                        """ Find lowest weight anchor that is compatible 
                            with chain size """
                        if (self._variables.compdict[comps]['item2'] 
                            == 'drag anchor'):  
                            if (self._variables.compdict[comps]['item7'][1]
                                > complim[0] 
                                and self._variables.compdict[comps]['item6'][3] 
                                == complim[1]):
                                """ Use dry mass of anchor """
                                ancweights.append((comps,
                                    self._variables.compdict[comps]['item7'][0]))                    
                    if not ancweights: 
                        """ If anchor cannot be selected, allow connector size 
                            to be increased """                               
                        for comps in self._variables.compdict: 
                            if (self._variables.compdict[comps]['item2'] 
                                == 'drag anchor'):
                                """ Find lowest weight anchor """
                                if (self._variables.compdict[comps]['item7'][1]
                                    > complim[0] 
                                    and self._variables.compdict[comps]['item6'][3] 
                                    > complim[1]):
                                    """ Use dry mass of anchor """
                                    ancweights.append((comps,
                                     self._variables.compdict[comps]['item7'][0]))   
                        if not ancweights:
                            """ If a suitable anchor still can not be found, abort search """
                            ancweights = [0, 0]
                    
                    ancminweight = min(ancweights, key=operator.itemgetter(1))
                    
                    
                    if ancminweight[0] != 0:
                        if self.soiltyp[j] in ('vsc', 'sc'):
                            """ Coefficients for soft clays and muds """
                            anccoef = self._variables.compdict[ancminweight[0]]['item9']['soft']
                        elif self.soiltyp[j] in ('fc', 'stc', 'ls', 'ms', 'ds'):
                            """ Coefficients for stiff clays and sands """
                            anccoef = self._variables.compdict[ancminweight[0]]['item9']['sand']
                    else:
                        anccoef = [0.0, 0.0]
                    return ancminweight, anccoef
                    
        def buriedline(embeddepth, resdesload, resdesloadang, omega):
            """ Contribution of buried line to load reduction (for direct 
                embeddment and pile anchors) """
            
            """ Bearing resistance based on bearing area (chain, DNV-RP-E301), 
                bearing capacity factor and average undrained shear 
                strength """                

            """ Line bearing capacity factor """
            if self.soiltyp[j] in ('vsc', 'sc', 'fc', 'stc'):
                """ Shear strength averaged over embeddment depth """
                unshstrav = ((self.unshstr[1] 
                    + self.unshstr[1] + self.unshstr[0] 
                    * embeddepth) / 2.0)
                """ Default line bearing capacity factor and bearing area from 
                    DNV-RP-E301. Note: depth variation not included """
                linebcfnc = 11.5
                """ Bearing resistance per unit length of chain """
                bearres = 2.5 * self.moorconnsize * linebcfnc * unshstrav                
            elif self.soiltyp[j] in ('ls', 'ms', 'ds'):
                """ Linear interpolation using bearing capacity factors from 
                    Meyerhoff and Adams 1968  """
                linebcfnqint = interpolate.interp1d(self._variables.linebcf[:,0], 
                                                    self._variables.linebcf[:,1],
                                                    bounds_error=False)
                linebcfnq = linebcfnqint(self.dsfangrad * 180.0 / math.pi)
                """ Bearing resistance per unit length of chain """
                bearres = linebcfnq * self.moorconnsize * self.soilweight * embeddepth 
                
            """ Friction coefficient between chain and soil """
            if self._variables.seaflrfriccoef:
                friccoef = self._variables.seaflrfriccoef
            else:
                friccoef = self.seaflrfriccoef
            """ Analytical approach to find line angle at padeye from Neubecker 
                and Randolph 1995 """
            def buriedfunc(theta_a): 
                if theta_a > math.pi / 4.0:
                    theta_a = math.pi / 4.0
                return math.sqrt((2.0 * embeddepth * bearres * math.exp(
                    friccoef * (theta_a - resdesloadang) )) / resdesload 
                    + resdesloadang ** 2.0)
                
            theta_a = optimize.fixed_point(buriedfunc, 0.0)                
            
            """ Maximum angle from horizontal is 90 degrees """
            if theta_a > math.pi / 4.0:
                theta_a = math.pi / 4.0
            
            """ Distance between anchor and dip down point. Horizontal length 
                calculated 1mm below seafloor  """                    
            x = -embeddepth * math.log(1.0e-3 / embeddepth) / theta_a
            """ Embedded length of line """
            le = (x + (embeddepth - omega / (bearres / embeddepth)) 
                    * theta_a / 4.0)     
            ten_a = ((resdesload - friccoef * omega * le) / math.exp(
                friccoef * (theta_a - resdesloadang))) 
                

            return ten_a, theta_a    
            
        def shallowdes(basewidth, baseleng, shearkeys, posftyps): 
            """ Design loop """ 
            mcoef = [0 for col in range(2)]
            loadincc = [0 for col in range(2)]
            loadincq = [0 for col in range(2)]
            loadincgamma = [0 for col in range(2)]
            kcorrfacc = [0 for col in range(2)]
            kcorrfacq = [0 for col in range(2)]
            kcorrfacgamma = [0 for col in range(2)]
            bearcapst = [0 for col in range(2)]
            maxbearstress = [0 for col in range(2)]
            transdepth = [0 for col in range(2)]
            depthattfac = [0 for col in range(2)]
            totnormloads = [0 for col in range(2)]
            shkeyres = [0 for col in range(2)]
            nshkeys = [0 for col in range(2)]
            shkeyspc = [0 for col in range(2)]
            resdesload = [0 for row in range(self.quanfound)]
            
            """ Note: Only lateral and eccentricity checks are carried out for 'other' soil types """
            
            latcheck = 'False'
            ecccheck = 'False'                     
            bcapcheck = 'False'
            if posftyps[0] == 'shallowfoundation':
                shkcheck = 'False'
            else: shkcheck = 'True'
            basewidth = 1.0
            baseleng = 1.0
            """ Soil friction coefficient """ 
            if self.soilgroup[j] == 'other':
                """ Friction coefficient base and seafloor """
                if self._variables.seaflrfriccoef:
                    self.soilfric = self._variables.seaflrfriccoef
                else:
                    self.soilfric = self.seaflrfriccoef
            else:
                if shearkeys == 'True':
                    self.soilfric = math.tan(self.dsfangrad)
                elif shearkeys == 'False':
                    self.soilfric = math.tan(self.dsfangrad - 5.0 * math.pi / 180.0)
            resdesload[j] = math.sqrt(self.horzfoundloads[j][
                                self.maxloadindex[j]] ** 2.0 
                                    + self.vertfoundloads[j] ** 2.0)
            
            for k in range(1,klim):  
                if resdesload[j] == 0.0:
                    module_logger.warn('WARNING: Resultant design load at foundation equal to zero, setting minimum foundation size')                        
                    basedim = [basewidth, baseleng, 0.1 * basewidth, 
                               0.0, 0.0, 0, 
                               0.0, 0.1 * basewidth * basewidth * baseleng * self._variables.conden, 0.0]  
                    break
                for l in range(1,klim):                                 
                    """ Base width orientated to largest load 
                        direction """                                           
                    basearea = basewidth * baseleng                    
                    if shearkeys == 'True':
                        skirtheight = 0.1 * basewidth
                        """ Full embedment assumed """
                        embeddepth = skirtheight
                    elif shearkeys == 'False':
                        skirtheight = 0.0
                        embeddepth = 0.0
                    """ Short-term (undrained) resistance to 
                        sliding """                    
                    if self.soilgroup[j] == 'other': 
                        trapsoilweight = 0.0    
                    else:
                        """ Buoyant weight of soil trapped in 
                            skirt """                        
                        trapsoilweight = (self.soilweight 
                            * basearea * embeddepth)                            
                    """ Static short-term loading. Note: long-term 
                        loading relevant when significant uplift 
                        possible """
                    if self.soilgroup[j] == 'cohesive': 
                        """ Undrained shear strength at shear key tip 
                        and average undrained shear strength """          
                        unshstrz = (self.unshstr[1] 
                            + self.unshstr[0] * embeddepth)
                        unshstrav = (self.unshstr[1] 
                            + self.unshstr[0] * 0.5 
                            * embeddepth)
                        if shearkeys == 'True':
                            keyheight = 0.1 * basewidth  
                            """ Minimum foundation weight required 
                                to resist sliding in maximum load 
                                direction """   
                            reqfoundweight[j] = ((((
                                self._variables.foundsf 
                                + self.soilfric * math.tan(
                                self.seabedslp[j][
                                self.maxloadindex[j]])) 
                                * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]]) 
                                - (self.draincoh 
                                * basearea 
                                / math.cos(self.seabedslp[j][
                                self.maxloadindex[j]]))) 
                                / (self.soilfric 
                                - self._variables.foundsf 
                                * math.tan(self.seabedslp[j][
                                self.maxloadindex[j]]))) 
                                + self.vertfoundloads[j] 
                                - trapsoilweight)                            
                        else: 
                            keyheight = 0.0
                            reqfoundweight[j] = ((((
                                self._variables.foundsf 
                                + self.soilfric * math.tan(
                                self.seabedslp[j][
                                self.maxloadindex[j]])) 
                                * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]]) 
                                - (self.draincoh 
                                * basearea 
                                / math.cos(self.seabedslp[j][
                                self.maxloadindex[j]]))) 
                                / (self.soilfric 
                                - self._variables.foundsf 
                                * math.tan(self.seabedslp[j][
                                self.maxloadindex[j]]))) 
                                + self.vertfoundloads[j])

                        """ If required foundation weight is 
                            negative (i.e. when the foundation is 
                            subjected to zero or large negative 
                            loads), weight is calculated using base 
                            size and density of concrete """
                        if reqfoundweight[j] < 0:
                            reqfoundweight [j] = (basearea 
                                * embeddepth 
                                * self._variables.conden 
                                * self._variables.gravity)
                        """ Lateral resistance to sliding (short-term) """                                 
                        latresst[j][self.maxloadindex[j]] = (
                            unshstrz 
                            * basearea + 2.0 * unshstrav * embeddepth 
                            * baseleng)
                        latresst[j][self.minloadindex[j]] = (
                            unshstrz 
                            * basearea + 2.0 * unshstrav * embeddepth 
                            * basewidth) 
                        if shearkeys == 'True':                            
                            """ Contribution from shear keys """                               
                            shkeyres[self.maxloadindex[j]] = ((0.5 
                                * self.soilweight 
                                * keyheight ** 2.0
                                + 2.0 * unshstrav * keyheight) 
                                * basewidth)
                            shkeyres[self.minloadindex[j]] = ((0.5 
                                * self.soilweight 
                                * keyheight ** 2.0 
                                + 2.0 * unshstrav * keyheight) 
                                * baseleng) 
                        else:      
                            shkeyres[self.maxloadindex[j]] = 0.0
                            shkeyres[self.minloadindex[j]] = 0.0
                        """ Lateral resistance to sliding (long-term) """    
                        latreslt[j][self.maxloadindex[j]] = ((self.draincoh 
                                                * basearea)  
                                                + (self.soilfric
                                                * ((reqfoundweight[j] 
                                                + trapsoilweight 
                                                - self.vertfoundloads[j]) * math.cos(
                                                self.seabedslp[j][
                                                self.maxloadindex[j]]) 
                                                - self.horzfoundloads[j][
                                                self.maxloadindex[j]] * math.sin(
                                                self.seabedslp[j][
                                                self.maxloadindex[j]]))))
                        latreslt[j][self.minloadindex[j]] = ((self.draincoh 
                                                * basearea)  
                                                + (self.soilfric
                                                * ((reqfoundweight[j] 
                                                + trapsoilweight 
                                                - self.vertfoundloads[j]) * math.cos(
                                                self.seabedslp[j][
                                                self.minloadindex[j]]) 
                                                - self.horzfoundloads[j][
                                                self.minloadindex[j]] * math.sin(
                                                self.seabedslp[j][
                                                self.minloadindex[j]]))))
                        """ Determine if short- or long-term loading is the limiting design case """
                        latresst[j][self.maxloadindex[j]] = max(math.fabs(latresst[j][self.maxloadindex[j]]),
                                                                math.fabs(latreslt[j][self.maxloadindex[j]]))
                        latresst[j][self.minloadindex[j]] = max(math.fabs(latresst[j][self.minloadindex[j]]),
                                                                math.fabs(latreslt[j][self.minloadindex[j]]))
                    elif self.soilgroup[j] == 'cohesionless':  
                        if shearkeys == 'True':
                            keyheight = 0.05 * basewidth                                    
                        else: keyheight = 0.0                                   
                        """ Minimum foundation weight required to 
                            resist sliding in maximum load 
                            direction """   
                        if shearkeys == 'True': 
                            reqfoundweight[j] = ((((
                                self._variables.foundsf 
                                + self.soilfric * math.tan(
                                self.seabedslp[j][
                                self.maxloadindex[j]])) 
                                * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]])) 
                                / (self.soilfric 
                                - self._variables.foundsf 
                                * math.tan(self.seabedslp[j][
                                self.maxloadindex[j]])))
                                + self.vertfoundloads[j] 
                                - trapsoilweight)
                        else:
                            """ Without shear keys sliding more 
                                likely to occur at foundation 
                                base """
                            reqfoundweight[j] = ((((
                                self._variables.foundsf 
                                + self.soilfric * math.tan(
                                self.seabedslp[j][
                                self.maxloadindex[j]])) 
                                * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]])) 
                                / (self.soilfric 
                                - self._variables.foundsf 
                                * math.tan(self.seabedslp[j][
                                self.maxloadindex[j]]))) 
                                + self.vertfoundloads[j])
                        if reqfoundweight[j] <= 0.0:
                            reqfoundweight [j] = (basearea 
                                * embeddepth 
                                * self._variables.conden 
                                * self._variables.gravity)
                        if shearkeys == 'True':
                            passlatpresscoef = ((math.tan(math.pi / 4.0 
                                        + self.dsfangrad / 2.0)) ** 2.0)
                            shkeyres[self.maxloadindex[j]] = (0.5 
                                * passlatpresscoef 
                                * self.soilweight 
                                * keyheight ** 2.0 * basewidth)
                            shkeyres[self.minloadindex[j]] = (0.5 
                                * passlatpresscoef 
                                * self.soilweight 
                                * keyheight ** 2.0 * baseleng)
                        else:
                            shkeyres[self.maxloadindex[j]] = 0.0
                            shkeyres[self.minloadindex[j]] = 0.0
                        """ Lateral resistance to sliding """                                 
                        latresst[j][self.maxloadindex[j]] = (
                            self.soilfric
                            * ((reqfoundweight[j] 
                            + trapsoilweight 
                            - self.vertfoundloads[j]) * math.cos(
                            self.seabedslp[j][
                            self.maxloadindex[j]]) 
                            - self.horzfoundloads[j][
                            self.maxloadindex[j]] * math.sin(
                            self.seabedslp[j][
                            self.maxloadindex[j]])) 
                            + shkeyres[self.maxloadindex[j]])
                        latresst[j][self.minloadindex[j]] = (
                            self.soilfric
                            * ((reqfoundweight[j]
                            + trapsoilweight 
                            - self.vertfoundloads[j]) * math.cos(
                            self.seabedslp[j][
                            self.minloadindex[j]]) 
                            - self.horzfoundloads[j][
                            self.minloadindex[j]] * math.sin(
                            self.seabedslp[j][
                            self.minloadindex[j]]))
                            + shkeyres[self.minloadindex[j]])
                    elif self.soilgroup[j] == 'other':  
                        """ Sliding resistance on hard surfaces. 
                            Note: this formulation needs to be 
                            checked """
                        reqfoundweight[j] = ((((
                            self._variables.foundsf 
                            + self.soilfric * math.tan(
                            self.seabedslp[j][
                            self.maxloadindex[j]])) 
                            * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]])) / (self.soilfric 
                            - self._variables.foundsf * math.tan(
                            self.seabedslp[j][
                            self.maxloadindex[j]]))) 
                            + self.vertfoundloads[j])
                        keyheight = 0.0                        
                        """ If required foundation weight is 
                            negative (i.e. when the foundation is 
                            subjected to zero or large negative 
                            loads), weight is calculated using base 
                            size and density of concrete """
                        if reqfoundweight[j] < 0:
                            reqfoundweight[j] = (basearea 
                                * embeddepth * self._variables.conden 
                                * self._variables.gravity)
                        """ Lateral resistance to sliding """ 
                        baseleng = ((4.0 * reqfoundweight[j])/self._variables.conden) ** (1.0 / 3.0)
                        basewidth = baseleng
                        baseheight = 0.25 * baseleng

                        reqsteelvol[j] = 0.0
                        reqconvol[j] = basewidth * baseleng * baseheight
                        if baseleng > 8.0:
                            baseleng  = ((4.0 * reqfoundweight[j])/((2.0 * self._variables.conden + self._variables.steelden)) / 3.0) ** (1.0 / 3.0)
                            basewidth = baseleng
                            baseheight = 0.25 * baseleng
                            reqsteelvol[j] = basewidth * baseleng * baseheight / 3.0
                            reqconvol[j] = basewidth * baseleng * baseheight * 2.0 / 3.0
                        latcheck = 'True'
 
                        break
                    """ Lateral resistance check. Modified to 
                        account for down and upslope sliding """    
                    if self.soilgroup in ('cohesive', 'cohesionless'):
                        if (latresst[j][self.maxloadindex[j]] 
                            < math.fabs(self._variables.foundsf 
                            * (self.totfoundslopeloads[j][
                            self.maxloadindex[j]] + (reqfoundweight[j] 
                            + trapsoilweight) * math.sin(
                            self.seabedslp[j][self.maxloadindex[j]]))) or (latresst[j][self.minloadindex[j]] 
                                < math.fabs(self._variables.foundsf 
                                * (self.totfoundslopeloads[j][
                                self.minloadindex[j]] + (reqfoundweight[j] 
                                + trapsoilweight) * math.sin(
                                self.seabedslp[j][self.minloadindex[j]]))))):
                            
                            if (latresst[j][self.maxloadindex[j]] 
                                < math.fabs(self._variables.foundsf 
                                * (self.totfoundslopeloads[j][
                                self.maxloadindex[j]] + (reqfoundweight[j] 
                                + trapsoilweight) * math.sin(
                                self.seabedslp[j][self.maxloadindex[j]])))):
                                # print 'Lateral resistance insufficient - Increase width'
                                basewidth = basewidth + 0.25
                                if self.uniformfound == 'True':
                                    baseleng = basewidth                        
                            if (latresst[j][self.minloadindex[j]] 
                                < math.fabs(self._variables.foundsf 
                                * (self.totfoundslopeloads[j][
                                self.minloadindex[j]] + (reqfoundweight[j] 
                                + trapsoilweight) * math.sin(
                                self.seabedslp[j][self.minloadindex[j]])))):
                                # print 'Lateral resistance insufficient - Increase length'                            
                                baseleng = baseleng + 0.25
                                if self.uniformfound == 'True':
                                    basewidth = baseleng                        
                    else:
                        """ Structure weight alone is suitable, no additional weight required """
                        if reqfoundweight[j] == 0.0:
                            break
                        latcheck = 'True'
                        # print 'Lateral resistance check passed'
                        """ Find height of concrete base required 
                            to achieve min foundation weight """
                        baseheight = (reqfoundweight[j] 
                            / (self._variables.gravity 
                            * self._variables.conden 
                            * basearea))                                                             
                        """ Recommended height limit for stability 
                            is 0.25B """
                        if baseheight > 0.25 * basewidth:  
                            baseheight = 0.25 * basewidth
                            reqvol = basearea * baseheight                                        
                            if (self._variables.gravity * reqvol 
                                * self._variables.steelden < 
                                reqfoundweight[j]):
                                basewidth = basewidth + 0.25
                                baseleng = basewidth
                            else:
                                reqbaseden =  (reqfoundweight[j] 
                                    / (reqvol 
                                    * self._variables.gravity))
                                """ Add steel to base """
                                reqsteelvol[j] = (reqvol 
                                    * (reqbaseden 
                                    - self._variables.conden) 
                                    / (self._variables.steelden 
                                    - self._variables.conden))
                                reqconvol[j] = (reqvol 
                                    - reqsteelvol[j])
                                break
                        else:
                            reqsteelvol[j] = 0.0    
                            reqconvol[j] = (reqfoundweight[j] 
                                / (self._variables.gravity 
                                * self._variables.conden))
                            break
                """ Structure weight alone is suitable, no additional weight required """
                if (reqfoundweight[j] == 0.0 and self._variables.systype in ('tidefixed','wavefixed')):                                     
                    module_logger.warn('WARNING: Structure weight alone is suitable, no additional weight required')
                    self.foundnotreqflag[posftyps[0]][j] = 'True'                    
                    
                    basedim = [0.0, 0.0, 0.0, 
                                   0.0, 0.0, 0, 
                                   0.0, 0.0, 0.0]
                    break
                if latcheck == 'True':
                    """ Load perpendicular to slope """
                    normload[j][self.maxloadindex[j]] = ((
                        reqfoundweight[j] 
                        + trapsoilweight - self.vertfoundloads[j]) 
                        * math.cos(self.seabedslp[j][
                        self.maxloadindex[j]]) 
                        - self.horzfoundloads[j][
                        self.maxloadindex[j]] * math.sin(
                        self.seabedslp[j][self.maxloadindex[j]]))
                    normload[j][self.minloadindex[j]] = ((
                        reqfoundweight[j] 
                        + trapsoilweight - self.vertfoundloads[j]) 
                        * math.cos(self.seabedslp[j][
                        self.minloadindex[j]]) 
                        - self.horzfoundloads[j][
                        self.minloadindex[j]] * math.sin(
                        self.seabedslp[j][self.minloadindex[j]]))
                    """ Moments about centre of shear key base """                        
                    momsbase[j][self.maxloadindex[j]] = (
                        trapsoilweight 
                        * 0.5 * embeddepth * math.sin(math.fabs(
                        self.seabedslp[j][self.maxloadindex[j]])) 
                        + reqfoundweight[j] * (embeddepth + 0.5 
                        * baseheight) * math.sin(math.fabs(
                        self.seabedslp[j][self.maxloadindex[j]])) 
                        - self.vertfoundloads[j] * (embeddepth 
                        + baseheight) * math.sin(math.fabs(
                        self.seabedslp[j][self.maxloadindex[j]])) 
                        + self.horzfoundloads[j][
                        self.maxloadindex[j]] * (embeddepth 
                        + baseheight) * math.sin(math.fabs(
                        self.seabedslp[j][self.maxloadindex[j]])))
                    momsbase[j][self.minloadindex[j]] = (
                        trapsoilweight 
                        * 0.5 * embeddepth * math.sin(math.fabs(
                        self.seabedslp[j][self.minloadindex[j]])) 
                        + reqfoundweight[j] * (embeddepth + 0.5 
                        * baseheight) * math.sin(math.fabs(
                        self.seabedslp[j][self.minloadindex[j]])) 
                        - self.vertfoundloads[j] * (embeddepth 
                        + baseheight) * math.sin(math.fabs
                        (self.seabedslp[j][self.minloadindex[j]])) 
                        + self.horzfoundloads[j][
                        self.minloadindex[j]] * (embeddepth 
                        + baseheight) * math.sin(math.fabs(
                        self.seabedslp[j][self.minloadindex[j]])))
                    """ Calculated eccentricity """
                    if normload[j][self.maxloadindex[j]] != 0.0:
                        eccen[j][self.maxloadindex[j]] = (
                            momsbase[j][self.maxloadindex[j]] 
                            / normload[j][self.maxloadindex[j]])
                    else:
                        eccen[j][self.maxloadindex[j]] = 0.0
                    if normload[j][self.minloadindex[j]] != 0.0:
                        eccen[j][self.minloadindex[j]] = (
                            momsbase[j][self.minloadindex[j]] 
                            / normload[j][self.minloadindex[j]])
                    else:
                        eccen[j][self.minloadindex[j]] = 0.0
                    if (abs(eccen[j][self.maxloadindex[j]]) 
                        > basewidth / 6.0):
                        #print """Eccentricity not acceptable - Increase base width"""
                        basewidth = basewidth + 0.25
                        if self.uniformfound == 'True':
                            baseleng = basewidth
                    elif (abs(eccen[j][self.minloadindex[j]]) 
                        > baseleng / 6.0):
                        baseleng = baseleng + 0.25
                        if self.uniformfound == 'True':
                            basewidth = baseleng
                        #print """Eccentricity not acceptable - Increase base length"""
                    else:
                        ecccheck = 'True'                                   
                        #print """Eccentricity check passed"""
                        if self.soilgroup[j] == 'other':
                            """ For rock soils, once lateral and eccentricity 
                                checks have passed break loop """
                            bcapcheck = 'True' 
                            shkcheck = 'True'
                            
                    if self.soilgroup[j] in ('cohesive', 'cohesionless'):
                        """ Effective base width and length accounting 
                            for load eccentricity """
                        effbaseleng = (baseleng 
                            - 2.0 * eccen[j][self.maxloadindex[j]])
                        effbasewidth = (basewidth 
                            - 2.0 * eccen[j][self.minloadindex[j]])
                        effbasearea = effbaseleng * effbasewidth                        
                        
                        if self.soilgroup[j] == 'cohesive':
                            """ Calculate undrained shear strength averaged 
                                over 0.7 x effective base width below shear 
                                keys """
                            unshstrz = ((self.unshstr[1] 
                                + self.unshstr[0] 
                                * embeddepth + self.unshstr[1] 
                                + self.unshstr[0] * (embeddepth 
                                + 0.7 * effbasewidth)) / 2)
                            
                        """ Correction factor for cohesion, overburden 
                            and density """                                
                        mcoef[self.maxloadindex[j]] = (((2.0 
                            + (effbaseleng / effbasewidth)) / (1.0 
                            + (effbaseleng / effbasewidth))) 
                            * (math.cos(math.pi/2)) ** 2.0 + ((2.0 
                            + (effbasewidth / effbaseleng)) / (1.0 
                            + (effbasewidth / effbaseleng))) 
                            * (math.sin(math.pi/2)) ** 2.0)
                        mcoef[self.minloadindex[j]] = (((2.0 
                            + (effbaseleng / effbasewidth)) / (1.0 
                            + (effbaseleng / effbasewidth))) 
                            * (math.cos(math.pi/2)) ** 2.0 + ((2.0 
                            + (effbasewidth / effbaseleng)) / (1.0 
                            + (effbasewidth / effbaseleng))) 
                            * (math.sin(math.pi/2)) ** 2.0)
                        """ Load inclination coefficients """ 
                        totvertfoundloads[j] = (reqfoundweight[j] 
                            + trapsoilweight - self.vertfoundloads[j])
                        loadincq[self.maxloadindex[j]] = ((1.0 
                            - (self.horzfoundloads[j][
                            self.maxloadindex[j]] 
                            / (totvertfoundloads[j] + effbasewidth 
                            * effbaseleng * self.draincoh 
                            / math.tan(self.dsfangrad)))) 
                            ** mcoef[self.maxloadindex[j]])
                        loadincq[self.minloadindex[j]] = (1.0 
                            - (self.horzfoundloads[j][
                            self.minloadindex[j]] 
                            / (totvertfoundloads[j] + effbasewidth 
                            * effbaseleng * self.draincoh 
                            / math.tan(self.dsfangrad))) 
                            ** mcoef[self.minloadindex[j]])                                    
                        loadincgamma[self.maxloadindex[j]] = ((1.0 
                            - (self.horzfoundloads[j][
                            self.maxloadindex[j]] 
                            / (totvertfoundloads[j] + effbasewidth 
                            * effbaseleng * self.draincoh 
                            / math.tan(self.dsfangrad)))) 
                            ** (mcoef[self.maxloadindex[j]] + 1))
                        loadincgamma[self.minloadindex[j]] = ((1.0 
                            - (self.horzfoundloads[j][
                            self.minloadindex[j]] 
                            / (totvertfoundloads[j] + effbasewidth 
                            * effbaseleng * self.draincoh 
                            / math.tan(self.dsfangrad)))) 
                            ** (mcoef[self.minloadindex[j]] + 1))
                        
                        """ Bearing capacity factors """
                        bcfntheta = ((math.tan(math.pi / 4.0 
                            + self.dsfangrad / 2)) ** 2.0)
                        bcfnq = (math.exp(math.pi * math.tan(
                            self.dsfangrad)) * bcfntheta)                           
                        bcfngamma = (2.0 * (1.0 + bcfnq) 
                            * math.tan(self.dsfangrad) 
                            * math.tan(math.pi / 4.0 
                            + self.dsfangrad / 5))
                               
                        if self.dsfangrad == 0.0:    
                            """ Purely cohesive soils """
                            bcfnc = 2 + math.pi
                            """ Undrained bearing capacity failure in 
                                cohesive soil (undrained friction angle 
                                = 0 deg) """
                            loadincc[self.maxloadindex[j]] = (1.0 
                                - mcoef[self.maxloadindex[j]] 
                                * math.fabs(self.horzfoundloads[j][
                                self.maxloadindex[j]]) / (effbasewidth 
                                * effbaseleng * unshstrav * bcfnc))
                            loadincc[self.minloadindex[j]] = (1.0 
                            - mcoef[self.minloadindex[j]] 
                            * math.fabs(self.horzfoundloads[j][
                                self.minloadindex[j]]) / (effbasewidth 
                            * effbaseleng * unshstrav * bcfnc))
                            """ Embedment depth coefficients """
                            depthcoefc = (1.0 + 2.0 * math.atan(
                                embeddepth / effbasewidth) 
                                * (1.0 / bcfnc))
                            depthcoefq = 1.0
                            depthcoefgamma = 1.0
                        elif self.dsfangrad > 0.0:
                            bcfnc = (bcfnq - 1.0) / math.tan(
                            self.dsfangrad)
                            """ Load inclination coefficients """ 
                            loadincc[self.maxloadindex[j]] = (
                                loadincq[self.maxloadindex[j]]
                                - ((1.0 - loadincq[self.maxloadindex[j]]) 
                                / (bcfnc * math.tan(
                                self.dsfangrad))))
                            loadincc[self.minloadindex[j]] = (
                                loadincq[self.minloadindex[j]] 
                                - ((1.0 - loadincq[self.minloadindex[j]]) 
                                / (bcfnc * math.tan(
                                self.dsfangrad))))                                
                            """ Embedment depth coefficients """
                            depthcoefc = (1.0 + 2.0 * ((1.0 
                                - math.sin(self.dsfangrad)) 
                                ** 2.0) * math.atan(embeddepth 
                                / effbasewidth) * (bcfnq / bcfnc))
                            depthcoefq = (1.0 + 2.0 * ((1.0 
                            - math.sin(self.dsfangrad)) ** 2.0) 
                            * math.atan(embeddepth / effbasewidth) 
                            * math.tan(self.dsfangrad))
                            depthcoefgamma = 1.0
                        """ Foundation shape coefficient """
                        shpcoefc = (1.0 + (effbasewidth / effbaseleng) 
                            * (bcfnq / bcfnc))
                        shpcoefq = (1.0 + (effbasewidth / effbaseleng) 
                            * math.tan(self.dsfangrad))
                        shpcoefgamma = (1.0 - 0.4 * (effbasewidth 
                            / effbaseleng))
                        """ Inclination of foundation base and seafloor 
                            (near-horizontal conditions assumed) """
                        basecoefc = 1.0
                        basecoefq = 1.0
                        basecoefgamma = 1.0
                        groundcoefc = 1.0
                        groundcoefq = 1.0
                        groundcoefgamma = 1.0
                            
                        """ Correction factor subgroups """
                        kcorrfacc[self.maxloadindex[j]] = (
                            loadincc[self.maxloadindex[j]] 
                            * shpcoefc * depthcoefc * basecoefc 
                            * groundcoefc)
                        kcorrfacq[self.maxloadindex[j]] = (
                            loadincq[self.maxloadindex[j]] 
                            * shpcoefq * depthcoefq * basecoefq 
                            * groundcoefq)
                        kcorrfacgamma[self.maxloadindex[j]] = (
                            loadincgamma[self.maxloadindex[j]]
                            * shpcoefgamma * depthcoefgamma 
                            * basecoefgamma * groundcoefgamma)
                        kcorrfacc[self.minloadindex[j]] = (
                            loadincc[self.minloadindex[j]] 
                            * shpcoefc * depthcoefc * basecoefc 
                            * groundcoefc)
                        kcorrfacq[self.minloadindex[j]] = (
                            loadincq[self.minloadindex[j]]
                            * shpcoefq * depthcoefq * basecoefq 
                            * groundcoefq)
                        kcorrfacgamma[self.minloadindex[j]] = (
                            loadincgamma[self.minloadindex[j]]
                            * shpcoefgamma * depthcoefgamma 
                            * basecoefgamma * groundcoefgamma)
                        
                        if self.soilgroup[j] == 'cohesive':  
                            """ For soft cohesive soil, sensitivity 
                                factor is approx. equal to 3 """
                            if self._variables.soilsen:
                                soilsen = self._variables.soilsen
                            else:
                                soilsen = self.soilsen
                            """ Short-term bearing capacity - Full 
                                embedment assumed """
                            bearcapst[self.maxloadindex[j]] = (
                                effbasearea 
                                * (unshstrz * bcfnc * kcorrfacc[
                                self.maxloadindex[j]] 
                                + self.soilweight 
                                * embeddepth) + (2 * basewidth + 2 
                                * baseleng) * embeddepth * (unshstrav 
                                / soilsen))
                            bearcapst[self.minloadindex[j]] = (
                                effbasearea 
                                * (unshstrz * bcfnc * kcorrfacc[
                                self.minloadindex[j]] 
                                + self.soilweight 
                                * embeddepth) + (2 * basewidth + 2 
                                * baseleng) * embeddepth * (unshstrav 
                                / soilsen))
                                                              
                        elif self.soilgroup[j] == 'cohesionless':                                    
                            """ Approximate fractional relative 
                                density """
                            fracrelden = ((self.soilweight 
                                - 2705.229741) / 550.621983)
                            """ Approximate critical confining 
                                pressure """
                            critconpres = (957603.447985 
                                * fracrelden ** 1.7)
                            """ Effective shear strength at critical 
                                confining pressure """
                            effshstr = ((critconpres * math.sin(
                                self.dsfangrad)) / (1 
                                - math.sin(self.dsfangrad)))
                            """ Maximum bearing stress """
                            maxbearstress[self.maxloadindex[j]] = (
                                effshstr * bcfnc 
                                *  kcorrfacc[self.maxloadindex[j]]) 
                            maxbearstress[self.minloadindex[j]] = (
                                effshstr * bcfnc 
                                *  kcorrfacc[self.minloadindex[j]])
                            """ Transition depth """
                            transdepth[self.maxloadindex[j]] = (
                                maxbearstress[self.maxloadindex[j]] 
                                / ((math.pi / 2) 
                                * self.soilweight * ((bcfnq 
                                * kcorrfacq[self.maxloadindex[j]] - 1) 
                                + ((effbasewidth / 2) / embeddepth) 
                                * bcfngamma 
                                * kcorrfacgamma[self.maxloadindex[j]])))
                            transdepth[self.minloadindex[j]] = (
                                maxbearstress[self.minloadindex[j]] 
                                / ((math.pi / 2) 
                                * self.soilweight * ((bcfnq 
                                * kcorrfacq[self.minloadindex[j]] - 1) 
                                + ((effbasewidth / 2) / embeddepth) 
                                * bcfngamma 
                                * kcorrfacgamma[self.minloadindex[j]])))
                            """ Depth attenuation factor """
                            depthattfac[self.maxloadindex[j]] = ((
                                math.atan(embeddepth 
                                / transdepth[self.maxloadindex[j]])) 
                                / (embeddepth 
                                / transdepth[self.maxloadindex[j]]))
                            depthattfac[self.minloadindex[j]] = ((
                                math.atan(embeddepth 
                                / transdepth[self.minloadindex[j]])) 
                                / (embeddepth 
                                / transdepth[self.minloadindex[j]]))                                
                            """ Effective friction angle alongside 
                                footing. Smooth sided foundation 
                                assumed but could be overriden """
                            efffricang = 0.0                                    
                            bearcapst[self.maxloadindex[j]] = (
                                effbasearea 
                                * self.soilweight 
                                * (embeddepth * (1.0 + (bcfnq 
                                * kcorrfacq[self.maxloadindex[j]] - 1) 
                                * depthattfac[self.maxloadindex[j]]) 
                                + (effbasewidth / 2) * bcfngamma 
                                * kcorrfacgamma[self.maxloadindex[j]] 
                                *  depthattfac[self.maxloadindex[j]]) 
                                + (2 * basewidth + 2 * baseleng) 
                                * (embeddepth) 
                                * self.soilweight 
                                * math.tan(efffricang) * (embeddepth 
                                + 0.0) / 2)
                            bearcapst[self.minloadindex[j]] = (
                            effbasearea 
                            * self.soilweight * (embeddepth 
                            * (1.0 + (bcfnq 
                            * kcorrfacq[self.minloadindex[j]] - 1) 
                            * depthattfac[self.minloadindex[j]]) 
                            + (effbasewidth / 2) * bcfngamma 
                            * kcorrfacgamma[self.minloadindex[j]] 
                            *  depthattfac[self.minloadindex[j]]) 
                            + (2 * basewidth + 2 * baseleng) 
                            * (embeddepth) * self.soilweight 
                            * math.tan(efffricang) * (embeddepth 
                            + 0.0) / 2)
                                                                    
                        """ Total normal loads """                            
                        totnormloads[self.maxloadindex[j]] = (
                            totvertfoundloads[j] 
                            * math.cos(self.seabedslp[j][
                            self.maxloadindex[j]]) 
                            - self.horzfoundloads[j][
                            self.maxloadindex[j]] * math.sin(
                            self.seabedslp[j][self.maxloadindex[j]]))
                        totnormloads[self.minloadindex[j]] = (
                            totvertfoundloads[j] 
                            * math.cos(self.seabedslp[j][
                            self.minloadindex[j]]) 
                            - self.horzfoundloads[j][
                            self.minloadindex[j]] * math.sin(
                            self.seabedslp[j][self.minloadindex[j]]))
                        if (bearcapst[self.maxloadindex[j]] 
                            < self._variables.foundsf 
                            * totnormloads[self.maxloadindex[j]]):
                            #print """ Insufficient bearing capacity """
                            basewidth = basewidth + 0.25  
                            if self.uniformfound == 'True':
                                baseleng = basewidth
                        elif (bearcapst[self.minloadindex[j]] 
                            < self._variables.foundsf 
                            * totnormloads[self.minloadindex[j]]):
                            #print """ Insufficient bearing capacity """
                            baseleng = baseleng + 0.25  
                            if self.uniformfound == 'True':
                                basewidth = baseleng
                        else: 
                            bcapcheck = 'True'
                            #print """Bearing capacity check passed """
                                     
                        if (shearkeys == 'True' and bcapcheck == 'True'):                            
                            """ Shear key resistance """
                            if self.soilgroup[j] == 'cohesive': 
                                """ Determine number of shear keys """ 
                                minspc = 1.0 * keyheight                                
                                unshstrav = (self.unshstr[1] 
                                    + self.unshstr[0] * 0.5 
                                    * keyheight)                                
                                shkeyres[self.maxloadindex[j]] = ((0.5 
                                    * self.soilweight 
                                    * keyheight ** 2.0
                                    + 2.0 * unshstrav * keyheight) 
                                    * basewidth)
                                shkeyres[self.minloadindex[j]] = ((0.5 
                                    * self.soilweight 
                                    * keyheight ** 2.0 
                                    + 2.0 * unshstrav * keyheight) 
                                    * baseleng) 
                            elif self.soilgroup[j] == 'cohesionless':
                                minspc = 2.0 * keyheight
                                passlatpresscoef = ((math.tan(math.pi 
                                    / 4.0 + self.dsfangrad / 2.0)) ** 2.0)                              
                                shkeyres[self.maxloadindex[j]] = (0.5 
                                    * passlatpresscoef 
                                    * self.soilweight 
                                    * keyheight ** 2.0 * basewidth)
                                shkeyres[self.minloadindex[j]] = (0.5 
                                    * passlatpresscoef 
                                    * self.soilweight 
                                    * keyheight ** 2.0 * baseleng)
                            """ Number of shear keys """
                            nshkeys[self.maxloadindex[j]] = math.floor(baseleng / minspc)
                            nshkeys[self.minloadindex[j]] = math.floor(basewidth / minspc)

                            """ Shear key spacing """  
                            if ((nshkeys[self.maxloadindex[j]] - 1) * shkeyres[self.maxloadindex[j]] > 2.0 * (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.maxloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.maxloadindex[j]]))
                                and ((nshkeys[self.minloadindex[j]] - 1) * shkeyres[self.minloadindex[j]] > 2.0 * (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.minloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.minloadindex[j]])))):

                                
                                baseleng = 0.95 * baseleng
                                basewidth = 0.95 * basewidth
                            if ((nshkeys[self.maxloadindex[j]] - 1) * shkeyres[self.maxloadindex[j]] < (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.maxloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.maxloadindex[j]]))
                                or ((nshkeys[self.minloadindex[j]] - 1) * shkeyres[self.minloadindex[j]] < (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.minloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.minloadindex[j]])))):

                                
                                baseleng = 1.1 * baseleng
                                basewidth = 1.1 * basewidth 
                            if ((nshkeys[self.maxloadindex[j]] - 1) * shkeyres[self.maxloadindex[j]] > (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.maxloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.maxloadindex[j]]))
                                and ((nshkeys[self.minloadindex[j]] - 1) * shkeyres[self.minloadindex[j]] > (self._variables.foundsf 
                                * self.totfoundslopeloadsabs[j][
                                self.minloadindex[j]] 
                                + (reqfoundweight[j] - self.vertfoundloads[j]) * math.sin(
                                self.seabedslp[j][
                                self.minloadindex[j]])))):

                                
                                shkcheck = 'True'
                                

                if (latcheck == 'True' and ecccheck == 'True' 
                    and bcapcheck == 'True' and shkcheck == 'True'):   
                    if baseheight > 0.0:
                        basedim = [basewidth, baseleng, baseheight, 
                                   skirtheight, keyheight, nshkeys, 
                                   shkeyspc, reqconvol[j], reqsteelvol[j]]
                        logmsg = [""]
                        logmsg.append('Solution found {}'.format(basedim))
                        module_logger.info("\n".join(logmsg))
                        self.founddesfail[j] = 'False' 
                    elif baseheight == 0.0:
                        module_logger.warn('WARNING: Resultant design load at padeye equal to zero, setting minimum foundation size')                        
                        basedim = [basewidth, baseleng, baseheight, 
                                   skirtheight, keyheight, nshkeys, 
                                   shkeyspc, reqconvol[j], reqsteelvol[j]]  
                    break
                elif k == klim - 1:
                    """ For when a solution cannot be found """ 
                    module_logger.warn('WARNING: Solution not found within set number of iterations!')  
                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                    basedim = [basewidth, baseleng, 'n/a', 
                               0.0, 0.0, 0, 
                               0.0, reqconvol[j], reqsteelvol[j]]                  
                    if posftyps[0] not in self.unsuitftyps:
                        self.unsuitftyps.append(posftyps[0])  

            return basedim

        latresst = [[0 for col in range(2)] for row in range(self.quanfound)]
        latreslt = [[0 for col in range(2)] for row in range(self.quanfound)]
        reqfoundweight = [[0 for col in range(1)] for row in 
                            range(self.quanfound)]
        normload = [[0 for col in range(2)] for row in range(self.quanfound)]
        momsbase = [[0 for col in range(2)] for row in range(self.quanfound)]
        eccen  = [[0 for col in range(2)] for row in range(self.quanfound)]
        totvertfoundloads = [0 for row in range(self.quanfound)]
        reqsteelvol = [0 for row in range(self.quanfound)]
        reqconvol = [0 for row in range(self.quanfound)]
        horzdesload = [0 for row in range(self.quanfound)]
        vertdesload = [0 for row in range(self.quanfound)]
        self.horzdesload = [0 for row in range(self.quanfound)]
        self.vertdesload = [0 for row in range(self.quanfound)]
        loadattpnt = [0 for row in range(self.quanfound)]
        resdesload = [0 for row in range(self.quanfound)]
        resdesloadang = [0 for row in range(self.quanfound)]
        padeyedesload = [0 for row in range(self.quanfound)]
        padeyetheta = [0 for row in range(self.quanfound)]
        padeyehorzload = [0 for row in range(self.quanfound)]
        padeyevertload = [0 for row in range(self.quanfound)]
        self.gravbasedim = [0 for row in range(self.quanfound)]
        self.shallbasedim = [0 for row in range(self.quanfound)]
        self.piledim = [0 for row in range(self.quanfound)]
        self.caissdim = [0 for row in range(self.quanfound)]
        self.platedim = [0 for row in range(self.quanfound)]
        self.ancdim = [0 for row in range(self.quanfound)]
        self.seldraganc = [0 for row in range(self.quanfound)]
        self.ancpendep = [0 for row in range(self.quanfound)]
        self.pileitemcost = [0 for row in range(self.quanfound)] 
        self.caissitemcost = [0 for row in range(self.quanfound)] 
        self.ancitemcost = [0 for row in range(self.quanfound)]   
        self.pilevolsteel = [0 for row in range(self.quanfound)]
        self.shallvolsteel = [0 for row in range(self.quanfound)]
        self.caissvolsteel = [0 for row in range(self.quanfound)]
        self.gravvolsteel = [0 for row in range(self.quanfound)]
        self.platevolsteel = [0 for row in range(self.quanfound)]
        self.pilevolcon = [0 for row in range(self.quanfound)]
        self.shallvolcon = [0 for row in range(self.quanfound)]
        self.caissvolcon = [0 for row in range(self.quanfound)]
        self.gravvolcon = [0 for row in range(self.quanfound)]
        self.platevolcon = [0 for row in range(self.quanfound)]
        self.pilevolgrout = [0 for row in range(self.quanfound)]               
        self.pilegrouttyp = [0 for row in range(self.quanfound)] 
        self.seldragancsubtyp = [0 for row in range(self.quanfound)]
        self.seldeancsubtyp = [0 for row in range(self.quanfound)]
        self.selpilesubtyp = [0 for row in range(self.quanfound)]
        self.selshallsubtyp = [0 for row in range(self.quanfound)]
        self.selfoundsubtypdict = {}
        self.foundnotreqflag = {}
        foundnotreqflaglist = ['False' for row in range(0,self.quanfound)]
        self.foundnotfoundflag = {}
        foundnotfoundflaglist = ['False' for row in range(0,self.quanfound)]
        for j in range(0,self.quanfound):              
            for posftyps in self.possfoundtyp[j]:
                self.foundnotreqflag[posftyps[0]] = copy.deepcopy(foundnotreqflaglist)
                self.foundnotfoundflag[posftyps[0]] = copy.deepcopy(foundnotfoundflaglist)
            
        for j in range(0,self.quanfound):
            self.unsuitftyps = []                         
            self.dsfangrad = float(self._variables.soilprops.ix[self.soiltyp[j],'dsfang']) * math.pi / 180.0
            self.relsoilden = float(self._variables.soilprops.ix[self.soiltyp[j],'relsoilden'])
            self.soilweight = float(self._variables.soilprops.ix[self.soiltyp[j],'soilweight'])
            self.unshstr = [float(self._variables.soilprops.ix[self.soiltyp[j],'unshstr0']),
                            float(self._variables.soilprops.ix[self.soiltyp[j],'unshstr1'])]
            self.draincoh = float(self._variables.soilprops.ix[self.soiltyp[j],'draincoh'])
            self.seaflrfriccoef = float(self._variables.soilprops.ix[self.soiltyp[j],'seaflrfriccoef'])
            self.soilsen = float(self._variables.soilprops.ix[self.soiltyp[j],'soilsen'])
            self.rockcomstr = float(self._variables.soilprops.ix[self.soiltyp[j],'rockcompstr'])
            
            logmsg = [""]
            logmsg.append('_______________________________________________________________________')
            logmsg.append('Foundation {}'.format(j))
            logmsg.append('_______________________________________________________________________')
            module_logger.info("\n".join(logmsg))
                        
            for posftyps in self.possfoundtyp[j]:                 
                if posftyps[0] == 'shallowfoundation': 
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Shallow foundation/anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))                          
                    """ Assumptions
                    1) Foundation/anchors are orientated to direction of 
                        highest magnitude load (i.e. for shallow foundations 
                        shear keys are perpendicular to largest load 
                        direction """
                                            
                    """ Uniform foundation dimensions...base width = 
                        base length """
                    self.uniformfound = 'True'
                    self.shearkeys = 'True'
                    """ Trial foundation base width """
                    self.basewidth = 1.0   
                    self.baseleng = self.basewidth
                        
                    self.shallbasedim[j] = shallowdes(self.basewidth, 
                                    self.baseleng, self.shearkeys, posftyps) 
                    
                    if self.shallbasedim[j][5] > 0.0:
                        self.selshallsubtyp[j] = 'concrete/steel composite structure with shear keys'
                    elif self.shallbasedim[j][5] == 0.0:
                        self.selshallsubtyp[j] = 'concrete/steel composite structure without shear keys'
                    self.shallvolsteel[j] = self.shallbasedim[j][8]
                    self.shallvolcon[j] = self.shallbasedim[j][7]
                    self.foundradadd[j] = (self.foundrad[j] + max(0.5 
                        * self.shallbasedim[j][0], 
                        0.5 * self.shallbasedim[j][1]))
                    self.selfoundsubtypdict[posftyps[0]] = self.selshallsubtyp
                    self.foundvolgroutdict[j][posftyps[0]] = 0.0             
                    self.selfoundgrouttypdict[j][posftyps[0]] = 'n/a' 
                    self.foundvolsteeldict[j][posftyps[0]] = self.shallvolsteel[j]
                    self.foundvolcondict[j][posftyps[0]] = self.shallvolcon[j]          
                    self.founddimdict[j][posftyps[0]] = self.shallbasedim[j]
                    self.foundcostdict[j][posftyps[0]] = [self._variables.coststeel, 
                                                        self._variables.costcon] 
                                                        
                elif posftyps[0] == 'gravity':
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Gravity foundation/anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))      
                    """ Uniform foundation dimensions...
                        base width = base length """
                    self.uniformfound = 'True'                
                    # print 'Gravity'
                    self.shearkeys = 'False'
                    """ Trial foundation base width """
                    self.basewidth = 1.0   
                    self.baseleng = self.basewidth
                    """ Gravity foundation calculation uses shallow foundation 
                        formulae, with the exception that shear keys are not 
                        considered """
                    self.gravbasedim[j] = shallowdes(self.basewidth, 
                                    self.baseleng, self.shearkeys, posftyps)
                       
                    if posftyps[0] in self.unsuitftyps:
                        pass 
                    else:
                        self.gravvolsteel[j] = self.gravbasedim[j][8]
                        self.gravvolcon[j] = self.gravbasedim[j][7] 
                        self.foundradadd[j] = (self.foundrad[j] + max(0.5 
                            * self.gravbasedim[j][0], 0.5 
                            * self.gravbasedim[j][1]))
                        
                        self.selfoundsubtypdict[posftyps[0]] = ['concrete/steel composite structure' 
                                            for row in range(self.quanfound)]
                        self.foundvolgroutdict[j][posftyps[0]] = 0.0              
                        self.selfoundgrouttypdict[j][posftyps[0]] = 'n/a' 
                        self.foundvolsteeldict[j][posftyps[0]] = self.gravvolsteel[j]
                        self.foundvolcondict[j][posftyps[0]] = self.gravvolcon[j]       
                        self.founddimdict[j][posftyps[0]] = self.gravbasedim[j]
                        self.foundcostdict[j][posftyps[0]] = [
                                                    self._variables.coststeel, 
                                                    self._variables.costcon] 
                    
                elif posftyps[0] == 'pile':                     
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Pile foundation/anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))      
                    def piledes(complim):   
                        latloadcheck = 'False'
                        comploadcheck = 'False'
                        upliftcheck = 'False'
                        momcheck = 'False'
                        pilecloseend = 'False'
                        pilelengexceedflag = 'False'
                        pilemindimflag = 'False'
                        klim = 200      
                        """ Recalculation loop in case length has been 
                            altered """
                        for q in range(0,klim):                            
                            if posftyps[0] in self.unsuitftyps:
                                logmsg = [""]
                                logmsg.append('!!! Solution not found, foundation type unsuitable !!!')
                                module_logger.info("\n".join(logmsg)) 
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                piledim = ['n/a', 'n/a', 'n/a', 
                                               'n/a', 'n/a', 
                                               'n/a']                               
                                break
                            if pilemindimflag == 'True':
                                break
                            """ Pile load is applied at top of pile. Pile is 
                                therefore fully embedded (foundations) or 
                                possibly buried (anchors) """                               
                            loadattpnt[j] = 0.0
                            for k in range(0,klim):                                  
                                """ Determine design loads at pile head """
                                horzdesload[j] = (self._variables.foundsf 
                                    * math.sqrt(self.horzfoundloads[j][
                                    self.maxloadindex[j]] ** 2.0 
                                    + self.horzfoundloads[j][
                                    self.minloadindex[j]] ** 2.0))
                                vertdesload[j] = (self._variables.foundsf 
                                                * self.vertfoundloads[j]) 
                                self.horzdesload[j] = horzdesload[j]
                                self.vertdesload[j] = vertdesload[j]
                                """ Resultant design load """
                                resdesload[j] = math.sqrt(
                                    horzdesload[j] ** 2.0 
                                    + vertdesload[j] ** 2.0)
                                if math.fabs(resdesload[j]) == 0.0:
                                    module_logger.warn('WARNING: Resultant design load equal to zero, setting minimum pile size')
                                    # Swtich from True to False                                    
                                    self.foundnotfoundflag[posftyps[0]][j] = 'False'
                                    complim = 0.0 
                                    pilemindia, pilethk, pileyld, pilemod = (
                                                        foundcompret(complim))
                                    pilecompind = pilemindia[0] 
                                    piledia = pilemindia[1] 
                                    pileleng = 2.0 * piledia
                                    pilecloseend = 'False'
                                    if (self.soilgroup[j] in ('cohesive', 
                                                        'cohesionless')):
                                        pilegroutdia = 0.0
                                    elif self.soilgroup[j] in ('other'):
                                        groutdiathkratio = 40.0
                                        pilegroutdia = (piledia / (1.0 - 2.0 
                                                    / groutdiathkratio))                                       
                                    piledim = [piledia, pilethk, pileleng, 
                                               pilecloseend, pilecompind, 
                                               pilegroutdia]  
                                    pilemindimflag = 'True'
                                    break
                                """ Component retrieval """ 
                                if k == 0:                                
                                    """ First search for smallest diameter pile 
                                        in DB """
                                    complim = 0.0   
                                if q == 0:
                                    pilemindia, pilethk, pileyld, pilemod = (
                                                        foundcompret(complim))
                                    pilecompind = pilemindia[0] 
                                    piledia = pilemindia[1]  
                                    
                                if (q > 0 and k == 0):
                                    if pilelengexceedflag == 'True':
                                        """ Look for large diameter pile if length limit of 100m has been reached """
                                        complim = piledia                                            
                                        pilemindia, pilethk, pileyld, pilemod = foundcompret(complim)
                                        """ Break loop if a suitable pile cannot be found in the database """
                                        if pilemindia == 0.0:
                                            if posftyps[0] not in self.unsuitftyps:
                                                self.unsuitftyps.append(posftyps[0])                                     
                                            piledim = ['n/a', 'n/a', 'n/a', 
                                                       'n/a', 'n/a', 
                                                       'n/a']    
                                            module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                                            self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                            break
                                        else:                                        
                                            pilecompind = pilemindia[0] 
                                            piledia = pilemindia[1] 
                                            pileleng = 2.0 * piledia
                                       
                                pilestiff = (pilemod * math.pi * pilethk 
                                    * (piledia / 2.0) ** 3.0)
                                    
                                """ Break loop if a suitable pile cannot be found in the database """
                                if pilemindia == 0.0:                                    
                                    piledim = ['n/a', 'n/a', 'n/a', 
                                               'n/a', 'n/a', 
                                               'n/a']
                                    if posftyps[0] not in self.unsuitftyps:
                                        self.unsuitftyps.append(posftyps[0]) 
                                    module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                    break
                                    
                                if (self.soilgroup[j] in ('cohesive', 
                                                        'cohesionless')):
                                    pilegroutdia = 0.0
                                    """ Embedment depth """
                                    embeddepth = 2.0 * piledia
                                    if (systype in ("wavefloat",
                                                         "tidefloat")):                                     
                                        omega = (self.moorcomptab.ix[
                                            self.compblocks[0],'wet mass'] 
                                            * self._variables.gravity)                                        
                                        """ Angle from horizontal at dip down 
                                            point """
                                        resdesloadang[j] = math.tan(
                                            vertdesload[j] / horzdesload[j])
                                        padeyedesload[j], padeyetheta[j] = (
                                            buriedline(embeddepth, 
                                            resdesload[j], 
                                            resdesloadang[j], omega))
                                elif self.soilgroup[j] == 'other':
                                    if (systype in ("wavefloat",
                                                         "tidefloat")): 
                                        """ Padeye at seafloor level """
                                        embeddepth = 0.0
                                        padeyedesload[j] = resdesload[j] 
                                        resdesloadang[j] = math.tan(
                                            vertdesload[j] / horzdesload[j])
                                        padeyetheta[j] = resdesloadang[j]
                                        
                                if (self.soilgroup[j] in ('cohesive', 
                                                            'cohesionless')):
                                    """ Pile deflection criteria. 10% is 
                                        recommended for anchor piles this can 
                                        be lower for foundation piles """
                                    deflmax = 0.1 * piledia
                                    """ Coefficient of subgrade soil reaction. 
                                        Note for calcareous soils use 
                                        cohensionless approach with relative 
                                        soil density of 35% """
                                    def coefssr(piledia):
                                        if self.soilgroup[j] == 'cohesive':
                                            """ Undrained shear strength 
                                                averaged over a depth of 
                                                four diameters """
                                            unshstrav = ((
                                                self.unshstr[1] 
                                                + self.unshstr[1] 
                                                + self.unshstr[0] 
                                                * 4.0 * piledia) / 2.0)
                                            if (self.soiltyp[j] in ('vsc', 
                                                                'sc', 'mc')):
                                                k1coefcol = 1
                                            elif (self.soiltyp[j] in ('fc', 
                                                                      'stc')):
                                                k1coefcol = 2
                                            """ Linear interpolation """
                                            k1coefint = interpolate.interp1d(
                                                self._variables.k1coef[:,0], 
                                                self._variables.k1coef[:,k1coefcol],
                                                bounds_error=False)
                                            k1coef = k1coefint(100.0 * deflmax 
                                                    / piledia) 
                                            sgrcoef = (unshstrav * k1coef 
                                                        / piledia)
                                        elif self.soilgroup[j] == 'cohesionless':
                                            sgrcoefint = interpolate.interp2d(
                                                self._variables.subgradereaccoef[0,1:len(
                                                self._variables.subgradereaccoef)], 
                                                self._variables.subgradereaccoef[1:len(
                                                self._variables.subgradereaccoef),0], 
                                                self._variables.subgradereaccoef[1:len(
                                                self._variables.subgradereaccoef),1:len(
                                                self._variables.subgradereaccoef)])
                                            sgrcoef = sgrcoefint(
                                                self.relsoilden, 
                                                100.0 * deflmax / piledia)
                                        return sgrcoef
                                        
                                    sgrcoef = coefssr(piledia)
                                    """ Pile-soil stiffness """
                                    pilesoilstiff = ((pilestiff 
                                                        / sgrcoef) ** 0.2 )  
                                    if (k == 0 and q == 0): 
                                        """ Trial pile length, minimum of 
                                            3 x pile-soil stiffness is 
                                            recommended """
                                        pileleng = 3 * pilesoilstiff
                                    """ Maximum depth coefficient """
                                    maxdepthcoef  = pileleng / pilesoilstiff                            
                                    if (maxdepthcoef < min(
                                        self._variables.piledefcoef[:,0])): 
                                        piledefcoefay = self._variables.piledefcoef[0,1]
                                        piledefcoefby = self._variables.piledefcoef[0,2]
                                    elif (maxdepthcoef > max(
                                                    self._variables.piledefcoef[:,0])):
                                        piledefcoefay = self._variables.piledefcoef[-1,1]
                                        piledefcoefby = self._variables.piledefcoef[-1,2]
                                    else:
                                        """ Pile deflection coefficients """
                                        piledefcoefayint = interpolate.interp1d(
                                            self._variables.piledefcoef[:,0], 
                                            self._variables.piledefcoef[:,1],
                                            bounds_error=False)
                                        piledefcoefbyint = interpolate.interp1d(
                                            self._variables.piledefcoef[:,0], 
                                            self._variables.piledefcoef[:,2],
                                            bounds_error=False)
                                        piledefcoefay = piledefcoefayint(
                                                                maxdepthcoef)
                                        piledefcoefby = piledefcoefbyint(
                                                                maxdepthcoef)                                    
                                    """ Lateral load capacity """
                                    latloadcap = ((deflmax * pilestiff) 
                                        / (piledefcoefay * pilesoilstiff ** 3 
                                        + loadattpnt[j] * piledefcoefby 
                                        * pilesoilstiff ** 2.0))
                                
                                elif self.soilgroup[j] == 'other':
                                    if (k == 0 and q == 0):
                                        """ Trial pile length """
                                        pileleng = 3.0
                                    """ Note: Single layer of rock assumed """
                                    latloadcap = (self.rockcomstr 
                                                    * piledia * pileleng)
                                
                                """ Alter pile length or diameter if 
                                    required """
                                """ Note anchor head depth could also be 
                                    increased """
                                if latloadcap > 1.05 * horzdesload[j]: 
                                    pileleng = pileleng - 0.5
                                    if pileleng / piledia < 2.0:
                                        pileleng = 2.0 * piledia
                                    if pileleng < 2.0:
                                        """ Minimum length of pile is 2m """
                                        pileleng = 2.0
                                    if k == klim - 1:
                                        """ Shorter pile not feasible (probably 
                                            due to low design loads) """
                                        latloadcheck = 'True'
                                elif latloadcap < 0.95 * horzdesload[j]: 
                                    pileleng = pileleng + 0.5
                                    """ Pile length upper limit of 100m """
                                    if pileleng >= 100.0:
                                        pilelengexceedflag = 'True'
                                        break
                                elif (latloadcap >= 0.95 * horzdesload[j] 
                                    and latloadcap <= 1.05 * horzdesload[j]):
                                    """ 5% tolerance """
                                    latloadcheck = 'True'                            
                                    break
                            
                            if posftyps[0] in self.unsuitftyps:                                
                                piledim = ['n/a', 'n/a', 'n/a', 
                                               'n/a', 'n/a', 
                                               'n/a']                                
                                break
                            
                            """ Axial load analysis """
                            if (self.soilgroup[j] in ('cohesive', 
                                                            'cohesionless')):
                                """ Average effective overburden pressure """
                                effoverbdpressav = (self.soilweight 
                                                    * pileleng / 2.0) 
                                if self.soilgroup[j] == 'cohesionless':
                                    """ Note: check of maximum skin friction 
                                    resistance is based on non-calcareous 
                                    sands only """ 
                                    skinfricreslimint = interpolate.interp1d(
                                        self._variables.pilefricresnoncal[:,0], 
                                        self._variables.pilefricresnoncal[:,3],
                                        bounds_error=False)
                                    skinfricreslim = skinfricreslimint(
                                        self.dsfangrad * 180.0 / math.pi)
                                    
                                if vertdesload[j] > 0:
                                    pilecloseend = 'False' 
                                    for l in range(0,klim):        
                                        """ Skin frictional resistance per unit 
                                            pile length """
                                        if self.soilgroup[j] == 'cohesive':
                                            """ Undrained shear strength 
                                                averaged over pile length """
                                            unshstrav = ((
                                                self.unshstr[1] 
                                                + self.unshstr[1] 
                                                + self.unshstr[0] 
                                                * pileleng) / 2.0)
                                            if (unshstrav / effoverbdpressav 
                                                <= 0.4):
                                                """ Normally consolidated 
                                                    soil. Note adapted for 
                                                    metric units (pile 
                                                    length) """
                                                skinfricresavup = (
                                                    effoverbdpressav 
                                                    * (0.468 - 0.052 
                                                    * math.log(pileleng 
                                                    * 3.28 /2.0)))
                                                if skinfricresavup > unshstrav:
                                                    skinfricresavup = unshstrav
                                            else: 
                                                """ Overconsolidated soils """
                                                skinfricresavup = ((0.458 
                                                    - 0.155 * math.log(
                                                    unshstrav 
                                                    / effoverbdpressav)) 
                                                    * unshstrav)
                                                if (unshstrav 
                                                    / effoverbdpressav > 2.0):
                                                    skinfricresavup = (0.351 
                                                        * unshstrav)
                                        elif self.soilgroup[j] == 'cohesionless': 
                                            skinfricresavup = (0.5 
                                                * effoverbdpressav * math.tan(
                                                self.dsfangrad - 5.0 
                                                * math.pi / 180.0))
                                            if (skinfricresavup 
                                                > skinfricreslim):
                                                skinfricresavup = skinfricreslim    
                                        """ Calculate uplift capacity """
                                        pilesurfarea = (math.pi * piledia 
                                            * pileleng)
                                        upliftcap = (pilesurfarea 
                                            * skinfricresavup)
                                        if upliftcap < vertdesload[j]:
                                            """ Increase pile length by 1 metre 
                                                and recalculate """
                                            pileleng = pileleng + 1.0
                                            """ Pile length upper limit of 100m """
                                            if pileleng >= 100.0:
                                                pilelengexceedflag = 'True'
                                                break
                                        else:
                                            upliftcheck = 'True'
                                            comploadcheck = 'True'
                                            break                   
                                elif vertdesload[j] < 0:
                                    for m in range(0,klim):
                                        """ Effective overburden pressure at 
                                            pile tip """
                                        effoverbdpresstip = (
                                                    self.soilweight 
                                                    * pileleng)
                                        if self.soilgroup[j] == 'cohesive':
                                            """ Undrained shear strength 
                                                averaged at pile tip """
                                            unshstrtip = (
                                                self.unshstr[1] 
                                                + self.unshstr[0] 
                                                * pileleng)
                                            """ Unit soil bearing capacity of 
                                                pile tip """
                                            soilbearcaptip = 9 * unshstrtip
                                            """ Undrained shear strength 
                                                averaged over pile length """
                                            unshstrav = ((
                                                self.unshstr[1] 
                                                + self.unshstr[1] 
                                                + self.unshstr[0] 
                                                * pileleng) / 2.0)
                                            if (unshstrav / effoverbdpressav 
                                                <= 0.4):
                                                """ Normally consolidated 
                                                    soil. Note adapted for 
                                                    metric units (pile 
                                                    length) """
                                                skinfricresavcomp = (
                                                    effoverbdpressav 
                                                    * (0.468 - 0.052 
                                                    * math.log(pileleng * 3.28 
                                                    / 2.0)))
                                                if (skinfricresavcomp 
                                                    > unshstrav):
                                                    skinfricresavcomp = unshstrav
                                            else: 
                                                """ Overconsolidated soils """
                                                skinfricresavcomp = ((0.458 
                                                    - 0.155 * math.log(
                                                    unshstrav 
                                                    / effoverbdpressav)) 
                                                    * unshstrav)
                                                if (unshstrav 
                                                    / effoverbdpressav > 2.0):
                                                    skinfricresavcomp = (0.351 
                                                        * unshstrav)
                                        elif (self.soilgroup[j] == 
                                            'cohesionless'):
                                            skinfricresavcomp = (0.7 
                                                * effoverbdpressav * math.tan(
                                                self.dsfangrad - 5.0 
                                                * math.pi / 180.0))
                                            if (skinfricresavcomp 
                                                > skinfricreslim):
                                                skinfricresavcomp = skinfricreslim
                                            """ Bearing capacity factor 
                                                limiting value """
                                            bcflimint = interpolate.interp1d(
                                                self._variables.pilefricresnoncal[:,0], 
                                                self._variables.pilefricresnoncal[:,2],
                                                bounds_error=False)
                                            bcflim = bcflimint(self.dsfangrad 
                                                * 180.0 / math.pi)                                
                                            """ Unit soil bearing capacity 
                                                limiting value """
                                            soilbearcaplimint = interpolate.interp1d(
                                                self._variables.pilefricresnoncal[:,0], 
                                                self._variables.pilefricresnoncal[:,4],
                                                bounds_error=False)
                                            soilbearcaplim = soilbearcaplimint(
                                                self.dsfangrad * 180.0 
                                                / math.pi)      
                                            """ Unit soil bearing capacity of 
                                                pile tip """
                                            soilbearcaptip = (effoverbdpresstip 
                                                * bcflim) 
                                            if soilbearcaptip > soilbearcaplim:
                                                soilbearcaptip = soilbearcaplim                                
                                        """ Pile tip bearing capacity """
                                        pilecaparea = ((math.pi / 4.0) 
                                            * piledia ** 2.0) 
                                        pilesurfarea = (math.pi * piledia 
                                            * pileleng)
                                        """ Try open ended pile first """ 
                                        pilecloseend = 'False' 
                                        bearcap = (pilesurfarea 
                                            * skinfricresavcomp)
                                        bearcaptip = bearcap
                                        totbearcap = bearcap + bearcaptip
                                        if (totbearcap 
                                            < math.fabs(vertdesload[j])):
                                            """ Try closed end pile """
                                            pilecloseend = 'True'
                                            bearcaptip = (pilecaparea 
                                                * soilbearcaptip)
                                            totbearcap = bearcap + bearcaptip                                
                                        if (totbearcap 
                                            < math.fabs(vertdesload[j])):
                                            """ Increase pile length by 1 metre 
                                                and recalculate """
                                            pileleng = pileleng + 1.0
                                            """ Pile length upper limit of 20m """
                                            if pileleng >= 20.0:
                                                pilelengexceedflag = 'True'
                                                break
                                        else:
                                            comploadcheck = 'True'
                                            upliftcheck = 'True'
                                            break
                                elif vertdesload[j] == 0:
                                    comploadcheck = 'True'
                                    upliftcheck = 'True'

                            elif self.soilgroup[j] in ('other'):
                                """ Pile is open-ended """
                                for l in range(0,klim):
                                    """ Grout bond strength formulation based 
                                        on HSE pile/sleeve connection offshore 
                                        technology report 2001/016. Note: Grout 
                                        specified for entire length of embedded 
                                        pile. Shear connector design not 
                                        included. 
                                        Approach valid for pile geometries 
                                        24<=(D/t)<=40 and grout annulus  
                                        geometries 10<=(D/t)<=45 """
                                    
                                    """ Stiffness factor with modular ratio set 
                                        to 18 (long-term > 28 days) """
                                    modratio = 18.0
                                    groutdiathkratio = 40.0
                                    pilegroutdia = (piledia / (1.0 - 2.0 
                                                    / groutdiathkratio))
                                    """ Coefficient of grouted length to pile 
                                        diameter ratio """
                                    glpdratios = np.array([[2.0, 1.0], 
                                                           [4.0, 0.9], 
                                                            [8.0, 0.8], 
                                                            [12.0, 0.7]])
                                    glpdratiosint = interpolate.interp1d(
                                                            glpdratios[:,0],
                                                            glpdratios[:,1])
                                    
                                    
                                    if pileleng / piledia < 12.0:
                                        groutlengpilediaratio = glpdratiosint(
                                                            pileleng / piledia)
                                    elif pileleng / piledia >= 12.0:
                                        groutlengpilediaratio = 0.7
                                    """ Plain pile connection """                                        
                                    surfcondfactr = 0.6
                                    stifffact = ((modratio * groutdiathkratio) 
                                        ** -1.0 + (modratio * (piledia 
                                        / pilethk)) ** -1.0)
                                    """ Grout pile bond strength in N/m2 """
                                    groutpilebondstr = (1.0e6 * stifffact 
                                        * groutlengpilediaratio * 9.0 
                                        * surfcondfactr 
                                        * self._variables.groutstr ** 0.5)
                                    self.groutpilebondstr = groutpilebondstr
                                    """ Calculate uplift capacity. Note: It is 
                                        assumed that axial strength can be 
                                        satisfied in upward and downward 
                                        directions """
                                    pileperim = math.pi * piledia
                                    upliftcap = (groutpilebondstr * pileleng 
                                                                * pileperim)
                                    if (upliftcap < self._variables.groutsf
                                        * math.fabs(vertdesload[j])):
                                        """ Increase pile length by 1 metre and 
                                            recalculate """
                                        pileleng = pileleng + 1.0
                                        """ Pile length upper limit of 20m """
                                        if pileleng >= 20.0:
                                            pilelengexceedflag = 'True'
                                            break
                                    else:
                                        upliftcheck = 'True'
                                        comploadcheck = 'True'
                                        break                                                                 
                            
                            if posftyps[0] in self.unsuitftyps:                                
                                piledim = ['n/a', 'n/a', 'n/a', 
                                               'n/a', 'n/a', 
                                               'n/a']                                
                                break 
                                
                            if (self.soilgroup[j] in ('cohesive', 
                                                        'cohesionless')):
                                """ Pile moment coefficients. Note value will 
                                    be extrapolated if outside of range """
                                pilemomcoefamint = interpolate.interp2d(
                                    self._variables.pilemomcoefam[0,1:len(
                                    self._variables.pilemomcoefam)], 
                                    self._variables.pilemomcoefam[1:len(
                                    self._variables.pilemomcoefam),0], 
                                    self._variables.pilemomcoefam[1:len(
                                    self._variables.pilemomcoefam),1:len(
                                    self._variables.pilemomcoefam)])
                                pilemomcoefbmint = interpolate.interp2d(
                                    self._variables.pilemomcoefbm[0,1:len(
                                    self._variables.pilemomcoefbm)], 
                                    self._variables.pilemomcoefbm[1:len(
                                    self._variables.pilemomcoefbm),0], 
                                    self._variables.pilemomcoefbm[1:len(
                                    self._variables.pilemomcoefbm),1:len(
                                    self._variables.pilemomcoefbm)])
                                for n in range(0,klim):
                                    if pilemindimflag == 'True':
                                        break
                                    pilecsa = ((math.pi / 4.0) 
                                        * (piledia ** 2.0 - (piledia - 2.0 
                                        * pilethk) ** 2.0))
                                    pilesecmod = ((math.pi / (32.0 * piledia)) 
                                        * (piledia ** 4.0 - (piledia - 2.0 
                                        * pilethk) ** 4.0))
                                    pilelengs = np.linspace(0, pileleng, 20)                     
                                    pilemomcoefam = [0 for row 
                                                    in range(len(pilelengs))]
                                    pilemomcoefbm = [0 for row 
                                                    in range(len(pilelengs))]                           
                                    pilestiff = (pilemod * math.pi * pilethk
                                                * (piledia / 2.0) ** 3.0)
                                    """ Subgrade reaction coefficient 
                                        recalculated in case pile diameter 
                                        has been altered """                            
                                    sgrcoef = coefssr(piledia)
                                    """ Pile-soil stiffness """
                                    pilesoilstiff = ((pilestiff 
                                                        / sgrcoef) ** 0.2)
                                    """ Maximum depth coefficient """
                                    maxdepthcoef  = pileleng / pilesoilstiff                            
                                    for p in range(0, len(pilelengs)):
                                        """ Depth coefficient, try several 
                                            values """
                                        depthcoef = (pilelengs[p] 
                                                        / pilesoilstiff)
                                        pilemomcoefam[p] = pilemomcoefamint(
                                                    maxdepthcoef, depthcoef)
                                        pilemomcoefbm[p] = pilemomcoefbmint(
                                                    maxdepthcoef, depthcoef)
                                    pilemomcoefam = max(pilemomcoefam)
                                    pilemomcoefbm = max(pilemomcoefbm)
                                    """ Calculate maximum compression """
                                    maxmom = (pilemomcoefam * horzdesload[j] 
                                        * pilesoilstiff + pilemomcoefbm 
                                        * max(self.sysbasemom))
                                    if vertdesload[j] > 0:                        
                                        pileloadten = vertdesload[j]
                                    else: pileloadten = 0.0                         
                                    if vertdesload[j] < 0:                        
                                        pileloadcomp = math.fabs(
                                                                vertdesload[j])
                                    else: pileloadcomp = 0.0 
                                    """ Maximum stress in tension """                        
                                    maxstressten =  ((-pileloadten / pilecsa) 
                                                    - (maxmom / pilesecmod))
                                    """ Maximum stress in compression """  
                                    maxstresscomp =  ((pileloadcomp / pilecsa) 
                                                    + (maxmom / pilesecmod))
            
                                    """ Check that calculated compressive loads 
                                        are above allowable steel stress. Note 
                                        limit is 60% of yield stress """
                                    if ((math.fabs(maxstressten) > 0.6 
                                        * pileyld or math.fabs(maxstresscomp) 
                                        > 0.6 * pileyld)):
            #                                print 'Pile inadequate in compression'
                                        complim = piledia   
                                        pilemindia, pilethk, pileyld, pilemod = foundcompret(complim)                                        
                                        """ Break loop if a suitable pile cannot be found in the database """
                                        if pilemindia == 0.0:
                                            if posftyps[0] not in self.unsuitftyps:
                                                self.unsuitftyps.append(posftyps[0]) 
                                                piledim = ['n/a', 'n/a', 'n/a', 
                                                           'n/a', 'n/a', 
                                                           'n/a']
                                            break
                                        piledia = pilemindia[1]         
                                    else: 
                                        momcheck = 'True'                                    
                                        break 
                                   
                            if (self.soilgroup[j] in ('cohesive', 
                                                        'cohesionless')):                                                       
                                if (momcheck == 'True'
                                    and comploadcheck == 'True' 
                                    and latloadcheck == 'True' 
                                    and upliftcheck == 'True'):
                                    # print 'Solution found'                                
                                    piledim = [piledia, pilethk, pileleng, 
                                               pilecloseend, pilecompind, 
                                               pilegroutdia]
                                    break
                            elif self.soilgroup[j] == 'other':                                  
                                """ Moment calculations not carried out for 
                                    rock """
                                if (comploadcheck == 'True' 
                                    and latloadcheck == 'True' 
                                    and upliftcheck == 'True'):                                                                                                      
                                    piledim = [piledia, pilethk, pileleng, 
                                               pilecloseend, pilecompind, 
                                               pilegroutdia]
                                    logmsg = [""]
                                    logmsg.append('Solution found {}'.format(piledim))
                                    module_logger.info("\n".join(logmsg))   
                                    self.founddesfail[j] = 'False' 
                                    break
                                    
                            if (k == klim - 1 and q == klim - 1):
                                """ For when a solution cannot be found """ 
                                piledim = [piledia, pilethk, pileleng, 
                                           pilecloseend, pilecompind, 
                                           pilegroutdia] 
                                module_logger.warn('WARNING: Solution not found within set number of iterations!')   
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'                                
                                if posftyps[0] not in self.unsuitftyps:
                                    self.unsuitftyps.append(posftyps[0])            
                                               
                        return piledim
                        
                    complim = 0.0
                    self.piledim[j] = piledes(complim) 
                    
                    if posftyps[0] in self.unsuitftyps:
                        pass 
                    else:
                        if self.piledim[j][1] == 0.0:
                            self.selpilesubtyp[j] = 'pin pile'
                        else:
                            self.selpilesubtyp[j] = 'pipe pile'
                        if self.piledim[j][3] == 'True':
                            """ Pile has closed end """
                            self.pilevolsteel[j] = (self.piledim[j][2] 
                                * (math.pi / 4.0) * (self.piledim[j][0] ** 2.0 
                                - (self.piledim[j][0] - 2 
                                * self.piledim[j][1]) ** 2.0) 
                                + (math.pi / 4.0) * self.piledim[j][0] ** 2.0 
                                * self.piledim[j][1])
                        elif self.piledim[j][3] == 'False':
                            self.pilevolsteel[j] = (self.piledim[j][2] 
                                * (math.pi / 4.0) * (self.piledim[j][0] ** 2.0 
                                - (self.piledim[j][0] - 2 
                                * self.piledim[j][1]) ** 2.0)) 
                        self.pilevolcon[j] = 0.0 
                        if self.piledim[j][5] > 0.0:
                            self.pilevolgrout[j] = (self.piledim[j][2] 
                                * (math.pi / 4.0) * (self.piledim[j][5] ** 2.0 
                                - self.piledim[j][0] ** 2.0))       
                        else: self.pilevolgrout[j] = 0.0
                        self.foundradadd[j] = (self.foundrad[j] 
                                                    + 0.5 * self.piledim[j][0])

                        """ Grout selection from the database may be added here """
                        if self.pilevolgrout[j] > 0.0:
                            self.pilegrouttyp[j] = 'grout'
                        else:
                            self.pilegrouttyp[j] = 'n/a'
                        self.selfoundsubtypdict[posftyps[0]] = self.selpilesubtyp
                        self.foundvolgroutdict[j][posftyps[0]] = self.pilevolgrout[j]
                        self.selfoundgrouttypdict[j][posftyps[0]] = self.pilegrouttyp[j]
                        self.foundvolsteeldict[j][posftyps[0]] = self.pilevolsteel[j]
                        self.foundvolcondict[j][posftyps[0]] = self.pilevolcon[j]          
                        self.founddimdict[j][posftyps[0]] = self.piledim[j]
                        self.pileitemcost[j] = self._variables.compdict[
                                                self.piledim[j][4]]['item11'] 
                        self.foundcostdict[j][posftyps[0]] = self.pileitemcost[j]
                    
                elif posftyps[0] == 'suctioncaisson':
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Suction caisson anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))             
                    klim = 200      
                    def caissdes(complim): 
                        """ Load is applied at top of caisson. Caisson is 
                            therefore fully embedded (foundations) or possibly 
                            buried (anchors) """                               
                        loadattpnt[j] = 0.0
                        for k in range(0,klim):       
                            """ Determine design loads at dipdown point """
                            horzdesload[j] = (self._variables.foundsf 
                                * math.sqrt(self.horzfoundloads[j][
                                self.maxloadindex[j]] ** 2.0 
                                + self.horzfoundloads[j][
                                self.minloadindex[j]] ** 2.0))
                            vertdesload[j] = (self._variables.foundsf 
                                            * self.vertfoundloads[j])
                            self.horzdesload[j] = horzdesload[j]
                            self.vertdesload[j] = vertdesload[j]
                            """ Resultant design load """
                            resdesload[j] = math.sqrt(horzdesload[j] ** 2.0 
                                + vertdesload[j] ** 2.0)
                            """ Component retrieval """ 
                            if k == 0:                                
                                """ First search for smallest diameter caisson 
                                    in DB which is >= 3.0m (Note: Iskander et al. 
                                    2002 suggest a lower limit of 4.0m) """
                                complim = 3.0 
                                caissmindia, caissthk, caissyld, caissmod = (
                                                        foundcompret(complim))
                                if math.fabs(resdesload[j]) == 0.0:
                                    module_logger.warn('WARNING: Resultant design load at padeye equal to zero, setting minimum caisson size')
                                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                    caisscompind = caissmindia[0]
                                    caissdia = caissmindia[1] 
                                    """ Length diameter ratio typically between 
                                        1.0-5.0 """
                                    caisslengdiaratio = 3.0
                                    """ Trial caisson length """
                                    caissleng = caisslengdiaratio * caissdia
                                    caissdim = [caissdia, caissthk, caissleng, 
                                            caisscompind]                                      
                                    break
                                if caissmindia == 0.0:
                                    if posftyps[0] not in self.unsuitftyps:
                                        self.unsuitftyps.append(posftyps[0])                                     
                                    caissdim = ['n/a', 'n/a', 'n/a', 
                                               'n/a']                                            
                                    module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                    break
                                    
                                else:                                
                                    caissdia = caissmindia[1] 
                                    """ Length diameter ratio typically between 
                                        1.0-5.0 """
                                    caisslengdiaratio = 3.0
                                    """ Trial caisson length """
                                    caissleng = caisslengdiaratio * caissdia
                                    """ Maximum embedment depth is soil layer depth """
                                    if caissleng > self.soildep[j]:
                                        caissleng = self.soildep[j]
                            """ Attachment point set to 0.7 x caisson 
                            length as per Randolph and Gourvenec, 2011 """
                            attatchdepth = 0.7 * caissleng                                    
                            omega = (self.moorcomptab.ix[self.compblocks[0],
                                'wet mass'] * self._variables.gravity)
                                                                                        
                            """ Angle from horizontal at dip down point """
                            if (math.fabs(vertdesload[j]) > 0.0 and math.fabs(horzdesload[j]) > 0.0):                            
                                resdesloadang[j] = math.atan(vertdesload[j] 
                                                            / horzdesload[j])                         
                            elif (math.fabs(vertdesload[j]) > 0.0 and math.fabs(horzdesload[j]) == 0.0):
                                resdesloadang[j] = 90.0 * math.pi / 180.0
                            elif (math.fabs(vertdesload[j]) == 0.0 and math.fabs(horzdesload[j]) > 0.0):
                                resdesloadang[j] = 0.0
                            
                            logmsg = [""]                       
                            logmsg.append('------------------------------------------------------------')                                
                            logmsg.append('Resultant design load [N] and angle [rad] at the seafloor {}'.format([resdesload[j], resdesloadang[j]]))                                
                            logmsg.append('------------------------------------------------------------')                                
                            module_logger.info("\n".join(logmsg))
                            padeyedesload[j], padeyetheta[j] = buriedline(
                                attatchdepth, resdesload[j], resdesloadang[j], 
                                omega)
                            
                            padeyehorzload[j] = (padeyedesload[j] * math.cos(
                                                    padeyetheta[j]))
                            padeyevertload[j] = (padeyedesload[j] * math.sin(
                                                    padeyetheta[j]))
                            caissarea = (math.pi / 4.0) * caissdia ** 2.0
                            
                            """ Uplift capacity factor ranges from 7.0 to 11.0. 
                                Typically 9.0 is used (Taiebat et al. 2005, 
                                Randolph et al. 2011) """                            
                            upliftcapfacnp = 9.0
                            """ Vertical embedment factor """
                            zetace = 1.0 + 0.4 * (caissleng / caissdia)
                            """ Shape factor. Cylinder assumed """
                            zetas = 1.2
                            """ Undrained shear strength averaged over a depth 
                                equivalent to caisson length """
                            unshstrav = ((self.unshstr[1] 
                                + self.unshstr[1] 
                                + self.unshstr[0] * caissleng) 
                                / 2.0)
                            """ Vertical capacity (Taiebat et al. 2005) """
                            upliftcap = (upliftcapfacnp * zetace * zetas 
                                        * caissarea * unshstrav)
                            """ Lateral capacity factor (Deng et al. 2001) """
                            latloadcapfacnh  = (3.6 / math.sqrt((0.75 
                                                - (attatchdepth / caissleng)) ** 2.0 + (0.45 
                                                * attatchdepth / caissleng) ** 2.0))
                            """  Lateral load capacity  """
                            latloadcap = (caissleng * caissdia 
                                            * latloadcapfacnh * unshstrav)
                            """ Check if loads fit within (elliptical) failure 
                                envelope (Randolph et al. 2011) """
                            envcoefa = (caissleng / caissdia) + 0.5
                            envcoefb = (caissleng / (3.0 * caissdia)) + 4.5                            
                            failenv = (((padeyehorzload[j] / latloadcap) 
                                ** envcoefa) + ((padeyevertload[j] / upliftcap) 
                                ** envcoefb))
                            
                            if failenv > 1.0:     
                                """ Caisson under-designed """
                                complim = caissdia   
                                caissmindia, caissthk, caissyld, caissmod = (
                                                        foundcompret(complim))
                                if caissmindia == 0.0:
                                    if posftyps[0] not in self.unsuitftyps:
                                        self.unsuitftyps.append(posftyps[0])                                     
                                    caissdim = ['n/a', 'n/a', 'n/a', 
                                               'n/a']                                            
                                    module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                    break
                                else:  
                                    caissdia = caissmindia[1]  
                                    caissleng = caisslengdiaratio * caissdia
                                    """ Maximum embedment depth is soil layer depth """ 
                                    if caissleng > self.soildep[j]:
                                        caissleng = self.soildep[j]
                            elif failenv < 0.01:
                                """ Caisson over-designed """
                                caissleng = caissleng - 1.0 
                                if caissleng < caissdia:
                                    caissleng = caissdia
                                    caisscompind = caissmindia[0] 
                                    caissdim = [caissdia, caissthk, caissleng, 
                                                caisscompind]
                                    break
                            else:
                                caisscompind = caissmindia[0] 
                                caissdim = [caissdia, caissthk, caissleng, 
                                            caisscompind]
                                logmsg = [""]
                                logmsg.append('Solution found {}'.format(caissdim))
                                module_logger.info("\n".join(logmsg))  
                                self.founddesfail[j] = 'False' 
                                break
                            if k == klim - 1:
                                """ For when a solution cannot be found """ 
                                caissdim = [caissdia, caissthk, caissleng, 
                                            caisscompind]
                                if posftyps[0] not in self.unsuitftyps:
                                    self.unsuitftyps.append(posftyps[0])    
                                module_logger.warn('WARNING: Solution not found within set number of iterations!')              
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'                                
                        return caissdim                        
                     
                        
                    complim = 0.0
                    self.caissdim[j] = caissdes(complim) 
                    
                    if posftyps[0] in self.unsuitftyps:
                        pass 
                    else:
                        self.foundradadd[j] = (self.foundrad[j] 
                                                    + 0.5 * self.caissdim[j][0])

                        self.caissvolsteel[j] = (self.caissdim[j][2] 
                            * (math.pi / 4.0) * (self.caissdim[j][0] ** 2.0 
                            - (self.caissdim[j][0] - 2 
                            * self.caissdim[j][1]) ** 2.0))
                        self.caissvolcon[j] = 0.0                        
                        self.selfoundsubtypdict[posftyps[0]] = ['closed top' 
                                            for row in range(self.quanfound)]
                        self.foundvolgroutdict[j][posftyps[0]] = 0.0            
                        self.selfoundgrouttypdict[j][posftyps[0]] = 'n/a'
                        self.foundvolsteeldict[j][posftyps[0]] = self.caissvolsteel[j]
                        self.foundvolcondict[j][posftyps[0]] = self.caissvolcon[j]           
                        self.founddimdict[j][posftyps[0]] = self.caissdim[j]
                        self.caissitemcost[j] = self._variables.compdict[
                                                self.caissdim[j][3]]['item11'] 
                        self.foundcostdict[j][posftyps[0]] = self.caissitemcost[j]
                        
                    
                elif posftyps[0] == 'directembedment':
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Direct embedment anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))       
                    
                    def directembedanc(horzfoundloads, vertfoundloads):
                        """ Buoyant weight of embedded chain per unit 
                            length """
                        omega = (self.moorcomptab.ix[self.compblocks[0],
                                                    'wet mass'] 
                                                    * self._variables.gravity)
                        """ Determine design loads at connection point with 
                            anchor. Includes approximate contribution of buried 
                            mooring line """
                        horzdesload[j] = self.moorsf * math.sqrt(
                            horzfoundloads[j][self.maxloadindex[j]] ** 2.0 
                            + horzfoundloads[j][self.minloadindex[j]] ** 2.0)
                        vertdesload[j] = self.moorsf * vertfoundloads[j] 
                        self.horzdesload[j] = horzdesload[j]
                        self.vertdesload[j] = vertdesload[j]
                        resdesload[j] = math.sqrt(horzdesload[j] ** 2.0 
                            + vertdesload[j] ** 2.0)
                        """ Angle with vertical at dip down point """
                        if (math.fabs(vertdesload[j]) > 0.0 and math.fabs(horzdesload[j]) > 0.0):                            
                            resdesloadang[j] = math.atan(vertdesload[j] 
                                                            / horzdesload[j])                         
                        elif (math.fabs(vertdesload[j]) > 0.0 and math.fabs(horzdesload[j]) == 0.0):
                            resdesloadang[j] = 90.0 * math.pi / 180.0
                        elif (math.fabs(vertdesload[j]) == 0.0 and math.fabs(horzdesload[j]) > 0.0):
                            resdesloadang[j] = 0.0
                        logmsg = [""]                       
                        logmsg.append('------------------------------------------------------------')                                
                        logmsg.append('Resultant design load [N] and angle [rad] at the seafloor {}'.format([resdesload[j], resdesloadang[j]]))                                
                        logmsg.append('------------------------------------------------------------')                                
                        module_logger.info("\n".join(logmsg))
                        """ Trial base width and length """
                        platewidth = 1.0
                        plateleng = 1.75 * platewidth
                        platethk = 0.05 * platewidth                        
                        
                        for k in range(0,klim):
                            if k == 0:                                
                                """ Trial penetration depth """
                                pendepth = 6.0 * platewidth 
                                """ Embedment depth after keying """
                                if self.soilgroup[j] == 'cohesive':
                                    embeddepth = pendepth - 2.0 * plateleng
                                elif self.soilgroup[j] == 'cohesionless':
                                    embeddepth = pendepth - 1.5 * plateleng
                                if math.fabs(resdesload[j]) == 0.0:
                                    module_logger.warn('WARNING: Resultant design load at padeye equal to zero, setting minimum plate size')
                                    self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                    platedim = [platewidth, plateleng, 
                                                    platethk, embeddepth]                                  
                                    break
                            elif k > 0:
                                """ Embedment depth after keying """
                                if self.soilgroup[j] == 'cohesive':
                                    embeddepth = embeddepth + 2.0 * plateleng
                                elif self.soilgroup[j] == 'cohesionless':
                                    embeddepth = embeddepth + 1.5 * plateleng
                            
                            """ Maximum embedment depth is soil layer depth """
                            if embeddepth > self.soildep[j]:
                                embeddepth = self.soildep[j]
                            
                            """ Contribution of buried mooring line """                                    
                            padeyedesload[j], padeyetheta[j] = buriedline(
                                embeddepth, resdesload[j], resdesloadang[j], 
                                omega)

                            if self.soilgroup[j] == 'cohesive':
                                """ Undrained shear strength at embedment 
                                    depth """
                                unshstrz = (self.unshstr[1] 
                                            + self.unshstr[0] 
                                            * embeddepth)
                                """ Short-term (ncs) and long-term (nc) holding 
                                    capacity factors. Note: short-term assumes 
                                    full suction """
                                if unshstrz < 5171.07:
                                    hcfncs = (5.4219 * (embeddepth 
                                        / platewidth) + 6.245)
                                    hcfnc = (5.4649 * (embeddepth 
                                        / platewidth) + 0.3614)
                                elif (unshstrz >= 5171.07 
                                    and unshstrz < 6894.75729):
                                    hcfncsint = interpolate.interp1d(
                                        [5171.07, 6894.75729], 
                                        [5.4219 * (embeddepth / platewidth) 
                                        + 6.245, 3.4936 * (embeddepth 
                                        / platewidth) + 6.0271])
                                    hcfncs = hcfncsint(unshstrz)
                                    hcfncint = interpolate.interp1d(
                                        [5171.07, 6894.75729], 
                                        [5.4649 * (embeddepth / platewidth) 
                                        + 0.3614, 3.4821 * (embeddepth 
                                        / platewidth) + 0.2221])
                                    hcfnc = hcfncint(unshstrz)
                                elif (unshstrz >= 6894.75729 
                                    and unshstrz < 10342.13593):
                                    hcfncsint = interpolate.interp1d(
                                        [6894.75729, 10342.13593], 
                                        [3.4936 * (embeddepth / platewidth) 
                                        + 6.0271, 2.654 * (embeddepth 
                                        / platewidth) + 6.0806])
                                    hcfncs = hcfncsint(unshstrz)
                                    hcfncint = interpolate.interp1d(
                                        [6894.75729, 10342.13593], 
                                        [3.4821 * (embeddepth / platewidth) 
                                        + 0.2221, 2.6316 * (embeddepth 
                                        / platewidth) + 0.3424])
                                    hcfnc = hcfncint(unshstrz)
                                elif (unshstrz >= 10342.13593 
                                    and unshstrz < 27579.02916):
                                    hcfncsint = interpolate.interp1d(
                                        [10342.13593, 27579.02916], 
                                        [2.654 * (embeddepth / platewidth) 
                                        + 6.0806, 1.8459 * (embeddepth 
                                        / platewidth) + 6.0792])
                                    hcfncs = hcfncsint(unshstrz)
                                    hcfncint = interpolate.interp1d(
                                        [10342.13593, 27579.02916], 
                                        [2.6316 * (embeddepth / platewidth) 
                                        + 0.3424, 1.8602 * (embeddepth 
                                        / platewidth) + 0.2535])
                                    hcfnc = hcfncint(unshstrz)
                                elif unshstrz > 27579.02916:
                                    hcfncs = (1.8459 * (embeddepth 
                                                / platewidth) + 6.0792)
                                    hcfnc = (1.8602 * (embeddepth 
                                                / platewidth) + 0.2535)
                                """ Maximum value """
                                if hcfncs > 15.0:
                                    hcfncs = 15.0
                                if hcfnc > 10.0:
                                    hcfnc = 10.0
                                """ Strength reduction factor from Valent 1978. 
                                    Note: Only values for soft normally 
                                    consolidated silty clay 
                                    unshstrz ~ 13789.51458, 
                                    soil sensitivity = 3.0 are included, 
                                    covering several clay types """
                                strredfactor = 0.8
                                
                            platearea = platewidth * plateleng
                            
                            if (embeddepth / platewidth >= 1.0):
                                """ Holding capacity factors for drained soil 
                                    condition. Values from HMGE """                        
                                hcfnqint = interpolate.interp2d(
                                    self._variables.hcfdrsoil[0,1:],self._variables.hcfdrsoil[1:,0],
                                    self._variables.hcfdrsoil[1:,1:])
                                hcfnq = hcfnqint(self.dsfangrad * 180.0 / math.pi, 
                                                 embeddepth / platewidth)
                            
                            if self.soilgroup[j] == 'cohesive':
                                """ Short- and long-term anchor holding
                                    capacity """                        
                                stancholdcap = (platearea * unshstrz 
                                    * strredfactor * hcfncs * (0.84 + 0.16 
                                    * platewidth / plateleng))
                                """ Note: this does not account for very soft 
                                    underconsolidated sediments such as delta 
                                    muds """
                                ltancholdcap = (platearea * (
                                    self.draincoh 
                                    * hcfnc + self.soilweight 
                                    * embeddepth * hcfnq) * (0.84 + 0.16 
                                    * (platewidth / plateleng)))
                                """ Short-term loading capacity is limiting 
                                    case """
                                if ltancholdcap > stancholdcap:
                                    ltancholdcap = stancholdcap
                                if (stancholdcap > padeyedesload[j] 
                                    and ltancholdcap > padeyedesload[j]):
                                    # print "Sufficient short- and long-term holding capacity"
                                    if (plateleng / platewidth >= 1.5 
                                        and plateleng / platewidth <= 2.0):
                                        # print "Length to width ratio acceptable"
                                        """ Plate thickness. Note: No 
                                            structural modelling carried 
                                            out """
                                        platethk = 0.05 * platewidth                                    
                                        platedim = [platewidth, plateleng, 
                                                    platethk, embeddepth]
                                        break
                                else:
                                    # print "Loading capacity not sufficient, increase plate size """
                                    platewidth = platewidth + 1.0
                                    plateleng = 1.75 * platewidth   
                                    
                            elif self.soilgroup[j] == 'cohesionless':                            
                                """ Combined short- and long-term anchor 
                                    holding capacity """  
                                stltancholdcap = (platearea 
                                    * self.soilweight
                                    * embeddepth * hcfnq * (0.84 + 0.16 
                                    * platewidth / plateleng))
                                if (stltancholdcap > padeyedesload[j]):
                                    # print "Sufficient short/long-term holding capacity"
                                    """ Plate thickness. Note: No 
                                        structural modelling carried 
                                        out """
                                    platethk = 0.05 * platewidth                                    
                                    platedim = [platewidth, plateleng, 
                                                platethk, embeddepth]
                                    logmsg = [""]                                    
                                    logmsg.append('Solution found {}'.format(platedim))
                                    module_logger.info("\n".join(logmsg))  
                                    self.founddesfail[j] = 'False'                                    
                                    break
                                else:
                                    # print "Loading capacity not sufficient, increase plate size """
                                    platewidth = platewidth + 1.0
                                    plateleng = 1.75 * platewidth 
                            elif k == klim - 1:
                                """ For when a solution cannot be found """   
                                module_logger.warn('WARNING: Solution not found within set number of iterations!')  
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'
                                platedim = [platewidth, plateleng, platethk, 
                                            embeddepth]
                                if posftyps[0] not in self.unsuitftyps:
                                    self.unsuitftyps.append(posftyps[0])                                   
                        return platedim    
                        
                    self.platedim[j] = directembedanc(self.horzfoundloads, 
                                        self.vertfoundloads)
                    
                    if posftyps[0] in self.unsuitftyps:
                        pass 
                    else:
                        self.platevolsteel[j] = (self.platedim[j][0] 
                                    * self.platedim[j][1] * self.platedim[j][2])
                        self.platevolcon[j] = 0.0
                        self.foundradadd[j] = (self.foundrad[j] + max(
                            0.5 * self.platedim[j][0], 0.5 
                            * self.platedim[j][1]))

                        self.seldeancsubtyp[j] = 'rectangular plate' 
                        self.selfoundsubtypdict[posftyps[0]] = self.seldeancsubtyp
                        self.foundvolgroutdict[j][posftyps[0]] = 0.0             
                        self.selfoundgrouttypdict[j][posftyps[0]] = 'n/a' 
                        self.foundvolsteeldict[j][posftyps[0]] = self.platevolsteel[j]
                        self.foundvolcondict[j][posftyps[0]] = self.platevolcon[j]           
                        self.founddimdict[j][posftyps[0]] = self.platedim[j]
                        self.foundcostdict[j][posftyps[0]] = self._variables.coststeel                    
                 
                elif posftyps[0] == 'drag':                  
                    logmsg = [""]
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    logmsg.append('Drag embedment anchor design')
                    logmsg.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    module_logger.info("\n".join(logmsg))                         
                    """ Holding capacity coefficients used in this routine 
                        include the contribution of the embedded mooring 
                        line """
                    def dragembedanc(horzfoundloads, vertfoundloads):
                        reqmoorconnsize = self.moorconnsize 
                        """ Determine design loads at anchor shackle """
                        horzdesload[j] = (self.moorsf * math.sqrt(
                            self.horzfoundloads[j][self.maxloadindex[j]] ** 2.0 
                            + self.horzfoundloads[j][self.minloadindex[j]] 
                            ** 2.0))
                        vertdesload[j] = self.moorsf * self.vertfoundloads[j] 
                        self.horzdesload[j] = horzdesload[j]
                        self.vertdesload[j] = vertdesload[j]
                        resdesload[j] = math.sqrt(horzdesload[j] ** 2.0 
                                                + vertdesload[j] ** 2.0)                        
                        for k in range(0,klim): 
                            """ Component retrieval """ 
                            if k == 0:
                                """ First search for lowest weight anchor with 
                                compatible shackle size in DB """
                                complim = [0.0, reqmoorconnsize] 
                            ancminweight, anccoef = foundcompret(complim) 
                            if math.fabs(resdesload[j]) == 0.0:
                                module_logger.warn('WARNING: Resultant design load at forerunner equal to zero, setting minimum anchor size')                                                                                           
                            if ancminweight[0] == 0:   
                                module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'   
                                if posftyps[0] not in self.unsuitftyps:
                                    self.unsuitftyps.append(posftyps[0])
                                break   
                            
                            """ Power law method """ 
                            """ Holding capacity coefficients """
                            ancholdcap = (self._variables.gravity * 1.0e3 
                                        * (anccoef[0] 
                                        * (ancminweight[1] / 1.0e3) 
                                        ** anccoef[1]))
                            if ancholdcap < resdesload[j]:
                                """ If anchor holding capacity is too low 
                                    increase minimum anchor weight """
                                complim = [ancminweight[1], reqmoorconnsize]
                            elif ancholdcap >= resdesload[j]:                                                       
                                logmsg = [""]                                    
                                logmsg.append('Solution found {}'.format(ancminweight))
                                module_logger.info("\n".join(logmsg))
                                break   
                            elif k == klim - 1:
                                """ For when a solution cannot be found """ 
                                module_logger.warn('WARNING: Solution not found within set number of iterations!') 
                                self.foundnotfoundflag[posftyps[0]][j] = 'True'                                 
                                if posftyps[0] not in self.unsuitftyps:
                                    self.unsuitftyps.append(posftyps[0])
                        
                        if anccoef[0] != 0:
                            """ Maximum anchor penetration in metres """
                            if (anccoef[2] > 0.0 and anccoef[3] > 0.0):
                                ancpen = (anccoef[2] * (ancminweight[1] / 1.0e3) 
                                        ** anccoef[3])
                                """ Maximum penetration depth is soil layer depth """
                                if ancpen > self.soildep[j]:
                                    module_logger.warn('WARNING: Full anchor penetration not possible!')   
                                    ancpen = self.soildep[j]                                
                            else: 
                                module_logger.warn('WARNING: Penetration coefficients not available for anchor!')  
                                ancpen = 'n/k'
                        else:
                            ancpen = 'n/a'
                        return ancminweight[0], ancpen
                        
                    self.seldraganc[j], self.ancpendep[j] = dragembedanc(
                                    self.horzfoundloads, self.vertfoundloads)
                                         
                    if self.seldraganc[j] != 0:                     
                        """ Flag if forerunner assembly needs to be altered to 
                            accomodate large anchor shackle size """
                        if (self._variables.compdict[self.seldraganc[j]]['item6'][3] 
                            > self.moorconnsize):
                            self.ancconnsize = self._variables.compdict[self.seldraganc[j]]['item6'][3] 
                    if posftyps[0] in self.unsuitftyps:
                        module_logger.warn('WARNING: Solution not found, foundation type unsuitable!')
                        self.foundnotfoundflag[posftyps[0]][j] = 'True'
                        pass 
                    else:                        
                        self.ancdim[j] = self._variables.compdict[self.seldraganc[j]]['item6'][0:3]
                        self.foundradadd[j] = (self.foundrad[j] 
                            + self._variables.compdict[self.seldraganc[j]]['item6'][2])
        
                        self.seldragancsubtyp[j] = self._variables.compdict[self.seldraganc[j]]['item3']
                        self.selfoundsubtypdict[posftyps[0]] = self.seldragancsubtyp 
                        self.foundvolgroutdict[j][posftyps[0]] = 0.0             
                        self.selfoundgrouttypdict[j][posftyps[0]] = 'n/a' 
                        self.foundvolsteeldict[j][posftyps[0]] = 'n/a' 
                        self.foundvolcondict[j][posftyps[0]] = 0.0     
                        self.founddimdict[j][posftyps[0]] = self.ancdim[j]
                        self.ancitemcost[j] = self._variables.compdict[self.seldraganc[j]]['item11']
                        self.foundcostdict[j][posftyps[0]] = self.ancitemcost[j]
                        self.founddesfail[j] = 'False'
            
            """ Remove unsuitable foundation entries from possible foundation 
                type list and associated dictionaries """            
            
            for unsuitftyps in self.unsuitftyps:    
                if unsuitftyps in self.foundvolgroutdict[j]:
                    del self.foundvolgroutdict[j][unsuitftyps]
                if unsuitftyps in self.selfoundgrouttypdict[j]:
                    del self.selfoundgrouttypdict[j][unsuitftyps]
                if unsuitftyps in self.foundvolsteeldict[j]:
                    del self.foundvolsteeldict[j][unsuitftyps]
                if unsuitftyps in self.foundvolcondict[j]:
                    del self.foundvolcondict[j][unsuitftyps]
                if unsuitftyps in self.founddimdict[j]:
                    del self.founddimdict[j][unsuitftyps]
                if unsuitftyps in self.foundcostdict[j]:
                    del self.foundcostdict[j][unsuitftyps]
                self.possfoundtyp[j] = [(name, score) for name, score in self.possfoundtyp[j] if name != unsuitftyps]

            """ Calculate approximate foundation footprint area and volume of 
                configuration """
            if systype in ('wavefloat', 'wavefixed', 'tidefloat', 'tidefixed'):
                self.devconfigfootarea = (math.pi / 4.0) * max(self.foundrad) ** 2.0
                self.devconfigvol = ((math.pi / 4.0) * max(self.foundrad) ** 2.0 
                                        * self.bathysysorig) 
            elif systype in ('substation'):
                self.substconfigfootarea = (math.pi / 4.0) * max(self.foundrad) ** 2.0
                self.substconfigvol = ((math.pi / 4.0) * max(self.foundrad) ** 2.0 
                                        * self.bathysysorig)
        if self.piledim:
            logmsg.append('self.piledim {}'.format(self.piledim))
        if self.pileyld:
            logmsg.append('self.pileyld {}'.format(self.pileyld))
            if 'other' in self.soilgroup:
                logmsg.append('self.groutpilebondstr {}'.format(self.groutpilebondstr))
        module_logger.info("\n".join(logmsg))
                                        
    def foundcost(self):
        """ Foundation capital cost calculations based on number of foundation 
            points and sizes """
        self.totfoundcostdict = {}
        self.pilecost = [0 for row in range(self.quanfound)] 
        self.anccost = [0 for row in range(self.quanfound)] 
        self.platecost = [0 for row in range(self.quanfound)] 
        self.gravcost = [0 for row in range(self.quanfound)] 
        self.caisscost = [0 for row in range(self.quanfound)] 
        self.shallcost = [0 for row in range(self.quanfound)] 
        self.sorttotfoundcost = [('dummy',1.0e10) for row 
                                                in range(self.quanfound)] 
        self.selfoundtyp = [0 for row in range(self.quanfound)]
        
        for j in range(0,self.quanfound):  
            """ Fabrication cost factor included for all foundation types apart from drag anchors """
            # logmsg = [""]
            # logmsg.append('------------------------------------------------------------------------')
            # logmsg.append('Foundation ' + str(j) + ': Identified suitable foundation type(s) {}'.format(self.possfoundtyp[j]))
            # logmsg.append('------------------------------------------------------------------------')
            # module_logger.info("\n".join(logmsg))
                        
            for posftyps in self.possfoundtyp[j]:
                if posftyps[0] == 'shallowfoundation': 
                    self.shallcost[j] = (self.foundcostdict[j][posftyps[0]][1] 
                        * self._variables.conden 
                        * self.foundvolcondict[j][posftyps[0]] 
                        + self.foundcostdict[j][posftyps[0]][0] 
                        * self._variables.steelden 
                        * self.foundvolsteeldict[j][posftyps[0]]) 
                    if self._variables.fabcost:
                        self.shallcost[j] =  self.shallcost[j] * (1.0 + self._variables.fabcost)                        
                    self.totfoundcostdict[posftyps[0]] = self.shallcost 
                elif posftyps[0] == 'gravity': 
                    self.gravcost[j] = (self.foundcostdict[j][posftyps[0]][1] 
                    * self._variables.conden 
                    * self.foundvolcondict[j][posftyps[0]] 
                    + self.foundcostdict[j][posftyps[0]][0] 
                    * self._variables.steelden 
                    * self.foundvolsteeldict[j][posftyps[0]]) 
                    if self._variables.fabcost:
                        self.gravcost[j] = self.gravcost[j] * (1.0 + self._variables.fabcost)                    
                    self.totfoundcostdict[posftyps[0]] = self.gravcost
                elif posftyps[0] == 'pile': 
                    self.pilecost[j] = (self.foundcostdict[j][posftyps[0]] 
                        * self.piledim[j][2] 
                        + self._variables.costgrout * self._variables.groutden 
                        * self.foundvolgroutdict[j][posftyps[0]])
                    if self._variables.fabcost:
                        self.pilecost[j] = self.pilecost[j] * (1.0 + self._variables.fabcost)                    
                    self.totfoundcostdict[posftyps[0]] = self.pilecost                     
                elif posftyps[0] == 'suctioncaisson':                 
                    self.caisscost[j] =  (self.foundcostdict[j][posftyps[0]] 
                                           * self.caissdim[j][2])
                    if self._variables.fabcost:
                        self.caisscost[j] = self.caisscost[j] * (1.0 + self._variables.fabcost)                    
                    self.totfoundcostdict[posftyps[0]] = self.caisscost   
                elif posftyps[0] == 'directembedment': 
                    self.platecost[j] =  (self.foundcostdict[j][posftyps[0]]
                        * self._variables.steelden 
                        * self.foundvolsteeldict[j][posftyps[0]])
                    if self._variables.fabcost:
                        self.platecost[j] = self.platecost[j] * (1.0 + self._variables.fabcost)                    
                    self.totfoundcostdict[posftyps[0]] = self.platecost                        
                elif posftyps[0] == 'drag':                    
                    self.anccost[j] = self._variables.compdict[
                                                self.seldraganc[j]]['item11']
                    self.totfoundcostdict[posftyps[0]] = self.anccost
                  
        for j in range(0,self.quanfound):            
            if self.possfoundtyp[j]:
                for posftyps in self.possfoundtyp[j]:
                    if self.possfoundtyp[j] != 0:                    
                        if (self.foundnotreqflag[posftyps[0]][j] == 'False' 
                            and self.foundnotfoundflag[posftyps[0]][j] == 'False'):                            
                            if (self.totfoundcostdict[posftyps[0]][j]
                                < self.sorttotfoundcost[j][1]):
                                self.sorttotfoundcost[j] = (posftyps[0], 
                                                    self.totfoundcostdict[posftyps[0]][j])  
                            """ Select lowest capital cost solution which is suitable """        
                            self.selfoundtyp[j] = self.sorttotfoundcost[j]
                        else:
                            if (self.foundnotreqflag[posftyps[0]][j] == 'True' 
                                and self._variables.systype in ('wavefixed', 'tidefixed')):
                                self.selfoundtyp[j] = ('Foundation not required', 0)    
            else:
                self.selfoundtyp[j] = ('Foundation solution not found', 0)
                                
                  

    def foundbom(self, deviceid):        
        """ Create foundation system bill of materials for the RAM and logistic functions """             
        self.foundecoparams = [0 for row in range(0,self.quanfound)]
        self.uniqfoundcomp = [0 for row in range(0,self.quanfound)]
        self.netlistuniqfoundcomp = [0 for row in range(0, self.quanfound)]
        foundmarkerlist = [0 for row in range(0, self.quanfound)] 
        self.foundrambomdict = {}         
        tabind = range(0,self.quanfound) 
        self.listfoundcomp = []        
        logmsg = [""]              
        logmsg.append('self.possfoundtyp {}'.format(self.possfoundtyp))            
        logmsg.append('self.selfoundtyp {}'.format(self.selfoundtyp))            
        module_logger.info("\n".join(logmsg))  
        for j in range(0, self.quanfound):            
            if (self.selfoundtyp[j][0] not in ('Foundation not required', 'Foundation solution not found')):
                if self.selfoundtyp[j][0] == 'pile':
                    self.listfoundcomp.append(self.piledim[j][4])
                    if self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]] == 'n/a':
                        pass 
                    else:
                        self.listfoundcomp.append(self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]])
                elif self.selfoundtyp[j][0] == 'suctioncaisson':
                    self.listfoundcomp.append(self.caissdim[j][3])
                else: self.listfoundcomp.append(self.selfoundtyp[j][0]) 
            else:
                self.listfoundcomp.append(self.selfoundtyp[j][0])
                
        self.quanfoundcomp = Counter(self.listfoundcomp)
        for j in range(0, self.quanfound):
            listuniqfoundcomp = []
            complabel = self.selfoundtyp[j][0] 
            if (self.selfoundtyp[j][0] not in ('Foundation not required', 'Foundation solution not found')):    
                """ Create RAM BOM """                
                self.uniqfoundcomp[j] = '{0:04}'.format(self.netuniqcompind)
                self.netuniqcompind = self.netuniqcompind + 1  
                devind = int(float(deviceid[-3:]))                
                if self.selfoundtyp[j][0] == 'drag': 
                    compid = self.seldraganc[j]
                elif self.selfoundtyp[j][0] == 'pile':
                    compid = self.piledim[j][4]
                elif self.selfoundtyp[j][0] == 'suctioncaisson':
                    compid = self.caissdim[j][3]
                else:
                    compid = 'n/a'
                self.foundecoparams[j] = [compid, 
                                          1.0,
                                          self.sorttotfoundcost[j][1],
                                          self.projectyear]
                listuniqfoundcomp.append(self.netuniqcompind)               
                
                self.netlistuniqfoundcomp[j] = self.uniqfoundcomp[j]
                if self.selfoundtyp[j][0] == 'pile':
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:]),int(self.netlistuniqfoundcomp[j][-4:])+1]
                    self.netuniqcompind = self.netuniqcompind + 1      
                else:
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:])]
            else:
                self.uniqfoundcomp[j] = 'n/a'
                self.foundecoparams[j] = ['n/a',
                                          'n/a',
                                          0.0,
                                          'n/a']
                self.netlistuniqfoundcomp[j] = []
                foundmarkerlist[j] = []
            
        self.foundrambomdict['marker'] = foundmarkerlist
        self.foundrambomdict['quantity'] = self.quanfoundcomp
        if self.possfoundtyp:        
            """ Create economics BOM """
            self.foundecobomtab = pd.DataFrame(self.foundecoparams, 
                                             index=tabind, 
                                             columns=['compid [-]',
                                                      'quantity [-]',
                                                      'component cost [euros] [-]',
                                                      'project year'])                         
        
    def foundhierarchy(self):    
        """Create foundation system hierarchy """
        self.foundhier = [0 for row in range(self.quanfound)]
        for j in range(0, self.quanfound):
            if (self.selfoundtyp[j][0] == 'Foundation not required' or self.selfoundtyp[j][0] == 'Foundation solution not found'):
                self.foundhier[j] = ['n/a']
            else:
                if self.selfoundtyp[j][0] == 'pile':
                    if self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]] == 'n/a':
                        self.foundhier[j] = [self.piledim[j][4]]
                    else:    
                        self.foundhier[j] = [self.piledim[j][4], self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]]]
                elif self.selfoundtyp[j][0] == 'suctioncaisson':
                    self.foundhier[j] = [self.caissdim[j][3]]
                elif self.selfoundtyp[j][0] == 'drag':
                    self.foundhier[j] = [self.seldraganc[j]]
                else: self.foundhier[j] = [self.selfoundtyp[j][0]]
    
    def foundinst(self,deviceid,systype,sysorig,foundloc,sysorienang):
        """ Foundation installation calculations """
        tabind = []
        self.listfoundcomp = []
        self.foundinstdep = [0 for row in range(self.quanfound)]
        self.foundweight = [0 for row in range(self.quanfound)]
        self.foundinstparams = [0 for row in range(self.quanfound)]
        self.layerlist = [0 for row in range(self.quanfound)]
        foundposglob = [[0 for col in range(0,3)] for row in range(self.quanfound)]
       
        foundloc = self.foundloc
        for j in range(0,self.quanfound):            
            """ Pandas table index """
            if systype in ("wavefloat","tidefloat","wavefixed","tidefixed"): 
                devind = int(float(deviceid[-3:]))
                tabind.append(j + (devind - 1) * self.quanfound)
            elif systype == 'substation':
                tabind.append(len(self.sysfoundinsttab.index))            
            if self.selfoundtyp[j][0] == 'Foundation not required':
                self.foundinstparams[j] = [deviceid, 
                    'foundation' + '{0:03}'.format(self.foundnum), 
                    'Foundation not required', 
                    'n/a',
                    'n/a',
                    'n/a', 
                    'n/a', 
                    'n/a',
                    'n/a', 
                    'n/a', 
                    'n/a', 
                    'n/a',
                    'n/a',
                    'n/a', 
                    'n/a', 
                    'n/a'] 
                
            elif self.selfoundtyp[j][0] == 'Foundation solution not found':
                self.foundinstparams[j] = [deviceid, 
                    'foundation' + '{0:03}'.format(self.foundnum), 
                    'Foundation solution not found', 
                    'n/a', 
                    'n/a',
                    'n/a',
                    'n/a', 
                    'n/a',
                    'n/a', 
                    'n/a', 
                    'n/a', 
                    'n/a',
                    'n/a',
                    'n/a', 
                    'n/a', 
                    'n/a']
                
            else:
                laylst = []
                for layer in range(0,(len(self._variables.soiltypgrid[0]) - 2) / 2):
                    laylst.append((layer, self.soiltyp[j], self.soildep[j]))
                self.layerlist[j] = laylst                
                
                if self.selfoundtyp[j][0] == 'drag':
                    self.foundinstdep[j] = self.ancpendep[j] 
                    self.foundweight[j] = self._variables.compdict[
                                            self.seldraganc[j]]['item7'][0]
                elif self.selfoundtyp[j][0] == 'pile':
                    if (systype in ("wavefloat","tidefloat") 
                        and self.soilgroup[j] in ('cohesive', 'cohesionless')):
                        """ Note: Pile head installed 2x pile diameter below seafloor """
                        self.foundinstdep[j] = (self.piledim[j][2] + 2.0 
                                                * self.piledim[j][0])
                    elif (systype in ("wavefixed","tidefixed",'substation') 
                                                or self.soilgroup[j] == 'other'):
                        """ Note: Assumes piles are installed to full depth (i.e. seafloor mounted system) """
                        self.foundinstdep[j] = self.piledim[j][2]
                    self.foundweight[j] = (self.foundvolsteeldict[j][
                        self.selfoundtyp[j][0]] * self._variables.steelden 
                        + self.foundvolgroutdict[j][self.selfoundtyp[j][0]]
                        * self._variables.groutden)
                elif self.selfoundtyp[j][0] == 'suctioncaisson':               
                    self.foundinstdep[j] = self.caissdim[j][2]
                    self.foundweight[j] = (self.foundvolsteeldict[j][
                                    self.selfoundtyp[j][0]] 
                                    * self._variables.steelden)
                elif self.selfoundtyp[j][0] == 'directembedment':
                    self.foundinstdep[j] = self.platedim[j][3]
                    self.foundweight[j] = (self.foundvolsteeldict[j][
                                    self.selfoundtyp[j][0]] 
                                    * self._variables.steelden)
                elif self.selfoundtyp[j][0] in ('shallowfoundation', 'gravity'):
                    self.foundinstdep[j] = 'n/a'
                    self.foundweight[j] = (self.foundvolsteeldict[j][
                        self.selfoundtyp[j][0]] * self._variables.steelden 
                        + self.foundvolcondict[j][self.selfoundtyp[j][0]] 
                        * self._variables.conden)
                     
                if math.fabs(sysorienang) > 0.0:
                    self.foundlocglob[j] = [foundloc[j][0] * math.cos(-sysorienang 
                                            * math.pi / 180.0) - foundloc[j][1] 
                                            * math.sin(-sysorienang * math.pi 
                                            / 180.0), foundloc[j][0] 
                                            * math.sin(-sysorienang * math.pi 
                                            / 180.0) + foundloc[j][1] * math.cos(
                                            -sysorienang * math.pi / 180.0), foundloc[j][2]]
                else: 
                    self.foundlocglob[j] = foundloc[j]
                if deviceid[0:6] == 'device':
                    foundposglob[j][0] =  self.foundlocglob[j][0] + sysorig[deviceid][0]
                    foundposglob[j][1] =  self.foundlocglob[j][1] + sysorig[deviceid][1]
                    foundposglob[j][2] =  self.foundlocglob[j][2]
                else:
                    foundposglob[j][0] =  self.foundlocglob[j][0] + sysorig[0]
                    foundposglob[j][1] =  self.foundlocglob[j][1] + sysorig[1]
                    foundposglob[j][2] =  self.foundlocglob[j][2] 
                if self.selfoundtyp[j][0] in ('pile', 'suctioncaisson'):
                    self.foundinstparams[j] = [deviceid, 
                         'foundation' + '{0:03}'.format(self.foundnum), 
                         self.selfoundtyp[j][0], 
                         self.selfoundsubtypdict[self.selfoundtyp[j][0]][j],
                         self.netlistuniqfoundcomp[j],
                         foundposglob[j][0],
                         foundposglob[j][1],
                         foundposglob[j][2],
                         self.founddimdict[j][self.selfoundtyp[j][0]][0], 
                         self.founddimdict[j][self.selfoundtyp[j][0]][0], 
                         self.founddimdict[j][self.selfoundtyp[j][0]][2], 
                         self.foundinstdep[j], 
                         self.layerlist[j],
                         self.foundweight[j], 
                         self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]], 
                         self.foundvolgroutdict[j][self.selfoundtyp[j][0]]]
                else: 
                    self.foundinstparams[j] = [deviceid, 
                    'foundation' + '{0:03}'.format(self.foundnum), 
                    self.selfoundtyp[j][0], 
                    self.selfoundsubtypdict[self.selfoundtyp[j][0]][j], 
                    self.netlistuniqfoundcomp[j],
                    foundposglob[j][0],
                    foundposglob[j][1],
                    foundposglob[j][2],
                    self.founddimdict[j][self.selfoundtyp[j][0]][1], 
                    self.founddimdict[j][self.selfoundtyp[j][0]][0], 
                    self.founddimdict[j][self.selfoundtyp[j][0]][2], 
                    self.foundinstdep[j],
                    self.layerlist[j],
                    self.foundweight[j], 
                    self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]], 
                    self.foundvolgroutdict[j][self.selfoundtyp[j][0]]] 
            self.foundnum = self.foundnum + 1
        if (self.founduniformflag == 'True' and systype in ("wavefloat", "tidefloat","wavefixed","tidefixed")):
            """ Each device to have the same size/weight foundation (largest) """
            self.fwmaxinddev = self.foundweight.index(max(
                                        self.foundweight))
            
            """ Update installation parameters, RAM hierarchy, RAM bill of materials and economics tables"""
            for j in range(0,self.quanfound): 
                if self.selfoundtyp[self.fwmaxinddev][0] in ('pile', 'suctioncaisson'):
                        self.foundinstparams[j] = [deviceid, 
                             'foundation' + '{0:03}'.format(j), 
                             self.selfoundtyp[self.fwmaxinddev][0], 
                             self.selfoundsubtypdict[self.selfoundtyp[self.fwmaxinddev][0]][self.fwmaxinddev],
                             self.netlistuniqfoundcomp[j],
                             foundposglob[j][0],
                             foundposglob[j][1],
                             foundposglob[j][2],
                             self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][0], 
                             self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][0], 
                             self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][2], 
                             self.foundinstdep[self.fwmaxinddev], 
                             self.layerlist[j],
                             self.foundweight[self.fwmaxinddev], 
                             self.selfoundgrouttypdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]], 
                             self.foundvolgroutdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]]]
                else: 
                    self.foundinstparams[j] = [deviceid, 
                    'foundation' + '{0:03}'.format(j), 
                    self.selfoundtyp[self.fwmaxinddev][0], 
                    self.selfoundsubtypdict[self.selfoundtyp[self.fwmaxinddev][0]][self.fwmaxinddev], 
                    self.netlistuniqfoundcomp[j],
                    foundposglob[j][0],
                    foundposglob[j][1],
                    foundposglob[j][2],
                    self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][1], 
                    self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][0], 
                    self.founddimdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]][2], 
                    self.foundinstdep[self.fwmaxinddev],
                    self.layerlist[j],
                    self.foundweight[self.fwmaxinddev], 
                    self.selfoundgrouttypdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]], 
                    self.foundvolgroutdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]]] 
                """ Update hierarchy """    
                self.foundhier[j] = self.foundhier[self.fwmaxinddev]
                
                """ Update RAM bill of materials table """
                self.listfoundcomp = []
                if self.selfoundtyp[self.fwmaxinddev][0] == 'pile':
                    self.listfoundcomp.append(self.piledim[self.fwmaxinddev][4])
                    if self.selfoundgrouttypdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]] == 'n/a':
                        pass 
                    else:
                        self.listfoundcomp.append(self.selfoundgrouttypdict[self.fwmaxinddev][self.selfoundtyp[self.fwmaxinddev][0]])
                elif self.selfoundtyp[self.fwmaxinddev][0] == 'suctioncaisson':
                    self.listfoundcomp.append(self.caissdim[self.fwmaxinddev][3])
                else: self.listfoundcomp.append(self.selfoundtyp[self.fwmaxinddev][0])
                self.quanfoundcomp = Counter(self.listfoundcomp)
                self.foundrambomdict['quantity'] = self.quanfoundcomp
                
                """ Update economics table """
                if self.selfoundtyp[self.fwmaxinddev][0] == 'drag': 
                    compid = self.seldraganc[j]
                elif self.selfoundtyp[self.fwmaxinddev][0] == 'pile':
                    compid = self.piledim[j][4]
                elif self.selfoundtyp[self.fwmaxinddev][0] == 'suctioncaisson':
                    compid = self.caissdim[j][3]
                else:
                    compid = 'n/a'
                self.foundecoparams[j] = [compid, 
                                          1.0,
                                          self.sorttotfoundcost[self.fwmaxinddev][1],
                                          self.projectyear]  
                self.foundecobomtab = pd.DataFrame(self.foundecoparams, 
                                                   index=tabind, 
                                                   columns=['compid [-]', 
                                                            'quantity [-]',
                                                            'component cost [euros] [-]',
                                                            'project year']) 
        
        self.foundinsttab = pd.DataFrame(self.foundinstparams, 
                                         index=tabind, 
                                         columns=['devices [-]', 
                                         'foundations [-]', 
                                         'type [-]', 
                                         'subtype [-]',
                                         'marker [-]',
                                         'x coord [m]', 
                                         'y coord [m]',
                                         'bathymetry at MSL [m]',
                                         'length [m]', 
                                         'width [m]', 
                                         'height [m]', 
                                         'installation depth [m]',
                                         'layer information (layer number, soil type, soil depth) [-,-,m]',
                                         'dry mass [kg]', 
                                         'grout type [-]', 
                                         'grout volume [m3]'])       
        