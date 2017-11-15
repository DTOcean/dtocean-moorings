"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter
"""


# Built in modulesrad
import math
import logging
from collections import Counter

# External module import
import numpy as np
import pandas as pd

# Start logging
module_logger = logging.getLogger(__name__)


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
        super(Umb, self).__init__(variables)
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
                    logmsg = ('Umbilical converged, max tension: '
                              '{}').format([max(Tumb), umbleng])
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
                
                logmsg = ('Umbilical upper end loads and horizontal angle '
                          '{}').format([HumbloadX,
                                        HumbloadY,
                                        Vumbload,
                                        Humbloadang])
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
        