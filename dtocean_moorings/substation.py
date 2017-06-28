"""
Release Version of the DTOcean: Moorings and Foundations module: 17/10/16
Developed by: Renewable Energy Research Group, University of Exeter
"""

# Built in modulesrad
import logging
from collections import Counter

# External module import
import pandas as pd

# Local imports
from .loads import Loads

# Start logging
module_logger = logging.getLogger(__name__)


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

        