# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Sam Weller, Jon Hardwick
#    Copyright (C) 2017-2018 Mathew Topper
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
.. moduleauthor:: Sam Weller <s.weller@exeter.ac.uk>
.. moduleauthor:: Jon Hardwick <j.p.hardwick@exeter.ac.uk>
.. moduleauthor:: Mathew Topper <mathew.topper@dataonlygreater.com>
"""

# Built in modules
import copy
import math
import logging
import operator
from collections import Counter

# External module import
import numpy as np
import pandas as pd
from scipy import interpolate, optimize

# Local imports
from .loads import Loads
from .moorings import Moor

# Start logging
module_logger = logging.getLogger(__name__)

        
class Found(Moor, Loads):
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
        self.groutpilebondstr = None
        self.prefound = None
        self.selfoundtyp = None
        
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
                          self.foundloc[:,0]],
                         dtype='float')
            b = np.array([math.fabs(self.totsysstatloads[2]),
                          self.sysbasemom[1],self.sysbasemom[0]],
                         dtype='float')
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
            
            # Collect local coords
            gpnear = self.gpnear[j]
                
            self.deltabathyy = (np.mean([gpnear[2][2], gpnear[3][2]])  
                                    - np.mean([gpnear[0][2], gpnear[1][2]]))
            self.deltabathyx = (np.mean([gpnear[1][2], gpnear[2][2]]) 
                                    - np.mean([gpnear[0][2], gpnear[3][2]]))
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
        
    def foundsel(self, systype): 
        """ Select suitable foundation types """
        
        self.possfoundtyp = [0 for row in range(self.quanfound)]
        self.foundtyps = {'shallowfoundation', 
                          'gravity', 
                          'pile', 
                          'suctioncaisson', 
                          'directembedment', 
                          'drag'}
        
        self.founduniformflag = 'False'
        
        # Check for preferred foundation type or uniary design
        if self._variables.prefound is not None: 
            
            if self._variables.prefound[0:3] == 'uni':
                
                self.founduniformflag = 'True'
                self.prefound = self._variables.prefound[6:]
                
                msgStr = ('Uniary foundation design requested')
                module_logger.info(msgStr)
                
            else:
                
                self.prefound = self._variables.prefound
                
            if not self.prefound:
                
                self.prefound = None
                
            else:
            
                msgStr = ('Prefered foundation type {} '
                          'selected').format(self.prefound)
                module_logger.info(msgStr)
        
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
            if (self.soildepk == 'shallow' 
                and self.soiltyp[j] in ('vsc','sc') 
                and systype in ("wavefloat","tidefloat")):
                self.foundsoildepdict.get('shallow')['drag'] = 88
            elif (self.soildepk == 'moderate' 
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
            
            # Remove unsuitable foundations with scores above maximum 
            # permissible limit of 10 unless its the preferred foundation
            for posftyps in self.foundmatsumdict: 
                
                if self.prefound is not None and posftyps == self.prefound:
                    continue
                
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
            
        return
                
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
                            ancweights = [(0, 0)]
                    
                    ancminweight = min(ancweights, key=operator.itemgetter(1))
                    
                    if ancminweight[0] == 0:
                        return ancminweight, [0.0, 0.0]
                    
                    item9 = self._variables.compdict[ancminweight[0]]['item9']
                        
                    if self.soiltyp[j] in ('vsc', 'sc'):
                        # Coefficients for soft clays and muds
                        anccoef = item9['soft']
                    elif self.soiltyp[j] in ('fc', 'stc', 'ls', 'ms', 'ds'):
                        # Coefficients for stiff clays and sands
                        anccoef = item9['sand']
                    else:
                        warnMsg = ("Drag anchors are not applicable to "
                                   "soil type: {}").format(self.soiltyp[j])
                        module_logger.warning(warnMsg)
                        
                        return ancminweight, [0.0, 0.0]
                    
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
                                                    fill_value="extrapolate")
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
                        # Sliding resistance on hard surfaces.
                        maxloadindex = self.maxloadindex[j]
                        slope_angle = self.seabedslp[j][maxloadindex]
                        horz_load = self.horzfoundloads[j][maxloadindex]
                        resistance = self._get_sliding_resistance(slope_angle,
                                                                  horz_load)
                        
                        # Abort if no resistance found
                        if not resistance:
                            
                            self.foundnotfoundflag[posftyps[0]][j] = 'True'
                            basedim = [np.nan, np.nan, 'n/a', 
                                       0.0, 0.0, 0, 
                                       0.0, np.nan, np.nan]                  
                            if posftyps[0] not in self.unsuitftyps:
                                self.unsuitftyps.append(posftyps[0])  
                                
                            return basedim
                        
                        reqfoundweight[j] = resistance + self.vertfoundloads[j]
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
                        logmsg = 'Solution found {}'.format(basedim)
                        module_logger.info(logmsg)
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
                                logmsg = ('!!! Solution not found, foundation '
                                          'type unsuitable !!!')
                                module_logger.info(logmsg)
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
                                                fill_value="extrapolate")
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
                                            fill_value="extrapolate")
                                        piledefcoefbyint = interpolate.interp1d(
                                            self._variables.piledefcoef[:,0], 
                                            self._variables.piledefcoef[:,2],
                                            fill_value="extrapolate")
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
                                        fill_value="extrapolate")
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
                                                fill_value="extrapolate")
                                            bcflim = bcflimint(self.dsfangrad 
                                                * 180.0 / math.pi)                                
                                            """ Unit soil bearing capacity 
                                                limiting value """
                                            soilbearcaplimint = interpolate.interp1d(
                                                self._variables.pilefricresnoncal[:,0], 
                                                self._variables.pilefricresnoncal[:,4],
                                                fill_value="extrapolate")
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
                                    logmsg = 'Solution found {}'.format(piledim)
                                    module_logger.info(logmsg)
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
                                logmsg = 'Solution found {}'.format(caissdim)
                                module_logger.info(logmsg)
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
                                    logmsg = 'Solution found {}'.format(platedim)
                                    module_logger.info(logmsg)
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
                                logmsg = 'Solution found {}'.format(ancminweight)
                                module_logger.info(logmsg)
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
        
        logmsg = []
        if self.piledim:
            logmsg.append('self.piledim: {}'.format(self.piledim))
        if self.pileyld:
            logmsg.append('self.pileyld: {}'.format(self.pileyld))
            if 'other' in self.soilgroup and self.groutpilebondstr is not None:
                logmsg.append('self.groutpilebondstr {}'.format(
                                                        self.groutpilebondstr))
        module_logger.debug("\n".join(logmsg))
                                        
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
                        
                        if (self.foundnotreqflag[posftyps[0]][j] == 'False' and
                            self.foundnotfoundflag[posftyps[0]][j] == 'False'):
                            
                            if (self.prefound is not None and
                                posftyps[0] == self.prefound):
                                
                                self.sorttotfoundcost[j] = (
                                        posftyps[0], 
                                        self.totfoundcostdict[posftyps[0]][j])
                                
                                self.selfoundtyp[j] = self.sorttotfoundcost[j]
                                
                                break
                            
                            elif (self.totfoundcostdict[posftyps[0]][j]
                                                < self.sorttotfoundcost[j][1]):
                                
                                self.sorttotfoundcost[j] = (
                                        posftyps[0], 
                                        self.totfoundcostdict[posftyps[0]][j])
                                
                            # Select lowest capital cost solution which is
                            # suitable   
                            self.selfoundtyp[j] = self.sorttotfoundcost[j]
                            
                        elif (self._variables.systype in ('wavefixed',
                                                          'tidefixed') and 
                              self.foundnotreqflag[posftyps[0]][j] == 'True'):
                                    
                                self.selfoundtyp[j] = \
                                                ('Foundation not required', 0) 
                                                
                                if (self.prefound is not None and
                                    posftyps[0] == self.prefound): break
            
            else:
                
                self.selfoundtyp[j] = ('Foundation solution not found', 0)
              
        return

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
        module_logger.debug(" ".join(logmsg))
        for j in range(0, self.quanfound):            
            if self.selfoundtyp[j][0] != 'Foundation solution not found': #not in ('Foundation not required', 'Foundation solution not found'):
                if self.selfoundtyp[j][0] == 'pile':
                    self.listfoundcomp.append(self.piledim[j][4])
                    if self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]] == 'n/a':
                        pass 
                    else:
                        self.listfoundcomp.append(self.selfoundgrouttypdict[j][self.selfoundtyp[j][0]])
                elif self.selfoundtyp[j][0] == 'suctioncaisson':
                    self.listfoundcomp.append(self.caissdim[j][3])
                elif self.selfoundtyp[j][0] == 'drag':
                    self.listfoundcomp.append(self.seldraganc[j])
                else:
                    self.listfoundcomp.append(self.selfoundtyp[j][0]) 
            else:
                self.listfoundcomp.append(self.selfoundtyp[j][0])
                
        self.quanfoundcomp = Counter(self.listfoundcomp)
        
        for j in range(0, self.quanfound):
            listuniqfoundcomp = []            
            if self.selfoundtyp[j][0] != 'Foundation solution not found': #not in ('Foundation not required', 'Foundation solution not found'):    
                # Create markers              
                self.uniqfoundcomp[j] = '{0:04}'.format(self.netuniqcompind)
                self.netuniqcompind = self.netuniqcompind + 1  
                listuniqfoundcomp.append(self.netuniqcompind)               
                
                self.netlistuniqfoundcomp[j] = self.uniqfoundcomp[j]
                if self.selfoundtyp[j][0] == 'pile':
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:]),int(self.netlistuniqfoundcomp[j][-4:])+1]
                    self.netuniqcompind = self.netuniqcompind + 1      
                else:
                    foundmarkerlist[j] = [int(self.netlistuniqfoundcomp[j][-4:])]
            else:
                self.uniqfoundcomp[j] = 'n/a'
                self.netlistuniqfoundcomp[j] = []
                foundmarkerlist[j] = []
            
            
            if self.selfoundtyp[j][0] not in ('Foundation not required',
                                              'Foundation solution not found'):    
                # Create RAM BOM              
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
                
            else:
                self.foundecoparams[j] = ['n/a',
                                          'n/a',
                                          0.0,
                                          'n/a']
            
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
                
            laylst = []
            
            for layer in range(0,(len(self._variables.soiltypgrid[0]) - 2) / 2):
                laylst.append((layer, self.soiltyp[j], self.soildep[j]))
            self.layerlist[j] = laylst      
            
            if deviceid[0:6] == 'device':
                foundposglob[j][0] =  self.foundlocglob[j][0] + sysorig[deviceid][0]
                foundposglob[j][1] =  self.foundlocglob[j][1] + sysorig[deviceid][1]
                foundposglob[j][2] =  self.foundlocglob[j][2]
            else:
                foundposglob[j][0] =  self.foundlocglob[j][0] + sysorig[0]
                foundposglob[j][1] =  self.foundlocglob[j][1] + sysorig[1]
                foundposglob[j][2] =  self.foundlocglob[j][2]
                
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
        
        active_founds = set([x[0] for x in self.selfoundtyp
                                     if x[0] != 'Foundation not required'])
        
        if (self.founduniformflag == 'True' and len(active_founds) == 1):
            """ Each device to have the same size/weight foundation (largest) """
            self.fwmaxinddev = self.foundweight.index(max(
                                        self.foundweight))
            
            self.foundinstparams = [0 for row in range(self.quanfound)]
            self.foundecoparams = [0 for row in range(0,self.quanfound)]
            
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
                elif self.selfoundtyp[self.fwmaxinddev][0] == 'drag':
                    self.listfoundcomp.append(self.seldraganc[self.fwmaxinddev])
                else:
                    self.listfoundcomp.append(self.selfoundtyp[self.fwmaxinddev][0])
                    
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
        
    def _get_sliding_resistance(self, slope_angle,
                                      horizontal_load):
        
        """Sliding resistance on hard surfaces.
        Note: this formulation needs to be checked
        """
        
        numerator = 1 + self.soilfric * math.tan(slope_angle)
        denominator = self.soilfric - math.tan(slope_angle)              
        
        if denominator <= 0.:
            
            angle = math.degrees(slope_angle)
            
            warnStr = ("Slope angle {} exceeds limiting friction "
                       "angle").format(angle)
            module_logger.warning(warnStr)
            
            return False
        
        horizontal_required = math.fabs(horizontal_load) * \
                                           numerator / denominator
                                                              
        result = self._variables.foundsf * horizontal_required
        
        return result
