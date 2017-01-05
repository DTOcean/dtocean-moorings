
import pytest

import os
import time
import pprint
import cPickle as pickle
import numpy as np
import pandas as pd

from dtocean_moorings import start_logging
from dtocean_moorings.main import Variables, Main

this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(this_dir, "..", "sample_data")

def main():    
    
    """ Sound of Islay dummy scenario with 10x fixed tidal turbines loosely based on the HS1000 
        turbine. Environmental conditions based on http://www.renewables-atlas.info/. Note: return 
        period values not available. ***Denotes information not available. """

    # Fix substations parameters
    substparams = pd.read_csv(os.path.join(data_dir, 'substparams_aegir.txt'),
                              sep='\t',
                              index_col = 0,
                              header = 0) #substation parameters
                              
    subcog = substparams['subcog']
    suborig = substparams['suborig']
    substloc = substparams['substloc']
    
    subcog_list = []
    suborig_list = []
    substloc_list = []
    
    for i in xrange(len(substparams)):
        
        subcog_list.append(eval(subcog .ix[i]))
        suborig_list.append(eval(suborig.ix[i]))
        substloc_list.append(eval(substloc.ix[i]))
        
    substparams['subcog'] = subcog_list
    substparams['suborig'] = suborig_list
    substparams['substloc'] = substloc_list    
    
    input_variables = Variables(['device001'], #'device002', 'device003', 'device004', 'device005'],#, 'device006', 'device007', 'device008', 'device009', 'device010', 'device011', 'device012', 'device013', 'device014'], #device list
                                9.80665, #gravity
                                1025.0, #sea water density
                                1.226, #air density
                                7851.1, #steel density
                                2400.0, #concrete density
                                2450.0, #grout density
                                eval(open(os.path.join(data_dir, 'dummycompdb.txt')).read()), #component database
                                np.genfromtxt(os.path.join(data_dir, 'aegirsoil2.txt'), dtype= None), #soil grid 
                                [], #seafloor friction coefficient (optional)
                                np.loadtxt(os.path.join(data_dir, 'aegirbath2.txt'), delimiter="\t"), #bathymetry grid
                                15.54, #grid deltax
                                30.93, #grid deltay
                                0.0, #***water level maximum offset
                                0.0, #***water level minimum offset
                                0.7, #current velocity
                                0.0, #***current direction
                                "1/7 power law", #current profile alternatives: "uniform" "1/7 power law"
                                [0.0], #***wave direction
                                [13.0], #significant wave height. Leave the square bracket blank if there are no wave conditions to analyse
                                [14.8], #***peak wave period
                                [3.3], #***jonswap gamma
                                39.8, #wind velocity *Note: 100m above water level
                                0.0, #***wind direction
                                0.0, #***wind gust velocity
                                0.0, #***wind gust direction
                                pd.read_csv(os.path.join(data_dir, 'soilprops.txt'), sep='\t', index_col = 0, header = 0), #default soil properties                                
                                np.loadtxt(os.path.join(data_dir, 'linebcf.txt'), delimiter="\t"), #buried line bearing capacity factors
                                np.loadtxt(os.path.join(data_dir, 'subgradereactioncoefficientk1_cohesive.txt'), delimiter="\t"), #subgrade reaction coefficients
                                3.0, #soil sensitivity
                                np.loadtxt(os.path.join(data_dir, 'subgradereactioncoefficient_cohesionless.txt'), delimiter="\t") , #subgrade soil reaction coefficients cohesionless
                                np.loadtxt(os.path.join(data_dir, 'piledeflectioncoefficients.txt'), delimiter="\t") , #pile deflection coefficients
                                np.loadtxt(os.path.join(data_dir, 'pilemomentcoefficientsam.txt'), delimiter="\t") , #pile moment coefficients am
                                np.loadtxt(os.path.join(data_dir, 'pilemomentcoefficientsbm.txt'), delimiter="\t") , #pile moment coefficients bm
                                np.loadtxt(os.path.join(data_dir, 'pilelimitingvaluesnoncalcareous.txt'), delimiter="\t"), #pile limiting values non calcaeous soils
                                np.loadtxt(os.path.join(data_dir, 'holdingcapacityfactorsplateanchors.txt'), delimiter="\t"), #plate anchor holding capacity factors
                                'wavefloat', #device type options: 'tidefloat', 'tidefixed', 'wavefloat', 'wavefixed'
                                False, #depth variation permitted
                                "rectangular",   #device profile options: "cylindrical" "rectangular"
                                1400.0e3, #device mass
                                [0.0, 0.0, 2.0], #device centre of gravity. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                1440.0, #device displaced volume
                                4.0, #device height
                                4.0, #device width
                                180.0, #device length
                                0.9e-2, #device surface roughness
                                {'device001': [584946.00,6650564.00,0.0], 'device002': [584946.00,6651064.00,0.0], 'device003': [584946.00,6651564.00,0.0], 'device004': [584946.00,6652064.00,0.0], 'device005': [584946.00,6652564.00,0.0], 'device006': [584946.00,6653064.00,0.0], 'device007': [584946.00,6653564.00,0.0], 'device008': [585946.00,6650814.00,0.0], 'device009': [585946.00,6651314.00,0.0], 'device010': [585946.00,6651814.00,0.0], 'device011': [585946.00,6652314.00,0.0], 'device012': [585946.00,6652814.00,0.0], 'device013': [585946.00,6653314.00,0.0], 'device014': [585946.00,6653814.00,0.0]}, #device global locations. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                np.array([[0.0,-219.02,-5.0],[1.732,-216.02,-5.0],[-1.732,-216.02,-5.0]]), #fairlead locations (from device origin)
                                np.array([[0.0, -471.05, 0.0],[220.0, -90.0, 0.0],[-220.0, -90.0, 0.0]]), #foundation locations (from device origin)
                                [0.0, 0.0, 0.0], #device umbilical connection point. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                8.0, #device dry frontal area
                                360.0, #device dry beam area
                                np.loadtxt(os.path.join(data_dir, 'dragcoefcyl.txt'), delimiter="\t"), #cylinder drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'wakeamplificationfactorcyl.txt'), delimiter="\t"), #cylinder wake amplification factors
                                np.loadtxt(os.path.join(data_dir, 'winddragcoefrect.txt'), delimiter="\t"), #rectangular wind drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'currentdragcoeffrect.txt'), delimiter="\t"), #rectangular current drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'driftcoefficientfloatrect.txt'), delimiter="\t"), #rectangular wave drift coefficients
                                (0.0, 0.0), #rotor diameter and hub offset from system centreline
                                np.loadtxt(os.path.join(data_dir, 'thrustcurv.txt'), delimiter="\t"), #thrust curve
                                0.0, #hub height. The vertical height from the system origin (fixed devices: seafloor, floating devices: base of device)
                                0.0, #device orientation angle
                                pickle.load(open(str(data_dir)+'\\fex_pelamiscuboid.pkl','rb')), #first-order wave excitation forces with analysed frequencies and directions
                                'catenary', #predefined mooring system type options: 'catenary', 'taut'
                                [20.0, 20.0, 10.0], #device maximum displacements in surge, sway and heave
                                [], #predefined foundation type options: 'shallowfoundation', 'gravity', 'pile', 'suctioncaisson', 'directembedment', 'drag'
                                1.0, #steel cost
                                0.25, #grout cost
                                0.24, #concrete cost
                                125.0, #grout strength
                                'id742', #predefined umbilical type (database identification number)
                                1.4925, #umbilical safety factor from DNV-RP-F401
                                1.5, #foundation safety factor
                                [], #predefined footprint radius
                                {'device001': [585146.00,6650764.00,-121.0000], 'device002': [585146.00,6650864.00,-121.5080], 'device003': [585146.00,6651364.00,-122.3330], 'device004': [585146.00,6651864.00,-124.0000], 'device005': [585146.00,6652364.00,-126.7470], 'device006': [585146.00,6652864.00,-126.7600], 'device007': [585146.00,6653364.00,-126.8200], 'device008': [585746.00,6650614.00,-115.0000], 'device009': [585746.00,6651114.00,-116.8230], 'device010': [585746.00,6651614.00,-119.8120], 'device011': [585746.00,6652114.00,-115.2470], 'device012': [585746.00,6652614.00,-120.3720], 'device013': [585746.00,6653114.00,-122.0000], 'device014': [585746.00,6653614.00,-122.4270]}, #global subsea cable connection point
                                substparams, #substation parameters
                                1.7, #mooring ultimate limit state safety factor from DNV-OS-E301
                                1.1, #mooring accident limit state safety factor from DNV-OS-E301
                                6.0, #grout safety factor
                                8.0, #device wet frontal area
                                360.0, #device wet beam area
                                2.0, #device equilibrium draft without mooring system
                                np.loadtxt(os.path.join(data_dir, 'waveinertiacoefrect.txt'), delimiter="\t"), #wave inertia coefficient for rectangular structures
                                [], #predefined mooring line component list e.g. ['shackle001','rope','shackle002']
                                1.0) #optional fabrication cost factor
    test = Main(input_variables)    
    devices = test()
    
    return devices
    
# def plot(rsystime):
    
    # plt.plot(rsystime)
    # plt.ylabel('System reliability', fontsize=10)
    # plt.xlabel('Time [hours]', fontsize=10)     
    # plt.show()
    
if __name__ == "__main__":

    start_logging(level="DEBUG")
    
    devices = main()
    
    pprint.pprint(devices)
    
    # plot(rsystime)

