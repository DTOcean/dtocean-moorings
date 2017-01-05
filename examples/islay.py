
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
        turbine and 3x substations. Environmental conditions based on the Atlas of UK Marine Renewable Energy. Note: 
        return period values are not available for this site. ***Denotes information not available. """
    
    # Fix substations parameters
    substparams = pd.read_csv(os.path.join(data_dir, 'substparams.txt'),
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
    
    input_variables = Variables(['device001', 'device002', 'device003', 'device004', 'device005', 'device006', 'device007', 'device008', 'device009', 'device010'], #device list
                                9.80665, #gravity
                                1025.0, #sea water density
                                1.226, #air density
                                7851.1, #steel density
                                2400.0, #concrete density
                                2450.0, #grout density
                                eval(open(os.path.join(data_dir, 'dummycompdb.txt')).read()), #component database
                                np.genfromtxt(os.path.join(data_dir, 'islaysoil2.txt'), dtype= None), #soil grid 
                                [], #seafloor friction coefficient (optional)
                                np.loadtxt(os.path.join(data_dir, 'islaybath2.txt'), delimiter="\t"), #bathymetry grid
                                17.39, #grid deltax
                                30.90, #grid deltay
                                0.0, #***water level maximum offset
                                0.0, #***water level minimum offset
                                3.0, #current velocity
                                0.0, #***current direction
                                "1/7 power law", #current profile alternatives: "uniform" "1/7 power law"
                                [0.0], #***wave direction
                                [1.5], #significant wave height. Leave the square bracket blank if there are no wave conditions to analyse
                                [10.0], #***peak wave period
                                [1.0], #***jonswap gamma
                                7.0, #wind velocity *Note: 100m above water level
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
                                'tidefixed', #device type options: 'tidefloat', 'tidefixed', 'wavefloat', 'wavefixed'
                                False, #depth variation permitted
                                "cylindrical",   #device profile options: "cylindrical" "rectangular"
                                310.0e3, #device mass
                                [0.0, 0.0, 11.0], #device centre of gravity. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                250.59, #device displaced volume
                                33.5, #device height
                                15.0, #device width
                                22.0, #device length
                                0.9e-2, #device surface roughness
                                {'device001': [681556.49,6192736.96,0.0], 'device002': [681603.21,6192751.02,0.0], 'device003': [681682.78,6192280.76,0.0], 'device004': [681722.74,6192292.58,0.0], 'device005': [681721.39,6191793.12,0.0], 'device006': [681763.74,6191804.41,0.0], 'device007': [681806.15,6191814.92,0.0], 'device008': [681771.13,6191326.04,0.0], 'device009': [681813.78,6191334.25,0.0], 'device010': [681888.09,6191351.89,0.0]}, #device global locations. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                np.array([[0.0,1.0,-1.5],[1.0,0.0,-1.5],[0.0,-1.0,-1.5],[-1.0,0.0,-1.5]]), #fairlead locations (from device origin)
                                np.array([[0.0, 13.333, 0],[7.5, -6.667, 0],[-7.5, -6.667, 0]]), #foundation locations (from device origin)
                                [0.0, 0.0, 22.0], #device umbilical connection point. Note for both fixed and floating devices the device origin is assumed to be at the base of the device
                                0.0, #device dry frontal area
                                0.0, #device dry beam area
                                np.loadtxt(os.path.join(data_dir, 'dragcoefcyl.txt'), delimiter="\t"), #cylinder drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'wakeamplificationfactorcyl.txt'), delimiter="\t"), #cylinder wake amplification factors
                                np.loadtxt(os.path.join(data_dir, 'winddragcoefrect.txt'), delimiter="\t"), #rectangular wind drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'currentdragcoeffrect.txt'), delimiter="\t"), #rectangular current drag coefficients
                                np.loadtxt(os.path.join(data_dir, 'driftcoefficientfloatrect.txt'), delimiter="\t"), #rectangular wave drift coefficients
                                (23.0, 0.0), #rotor diameter and hub offset from system centreline
                                np.loadtxt(os.path.join(data_dir, 'thrustcurv.txt'), delimiter="\t"), #thrust curve
                                22.0, #hub height. The vertical height from the system origin (fixed devices: seafloor, floating devices: base of device)
                                0.0, #device orientation angle
                                pickle.load(open(str(data_dir)+'\\fex_wp4.pkl','rb')), #first-order wave excitation forces with analysed frequencies and directions
                                'catenary', #predefined mooring system type options: 'catenary', 'taut'
                                [20.0, 20.0, 10.0], #device maximum displacements in surge, sway and heave
                                [], #predefined foundation type options: 'shallowfoundation', 'gravity', 'pile', 'suctioncaisson', 'directembedment', 'drag'
                                1.0, #steel cost
                                0.25, #grout cost
                                0.24, #concrete cost
                                125.0, #grout strength
                                'id737', #predefined umbilical type (database identification number)
                                1.4925, #umbilical safety factor from DNV-RP-F401
                                1.5, #foundation safety factor
                                [], #predefined footprint radius
                                {'device001': [681554.13340049,6192718.74251017,-57.73], 'device002': [681606.28865251,6192720.92917814,-56.28], 'device003': [681677.88486354,6192259.60929421,-43.63], 'device004': [681730.04567476,6192261.79734614,-48.86], 'device005': [681716.00975873,6191765.93122009,-50.36], 'device006': [681766.88006262,6191799.02005986,-49.27], 'device007': [681801.65765063,6191800.47933450,-48.26], 'device008': [681770.23216186,6191303.88351347,-42.84], 'device009': [681822.40450582,6191306.07245544,-48.88], 'device010': [681890.67046623,6191339.89243383,-52.15]}, #global subsea cable connection point
                                substparams, #substation parameters
                                1.7, #mooring ultimate limit state safety factor from DNV-OS-E301
                                1.1, #mooring accident limit state safety factor from DNV-OS-E301
                                6.0, #grout safety factor
                                98.6, #device wet frontal area
                                132.5, #device wet beam area
                                0.0, #device equilibrium draft without mooring system
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

