# -*- coding: utf-8 -*-

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

import pandas as pd

from dtocean_moorings.utils import dummy_to_dataframe, dataframe_to_dummy


def test_dummy_to_dataframe():
    
    dummydb = {'id1': {'cost': 37.5,
                       'item1': 'mooring system',
                       'item2': 'chain',
                       'item3': 'studlink chain',
                       'item4': 'grade 1',
                       'mbl': 417000.0,
                       'size': 32.0,
                       'weight': 23.9},
               'id2': {'cost': 30.5,
                       'item1': 'mooring system',
                       'item2': 'shackle',
                       'item3': 'safety bow shackle',
                       'item4': 'g-2140',
                       'mbl': 883000.0,
                       'size': 32.0,
                       'weight': 6.4}
                }
                
    result = dummy_to_dataframe(dummydb)
    
    assert isinstance(result, pd.DataFrame)
    assert 'cost' in result.columns


def test_dataframe_to_dummy():
    
    dummy_dict = {'cost': {'id1': 37.5, 'id2': 30.5},
                  'item1': {'id1': 'mooring system', 'id2': 'mooring system'},
                  'item2': {'id1': 'chain', 'id2': 'shackle'},
                  'item3': {'id1': 'studlink chain',
                            'id2': 'safety bow shackle'},
                  'item4': {'id1': 'grade 1', 'id2': 'g-2140'},
                  'mbl': {'id1': 417000.0, 'id2': 883000.0},
                  'size': {'id1': 32.0, 'id2': 32.0},
                  'weight': {'id1': 23.899999999999999,
                             'id2': 6.4000000000000004}}
    df = pd.DataFrame(dummy_dict)
    
    result = dataframe_to_dummy(df)
    
    assert set(result.keys()) == set(['id1', 'id2'])
    