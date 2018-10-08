# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Mathew Topper
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


def dummy_to_dataframe(dummy_db):

    dfindex = []
    dfcols = {}
    
    for dummy_index, dummy_row in dummy_db.iteritems():
        dfindex.append(dummy_index)
        for key, value in dummy_row.iteritems():
            if key not in dfcols:
                dfcols[key] = [value]
            else:
                dfcols[key].append(value)
                
    df = pd.DataFrame(dfcols, index=dfindex)
    
    return df


def dataframe_to_dummy(df):

    dummy_db = {}

    for idx in df.index:
        
        df_row = df.ix[idx]
        dummy_row = df_row.to_dict()
        dummy_db[idx] = dummy_row
        
    return dummy_db
