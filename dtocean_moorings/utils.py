
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

