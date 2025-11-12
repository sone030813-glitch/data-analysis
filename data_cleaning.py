import pandas as pd

def get_df(): 
    path = 'Postcode_level_all_meters_electricity_2023.csv'
    path2 = 'SmallUser.csv'
    df = pd.read_csv(path)
    df_la = pd.read_csv(path2) # including altitude and longitude data

    df = df[['Postcode','Total_cons_kwh','Mean_cons_kwh', 'Median_cons_kwh']]
    df_la = df_la[['Postcode', 'Latitude', 'Longitude']]
    df_merged = df.merge(df_la, on="Postcode", how="left")

    df_merged = df_merged.dropna(subset = ['Latitude', 'Longitude'],  how='any') #drop rows including nan
    return df_merged



