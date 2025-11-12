import pandas as pd
import data_cleaning as dc
import numpy as np
from pyproj import Proj, Transformer
from scipy.ndimage import gaussian_filter

import pygmt
from scipy.ndimage import gaussian_filter

"""
1, i need to transfer postcode to longitude and altitude (finished)
2, dataframe has same differences each elements
2, use gaussion to smooth data
3, build a coordinate system
4, aggregation data
5, smooth data by gaussion
6, build heat map and contour
"""
df_merged = dc.get_df()
print(df_merged)

# ======================== (0) 配置参数 ========================
# val_col:      你要可视化的指标列（例如 'Total_cons_kwh'）
# grid_size_m:  栅格分辨率（米）。越小越细，但噪声更明显、插值更慢。城市级常用 200–500m
# sigma_m:      高斯平滑的标准差（米）。常取 1~2 × grid_size_m。越大越“糊”、越平滑
# winsor_p:     温莎化上限分位（%）。把极端大值“截顶”到该分位，避免色带被少数异常值主导
# levels_pct:   等高线使用的若干分位（%），更稳健，适合不同量纲/区域对比
val_col    = "Total_cons_kwh"
grid_size_m= 300
sigma_m    = 450
winsor_p   = 99
levels_pct = [50, 75, 90, 95, 99]

# 图幅设置（你可以改）
fig_width_in  = 8.0      # 图宽（英寸）
fig_height_in = 6.0      # 图高（英寸）
dpi = 200                # 导出分辨率


def transfer_unit(df):
    """
    1,find maxium and minimum latitude and longitude to calculate distance,   
    2,calculate ratio 
    """
    maximum_lat = df['Latitude'].max()
    maximum_long = df['Longitude'].max()
    minimum_lat = df['Latitude'].min()
    minimum_long = df['Longitude'].min()
    dis_lat = maximum_lat - minimum_lat
    dis_long = maximum_long - minimum_long
    return dis_lat, dis_long


_tf_utm30 = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True) # a transformer, change long and la to m scale
_M2FT = 3.280839895013123 # M -> FT
_M2IN = 39.37007874015748 # M -> IN

def latlon_to_utm_feet(lat, lon):

    x_m, y_m = _tf_utm30.transform(lat, lon)  # get FT coordinate location
    return x_m * _M2FT, y_m * _M2FT


def contour_array():
    """
    2,find a suitable scale
    3,create an array for each value in coordinate
     
    """

    dis_lat, dis_long = transfer_unit(df = df_merged)
    FT_x, FT_y = latlon_to_utm_feet(df_merged['Longitude'], df_merged["Latitude"])

    df_merged['FT_x'] = FT_x
    df_merged['FT_y'] = FT_y
    val = df_merged["Total_cons_kwh"].to_numpy()
    return val, df_merged

val, df_merged = contour_array()
print(val)
print(df_merged)



