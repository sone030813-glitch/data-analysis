import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_2d
from pyproj import Transformer
import data_cleaning as dc

# 确保 Transformer 初始化正确
# always_xy=True 意味着 transform 接受 (Longitude, Latitude)
_tf_utm30 = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
_M2FT = 3.280839895013123

def latlon_to_utm_feet(lat_arr, lon_arr):
    """
    修正后的转换函数：
    输入：纬度数组 (lat), 经度数组 (lon)
    输出：英尺坐标 x, y
    """
    # ⚠️ 核心修正：pyproj transform(x, y) -> transform(lon, lat)
    # 即使函数参数叫 lat/lon，传给 transformer 时必须是 lon 在前！
    x_m, y_m = _tf_utm30.transform(lon_arr, lat_arr)
    return x_m * _M2FT, y_m * _M2FT

def contour_array(
    val_col="Total_cons_kwh",
    grid_ft=3000.0,            
    pad_cells=1,
    agg="median",              
    winsor_p=99,               
    use_log1p=True,
    max_cells=2_000_000        
):
    
    df = dc.get_df().copy()

    # 1. 清洗
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df[val_col]     = pd.to_numeric(df[val_col],     errors="coerce")
    df = df.dropna(subset=["Latitude","Longitude",val_col])
    
    if df.empty: raise ValueError("无有效数据")

    # 2. 经纬度 -> 英尺 (已在函数内部修正为 Lon, Lat 顺序)
    FT_x, FT_y = latlon_to_utm_feet(
        df["Latitude"].to_numpy(), 
        df["Longitude"].to_numpy()
    )
    
    x = FT_x.astype(float)
    y = FT_y.astype(float)
    v = df[val_col].to_numpy(dtype=float)

    # 3. 预处理：winsor + log1p
    if winsor_p is not None:
        hi = np.nanpercentile(v, winsor_p)
        v = np.clip(v, None, hi)
    if use_log1p:
        v = np.log1p(v)

    # 4. 范围 + 留边
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = pad_cells * grid_ft
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    # 5. 自动调整网格大小 (Autotune)
    nx = max(2, int(np.ceil((xmax - xmin) / grid_ft)))
    ny = max(2, int(np.ceil((ymax - ymin) / grid_ft)))
    cells = nx * ny
    
    if cells > max_cells:
        scale = (cells / max_cells) ** 0.5
        grid_ft *= scale
        # 重新计算 nx, ny
        nx = max(2, int(np.ceil((xmax - xmin) / grid_ft)))
        ny = max(2, int(np.ceil((ymax - ymin) / grid_ft)))
        print(f"[autotune] grid_ft -> {grid_ft:.1f} ft, nx={nx}, ny={ny}, cells≈{nx*ny:,}")

    # 6. 2D 分箱聚合 (binned_statistic_2d)
    # 注意：statistic 也可以传自定义函数，但 'mean'/'median' 最快
    stat = "median" if agg == "median" else "mean"
    
    H, xedges, yedges, bin_numbers = binned_statistic_2d(
        x, y, v, 
        statistic=stat,
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    
    # H 的形状是 (nx, ny)，但 xarray 和图片通常期望 (y, x)
    # 所以我们需要转置 H
    Z = H.T  # 形状变为 (ny, nx)
    
    # 计算格心坐标
    xc = (xedges[:-1] + xedges[1:]) / 2.0
    yc = (yedges[:-1] + yedges[1:]) / 2.0

    # 统计填充率
    filled = int(np.isfinite(Z).sum())
    total = Z.size
    print(f"[grid] filled bins: {filled}/{total} ({filled/total:.1%})")

    # 7. 构建 xarray
    da = xr.DataArray(
        Z.astype("float32"),
        dims=("y","x"),
        coords={"y": ("y", yc.astype(float)), "x": ("x", xc.astype(float))},
        name=val_col,
        attrs={
            "grid_size_ft": float(grid_ft),
            "x_unit": "ft", "y_unit": "ft", 
            "agg": agg,
            "crs": "EPSG:32630 feet",
            "pad_cells": int(pad_cells),
            "winsor_p": winsor_p, 
            "log1p": use_log1p
        }
    )
    return da


da = contour_array(grid_ft=3000)
print(da)
da.plot(cmap='viridis') # 快速画图检查

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- 接在你的 da = contour_array(...) 之后 ---

# 1. 准备绘图数据：填充空值为0 (为了平滑)，进行高斯平滑
# sigma=2.0 控制模糊程度，数值越大越平滑
grid_val = da.fillna(0).values
grid_smooth = gaussian_filter(grid_val, sigma=2.0)

# 2. 创建一个新的 DataArray 存放平滑后的数据 (为了利用 xarray 的绘图功能)
da_smooth = xr.DataArray(
    grid_smooth,
    coords=da.coords,
    dims=da.dims,
    name="Smoothed Energy Consumption"
)

# 3. 开始画图
print("正在生成图片...")
plt.figure(figsize=(10, 12)) # 设置画布大小 (宽, 高)

# 画等高线填充图 (Contourf)
# levels=20 表示把颜色分成 20 层
# cmap="Spectral_r" 是红-黄-蓝颜色条 (r表示反转，红色代表高耗能)
plot = da_smooth.plot.contourf(
    levels=20, 
    cmap="Spectral_r", 
    add_colorbar=True,
    cbar_kwargs={'label': 'Log(Total Consumption)'} 
)

plt.title("Energy Consumption Contour Map")
plt.axis('equal')  # ⚠️ 关键：保持地图的物理比例，不然英国会变扁
plt.xlabel("Easting (ft)")
plt.ylabel("Northing (ft)")

# 4. 保存图片 (Codespace 必须这一步)
output_filename = "contour_map_result.png"
plt.savefig(output_filename, dpi=150)
print(f"✅ 图片已保存为: {output_filename}")
print("请在左侧文件浏览器中点击该图片查看结果。")