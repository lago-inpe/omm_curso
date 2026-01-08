# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 18:41:58 2025

@author: fedeo
"""

#MONAN

import os
import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from metpy.units import units
from matplotlib.gridspec import GridSpec

# ==============================
# Configuración
# ==============================
path = 'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071900/'

#Definition for specific plotting variables
exp = 1
smooth_flag = 1

# Definir la sección transversal
start = (-33.0, -73.0)
end   = (-33.0, -67.0)

# Colores o estilo para diferenciar experimentos (opcional)
colors = plt.cm.tab10.colors

ds = xr.open_dataset(path + 'MONAN_DIAG_R_POS_GFS_2023071900_2023072120.00.00.x1.65536002L55.nc#mode=bytes').metpy.parse_cf()

# === Extraer metadatos ===
nlevs = ds.sizes.get('level', 'unknown')  # Usa .sizes en lugar de .dims

# === Sección transversal ===
# ✅ Usar la verdadera topografía del modelo
true_topo = ds['ter']
cross_topo = cross_section(true_topo, start, end)
cross_ds = cross_section(ds[['temperature', 'spechum', 'uzonal', 'umeridional', 'zgeo']], start, end)

# Eliminar dimensiones de tamaño 1
cross_topo = cross_topo.squeeze()
cross_ds = cross_ds.squeeze()

# Extraer
topo_line = cross_topo.values  
lons = cross_ds['longitude'].values
u_data = cross_ds['uzonal'].values          
t_data = cross_ds['temperature'].values
q_data = cross_ds['spechum'].values
height_3d = cross_ds['zgeo'].values          

# === Eliminar columnas con NaN (bordes del dominio) ===
valid_cols = ~np.isnan(height_3d).all(axis=0)

lons = lons[valid_cols]
topo_line = topo_line[valid_cols]
u_data = u_data[:, valid_cols]
t_data = t_data[:, valid_cols]
q_data = q_data[:, valid_cols]
height_3d = height_3d[:, valid_cols]

# Usar alturas del primer punto como eje vertical
height_levels = height_3d[:, 0]

# === Calcular temperatura potencial (theta) ===
std_pressure = mpcalc.height_to_pressure_std(height_3d * units.m)
theta = mpcalc.potential_temperature(std_pressure, t_data * units.K).magnitude

# === Rellenar debajo de la topografía (opcional, mejora visual) ===
def fill_below_topo_3d(var_3d, height_3d, topo_1d):
    var_filled = np.copy(var_3d)
    for i in range(var_3d.shape[1]):
        topo_h = topo_1d[i]
        if np.isnan(topo_h):
            continue
        levels_above = np.where(height_3d[:, i] >= topo_h)[0]
        if len(levels_above) > 0:
            first_idx = levels_above[0]
            var_filled[:first_idx, i] = var_filled[first_idx, i]
    return var_filled


# === Graficar ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel 1: Viento zonal y theta
cf1 = ax1.contourf(lons, height_levels, u_data, 
                   levels=np.arange(-10, 50, 2), cmap='RdBu_r', extend='both')
fig.colorbar(cf1, ax=ax1, pad=0.01).set_label('Zonal wind (m s⁻¹)', fontsize=10)
ax1.contour(lons, height_levels, theta, 
            levels=np.arange(200, 400, 3), colors='k', linewidths=1.5, alpha=0.7)

# ✅ Topografía visible
ax1.fill_between(lons, 0, topo_line, color='saddlebrown', zorder=100)
ax1.plot(lons, topo_line, color='black', lw=2, zorder=101)

# Ajustar ylim para ver la cordillera
ax1.set_ylim(0, 17000)
ax1.set_ylabel('Height (m)', fontsize=11)
ax1.set_title(f'{exp} - Zonal Wind & Potential Temperature')

# Panel 2: Humedad específica
q_gkg = q_data * 1000
cf2 = ax2.contourf(lons, height_levels, q_gkg, 
                   levels=np.arange(0, 6, 0.25), cmap='BrBG', extend='max')
fig.colorbar(cf2, ax=ax2, pad=0.01).set_label('Specific Humidity (g kg⁻¹)', fontsize=10)

# ✅ Topografía visible
ax2.fill_between(lons, 0, topo_line, color='saddlebrown', zorder=100)
ax2.plot(lons, topo_line, color='black', lw=2, zorder=101)

ax2.set_xlabel('Longitude (°)', fontsize=11)
ax2.set_ylabel('Height (m)', fontsize=11)
ax2.set_ylim(0, 8000)
ax2.set_title('Specific Humidity')

# Texto informativo
info_text = f'Vertical levels: {nlevs}\n{smooth_flag}'
ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes,
         fontsize=10, va='top', ha='left',
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

############################################################################
# U y theta

# === Gráfico único ===
fig, ax = plt.subplots(figsize=(16, 8))

# Viento zonal (U)
u_levels = np.arange(-10, 50, 2)
cf = ax.contourf(lons, height_levels, u_data, levels=u_levels, cmap='RdBu_r', extend='both')
cb = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
cb.set_label('Zonal wind (m s⁻¹)', fontsize=12)

# Temperatura potencial (θ)
theta_min = np.nanmin(theta)
theta_max = np.nanmax(theta)
theta_levels = np.arange(int(theta_min) - 2, int(theta_max) + 3, 3)
ct = ax.contour(lons, height_levels, theta, levels=theta_levels, colors='k', linewidths=1.0, alpha=0.8)
ax.clabel(ct, ct.levels[::2], inline=True, fontsize=10, fmt='%d')

# Topografía (cordillera)
ax.fill_between(lons, 0, topo_line, color='saddlebrown', zorder=100, alpha=0.9)
ax.plot(lons, topo_line, color='black', lw=2.5, zorder=101)

# Etiquetas y formato
ax.set_xlabel('Longitude (°)', fontsize=13)
ax.set_ylabel('Height (m)', fontsize=13)
ax.set_ylim(0, 17000)
ax.set_title(f'{exp} — Cross-section: Zonal Wind (U) and Potential Temperature (θ)', fontsize=14)

# Texto informativo
info_text = f'Vertical levels: {nlevs}\n{smooth_flag}'
ax.text(0.02, 0.96, info_text, transform=ax.transAxes,
        fontsize=11, va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()
