# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 19:51:39 2025

@author: fedeo
"""

# -*- coding: utf-8 -*-
"""
Diagrama tiempo-altura para salida MONAN
Adaptado de script WRF original
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import get_cmap
from matplotlib.ticker import ScalarFormatter
import metpy.calc as mpcalc
from metpy.units import units
import requests
from bs4 import BeautifulSoup

def get_url_paths(url, ext='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

# ==============================
# Configuración
# ==============================
# Ruta a los archivos MONAN (deben tener dimensión Time)
#path = 'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/'  # ⚠️ Cambia esto"  # ⚠️ Ajusta ruta
path = '/pesq/share/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/'  # ⚠️ Cambia esto"  # ⚠️ Ajusta ruta

#path_salida = os.path.join(path, "figuras/")
path_salida = "figuras/"
os.makedirs(path_salida, exist_ok=True)

# Punto de interés
lat1 = -32.833
lon1 = -68.8

# ==============================
# Paso 1: Cargar todos los archivos MONAN
# ==============================
lista_archivos = sorted(glob.glob(os.path.join(path, "MONAN_DIAG*.nc")))
#ext = '.nc'
#lista_archivos = get_url_paths(path, ext)
#lista_archivos = [s + "#mode=bytes" for s in lista_archivos]

#print(lista_archivos)
print(f"Cargando {len(lista_archivos)} archivos...")

# Abrir conjunto de datos múltiples
ds = xr.open_mfdataset(lista_archivos, combine='by_coords')

# Verificar dimensiones
print("Dimensiones:", ds.dims)
print("Variables:", list(ds.variables))


# ==============================
# Paso 3: Extraer series temporales en el punto
# ==============================
# Seleccionar el punto
sounding_lat = -32.86 # SACO: aprox
sounding_lon = -68.8

ds_point = ds.sel(latitude=sounding_lat, longitude=sounding_lon, method = 'nearest')

# Variables necesarias
z = ds_point['zgeo']              # Altura (m)
u = ds_point['uzonal']            # Viento zonal (m/s)
v = ds_point['umeridional']       # Viento meridional (m/s)
# theta = ds_point['theta']         # Temperatura potencial (K) - si existe
# Si no tienes 'theta', calcúlala:
if 'theta' not in ds:
    # Asumir que 'level' está en Pa
    pressure = ds_point['level'] * units.Pa
    temperature = ds_point['temperature'] * units.K
    theta = mpcalc.potential_temperature(pressure, temperature)
    theta = theta.metpy.dequantify()  # Convertir a xarray sin unidades

# Calcular velocidad del viento
vel = np.sqrt(u**2 + v**2)

# Extraer tiempo (asumir que la dimensión temporal se llama 'Time')
time = ds_point['Time']

# ==============================
# Paso 4: Crear diagrama tiempo-altura (presión)
# ==============================
# Convertir presión a hPa si es necesario
if ds_point['level'].units == 'Pa':
    pressure_hPa = ds_point['level'] / 100
else:
    pressure_hPa = ds_point['level']

# Promediar en tiempo para el eje Y (presión o altura)
p_mean = pressure_hPa
z_mean = z.mean(dim='Time')

# Preparar grids para quiver
lons, lats = np.meshgrid(time, p_mean)

# Gráfico 1: Viento y theta (presión)
fig = plt.figure(figsize=(14, 6), dpi=200)
ax = plt.axes()

# Contornos de velocidad del viento
contour_levels = np.arange(0, 60, 2)
cf = ax.contourf(time, p_mean, vel.T, levels=contour_levels, cmap=get_cmap("Spectral_r"))
plt.colorbar(cf, ax=ax, pad=0.05, label="Wind speed (m s$^{-1}$)")

# Contornos de temperatura potencial
theta_levels = np.arange(300, 320, 1)
ct = ax.contour(time, p_mean, theta, levels=theta_levels, colors='dimgray', linewidths=0.5)
ax.clabel(ct, inline=True, fontsize=8)

# Barbs de viento
skip = 1
Q = ax.quiver(lons[::skip, ::skip], lats[::skip, ::skip],
              u.T[::skip, ::skip], v.T[::skip, ::skip],
              scale=900, width=0.002)

qk = ax.quiverkey(Q, 1.1, -0.05, 20, '20 m/s', labelpos='W')

# Formato del eje X (tiempo)
ax.tick_params(axis='x', labelsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y %H:00"))
plt.xticks(rotation=60)

# Formato del eje Y (presión)
ax.set_yscale('symlog')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks(np.linspace(100, 1000, 10))
ax.set_ylim(925, 100)

ax.set_xlabel("Time (UTC)", fontsize=12)
ax.set_ylabel("Pressure (hPa)", fontsize=12)

plt.savefig(os.path.join(path_salida, 'cross_section_time_height_pressure.png'),
            bbox_inches='tight', dpi=150, format='png')
plt.show()

# ==============================
# Paso 5: Diagrama tiempo-altura (altura en metros)
# ==============================
# Promediar altura en tiempo
z_mean = z.mean(dim='Time')

# Preparar grids para quiver
lons2, lats2 = np.meshgrid(time, z_mean)

# Gráfico 2: Viento y theta (altura)
fig = plt.figure(figsize=(14, 6), dpi=200)
ax = plt.axes()

# Contornos de velocidad del viento
cf = ax.contourf(time, z_mean, vel.T, cmap=get_cmap("BrBG_r"))
plt.colorbar(cf, ax=ax, pad=0.05, label="Wind speed (m s$^{-1}$)")

# Contornos de temperatura potencial
theta_levels = np.arange(298, 315, 2)
ct = ax.contour(time, z_mean, theta, levels=theta_levels, colors='dimgray', linewidths=0.5)
ax.clabel(ct, inline=True, fontsize=8)

# Barbs de viento
skip = 1
Q = ax.quiver(lons2[::skip, ::skip], lats2[::skip, ::skip],
              u.T[::skip, ::skip], v.T[::skip, ::skip],
              scale=700, width=0.002)

qk = ax.quiverkey(Q, 1.1, -0.05, 20, '20 m/s', labelpos='W')

# Formato del eje X (tiempo)
ax.tick_params(axis='x', labelsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y %H:00"))
plt.xticks(rotation=60)

# Formato del eje Y (altura)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_ylim(800, 8000)
ax.set_ylabel("Height (m)", fontsize=12)
ax.set_xlabel("Time (UTC)", fontsize=12)

plt.savefig(os.path.join(path_salida, 'cross_section_time_height_altura.png'),
            bbox_inches='tight', dpi=150, format='png')
plt.show()
