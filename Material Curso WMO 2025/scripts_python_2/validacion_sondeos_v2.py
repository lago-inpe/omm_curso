import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import xarray as xr
from siphon.simplewebservice.wyoming import WyomingUpperAir
import metpy.calc as mpcalc

# ==============================
# Configuración
# ==============================
# 1. Datos del radiosondeo
station = 'SCSN'        # Código de estación (ej: 'SACO' = San Antonio de los Cobres, Argentina)
date = pd.to_datetime('2023-07-21 12:00')  # Fecha y hora del lanzamiento

# 2. Archivo MONAN
#monan_file = 'D:/disco_EFI_WRF/datos_MONAN/nuevo_recorte/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_2023072112.00.00.x1.65536002L55_mendoza.nc'  # ⚠️ Cambia esto
monan_file = '/pesq/share/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_2023072112.00.00.x1.65536002L55.nc'
#monan_file = 'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_2023072112.00.00.x1.65536002L55.nc#mode=bytes'
# Coordenadas del sitio del radiosondeo (lat, lon)
sounding_lat = -33.65 # SACO: aprox
sounding_lon = -71.61


 # Station identifier: SCSN
 #                             Station number: 85586
 #                           Observation time: 251108/0000
 #                           Station latitude: -33.65
 #                          Station longitude: -71.61
 #                          Station elevation: 75.0

# ==============================
# Paso 1: Descargar radiosondeo real
# ==============================
print(f"Descargando radiosondeo de {station} en {date}...")
try:
    df_real = WyomingUpperAir.request_data(date, station)
    print("✓ Radiosondeo descargado.")
except Exception as e:
    print(f"❌ Error al descargar: {e}")
    raise

# Seleccionar solo variables necesarias y eliminar niveles con NaN
df_real = df_real[['pressure', 'height', 'temperature', 'dewpoint', 'u_wind', 'v_wind']].dropna()

# ==============================
# Paso 2: Leer y extraer perfil MONAN más cercano
# ==============================
print("Leyendo datos de MONAN...")
ds = xr.open_dataset(monan_file).metpy.parse_cf()

# Encontrar el punto de la cuadrícula más cercano al radiosondeo
distances = ((ds['latitude'] - sounding_lat)**2 + (ds['longitude'] - sounding_lon)**2)**0.5
min_idx = np.unravel_index(np.argmin(distances.values), distances.shape)
y_idx, x_idx = min_idx

# Extraer perfil en ese punto
ds_point = ds.isel(latitude=y_idx, longitude=x_idx)  
# Variables MONAN
pres_monan = ds_point['level'].values / 100  # Convertir Pa → hPa
z_monan = ds_point['zgeo'].values           # Altura en metros
temp_monan = ds_point['temperature'].values - 273.15  # K → °C
dewp_monan = mpcalc.dewpoint_from_relative_humidity((ds_point['t2m'].values + 273.15) * units.K, ds_point['q2'].values) #???
q_monan = ds_point['spechum'].values        # kg/kg
u_monan = ds_point['uzonal'].values
v_monan = ds_point['umeridional'].values

# --- Corregir dimensiones (eliminar Time=1) ---
def ensure_1d(arr):
    return np.squeeze(np.asarray(arr))

pres_monan = ensure_1d(pres_monan)
z_monan = ensure_1d(z_monan)
temp_monan = ensure_1d(temp_monan)
dewp_monan = ensure_1d(dewp_monan)
u_monan = ensure_1d(u_monan)
v_monan = ensure_1d(v_monan)

# --- Crear DataFrame ---
df_monan = pd.DataFrame({
    'pressure': pres_monan,
    'height': z_monan,
    'temperature': temp_monan,
    'dewpoint': dewp_monan,
    'u_wind': u_monan,
    'v_wind': v_monan
}).dropna().sort_values('pressure', ascending=False)  # De superficie a alto

# ==============================
# Paso 3: Interpolar MONAN al mismo perfil de presión del radiosondeo
# ==============================
# Interpolación logarítmica en presión (estándar en meteorología)
def interpolate_model_to_obs(model_df, obs_pressures):
    from scipy.interpolate import interp1d
    
    # Usar log(presión) para interpolación logarítmica
    log_p_model = np.log(model_df['pressure'].values)
    log_p_obs = np.log(obs_pressures)
    
    # Interpolar cada variable
    interp_vars = {}
    for var in ['height', 'temperature', 'dewpoint', 'u_wind', 'v_wind']:
        f = interp1d(log_p_model, model_df[var].values, 
                     bounds_error=False, fill_value=np.nan, kind='linear')
        interp_vars[var] = f(log_p_obs)
    
    # Crear DataFrame
    df_interp = pd.DataFrame({
        'pressure': obs_pressures,
        **interp_vars
    }).dropna()
    
    return df_interp

# Interpolar MONAN al perfil de presión del radiosondeo
df_monan_interp = interpolate_model_to_obs(df_monan, df_real['pressure'].values)

# ==============================
# Paso 4: Graficar Skew-T con ambos perfiles
# ==============================
fig = plt.figure(figsize=(10, 10))
skew = SkewT(fig, rotation=45)

# Preparar datos para gráfico
p_real = df_real['pressure'].values * units.hPa
t_real = df_real['temperature'].values * units.degC
td_real = df_real['dewpoint'].values * units.degC
u_real = df_real['u_wind'].values * units('m/s')
v_real = df_real['v_wind'].values * units('m/s')

p_monan = df_monan_interp['pressure'].values * units.hPa
t_monan = df_monan_interp['temperature'].values * units.degC
td_monan = df_monan_interp['dewpoint'].values * units.degC
u_monan_interp = df_monan_interp['u_wind'].values * units('m/s')
v_monan_interp = df_monan_interp['v_wind'].values * units('m/s')

# Graficar radiosondeo real → como LÍNEAS
skew.plot(p_real, t_real, 'r-', label='Radiosondeo (Real)', linewidth=2)
skew.plot(p_real, td_real, 'g-', label='Dewpoint (Real)', linewidth=1.5)
skew.plot_barbs(p_real, u_real, v_real, length=8, color='red', xloc = 0.95)

# Graficar MONAN → ya es línea
skew.plot(p_monan, t_monan, 'b-', label='MONAN', linewidth=2)
skew.plot(p_monan, td_monan, 'c-', label='Dewpoint (MONAN)', linewidth=1.5)  # cian para contraste
skew.plot_barbs(p_monan[::2], u_monan_interp[::2], v_monan_interp[::2], 
                length=8, color='blue', fill_empty=False)

# Configuración
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 50)
skew.ax.set_title(f'Validación MONAN vs Radiosondeo\nEstación: {station} - Fecha: {date.date()}', fontsize=14)
skew.ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==============================
# Paso 5: Calcular y graficar BIAS por niveles de presión
# ==============================

# Asegurar que ambos datasets estén interpolados y alineados
# (ya lo hicimos en df_monan_interp y df_real)

# Extraer presiones comunes y variables
p_common = df_monan_interp['pressure'].values  # hPa
temp_bias = df_monan_interp['temperature'].values - df_real['temperature'].values
dewp_bias = df_monan_interp['dewpoint'].values - df_real['dewpoint'].values

# Filtrar NaN
valid = ~(np.isnan(temp_bias) | np.isnan(dewp_bias) | np.isnan(p_common))
p_plot = p_common[valid]
temp_bias_plot = temp_bias[valid]
dewp_bias_plot = dewp_bias[valid]

# Graficar bias
fig, ax = plt.subplots(figsize=(6, 10))

# Bias de temperatura
ax.plot(temp_bias_plot, p_plot, 'r-', label='Bias T (MONAN - Real)', linewidth=2)
# Bias de punto de rocío
ax.plot(dewp_bias_plot, p_plot, 'g-', label='Bias Td (MONAN - Real)', linewidth=2)

# Línea cero
ax.axvline(0, color='k', linestyle='--', linewidth=1)

# Configuración
ax.set_ylim(1000, 100)
ax.set_xlabel('Bias (°C)', fontsize=12)
ax.set_ylabel('Pressure (hPa)', fontsize=12)
ax.set_title(f'Bias vertical — {station} {date.date()}', fontsize=13)
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(loc='best')

# Etiquetas en el eje X
ax.set_xlim(-8, 8)  # Ajusta según tus datos

plt.tight_layout()
plt.show()

# ==============================
# Paso 6: Estadísticas resumen de bias
# ==============================

print("=== Verificación de velocidades ===")
print(f"Viento real (min/max): {np.nanmin(wspd_real):.1f} / {np.nanmax(wspd_real):.1f} m/s")
print(f"Viento MONAN (min/max): {np.nanmin(wspd_monan):.1f} / {np.nanmax(wspd_monan):.1f} m/s")
print(f"Viento real (media): {np.nanmean(wspd_real):.1f} m/s")
print(f"Viento MONAN (media): {np.nanmean(wspd_monan):.1f} m/s")

print("\n" + "="*50)
print(f"VALIDACIÓN: MONAN vs Radiosondeo ({station} - {date.date()})")
print("="*50)
print(f"{'Variable':<15} {'Bias promedio (°C)':<20} {'RMSE (°C)':<15}")
print("-"*50)

def calc_stats(bias_vals):
    bias_mean = np.nanmean(bias_vals)
    rmse = np.sqrt(np.nanmean(bias_vals**2))
    return bias_mean, rmse

t_bias_mean, t_rmse = calc_stats(temp_bias_plot)
td_bias_mean, td_rmse = calc_stats(dewp_bias_plot)

print(f"{'Temperatura':<15} {t_bias_mean:>12.2f}{'':<8} {t_rmse:>10.2f}")
print(f"{'Punto de rocío':<15} {td_bias_mean:>12.2f}{'':<8} {td_rmse:>10.2f}")
print("="*50)

# ==============================
# viento
# ==============================

# --- Calcular velocidad del viento (m/s) ---
wspd_real = np.sqrt(df_real['u_wind']**2 + df_real['v_wind']**2)
wspd_monan = np.sqrt(df_monan_interp['u_wind']**2 + df_monan_interp['v_wind']**2)

# Bias de velocidad del viento
wspd_bias = wspd_monan.values - wspd_real.values

# Graficar bias
fig, ax = plt.subplots(figsize=(6, 10))

# Bias de temperatura
# ax.plot(temp_bias_plot, p_plot, 'r-', label='Bias T (°C)', linewidth=2)
# # Bias de punto de rocío
# ax.plot(dewp_bias_plot, p_plot, 'g-', label='Bias Td (°C)', linewidth=2)
# Bias de velocidad del viento
ax.plot(wspd_bias[valid], p_plot, 'b-', label='Bias Viento (m/s)', linewidth=2)

# Línea cero
ax.axvline(0, color='k', linestyle='--', linewidth=1)

# Configuración
ax.set_ylim(1000, 100)
ax.set_xlabel('Bias', fontsize=12)
ax.set_ylabel('Pressure (hPa)', fontsize=12)
ax.set_title(f'Bias vertical — {station} {date.date()}', fontsize=13)
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(loc='best')

# Ajustar límites según las variables (T/Td en ±8°C, viento en ±10 m/s)
# ax.set_xlim(-10, 10)  # Abarca viento hasta ±10 m/s

plt.tight_layout()
plt.show()






