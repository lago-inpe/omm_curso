# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:19:14 2025

@author: fedeo
"""

# ==============================
# IMPORTS
# ==============================
import os
import glob
import re
from datetime import datetime
import pandas as pd
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt

path_datos = "https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/Validation/Zonal_Wind_Mendoza/Federico_Luiz/"

#Mendoza_aero =  pd.read_excel (r'D:\disco_EFI_WRF\datos_MONAN\curso_WMO/MENDOZA_AERO_2023_2024.xlsx')
Mendoza_aero =  pd.read_excel (path_datos + 'datos_MENDOZA/MENDOZA_AERO_2023_2024.xlsx')
Mendoza_aero['FECHAUTC'] = pd.to_datetime(Mendoza_aero['FECHAUTC'], format='%d/%m/%Y %H:%M:%S')

#Mendoza_Obs =  pd.read_excel (r'D:\disco_EFI_WRF\datos_MONAN\curso_WMO/MENDOZA_OBSERVATORIO_2023_2024.xlsx')
Mendoza_Obs =  pd.read_excel (path_datos + 'datos_MENDOZA/MENDOZA_OBSERVATORIO_2023_2024.xlsx')
Mendoza_Obs['FECHAUTC'] = pd.to_datetime(Mendoza_Obs['FECHAUTC'], format='%d/%m/%Y %H:%M:%S')

#San_juan_aero =  pd.read_excel (r'D:\disco_EFI_WRF\datos_MONAN\curso_WMO/SAN_JUAN_AERO_2023_2024.xlsx')
San_juan_aero =  pd.read_excel (path_datos + 'datos_MENDOZA/SAN_JUAN_AERO_2023_2024.xlsx')
San_juan_aero['FECHAUTC'] = pd.to_datetime(San_juan_aero['FECHAUTC'], format='%d/%m/%Y %H:%M:%S')

# ==============================
# 1. CONFIGURACIÓN GENERAL
# ==============================
#path_monan = r'D:/disco_EFI_WRF/datos_MONAN/recortes_mendoza/10km/'
#path_monan = 'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind/'
path_monan = "/pesq/share/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind/"
inicio_objetivo = datetime(2023, 7, 21, 0)   # Desde 2023-07-21 00:00 UTC
fin_objetivo     = datetime(2023, 7, 22, 12) # Hasta 2023-07-22 12:00 UTC

# ==============================
# 2. CARGAR ESTACIONES
# ==============================
#ruta_estaciones = r'C:/Users/Usuario/Desktop/proyecto_SMN_ERA/estaciones_lat_lon.xlsx'
ruta_estaciones = path_datos + 'estaciones_lat_lon.xlsx'
estaciones_smn = pd.read_excel(ruta_estaciones, sheet_name='smn')
estaciones_ianigla = pd.read_excel(ruta_estaciones, sheet_name='ianigla')
estaciones_ceaza = pd.read_excel(ruta_estaciones, sheet_name='ceaza')
estaciones_meteochile = pd.read_excel(ruta_estaciones, sheet_name='meteochile')

# ==============================
# 3. CARGAR Y PROCESAR CORRIDAS MONAN
# ==============================
def cargar_corrida_monan(ruta_base, inicio_obj, fin_obj):
    archivos = sorted(glob.glob(ruta_base + 'MONAN_DIAG_R_POS_GFS_*.nc'))
#    archivos = sorted(glob.glob(ruta_base + 'MONAN_DIAG_R_POS_GFS_*.nc#mode=bytes'))
    if not archivos:
        return None

    fechas_validacion = []
    archivos_filtrados = []

    for archivo in archivos:
        nombre = os.path.basename(archivo)
        match = re.search(r'_(\d{10})\.\d{2}\.\d{2}\.x1', nombre)
        if match:
            fecha_str = match.group(1)
            fecha = datetime.strptime(fecha_str, '%Y%m%d%H')
            if inicio_obj <= fecha <= fin_obj:
                archivos_filtrados.append(archivo)
                fechas_validacion.append(fecha)

    if not archivos_filtrados:
        return None

    ds = xr.open_mfdataset(
        archivos_filtrados,
        combine='nested',
        concat_dim='Time',
        parallel=True,
        chunks={'Time': 1}
    )
    ds = ds.assign_coords(Time=('Time', fechas_validacion))
    return ds

def procesar_corrida_monan(ds):
    t2m_K = ds['t2m'].values
    q2 = ds['q2'].values
    u10 = ds['u10'].values
    v10 = ds['v10'].values
    psfc = ds['surface_pressure'].values

    coords = {k: v for k, v in ds['t2m'].coords.items()}
    dims = ds['t2m'].dims

    def limpiar_entrada(x, vmin, vmax):
        return np.where((np.isfinite(x)) & (x >= vmin) & (x <= vmax), x, np.nan)

    u10, v10 = limpiar_entrada(u10, -100, 100), limpiar_entrada(v10, -100, 100)
    t2m = limpiar_entrada(t2m_K - 273.15, -60, 60)
    q2 = limpiar_entrada(q2, 0, 0.1)
    psfc = limpiar_entrada(psfc, 50000, 110000)

    wind_speed = mpcalc.wind_speed(u10 * units('m/s'), v10 * units('m/s'))
    wind_dir = mpcalc.wind_direction(u10 * units('m/s'), v10 * units('m/s'), convention='from')

    rh2 = mpcalc.relative_humidity_from_specific_humidity(
        (psfc / 100.0) * units.hPa,
        (t2m + 273.15) * units.K,
        q2 * units('kg/kg')
    )
    rh2_percent = rh2 * 100.0

    dew2_K = mpcalc.dewpoint_from_relative_humidity((t2m + 273.15) * units.K, rh2)
    dew2_C = dew2_K.to('degC').magnitude

    def _array(data):
        return xr.DataArray(data, dims=dims, coords=coords)

    return {
        't2m': _array(t2m),
        'dew2': _array(dew2_C),
        'rh2': _array(rh2_percent.magnitude),
        'wind_speed': _array(wind_speed.magnitude),
        'wind_dir': _array(wind_dir.magnitude),
        'Time': ds['Time']
    }

# Cargar corridas
ds_18 = cargar_corrida_monan(path_monan + '2023071800/', inicio_objetivo, fin_objetivo)
ds_19 = cargar_corrida_monan(path_monan + '2023071900/', inicio_objetivo, fin_objetivo)
ds_20 = cargar_corrida_monan(path_monan + '2023072000/', inicio_objetivo, fin_objetivo)
ds_21 = cargar_corrida_monan(path_monan + '2023072100/', inicio_objetivo, fin_objetivo)

monan_data = []
for ds in [ds_18, ds_19, ds_20, ds_21]:
    if ds is not None:
        monan_data.append(procesar_corrida_monan(ds))
    else:
        monan_data.append(None)

monan_data_forecast1 = monan_data[0]
monan_data_forecast2 = monan_data[1]
monan_data_forecast3 = monan_data[2]
monan_data_forecast4 = monan_data[3]

# ==============================
# 4. FUNCIÓN PARA LÍMITES DINÁMICOS
# ==============================
def calcular_limites(obs_series, modelo_series=None, margen=0.1, min_margin=1.0, max_abs_value=100):
    todos_datos = obs_series.dropna()
    if modelo_series is not None:
        todos_datos = pd.concat([todos_datos, modelo_series.dropna()])
    
    if todos_datos.empty:
        return (0, 1)
    
    todos_datos = todos_datos[np.isfinite(todos_datos)]
    todos_datos = todos_datos[(todos_datos >= -max_abs_value) & (todos_datos <= max_abs_value)]
    
    if len(todos_datos) == 0:
        return (0, 1)
    
    vmin, vmax = todos_datos.min(), todos_datos.max()
    rango = vmax - vmin
    margen_abs = max(rango * margen, min_margin)
    return (vmin - margen_abs, vmax + margen_abs)

# ==============================
# 5. INICIALIZAR MATRICES
# ==============================
n_corridas = 4
n_vars = 4

rmse_smn = np.full((n_corridas, n_vars, len(estaciones_smn)), np.nan)
bias_smn = np.full((n_corridas, n_vars, len(estaciones_smn)), np.nan)
corr_smn = np.full((n_corridas, n_vars, len(estaciones_smn)), np.nan)

rmse_ianigla = np.full((n_corridas, n_vars, len(estaciones_ianigla)), np.nan)
bias_ianigla = np.full((n_corridas, n_vars, len(estaciones_ianigla)), np.nan)
corr_ianigla = np.full((n_corridas, n_vars, len(estaciones_ianigla)), np.nan)

rmse_ceaza = np.full((n_corridas, n_vars, len(estaciones_ceaza)), np.nan)
bias_ceaza = np.full((n_corridas, n_vars, len(estaciones_ceaza)), np.nan)
corr_ceaza = np.full((n_corridas, n_vars, len(estaciones_ceaza)), np.nan)

rmse_meteochile = np.full((n_corridas, n_vars, len(estaciones_meteochile)), np.nan)
bias_meteochile = np.full((n_corridas, n_vars, len(estaciones_meteochile)), np.nan)
corr_meteochile = np.full((n_corridas, n_vars, len(estaciones_meteochile)), np.nan)

# ==============================
# 6. CREAR CARPETA DE SALIDA
# ==============================
#os.makedirs('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/', exist_ok=True)
os.makedirs('figuras_validacion/', exist_ok=True)

# ==============================
# 7. PROCESAR ESTACIONES SMN
# ==============================
for i in range(len(estaciones_smn)):
    print(f"\nSMN: {estaciones_smn['Estacion'][i]}")
    try:
        lat_e = estaciones_smn['Latitud'][i]
        lon_e = estaciones_smn['Longitud'][i]

        df = eval(estaciones_smn['Estacion'][i]).copy()
        df['FECHAUTC'] = pd.to_datetime(df['FECHAUTC'])
        mask_obs = (df['FECHAUTC'] >= inicio_objetivo) & (df['FECHAUTC'] <= fin_objetivo)
        df = df.loc[mask_obs].reset_index(drop=True)
        if df.empty:
            continue
        df = df.set_index('FECHAUTC')
        df_plot = df.copy()

        df_temp = df.filter(regex='TEMPERATURA')
        df_humedad = df.filter(regex='HR')
        df_velocidad = df[['INTENSIDAD']] if 'INTENSIDAD' in df.columns else pd.DataFrame()
        df_direccion = df[['DIRECCION']] if 'DIRECCION' in df.columns else pd.DataFrame()
        df_rocio = df.filter(regex='TD')

        temp_obs = df_temp.iloc[:, 0] if df_temp.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        hum_obs = df_humedad.iloc[:, 0] if df_humedad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        vel_obs = df_velocidad.iloc[:, 0] * 0.514444 if df_velocidad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        dir_obs = df_direccion.iloc[:, 0] if df_direccion.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)

        if df_rocio.shape[1] == 1:
            dew_obs = df_rocio.iloc[:, 0]
        else:
            dew_vals = mpcalc.dewpoint_from_relative_humidity(
                temp_obs.values * units('degC'),
                hum_obs.values * units('%')
            ).to('degC').magnitude
            dew_obs = pd.Series(dew_vals, index=df_plot.index)

        corridas = [
            (monan_data_forecast1, 'MONAN 72h'),
            (monan_data_forecast2, 'MONAN 57h'),
            (monan_data_forecast3, 'MONAN 42h'),
            (monan_data_forecast4, 'MONAN 27h')
        ]
        corridas_punto = []
        for corrida, nombre in corridas:
            if corrida is not None:
                punto = {
                    't2m': corrida['t2m'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'dew2': corrida['dew2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'rh2': corrida['rh2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_speed': corrida['wind_speed'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_dir': corrida['wind_dir'].sel(latitude=lat_e, longitude=lon_e, method='nearest')
                }
                corridas_punto.append((punto, nombre))
            else:
                corridas_punto.append((None, nombre))

        fig, ax = plt.subplots(5, sharex=True, figsize=(12, 10))
        for punto, label in corridas_punto:
            if punto is not None:
                ax[0].plot(punto['t2m'].Time, punto['t2m'], alpha=0.8, linewidth=2, label=label)
        ax[0].plot(df_plot.index, temp_obs, color='black', linewidth=2.5, label='Estación', alpha=0.9)
        temp_modelos = [p['t2m'].values for p, _ in corridas_punto if p is not None]
        ylim_t = calcular_limites(temp_obs, pd.Series(np.concatenate(temp_modelos)) if temp_modelos else None)
        ax[0].set_ylabel('T (°C)'); ax[0].set_ylim(ylim_t); ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].legend(loc='upper left', fontsize=9)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[1].plot(punto['dew2'].Time, punto['dew2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[1].plot(df_plot.index, dew_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        td_modelos = [p['dew2'].values for p, _ in corridas_punto if p is not None]
        ylim_td = calcular_limites(dew_obs, pd.Series(np.concatenate(td_modelos)) if td_modelos else None)
        ax[1].set_ylabel('Td (°C)'); ax[1].set_ylim(ylim_td); ax[1].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[2].plot(punto['rh2'].Time, punto['rh2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[2].plot(df_plot.index, hum_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ax[2].set_ylabel('HR (%)'); ax[2].set_ylim(-5, 105); ax[2].set_yticks(np.arange(0,101,20)); ax[2].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[3].plot(punto['wind_speed'].Time, punto['wind_speed'], alpha=0.8, linewidth=2, label='_nolegend_')
        vel_obs_clean = vel_obs[(np.isfinite(vel_obs)) & (vel_obs >= 0) & (vel_obs <= 100)]
        ax[3].plot(df_plot.index, vel_obs_clean, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ws_modelos = [p['wind_speed'].values for p, _ in corridas_punto if p is not None]
        ylim_ws = calcular_limites(vel_obs_clean, pd.Series(np.concatenate(ws_modelos)) if ws_modelos else None)
        ax[3].set_ylabel('Viento (m/s)'); ax[3].set_ylim(max(0, ylim_ws[0]), ylim_ws[1]); ax[3].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[4].scatter(punto['wind_dir'].Time, punto['wind_dir'], alpha=0.7, s=30)
        ax[4].scatter(df_plot.index, dir_obs, color='black', alpha=0.8, s=30)
        ax[4].set_ylabel('Dirección (°)'); ax[4].set_ylim(0, 360); ax[4].set_yticks([0,90,180,270,360]); ax[4].grid(True, linestyle='--', alpha=0.5)

        ax[4].set_xlabel('Fecha')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f"MONAN vs SMN - {estaciones_smn['Estacion'][i]}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(f"D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/meteograma_MONAN_{estaciones_smn['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"figuras_validacion/meteograma_MONAN_{estaciones_smn['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        plt.close()

        for idx, (corrida, _) in enumerate(corridas):
            if corrida is None:
                continue
            punto = {k: v.sel(latitude=lat_e, longitude=lon_e, method='nearest') for k, v in corrida.items() if k != 'Time'}
            common = df_plot.index.intersection(punto['t2m'].Time)
            if len(common) == 0:
                continue

            obs_t = temp_obs.loc[common].values
            obs_td = dew_obs.loc[common].values
            obs_rh = hum_obs.loc[common].values
            obs_ws = vel_obs_clean.loc[common].values

            mod_t = punto['t2m'].sel(Time=common).values
            mod_td = punto['dew2'].sel(Time=common).values
            mod_rh = punto['rh2'].sel(Time=common).values
            mod_ws = punto['wind_speed'].sel(Time=common).values

            rmse_smn[idx, 0, i] = np.sqrt(np.mean((mod_t - obs_t) ** 2))
            rmse_smn[idx, 1, i] = np.sqrt(np.mean((mod_td - obs_td) ** 2))
            rmse_smn[idx, 2, i] = np.sqrt(np.mean((mod_rh - obs_rh) ** 2))
            rmse_smn[idx, 3, i] = np.sqrt(np.mean((mod_ws - obs_ws) ** 2))

            bias_smn[idx, 0, i] = np.mean(mod_t - obs_t)
            bias_smn[idx, 1, i] = np.mean(mod_td - obs_td)
            bias_smn[idx, 2, i] = np.mean(mod_rh - obs_rh)
            bias_smn[idx, 3, i] = np.mean(mod_ws - obs_ws)

            corr_smn[idx, 0, i] = np.corrcoef(mod_t, obs_t)[0, 1]
            corr_smn[idx, 1, i] = np.corrcoef(mod_td, obs_td)[0, 1]
            corr_smn[idx, 2, i] = np.corrcoef(mod_rh, obs_rh)[0, 1]
            corr_smn[idx, 3, i] = np.corrcoef(mod_ws, obs_ws)[0, 1]

    except Exception as e:
        print(f"  ❌ Error en SMN {estaciones_smn['Estacion'][i]}: {e}")

# ==============================
# 8. PROCESAR ESTACIONES IANIGLA
# ==============================
for i in range(len(estaciones_ianigla)):
    print(f"\nIANIGLA: {estaciones_ianigla['Estacion'][i]}")
    try:
        lat_e = estaciones_ianigla['Latitud'][i]
        lon_e = estaciones_ianigla['Longitud'][i]

#        df = pd.read_excel(f"D:/datos_IANIGLA/Est_{estaciones_ianigla['Estacion'][i]}.xlsx")
        df = pd.read_excel(path_datos + f"datos_IANIGLA/Est_{estaciones_ianigla['Estacion'][i]}.xlsx")
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')
        mask_obs = (df['date'] >= inicio_objetivo) & (df['date'] <= fin_objetivo)
        df = df.loc[mask_obs].reset_index(drop=True)
        if df.empty:
            continue
        df = df.set_index('date')
        df_plot = df.copy()

        df_temp = df.filter(regex='Temp_Aire')
        df_humedad = df.filter(regex='H_relativa')
        df_velocidad = df.filter(regex='Vel_Viento')
        df_direccion = df.filter(regex='Dir_Viento')
        df_rocio = df.filter(regex='Punto')

        temp_obs = df_temp.iloc[:, 0] if df_temp.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        hum_obs = df_humedad.iloc[:, 0] if df_humedad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        vel_obs = df_velocidad.iloc[:, 0] * 0.277778 if df_velocidad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        dir_obs = df_direccion.iloc[:, 0] if df_direccion.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)

        if df_rocio.shape[1] == 1:
            dew_obs = df_rocio.iloc[:, 0]
        else:
            dew_vals = mpcalc.dewpoint_from_relative_humidity(
                temp_obs.values * units('degC'),
                hum_obs.values * units('%')
            ).to('degC').magnitude
            dew_obs = pd.Series(dew_vals, index=df_plot.index)

        corridas = [
            (monan_data_forecast1, 'MONAN 72h'),
            (monan_data_forecast2, 'MONAN 57h'),
            (monan_data_forecast3, 'MONAN 42h'),
            (monan_data_forecast4, 'MONAN 27h')
        ]
        corridas_punto = []
        for corrida, nombre in corridas:
            if corrida is not None:
                punto = {
                    't2m': corrida['t2m'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'dew2': corrida['dew2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'rh2': corrida['rh2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_speed': corrida['wind_speed'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_dir': corrida['wind_dir'].sel(latitude=lat_e, longitude=lon_e, method='nearest')
                }
                corridas_punto.append((punto, nombre))
            else:
                corridas_punto.append((None, nombre))

        fig, ax = plt.subplots(5, sharex=True, figsize=(12, 10))
        for punto, label in corridas_punto:
            if punto is not None:
                ax[0].plot(punto['t2m'].Time, punto['t2m'], alpha=0.8, linewidth=2, label=label)
        ax[0].plot(df_plot.index, temp_obs, color='black', linewidth=2.5, label='Estación', alpha=0.9)
        temp_modelos = [p['t2m'].values for p, _ in corridas_punto if p is not None]
        ylim_t = calcular_limites(temp_obs, pd.Series(np.concatenate(temp_modelos)) if temp_modelos else None)
        ax[0].set_ylabel('T (°C)'); ax[0].set_ylim(ylim_t); ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].legend(loc='upper left', fontsize=9)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[1].plot(punto['dew2'].Time, punto['dew2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[1].plot(df_plot.index, dew_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        td_modelos = [p['dew2'].values for p, _ in corridas_punto if p is not None]
        ylim_td = calcular_limites(dew_obs, pd.Series(np.concatenate(td_modelos)) if td_modelos else None)
        ax[1].set_ylabel('Td (°C)'); ax[1].set_ylim(ylim_td); ax[1].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[2].plot(punto['rh2'].Time, punto['rh2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[2].plot(df_plot.index, hum_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ax[2].set_ylabel('HR (%)'); ax[2].set_ylim(-5, 105); ax[2].set_yticks(np.arange(0,101,20)); ax[2].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[3].plot(punto['wind_speed'].Time, punto['wind_speed'], alpha=0.8, linewidth=2, label='_nolegend_')
        vel_obs_clean = vel_obs[(np.isfinite(vel_obs)) & (vel_obs >= 0) & (vel_obs <= 100)]
        ax[3].plot(df_plot.index, vel_obs_clean, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ws_modelos = [p['wind_speed'].values for p, _ in corridas_punto if p is not None]
        ylim_ws = calcular_limites(vel_obs_clean, pd.Series(np.concatenate(ws_modelos)) if ws_modelos else None)
        ax[3].set_ylabel('Viento (m/s)'); ax[3].set_ylim(max(0, ylim_ws[0]), ylim_ws[1]); ax[3].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[4].scatter(punto['wind_dir'].Time, punto['wind_dir'], alpha=0.7, s=30)
        ax[4].scatter(df_plot.index, dir_obs, color='black', alpha=0.8, s=30)
        ax[4].set_ylabel('Dirección (°)'); ax[4].set_ylim(0, 360); ax[4].set_yticks([0,90,180,270,360]); ax[4].grid(True, linestyle='--', alpha=0.5)

        ax[4].set_xlabel('Fecha')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f"MONAN vs IANIGLA - {estaciones_ianigla['Estacion'][i]}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(f"D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/meteograma_MONAN_{estaciones_ianigla['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"figuras_validacion/meteograma_MONAN_{estaciones_ianigla['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        # plt.close()

        for idx, (corrida, _) in enumerate(corridas):
            if corrida is None:
                continue
            punto = {k: v.sel(latitude=lat_e, longitude=lon_e, method='nearest') for k, v in corrida.items() if k != 'Time'}
            common = df_plot.index.intersection(punto['t2m'].Time)
            if len(common) == 0:
                continue

            obs_t = temp_obs.loc[common].values
            obs_td = dew_obs.loc[common].values
            obs_rh = hum_obs.loc[common].values
            obs_ws = vel_obs_clean.loc[common].values

            mod_t = punto['t2m'].sel(Time=common).values
            mod_td = punto['dew2'].sel(Time=common).values
            mod_rh = punto['rh2'].sel(Time=common).values
            mod_ws = punto['wind_speed'].sel(Time=common).values

            rmse_ianigla[idx, 0, i] = np.sqrt(np.mean((mod_t - obs_t) ** 2))
            rmse_ianigla[idx, 1, i] = np.sqrt(np.mean((mod_td - obs_td) ** 2))
            rmse_ianigla[idx, 2, i] = np.sqrt(np.mean((mod_rh - obs_rh) ** 2))
            rmse_ianigla[idx, 3, i] = np.sqrt(np.mean((mod_ws - obs_ws) ** 2))

            bias_ianigla[idx, 0, i] = np.mean(mod_t - obs_t)
            bias_ianigla[idx, 1, i] = np.mean(mod_td - obs_td)
            bias_ianigla[idx, 2, i] = np.mean(mod_rh - obs_rh)
            bias_ianigla[idx, 3, i] = np.mean(mod_ws - obs_ws)

            corr_ianigla[idx, 0, i] = np.corrcoef(mod_t, obs_t)[0, 1]
            corr_ianigla[idx, 1, i] = np.corrcoef(mod_td, obs_td)[0, 1]
            corr_ianigla[idx, 2, i] = np.corrcoef(mod_rh, obs_rh)[0, 1]
            corr_ianigla[idx, 3, i] = np.corrcoef(mod_ws, obs_ws)[0, 1]

    except Exception as e:
        print(f"  ❌ Error en IANIGLA {estaciones_ianigla['Estacion'][i]}: {e}")

# ==============================
# 9. PROCESAR ESTACIONES CEAZA
# ==============================
for i in range(len(estaciones_ceaza)):
    print(f"\nCEAZA: {estaciones_ceaza['Estacion'][i]}")
    try:
        lat_e = estaciones_ceaza['Latitud'][i]
        lon_e = estaciones_ceaza['Longitud'][i]
        elevacion = estaciones_ceaza['elevacion'][i]

        nombre_est = estaciones_ceaza['Estacion'][i].replace(u'\xa0', u'')
#        df = pd.read_excel(f"D:/datos_CEAZA/20210818/{nombre_est}.xls", sheet_name='21072023')
#        print(path_datos + f"datos_CEAZA/20210818/{nombre_est}.xls")
        df = pd.read_excel(path_datos + f"datos_CEAZA/20210818/{nombre_est}.xls", sheet_name='21072023')
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
        mask_obs = (df['fecha'] >= inicio_objetivo) & (df['fecha'] <= fin_objetivo)
        df = df.loc[mask_obs].reset_index(drop=True)
        if df.empty:
            continue
        df = df.set_index('fecha')
        df_plot = df.copy()

        df2 = df.filter(regex='Prom')
        df_temp = df2.filter(regex='Temperatura')
        df_humedad = df2.filter(regex='Humedad')
        df_velocidad = df2.filter(regex='Velocidad')
        df_direccion = df2.filter(regex='Dir')
        df_rocio = df2.filter(regex='Punto')

        temp_obs = df_temp.iloc[:, 0] if df_temp.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        hum_obs = df_humedad.iloc[:, 0] if df_humedad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        dir_obs = df_direccion.iloc[:, 0] if df_direccion.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)

        vel_obs = pd.Series(np.nan, index=df_plot.index)
        if not df_velocidad.empty:
            col_vel = df_velocidad.columns[0]
            u_meas = df_velocidad[col_vel]
            if '[5m]' in col_vel:
                z0 = 0.03
                u10_kmh = u_meas * (np.log(10 / z0) / np.log(5 / z0))
            else:
                u10_kmh = u_meas
            vel_obs = u10_kmh * 0.277778

        if df_rocio.shape[1] == 1:
            dew_obs = df_rocio.iloc[:, 0]
        else:
            dew_vals = mpcalc.dewpoint_from_relative_humidity(
                temp_obs.values * units('degC'),
                hum_obs.values * units('%')
            ).to('degC').magnitude
            dew_obs = pd.Series(dew_vals, index=df_plot.index)

        corridas = [
            (monan_data_forecast1, 'MONAN 72h'),
            (monan_data_forecast2, 'MONAN 57h'),
            (monan_data_forecast3, 'MONAN 42h'),
            (monan_data_forecast4, 'MONAN 27h')
        ]
        corridas_punto = []
        for corrida, nombre in corridas:
            if corrida is not None:
                punto = {
                    't2m': corrida['t2m'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'dew2': corrida['dew2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'rh2': corrida['rh2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_speed': corrida['wind_speed'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_dir': corrida['wind_dir'].sel(latitude=lat_e, longitude=lon_e, method='nearest')
                }
                corridas_punto.append((punto, nombre))
            else:
                corridas_punto.append((None, nombre))

        fig, ax = plt.subplots(5, sharex=True, figsize=(12, 10))
        for punto, label in corridas_punto:
            if punto is not None:
                ax[0].plot(punto['t2m'].Time, punto['t2m'], alpha=0.8, linewidth=2, label=label)
        ax[0].plot(df_plot.index, temp_obs, color='black', linewidth=2.5, label='Estación', alpha=0.9)
        temp_modelos = [p['t2m'].values for p, _ in corridas_punto if p is not None]
        ylim_t = calcular_limites(temp_obs, pd.Series(np.concatenate(temp_modelos)) if temp_modelos else None)
        ax[0].set_ylabel('T (°C)'); ax[0].set_ylim(ylim_t); ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].legend(loc='upper left', fontsize=9)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[1].plot(punto['dew2'].Time, punto['dew2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[1].plot(df_plot.index, dew_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        td_modelos = [p['dew2'].values for p, _ in corridas_punto if p is not None]
        ylim_td = calcular_limites(dew_obs, pd.Series(np.concatenate(td_modelos)) if td_modelos else None)
        ax[1].set_ylabel('Td (°C)'); ax[1].set_ylim(ylim_td); ax[1].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[2].plot(punto['rh2'].Time, punto['rh2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[2].plot(df_plot.index, hum_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ax[2].set_ylabel('HR (%)'); ax[2].set_ylim(-5, 105); ax[2].set_yticks(np.arange(0,101,20)); ax[2].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[3].plot(punto['wind_speed'].Time, punto['wind_speed'], alpha=0.8, linewidth=2, label='_nolegend_')
        vel_obs_clean = vel_obs[(np.isfinite(vel_obs)) & (vel_obs >= 0) & (vel_obs <= 100)]
        ax[3].plot(df_plot.index, vel_obs_clean, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ws_modelos = [p['wind_speed'].values for p, _ in corridas_punto if p is not None]
        ylim_ws = calcular_limites(vel_obs_clean, pd.Series(np.concatenate(ws_modelos)) if ws_modelos else None)
        ax[3].set_ylabel('Viento (m/s)'); ax[3].set_ylim(max(0, ylim_ws[0]), ylim_ws[1]); ax[3].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[4].scatter(punto['wind_dir'].Time, punto['wind_dir'], alpha=0.7, s=30)
        ax[4].scatter(df_plot.index, dir_obs, color='black', alpha=0.8, s=30)
        ax[4].set_ylabel('Dirección (°)'); ax[4].set_ylim(0, 360); ax[4].set_yticks([0,90,180,270,360]); ax[4].grid(True, linestyle='--', alpha=0.5)

        ax[4].set_xlabel('Fecha')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f"MONAN vs CEAZA - {nombre_est}", fontsize=14)
        fig.text(0.95, 0.95, f"{elevacion} m", fontsize=12, ha='right', va='top')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(f"D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/meteograma_MONAN_{nombre_est}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"figuras_validacion/meteograma_MONAN_{nombre_est}.png", dpi=150, bbox_inches='tight')
        # plt.close()

        for idx, (corrida, _) in enumerate(corridas):
            if corrida is None:
                continue
            punto = {k: v.sel(latitude=lat_e, longitude=lon_e, method='nearest') for k, v in corrida.items() if k != 'Time'}
            common = df_plot.index.intersection(punto['t2m'].Time)
            if len(common) == 0:
                continue

            obs_t = temp_obs.loc[common].values
            obs_td = dew_obs.loc[common].values
            obs_rh = hum_obs.loc[common].values
            obs_ws = vel_obs_clean.loc[common].values

            mod_t = punto['t2m'].sel(Time=common).values
            mod_td = punto['dew2'].sel(Time=common).values
            mod_rh = punto['rh2'].sel(Time=common).values
            mod_ws = punto['wind_speed'].sel(Time=common).values

           # rmceaza[idx, 0, i] = np.sqrt(np.mean((mod_t - obs_t) ** 2))  # ← FIX: era rmceaza, debe ser rmse_ceaza
            rmse_ceaza[idx, 0, i] = np.sqrt(np.mean((mod_t - obs_t) ** 2))  # ← FIX: era rmceaza, debe ser rmse_ceaza
            rmse_ceaza[idx, 1, i] = np.sqrt(np.mean((mod_td - obs_td) ** 2))
            rmse_ceaza[idx, 2, i] = np.sqrt(np.mean((mod_rh - obs_rh) ** 2))
            rmse_ceaza[idx, 3, i] = np.sqrt(np.mean((mod_ws - obs_ws) ** 2))

            bias_ceaza[idx, 0, i] = np.mean(mod_t - obs_t)
            bias_ceaza[idx, 1, i] = np.mean(mod_td - obs_td)
            bias_ceaza[idx, 2, i] = np.mean(mod_rh - obs_rh)
            bias_ceaza[idx, 3, i] = np.mean(mod_ws - obs_ws)

            corr_ceaza[idx, 0, i] = np.corrcoef(mod_t, obs_t)[0, 1]
            corr_ceaza[idx, 1, i] = np.corrcoef(mod_td, obs_td)[0, 1]
            corr_ceaza[idx, 2, i] = np.corrcoef(mod_rh, obs_rh)[0, 1]
            corr_ceaza[idx, 3, i] = np.corrcoef(mod_ws, obs_ws)[0, 1]

    except Exception as e:
        print(f"  ❌ Error en CEAZA {estaciones_ceaza['Estacion'][i]}: {e}")

# ==============================
# 10. PROCESAR ESTACIONES METEOCHILE
# ==============================
for i in range(len(estaciones_meteochile)):
    print(f"\nMETEOCHILE: {estaciones_meteochile['Estacion'][i]}")
    try:
        lat_e = estaciones_meteochile['Latitud'][i]
        lon_e = estaciones_meteochile['Longitud'][i]
        elevacion = estaciones_meteochile['elevacion'][i]
#        print("\n" + path_datos + f"datos_METEOCHILE/{estaciones_meteochile['Estacion'][i]}.xlsx")
        df = pd.read_excel(
#            f"D:/datos_METEOCHILE/{estaciones_meteochile['Estacion'][i]}.xlsx",
            path_datos + f"datos_METEOCHILE/MONAN/{estaciones_meteochile['Estacion'][i]}.xlsx",
            sheet_name='21072023'
        )
        df['Momento UTC'] = pd.to_datetime(df['Momento UTC'], format='%d/%m/%Y %H:%M:%S')
        mask_obs = (df['Momento UTC'] >= inicio_objetivo) & (df['Momento UTC'] <= fin_objetivo)
        df = df.loc[mask_obs].reset_index(drop=True)
        if df.empty:
            continue
        df = df.set_index('Momento UTC')
        df_plot = df.copy()

        df_temp = df.filter(regex='Temperatura')
        df_humedad = df.filter(regex='Humedad')
        df_velocidad = df.filter(regex='Viento')
        df_direccion = df.filter(regex='Dir')
        df_rocio = df.filter(regex='Punto')

        temp_obs = df_temp.iloc[:, 0] if df_temp.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        hum_obs = df_humedad.iloc[:, 0] if df_humedad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        dir_obs = df_direccion.iloc[:, 0] if df_direccion.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)
        vel_obs = df_velocidad.iloc[:, 0] * 0.277778 if df_velocidad.shape[1] > 0 else pd.Series(np.nan, index=df_plot.index)

        if df_rocio.shape[1] == 1:
            dew_obs = df_rocio.iloc[:, 0]
        else:
            dew_vals = mpcalc.dewpoint_from_relative_humidity(
                temp_obs.values * units('degC'),
                hum_obs.values * units('%')
            ).to('degC').magnitude
            dew_obs = pd.Series(dew_vals, index=df_plot.index)

        corridas = [
            (monan_data_forecast1, 'MONAN 72h'),
            (monan_data_forecast2, 'MONAN 57h'),
            (monan_data_forecast3, 'MONAN 42h'),
            (monan_data_forecast4, 'MONAN 27h')
        ]
        corridas_punto = []
        for corrida, nombre in corridas:
            if corrida is not None:
                punto = {
                    't2m': corrida['t2m'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'dew2': corrida['dew2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'rh2': corrida['rh2'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_speed': corrida['wind_speed'].sel(latitude=lat_e, longitude=lon_e, method='nearest'),
                    'wind_dir': corrida['wind_dir'].sel(latitude=lat_e, longitude=lon_e, method='nearest')
                }
                corridas_punto.append((punto, nombre))
            else:
                corridas_punto.append((None, nombre))

        fig, ax = plt.subplots(5, sharex=True, figsize=(12, 10))
        for punto, label in corridas_punto:
            if punto is not None:
                ax[0].plot(punto['t2m'].Time, punto['t2m'], alpha=0.8, linewidth=2, label=label)
        ax[0].plot(df_plot.index, temp_obs, color='black', linewidth=2.5, label='Estación', alpha=0.9)
        temp_modelos = [p['t2m'].values for p, _ in corridas_punto if p is not None]
        ylim_t = calcular_limites(temp_obs, pd.Series(np.concatenate(temp_modelos)) if temp_modelos else None)
        ax[0].set_ylabel('T (°C)'); ax[0].set_ylim(ylim_t); ax[0].grid(True, linestyle='--', alpha=0.5)
        ax[0].legend(loc='upper left', fontsize=9)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[1].plot(punto['dew2'].Time, punto['dew2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[1].plot(df_plot.index, dew_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        td_modelos = [p['dew2'].values for p, _ in corridas_punto if p is not None]
        ylim_td = calcular_limites(dew_obs, pd.Series(np.concatenate(td_modelos)) if td_modelos else None)
        ax[1].set_ylabel('Td (°C)'); ax[1].set_ylim(ylim_td); ax[1].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[2].plot(punto['rh2'].Time, punto['rh2'], alpha=0.8, linewidth=2, label='_nolegend_')
        ax[2].plot(df_plot.index, hum_obs, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ax[2].set_ylabel('HR (%)'); ax[2].set_ylim(-5, 105); ax[2].set_yticks(np.arange(0,101,20)); ax[2].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[3].plot(punto['wind_speed'].Time, punto['wind_speed'], alpha=0.8, linewidth=2, label='_nolegend_')
        vel_obs_clean = vel_obs[(np.isfinite(vel_obs)) & (vel_obs >= 0) & (vel_obs <= 100)]
        ax[3].plot(df_plot.index, vel_obs_clean, color='black', linewidth=2.5, label='_nolegend_', alpha=0.9)
        ws_modelos = [p['wind_speed'].values for p, _ in corridas_punto if p is not None]
        ylim_ws = calcular_limites(vel_obs_clean, pd.Series(np.concatenate(ws_modelos)) if ws_modelos else None)
        ax[3].set_ylabel('Viento (m/s)'); ax[3].set_ylim(max(0, ylim_ws[0]), ylim_ws[1]); ax[3].grid(True, linestyle='--', alpha=0.5)

        for punto, label in corridas_punto:
            if punto is not None:
                ax[4].scatter(punto['wind_dir'].Time, punto['wind_dir'], alpha=0.7, s=30)
        ax[4].scatter(df_plot.index, dir_obs, color='black', alpha=0.8, s=30)
        ax[4].set_ylabel('Dirección (°)'); ax[4].set_ylim(0, 360); ax[4].set_yticks([0,90,180,270,360]); ax[4].grid(True, linestyle='--', alpha=0.5)

        ax[4].set_xlabel('Fecha')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f"MONAN vs METEOCHILE - {estaciones_meteochile['Estacion'][i]}", fontsize=14)
        fig.text(0.95, 0.95, f"{elevacion} m", fontsize=12, ha='right', va='top')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(f"D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/meteograma_MONAN_{estaciones_meteochile['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"figuras_validacion/meteograma_MONAN_{estaciones_meteochile['Estacion'][i]}.png", dpi=150, bbox_inches='tight')
        # plt.close()

        for idx, (corrida, _) in enumerate(corridas):
            if corrida is None:
                continue
            punto = {k: v.sel(latitude=lat_e, longitude=lon_e, method='nearest') for k, v in corrida.items() if k != 'Time'}
            common = df_plot.index.intersection(punto['t2m'].Time)
            if len(common) == 0:
                continue

            obs_t = temp_obs.loc[common].values
            obs_td = dew_obs.loc[common].values
            obs_rh = hum_obs.loc[common].values
            obs_ws = vel_obs_clean.loc[common].values

            mod_t = punto['t2m'].sel(Time=common).values
            mod_td = punto['dew2'].sel(Time=common).values
            mod_rh = punto['rh2'].sel(Time=common).values
            mod_ws = punto['wind_speed'].sel(Time=common).values

            rmse_meteochile[idx, 0, i] = np.sqrt(np.mean((mod_t - obs_t) ** 2))
            rmse_meteochile[idx, 1, i] = np.sqrt(np.mean((mod_td - obs_td) ** 2))
            rmse_meteochile[idx, 2, i] = np.sqrt(np.mean((mod_rh - obs_rh) ** 2))
            rmse_meteochile[idx, 3, i] = np.sqrt(np.mean((mod_ws - obs_ws) ** 2))

            bias_meteochile[idx, 0, i] = np.mean(mod_t - obs_t)
            bias_meteochile[idx, 1, i] = np.mean(mod_td - obs_td)
            bias_meteochile[idx, 2, i] = np.mean(mod_rh - obs_rh)
            bias_meteochile[idx, 3, i] = np.mean(mod_ws - obs_ws)

            corr_meteochile[idx, 0, i] = np.corrcoef(mod_t, obs_t)[0, 1]
            corr_meteochile[idx, 1, i] = np.corrcoef(mod_td, obs_td)[0, 1]
            corr_meteochile[idx, 2, i] = np.corrcoef(mod_rh, obs_rh)[0, 1]
            corr_meteochile[idx, 3, i] = np.corrcoef(mod_ws, obs_ws)[0, 1]

    except Exception as e:
        print(f"  ❌ Error en METEOCHILE {estaciones_meteochile['Estacion'][i]}: {e}")

# ==============================
# 11. GUARDAR RESULTADOS
# ==============================
print("\n✅ Validación completada. Guardando métricas...")

# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/rmse_smn.npy', rmse_smn)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/bias_smn.npy', bias_smn)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/corr_smn.npy', corr_smn)

# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/rmse_ianigla.npy', rmse_ianigla)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/bias_ianigla.npy', bias_ianigla)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/corr_ianigla.npy', corr_ianigla)

# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/rmse_ceaza.npy', rmse_ceaza)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/bias_ceaza.npy', bias_ceaza)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/corr_ceaza.npy', corr_ceaza)

# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/rmse_meteochile.npy', rmse_meteochile)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/bias_meteochile.npy', bias_meteochile)
# np.save('D:/disco_EFI_WRF/datos_MONAN/figuras_validacion/corr_meteochile.npy', corr_meteochile)

# print("💾 Métricas guardadas en .npy")
