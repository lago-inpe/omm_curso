# -*- coding: utf-8 -*-
"""
diagnostics_compare_monan_obs.py

Compacto: descarga radiosondeo WYOMING (o usa local) + extrae perfil MONAN (NetCDF)
y calcula/trae:
 - theta, N, N2
 - scorer parameter l^2 (usando U)
 - Richardson number (Ri) usando dU/dz
 - Froude (varias versiones)
Genera:
 - Skew-T comparativo (obs vs MONAN)
 - Perfiles superpuestos (U, T, theta, N, l2, Ri, Fr)
 - Bias (T, Td, wind)
Salida: carpeta outdir con PNGs y CSV

Requisitos: numpy, pandas, xarray, metpy, siphon, matplotlib, scipy

Autor: Adaptado para tu uso
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
import xarray as xr
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from scipy.ndimage import gaussian_filter1d

# ---------------------------
# CONFIG (ajustá aquí)
# ---------------------------
station = "SCSN"                   # Wyoming station code (string)
wy_date = datetime(2023, 7, 21, 12)  # fecha/hora del sondeo (UTC)
# MONAN file (poné tu ruta local al archivo .nc)
#monan_file = r"D:/disco_EFI_WRF/datos_MONAN/nuevo_recorte/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_2023072112.00.00.x1.65536002L55_mendoza.nc"
#monan_file= r'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_20230721200.mm.x20.835586L55.nc'
monan_file= 'https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/2023071800/MONAN_DIAG_R_POS_GFS_2023071800_2023072112.00.00.x1.65536002L55.nc#mode=bytes'
# coordenadas donde extraer MONAN (cercanas al radiosondeo)
sounding_lat = -33.65
sounding_lon = -71.61

outdir = "foehn_diag_compare"
os.makedirs(outdir, exist_ok=True)

# física
g = 9.81
R_div_cp = 0.286
epsilon = 0.622

# ---------------------------
# FUNCIONES FÍSICAS
# ---------------------------
def calc_theta(T_K, p_hPa):
    return T_K * (1000.0 / p_hPa) ** R_div_cp

def brunt_vaisala(theta, z):
    # theta (K), z (m)
    dtheta_dz = np.gradient(theta, z)
    N2 = (g / theta) * dtheta_dz
    N = np.sqrt(np.maximum(N2, 0.0))
    return N, N2, dtheta_dz

def scorer_parameter(N2, U, z, vert_smooth_sigma=1.0):
    # U (m/s), z (m)
    U_safe = U.copy()
    # fill nan edges
    mask = np.isfinite(U_safe)
    if mask.sum() < 3:
        return np.full_like(U_safe, np.nan)
    if not mask.all():
        first = np.argmax(mask)
        last = len(mask) - np.argmax(mask[::-1]) - 1
        U_safe[:first] = U_safe[first]
        U_safe[last+1:] = U_safe[last]
    U_smooth = gaussian_filter1d(U_safe, sigma=vert_smooth_sigma, mode='nearest')
    d2U = np.gradient(np.gradient(U_smooth, z), z)
    U_nonzero = np.where(np.abs(U_smooth) < 1e-6, 1e-6, U_smooth)
    l2 = (N2 / (U_nonzero**2)) - (d2U / U_nonzero)
    return l2

def richardson_number(theta, u, v, z):
    dtheta_dz = np.gradient(theta, z)
    du_dz = np.gradient(u, z)
    dv_dz = np.gradient(v, z)
    shear2 = du_dz**2 + dv_dz**2 + 1e-6
    Ri = (g / theta) * dtheta_dz / shear2
    return Ri

def froude_local(Uz, N, H_m):
    N_safe = np.where(N < 1e-4, 1e-4, N)
    Fr = Uz / (N_safe * H_m)
    return Fr

# ---------------------------
# LECTURA WYOMING
# ---------------------------
def download_sound_wyoming(station_code, dt):
    print(f"Descargando sondeo WYOMING: {station_code} {dt}")
    df = WyomingUpperAir.request_data(dt, station_code)
    # keep needed columns (pressure hPa, height m, temperature C, dewpoint C, u_wind m/s, v_wind m/s, speed, dir)
    keep = ['pressure', 'height', 'temperature', 'dewpoint', 'u_wind', 'v_wind']
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].dropna(subset=['height'])
    df.rename(columns={'pressure':'p', 'height':'z', 'temperature':'temp', 'dewpoint':'td',
                       'u_wind':'u', 'v_wind':'v'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ---------------------------
# LECTURA MONAN (xarray)
# ---------------------------
def read_monan_profile(monan_path, lat, lon):
    print("Leyendo MONAN:", monan_path)
    ds = xr.open_dataset(monan_path)
    # intenta parse CF si disponible
    try:
        ds = ds.metpy.parse_cf()
    except Exception:
        pass
    # localizar punto más cercano
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    # intentar extraer variables comunes (nombres que viste en tus scripts)
    # level (Pa), temperature(K), spechum (kg/kg), zgeo (m), uzonal (m/s), umeridional (m/s)
    def squeeze(x):
        return np.squeeze(np.asarray(x))
    # algunos datasets usan 'level' en Pa
    if 'level' in ds_point:
        pres = squeeze(ds_point['level'].values) / 100.0  # Pa -> hPa
    elif 'pressure' in ds_point:
        pres = squeeze(ds_point['pressure'].values)
    else:
        raise ValueError("No se encontró variable de nivel en MONAN (level/pressure).")

    # temperature
    if 'temperature' in ds_point:
        temp = squeeze(ds_point['temperature'].values) - 273.15
    elif 't2m' in ds_point:
        temp = squeeze(ds_point['t2m'].values) - 273.15
    else:
        raise ValueError("No se encontró temperatura en MONAN (temperature).")

    # specific humidity
    if 'spechum' in ds_point:
        q = squeeze(ds_point['spechum'].values)
    elif 'q' in ds_point:
        q = squeeze(ds_point['q'].values)
    else:
        q = np.full_like(temp, np.nan)

    # geopotential height or z
    if 'zgeo' in ds_point:
        z = squeeze(ds_point['zgeo'].values)
    elif 'height' in ds_point:
        z = squeeze(ds_point['height'].values)
    else:
        # approximate scale height if not present
        z = np.linspace(0, 20000, len(pres))

    # winds (MONAN may be in m/s)
    if 'uzonal' in ds_point:
        u = squeeze(ds_point['uzonal'].values) * 1.943842816446084
    elif 'u' in ds_point:
        u = squeeze(ds_point['u'].values) * 1.943842816446084
    else:
        u = np.full_like(temp, np.nan) * 1.943842816446084

    if 'umeridional' in ds_point:
        v = squeeze(ds_point['umeridional'].values)
    elif 'v' in ds_point:
        v = np.full_like(u, np.nan)
        v = squeeze(ds_point['v'].values)
    else:
        v = np.full_like(u, np.nan)

    # convert units if necessary (some MONAN were in m/s already, the earlier script had a mistaken *1.94384 -> knots; keep m/s)
    # Build DataFrame
    df = pd.DataFrame({
        'p': pres,
        'z': z,
        'temp': temp,
        'q': q,
        'u': u,
        'v': v
    })
    # compute dewpoint from specific humidity if available
    mask = np.isfinite(df['p']) & np.isfinite(df['q'])
    if mask.any():
        try:
            p_u = (df.loc[mask, 'p'].values * units.hPa)
            q_u = (df.loc[mask, 'q'].values * units('kg/kg'))
            w = mpcalc.mixing_ratio_from_specific_humidity(q_u)  # dimensionless
            e = (w * p_u) / (epsilon + w)  # vapor pressure (hPa)
            td = mpcalc.dewpoint(e)
            td_vals = np.full_like(df['temp'].values, np.nan, dtype=float)
            td_vals[mask] = td.m_as('degC')
            df['td'] = td_vals
        except Exception:
            df['td'] = np.nan
    else:
        df['td'] = np.nan

    # ensure sorting from surface -> top (descending pressure)
    df = df.dropna(subset=['p']).sort_values('p', ascending=False).reset_index(drop=True)
    return df

# ---------------------------
# UTILS: interpolación y alineado
# ---------------------------
def interp_to_common_levels(df_obs, df_mod, kind='linear'):
    # Interpolar model onto obs pressure levels (hPa)
    p_obs = df_obs['p'].values
    out_mod = {}
    for var in ['temp', 'td', 'u', 'v', 'z', 'q']:
        if var in df_mod.columns:
            out_mod[var] = np.interp(p_obs, df_mod['p'].values[::-1], df_mod[var].values[::-1])
        else:
            out_mod[var] = np.full_like(p_obs, np.nan)
    df_mod_interp = pd.DataFrame(out_mod)
    df_mod_interp['p'] = p_obs
    # use obs z for plotting vertical axis (approx)
    df_obs2 = df_obs.copy().reset_index(drop=True)
    df_mod_interp['z'] = df_obs2['z'].values
    return df_obs2, df_mod_interp

# ---------------------------
# MAIN processing (comparativo)
# ---------------------------
def process_compare(df_obs, df_mod, outdir):
    os.makedirs(outdir, exist_ok=True)
    # align levels: interpolate MONAN -> obs pressures
    df_o, df_m = interp_to_common_levels(df_obs, df_mod)

    # prepare arrays (in SI)
    p = df_o['p'].values
    z = df_o['z'].values.astype(float)
    # Observed
    T_o = df_o['temp'].values.astype(float)
    td_o = df_o.get('td', np.full_like(T_o, np.nan)).astype(float)
    u_o = df_o['u'].values.astype(float)
    v_o = df_o['v'].values.astype(float)
    U_o = np.sqrt(u_o**2 + v_o**2)
    # Model (interpolated)
    T_m = df_m['temp'].values.astype(float)
    td_m = df_m.get('td', np.full_like(T_m, np.nan)).astype(float)
    u_m = df_m['u'].values.astype(float)
    v_m = df_m['v'].values.astype(float)
    U_m = np.sqrt(u_m**2 + v_m**2)
    # pressures (hPa) -> convert T to K
    T_o_K = T_o + 273.15
    T_m_K = T_m + 273.15

    # thetas
    theta_o = calc_theta(T_o_K, p)
    theta_m = calc_theta(T_m_K, p)

    # N, N2
    N_o, N2_o, dtheta_o = brunt_vaisala(theta_o, z)
    N_m, N2_m, dtheta_m = brunt_vaisala(theta_m, z)

    # l2 (use U only)
    l2_o = scorer_parameter(N2_o, U_o, z)
    l2_m = scorer_parameter(N2_m, U_m, z)

    # Ri (using U only)
    Ri_o = richardson_number(theta_o, u_o, v_o, z)
    Ri_m = richardson_number(theta_m, u_m, v_m, z)

    # Froude char (use H ~ max(z)-min(z) from obs)
    H = max(1.0, np.nanmax(z) - np.nanmin(z))
    Fr_o = np.zeros_like(z) * np.nan
    Fr_m = np.zeros_like(z) * np.nan
    mask_o = N_o > 0
    mask_m = N_m > 0
    Fr_o[mask_o] = U_o[mask_o] / (N_o[mask_o] * H)
    Fr_m[mask_m] = U_m[mask_m] / (N_m[mask_m] * H)

    # Save diagnostics CSV
    df_out = pd.DataFrame({
        'p_hPa': p, 'z_m': z,
        'T_obs_C': T_o, 'T_monan_C': T_m,
        'td_obs_C': td_o, 'td_monan_C': td_m,
        'U_obs_ms': U_o, 'U_monan_ms': U_m,
        'theta_obs_K': theta_o, 'theta_monan_K': theta_m,
        'N_obs_1s': N_o, 'N_monan_1s': N_m,
        'l2_obs_1_m2': l2_o, 'l2_monan_1_m2': l2_m,
        'Ri_obs': Ri_o, 'Ri_monan': Ri_m,
        'Fr_obs': Fr_o, 'Fr_monan': Fr_m
    })
    df_out.to_csv(os.path.join(outdir, "diagnostics_compare.csv"), index=False)

    # ---------------------------
    # PLOTS
    # ---------------------------
    # 1) Skew-T comparativo
    try:
        fig = plt.figure(figsize=(8, 10))
        skew = SkewT(fig, rotation=45)
        p_o = (p * units.hPa)
        t_o = (T_o * units.degC)
        td_o_u = (td_o * units.degC)
        u_o_u = (u_o * units('m/s'))
        v_o_u = (v_o * units('m/s'))

        p_m = (p * units.hPa)
        t_m = (T_m * units.degC)
        td_m_u = (td_m * units.degC)
        u_m_u = (u_m * units('m/s'))
        v_m_u = (v_m * units('m/s'))

        skew.plot(p_o, t_o, 'r-', label='Obs T', linewidth=2)
        skew.plot(p_o, td_o_u, 'g-', label='Obs Td', linewidth=1.5)
        skew.plot_barbs(p_o, u_o_u, v_o_u, length=6, xloc=0.95, color='r')

        skew.plot(p_m, t_m, 'b-', label='MONAN T', linewidth=2)
        skew.plot(p_m, td_m_u, 'c-', label='MONAN Td', linewidth=1.5)
        skew.plot_barbs(p_m, u_m_u, v_m_u, length=6, xloc=0.82, color='b')

        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 50)
        skew.ax.set_title("Skew-T Obs vs MONAN")
        skew.ax.legend(loc='upper right')
        plt.tight_layout()
        # fig.savefig(os.path.join(outdir, "SkewT_obs_monan.png"), dpi=150)
        # plt.close(fig)
    except Exception as e:
        print("Warning SkewT:", e)

    # 2) Perfiles comparados (U, T, theta, N, l2, Ri, Fr)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharey=True)
    axU = axes[0,0]; axT = axes[0,1]; axtheta = axes[0,2]; axN = axes[0,3]
    axl2 = axes[1,0]; axRi = axes[1,1]; axFr = axes[1,2]; axBlank = axes[1,3]
    axes_flat = [axU, axT, axtheta, axN, axl2, axRi, axFr]

    # U
    axU.plot(U_o, z, 'r-', label='Obs')
    axU.plot(U_m, z, 'b--', label='MONAN')
    axU.set_xlabel('U (m/s)'); axU.set_ylabel('Height (m)')
    axU.grid(True); axU.legend()

    # T
    axT.plot(T_o, z, 'r-'); axT.plot(T_m, z, 'b--')
    axT.set_xlabel('T (°C)'); axT.grid(True)

    # theta
    axtheta.plot(theta_o-273.15, z, 'r-'); axtheta.plot(theta_m-273.15, z, 'b--')
    axtheta.set_xlabel('θ (°C)'); axtheta.grid(True)

    # N
    axN.plot(N_o, z, 'r-'); axN.plot(N_m, z, 'b--')
    axN.set_xlabel('N (1/s)'); axN.grid(True)

    # l2
    axl2.plot(l2_o, z, 'r-'); axl2.plot(l2_m, z, 'b--')
    axl2.axvline(0, color='k', ls='--', lw=0.7)
    axl2.set_xlabel('l² (1/m²)'); axl2.grid(True)

    # Ri
    axRi.plot(Ri_o, z, 'r-'); axRi.plot(Ri_m, z, 'b--')
    axRi.axvline(0.25, color='k', ls='--', lw=0.7)
    axRi.set_xlabel('Ri'); axRi.grid(True)

    # Fr (plot both)
    axFr.plot(Fr_o, z, 'r-'); axFr.plot(Fr_m, z, 'b--')
    axFr.axvline(1.0, color='k', ls='--', lw=0.7)
    axFr.set_xlabel('Fr'); axFr.grid(True)

    axBlank.axis('off')
    fig.suptitle("Obs (rojo) vs MONAN (azul) — perfiles comparados", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(outdir, "profiles_compare.png"), dpi=150)
    plt.close(fig)

    # 3) Bias plots (T, Td, U)
    # bias = model - obs
    temp_bias = T_m - T_o
    td_bias = td_m - td_o
    u_bias = U_m - U_o

    fig, ax = plt.subplots(1,3, figsize=(15,6), sharey=True)
    ax[0].plot(temp_bias, p)
    ax[0].axvline(0, color='k', ls='--'); ax[0].set_xlabel('Bias T (°C)'); ax[0].invert_yaxis(); ax[0].grid(True)
    ax[1].plot(td_bias, p); ax[1].axvline(0, color='k', ls='--'); ax[1].set_xlabel('Bias Td (°C)'); ax[1].grid(True)
    ax[2].plot(u_bias, p); ax[2].axvline(0, color='k', ls='--'); ax[2].set_xlabel('Bias U (m/s)'); ax[2].grid(True)
    ax[0].set_ylim(1000, 100); ax[0].invert_yaxis()
    fig.suptitle("Bias (MONAN - Obs)")
    # fig.savefig(os.path.join(outdir, "bias_profiles.png"), dpi=150)
    # plt.close(fig)

    # summary stats
    def stats(arr):
        return np.nanmean(arr), np.sqrt(np.nanmean(arr**2))
    tmean, trmse = stats(temp_bias)
    tdmean, tdrmse = stats(td_bias)
    umea, urmse = stats(u_bias)
    summary_txt = (
        f"Bias mean / RMSE\n"
        f"T: {tmean:.2f} / {trmse:.2f} °C\n"
        f"Td: {tdmean:.2f} / {tdrmse:.2f} °C\n"
        f"U: {umea:.2f} / {urmse:.2f} m/s\n"
    )
    with open(os.path.join(outdir, "summary_stats.txt"), 'w') as f:
        f.write(summary_txt)

    print(summary_txt)
    print("Figuras y CSV guardados en:", outdir)

# ---------------------------
# EJECUCIÓN PRINCIPAL
# ---------------------------
if __name__ == "__main__":
    # 1) descarga radiosondeo
    try:
        df_obs = download_sound_wyoming(station, wy_date)
    except Exception as e:
        print("No se pudo descargar WYOMING:", e)
        # si hay un archivo local (por ejemplo un xlsx), podrías leerlo aquí
        raise

    # 2) leer MONAN perfil
    try:
        df_monan = read_monan_profile(monan_file, sounding_lat, sounding_lon)
    except Exception as e:
        print("Error leyendo MONAN:", e)
        raise

    # 3) procesar y graficar comparativo
    process_compare(df_obs, df_monan, outdir)
