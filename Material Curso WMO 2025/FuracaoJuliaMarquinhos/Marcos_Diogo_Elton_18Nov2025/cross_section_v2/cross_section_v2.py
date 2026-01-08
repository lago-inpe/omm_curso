#!/usr/bin/python
# -*- coding: latin-1 -*-
# -*- coding: iso-8859-15 -*-
# -*- coding: ascii -*-

# ==============================================================================
#  SISTEMA COMPLETO: TRACKING AUTOM�TICO + CROSS-SECTION + MAPA FIXO
#  Altera��o: Mapa de Superf�cie com �rea fixa (TRACK_LAT/LON constants)
# ==============================================================================

import os
import csv
import time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === CONFIGURA��ES GERAIS ===
base_path = "/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia"
track_file = "track_julia_auto.csv" 

# Datas para processar
datas_para_processar = [
    "2022100700",
    "2022100800",
    "2022100900",
    "2022101000",
    "2022101100"
]

# �rea de Busca e AGORA TAMB�M �REA DO MAPA
TRACK_LAT_MIN, TRACK_LAT_MAX = 8, 18
TRACK_LON_MIN, TRACK_LON_MAX = -95, -60

# N�veis verticais
press_levels_pa = np.array([100000, 92500, 85000, 77500, 70000, 50000, 
                            40000, 30000, 25000, 20000, 15000, 10000])

# === SETUP DO LOGGING ===
log_filename = "processamento_julia_completo.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def log_stats(var_name, data, unit=""):
    d_min, d_max, d_mean = np.nanmin(data), np.nanmax(data), np.nanmean(data)
    logger.info(f"STATS > {var_name:.<25}: Min={d_min:7.2f} | Max={d_max:7.2f} | Mean={d_mean:7.2f} {unit}")

# ==============================================================================
#  PASSO 1: GERADOR DE TRACKING (AUTO-CSV)
# ==============================================================================
def gerar_track_automatico():
    logger.info("="*60)
    logger.info("INICIANDO GERA��O AUTOM�TICA DE TRACKING (VARREDURA)")
    logger.info(f"Buscando m�nimos de MSLP na caixa: Lat[{TRACK_LAT_MIN}:{TRACK_LAT_MAX}], Lon[{TRACK_LON_MIN}:{TRACK_LON_MAX}]")
    
    with open(track_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'lat', 'lon', 'min_mslp'])
        
        for rodada in datas_para_processar:
            arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{rodada}.00.00.x1.5898242L55.nc"
            caminho = Path(base_path) / rodada / arquivo
            
            if not caminho.exists():
                logger.warning(f"Arquivo n�o encontrado para tracking: {rodada}")
                continue
                
            try:
                nc = Dataset(caminho, 'r')
                lats = nc.variables['latitude'][:]
                lons = nc.variables['longitude'][:]
                mslp = nc.variables['mslp'][0, :, :] / 100.0 
                nc.close()
                
                lat2d, lon2d = np.meshgrid(lats, lons, indexing='ij')
                mask = (lat2d >= TRACK_LAT_MIN) & (lat2d <= TRACK_LAT_MAX) & \
                       (lon2d >= TRACK_LON_MIN) & (lon2d <= TRACK_LON_MAX)
                
                mslp_masked = np.where(mask, mslp, np.nan)
                idx_min = np.nanargmin(mslp_masked)
                i_lat, i_lon = np.unravel_index(idx_min, mslp_masked.shape)
                
                lat_found = lats[i_lat]
                lon_found = lons[i_lon]
                val_found = mslp_masked[i_lat, i_lon]
                
                logger.info(f"Data {rodada}: Centro encontrado em {lat_found:.2f}N, {lon_found:.2f}E ({val_found:.1f} hPa)")
                writer.writerow([rodada, lat_found, lon_found, val_found])
                
            except Exception as e:
                logger.error(f"Erro ao processar tracking para {rodada}: {e}")

    logger.info(f"Arquivo de tracking gerado com sucesso: {track_file}")
    logger.info("="*60)

# ==============================================================================
#  PASSO 2: PROCESSAMENTO GR�FICO
# ==============================================================================
def processar_figura(rodada, lat_centro, lon_centro):
    logger.info(f"Gerando figura para: {rodada} | Centro CSV: {lat_centro}�N, {lon_centro}�E")
    start_time = time.time()
    
    arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{rodada}.00.00.x1.5898242L55.nc"
    caminho = Path(base_path) / rodada / arquivo

    if not caminho.exists(): return

    dado = Dataset(caminho, 'r')
    lons = dado.variables['longitude'][:]
    lats = dado.variables['latitude'][:]
    levels_file = dado.variables['level'][:] 

    # Defini��o da Se��o
    lon_start, lon_end = lon_centro - 2.0, lon_centro + 2.0
    lat_fixed = lat_centro
    lat_idx = np.argmin(np.abs(lats - lat_fixed))
    
    lon_mask = (lons >= lon_start) & (lons <= lon_end)
    lon_indices = np.where(lon_mask)[0]
    lons_section = lons[lon_indices]

    # N�veis
    valid_levels = []
    level_indices = []
    for p in press_levels_pa:
        idx = np.argmin(np.abs(levels_file - p))
        if np.abs(levels_file[idx] - p) <= 500:
            valid_levels.append(levels_file[idx])
            level_indices.append(idx)
    press_plot = np.array(valid_levels) / 100.0

    # Extra��o
    lat_slice = slice(lat_idx - 1, lat_idx + 2)
    u_slab = dado.variables['uzonal'][0, level_indices, lat_slice, lon_indices]
    v_slab = dado.variables['umeridional'][0, level_indices, lat_slice, lon_indices]
    temp_slab = dado.variables['temperature'][0, level_indices, lat_slice, lon_indices]
    rh_slab = dado.variables['relhum'][0, level_indices, lat_slice, lon_indices]
    omega_slab = dado.variables['omega'][0, level_indices, lat_slice, lon_indices]
    ter_profile = dado.variables['ter'][lat_idx, lon_indices]
    
    mslp_full = dado.variables['mslp'][0, :, :] / 100.0

    log_stats("Vento U (Slab)", u_slab, "m/s")
    
    # C�lculos
    lats_slab = lats[lat_slice]
    lon_grid, lat_grid = np.meshgrid(lons_section, lats_slab)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon_grid, lat_grid)
    
    u_q = units.Quantity(u_slab, 'm/s')
    v_q = units.Quantity(v_slab, 'm/s')
    vort_slab = mpcalc.vorticity(u_q, v_q, dx=dx[None,:,:], dy=dy[None,:,:])
    div_slab = mpcalc.divergence(u_q, v_q, dx=dx[None,:,:], dy=dy[None,:,:])

    u_sect = u_slab[:, 1, :]
    v_sect = v_slab[:, 1, :]
    temp_sect = temp_slab[:, 1, :]
    rh_sect = rh_slab[:, 1, :]
    omega_sect = omega_slab[:, 1, :] * 100.0
    vort_sect = vort_slab[:, 1, :].magnitude * 1e5
    div_sect = div_slab[:, 1, :].magnitude * 1e5
    
    wind_kt = np.sqrt(u_sect**2 + v_sect**2) * 1.94384
    theta_k = temp_sect * (1000.0 / press_plot[:, None]) ** 0.2854
    
    sigma = 0.8
    vort_smooth = gaussian_filter(vort_sect, sigma)
    div_smooth = gaussian_filter(div_sect, sigma)
    omega_smooth = gaussian_filter(omega_sect, sigma)
    LON_MESH, PRESS_MESH = np.meshgrid(lons_section, press_plot)

    # --- PLOTAGEM ---
    logger.info("Gerando plot...")
    fig = plt.figure(figsize=(12, 28))

    # MAPA (�REA FIXA baseada em TRACK_LON/LAT_MIN/MAX)
    ax0 = fig.add_subplot(6, 1, 1, projection=ccrs.PlateCarree())
    
    # ALTERA��O AQUI: Fixando o extent para a �rea total de tracking
    ax0.set_extent([TRACK_LON_MIN, TRACK_LON_MAX, TRACK_LAT_MIN, TRACK_LAT_MAX], crs=ccrs.PlateCarree())
    
    ax0.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
    ax0.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray', linestyle='--')

    # Recorte de dados para o mapa tamb�m fixo na �rea de tracking
    mask_map_lon = (lons >= TRACK_LON_MIN) & (lons <= TRACK_LON_MAX)
    mask_map_lat = (lats >= TRACK_LAT_MIN) & (lats <= TRACK_LAT_MAX)
    ix_lon, ix_lat = np.where(mask_map_lon)[0], np.where(mask_map_lat)[0]
    mslp_zoom = mslp_full[np.ix_(ix_lat, ix_lon)]
    X_MAP, Y_MAP = np.meshgrid(lons[ix_lon], lats[ix_lat])
    
    # Contour Lines (Sem fundo cinza)
    cs_map = ax0.contour(X_MAP, Y_MAP, mslp_zoom, levels=np.arange(980, 1020, 4), colors='black', linewidths=1.2, transform=ccrs.PlateCarree())
    ax0.clabel(cs_map, inline=True, fontsize=10, fmt='%d')
    
    ax0.plot([lon_start, lon_end], [lat_fixed, lat_fixed], color='red', linewidth=3, marker='|', markersize=10, transform=ccrs.PlateCarree())
    ax0.plot(lon_centro, lat_centro, 'r*', markersize=15, markeredgecolor='k', transform=ccrs.PlateCarree())
    ax0.set_title(f"Mapa de Superf�cie (MSLP) | Centro: {lat_centro:.2f}�N, {lon_centro:.2f}�E", fontweight='bold')

    # PAINEIS VERTICAIS
    ax1 = fig.add_subplot(6, 1, 2)
    ax2 = fig.add_subplot(6, 1, 3, sharex=ax1)
    ax3 = fig.add_subplot(6, 1, 4, sharex=ax1)
    ax4 = fig.add_subplot(6, 1, 5, sharex=ax1)
    ax5 = fig.add_subplot(6, 1, 6, sharex=ax1)

    def setup_cb(im, ax, label):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="2%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label)

    def plot_terr(ax):
        ter_p = 1013.25 * np.exp(-ter_profile / 8000.0)
        ax.fill_between(lons_section, ter_p, 1050, color='gray', zorder=10)

    # 1. Vento
    cf1 = ax1.contourf(LON_MESH, PRESS_MESH, wind_kt, levels=np.arange(0, 101, 5), cmap='jet', extend='max')
    cs1 = ax1.contour(LON_MESH, PRESS_MESH, theta_k, levels=np.arange(280, 400, 4), colors='k', linewidths=1.0)
    ax1.clabel(cs1, inline=True, fontsize=9, fmt='%d')
    setup_cb(cf1, ax1, 'kt')
    ax1.set_title('Vento Horizontal (kt) + Temp. Potencial (K)', fontweight='bold')
    plot_terr(ax1)

    # 2. UR
    cf2 = ax2.contourf(LON_MESH, PRESS_MESH, rh_sect, levels=np.arange(40, 101, 5), cmap='BrBG', extend='both')
    setup_cb(cf2, ax2, '%')
    ax2.set_title('Umidade Relativa (%)', fontweight='bold')
    plot_terr(ax2)

    # 3. Vorticidade
    cf3 = ax3.contourf(LON_MESH, PRESS_MESH, vort_smooth, levels=np.arange(-20, 21, 2), cmap='RdBu_r', extend='both')
    setup_cb(cf3, ax3, '10?? s?�')
    ax3.set_title('Vorticidade Relativa', fontweight='bold')
    plot_terr(ax3)

    # 4. Diverg�ncia
    cf4 = ax4.contourf(LON_MESH, PRESS_MESH, div_smooth, levels=np.arange(-10, 11, 1), cmap='PuOr', extend='both')
    setup_cb(cf4, ax4, '10?? s?�')
    ax4.set_title('Diverg�ncia Horizontal', fontweight='bold')
    plot_terr(ax4)

    # 5. Omega
    cf5 = ax5.contourf(LON_MESH, PRESS_MESH, omega_smooth, levels=np.arange(-5, 5.1, 0.5), cmap='coolwarm', extend='both')
    cs5 = ax5.contour(LON_MESH, PRESS_MESH, omega_smooth, levels=np.arange(-6, 7, 1), colors='k', linewidths=0.8)
    ax5.clabel(cs5, inline=True, fontsize=8, fmt='%d')
    setup_cb(cf5, ax5, 'Pa/s')
    ax5.set_title('Velocidade Vertical (Omega)', fontweight='bold')
    plot_terr(ax5)
    ax5.set_xlabel('Longitude (�E)')

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_ylabel('Press�o (hPa)')
        ax.set_ylim(1000, 100)
        ax.set_xlim(lons_section[0], lons_section[-1])
        ax.grid(True, linestyle=':', alpha=0.6)
        if ax != ax5: ax.set_xticklabels([])

    plt.suptitle(f'An�lise Vertical e Superf�cie: Furac�o Julia | {rodada}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    outfile = f"cross_section_julia_auto_{rodada}.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Processamento conclu�do em {elapsed:.2f} segundos.")
    logger.info(f"Figura gerada: {outfile}")
    dado.close()

# === EXECU��O PRINCIPAL ===
if __name__ == "__main__":
    # 1. Gerar CSV
    gerar_track_automatico()
    
    # 2. Plotar
    if os.path.exists(track_file):
        logger.info("Iniciando ciclo de plotagem...")
        with open(track_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                try:
                    processar_figura(row[0], float(row[1]), float(row[2]))
                except Exception as e:
                    logger.error(f"Falha ao plotar {row[0]}: {e}")
    
    logger.info("="*60)
    logger.info("Script finalizado.")
