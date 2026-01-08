#!/usr/bin/python
# -*- coding: latin-1 -*-
# -*- coding: iso-8859-15 -*-
# -*- coding: ascii -*-

#========================

# ==============================================================
#  ANÁLISE VERTICAL DO FURACÃO JULIA (MONAN/GFS)
#  Versão com logging, centro correto, trajetória, legenda e colorbar robusto
# ==============================================================

import os
import logging
from pathlib import Path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import ticker
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset

# === CONFIGURAÇÕES DE LOGGING ===
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURAÇÕES GERAIS ===
#base_path = "/share/bam/dist/paulo.kubota/externo/Curso_da_OMM_2025_estudos_de_casos/Central_America_Hurricane_Julia"
base_path = "/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia/"
rodadas = ["2022100700", "2022100800", "2022100900"]

# Limites da área de interesse para PLOTAGEM E CÁLCULO DO CENTRO
lon_min, lon_max = -85, -65
lat_min, lat_max = 5, 20                     # Corrigido: lat_min < lat_max

# Fator de suavização (Sigma) para Vorticidade e Divergência
SIGMA_VORT_DIV = 1.5

# === CRIAÇÃO DA FIGURA COM SUBPLOTS ===
logger.info("Inicializando figura com subplots...")
fig, axs = plt.subplots(
    4, len(rodadas),
    figsize=(12, 26),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    constrained_layout=True
)

# --------------------------------------------------------------
#  Variáveis para colorbar e trajetória
# --------------------------------------------------------------
cf_first = [None, None, None]   # [Superfície, 850, 500] - armazena o primeiro contourf de cada linha
centros_lon = []                # para a trajetória
centros_lat = []                # para a trajetória

for j, rodada in enumerate(rodadas):
    logger.info(f"Processando rodada: {rodada}")

    # === MONTAGEM DO CAMINHO DO ARQUIVO ===
    arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{rodada}.00.00.x1.5898242L55.nc"
    caminho = Path(base_path) / rodada / arquivo

    if not caminho.exists():
        logger.warning(f"Arquivo não encontrado: {caminho}. Pulando coluna {j+1}.")
        for i in range(4):
            axs[i, j].set_visible(False)
        continue

    logger.info(f"Lendo arquivo: {caminho}")
    dado = Dataset(caminho, 'r')

    # Extração de coordenadas
    lons_all = dado.variables['longitude'][:]
    lats_all = dado.variables['latitude'][:]
    logger.info(f"Coordenadas extraídas: Lons shape={lons_all.shape}, Lats shape={lats_all.shape}")

    # === 1. RESTRIÇÃO DO CÁLCULO DO CENTRO À ÁREA DE INTERESSE ===
    logger.info("Calculando centro de baixa pressão...")
    mslp_all = dado.variables['mslp'][0, :, :] / 100.0

    lat2d, lon2d = np.meshgrid(lats_all, lons_all, indexing='ij')
    area_mask = (lon2d >= lon_min) & (lon2d <= lon_max) & \
                (lat2d >= lat_min) & (lat2d <= lat_max)

    mslp_area = np.where(area_mask, mslp_all, np.nan)

    if np.all(np.isnan(mslp_area)):
        logger.warning(f"Nenhum dado válido na área para {rodada}. Usando centro padrão.")
        lat_centro_sf, lon_centro_sf = (lat_min + lat_max)/2, (lon_min + lon_max)/2
    else:
        idx_min = np.nanargmin(mslp_area)
        i, k = np.unravel_index(idx_min, mslp_area.shape)
        lat_centro_sf = lats_all[i]
        lon_centro_sf = lons_all[k]

    # Guardar para a trajetória
    centros_lon.append(lon_centro_sf)
    centros_lat.append(lat_centro_sf)

    logger.info(f"Centro de Superfície {rodada[:8]} (Restrito): Lon={lon_centro_sf:.2f}, Lat={lat_centro_sf:.2f}")

    # Função para plotar o ponto central (a partir de 850 hPa)
    def plot_center_point(ax):
        ax.plot(lon_centro_sf, lat_centro_sf, 'o', color='red', markersize=7,
                markeredgecolor='black', zorder=10, transform=ccrs.PlateCarree())

    lons = lons_all
    lats = lats_all

    # Flip para setas de vento (hemisfério sul)
    flip = np.zeros((lats.shape[0], lons.shape[0]))
    flip[lats < 0] = 1

    # ========== 1 Superfície ==========
    logger.info("Plotando nível: Superfície")
    ax = axs[0, j]
    temp2m = dado.variables['t2m'][0, :, :] - 273.15
    u10 = dado.variables['uzonal'][0, 0, :, :]
    v10 = dado.variables['umeridional'][0, 0, :, :]

    cf = ax.contourf(lons, lats, temp2m, levels=np.arange(20, 33, 1),
                     cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree())
    if j == 0:
        cf_first[0] = cf  # ? Armazena para colorbar

    cs = ax.contour(lons, lats, mslp_all, levels=np.arange(980, 1020, 2),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%1.0f', inline=0, fontsize=10)
    ax.barbs(lons[::15], lats[::15], u10[::15, ::15], v10[::15, ::15],
             length=5.5, linewidth=0.8, pivot='middle', color='dimgray',
             flip_barb=flip[::15, ::15], transform=ccrs.PlateCarree())
    ax.set_title(f"{rodada[6:8]}/{rodada[4:6]}/{rodada[:4]} {rodada[8:10]}Z", fontsize=12)

    # ========== 2 850 hPa ==========
    logger.info("Plotando nível: 850 hPa")
    ax = axs[1, j]
    rh850 = dado.variables['relhum'][0, 2, :, :]
    z850 = dado.variables['zgeo'][0, 2, :, :]
    u850 = dado.variables['uzonal'][0, 2, :, :]
    v850 = dado.variables['umeridional'][0, 2, :, :]

    cf = ax.contourf(lons, lats, rh850, levels=np.arange(40, 101, 5),
                     cmap='YlGnBu', extend='both', transform=ccrs.PlateCarree())
    if j == 0:
        cf_first[1] = cf  # ? Armazena para colorbar

    cs = ax.contour(lons, lats, z850, levels=np.arange(1400, 1600, 10),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::15], lats[::15], u850[::15, ::15], v850[::15, ::15],
             length=5.0, linewidth=0.8, color='black',
             flip_barb=flip[::15, ::15], transform=ccrs.PlateCarree())
    ax.set_title(f"{rodada[6:8]}/{rodada[4:6]}/{rodada[:4]} {rodada[8:10]}Z", fontsize=12)
    plot_center_point(ax)

    # ========== 3 500 hPa ==========
    logger.info("Plotando nível: 500 hPa")
    ax = axs[2, j]
    z500 = dado.variables['zgeo'][0, 5, :, :]
    z1000 = dado.variables['zgeo'][0, 0, :, :]
    thickness = z500 - z1000
    u500 = dado.variables['uzonal'][0, 5, :, :]
    v500 = dado.variables['umeridional'][0, 5, :, :]

    u_q = units.Quantity(u500, "m/s")
    v_q = units.Quantity(v500, "m/s")
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    vort = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy).to('1/s').magnitude * 1e5
    vort_smooth = gaussian_filter(vort, sigma=SIGMA_VORT_DIV)

    cf = ax.contourf(lons, lats, vort_smooth, levels=np.arange(-10, 11, 1),
                     cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
    if j == 0:
        cf_first[2] = cf  # ? Armazena para colorbar

    cs = ax.contour(lons, lats, gaussian_filter(thickness, 8), cmap='seismic',
                    linestyles='dashed', linewidths=1.0,
                    levels=np.arange(4900, 5900, 20), transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::15], lats[::15], u500[::15, ::15], v500[::15, ::15],
             length=5.0, linewidth=0.8, color='dimgray',
             flip_barb=flip[::15, ::15], transform=ccrs.PlateCarree())
    ax.set_title(f"{rodada[6:8]}/{rodada[4:6]}/{rodada[:4]} {rodada[8:10]}Z", fontsize=12)
    plot_center_point(ax)

    # ========== 4 250 hPa ==========
    logger.info("Plotando nível: 250 hPa")
    ax = axs[3, j]
    u250 = dado.variables['uzonal'][0, 8, :, :]
    v250 = dado.variables['umeridional'][0, 8, :, :]

    div_nivel = mpcalc.divergence(u250, v250, dx=dx, dy=dy)
    div_nivel = gaussian_filter(div_nivel, sigma=SIGMA_VORT_DIV)

    mask_div = ma.masked_less_equal(div_nivel, 0).mask
    div_nivel[mask_div] = np.nan

    cf = ax.contour(lons, lats, div_nivel * 1e5, cmap='gnuplot',
                    linewidths=1.0, linestyles='solid',
                    extend='both', transform=ccrs.PlateCarree())
    ax.streamplot(lons, lats, u250, v250, density=[2, 2],
                  linewidth=1, color='gray', transform=ccrs.PlateCarree())
    ax.set_title(f"{rodada[6:8]}/{rodada[4:6]}/{rodada[:4]} {rodada[8:10]}Z", fontsize=12)
    plot_center_point(ax)

    # --- Elementos comuns a todos os eixos ---
    logger.info("Adicionando elementos comuns (costas, bordas)...")
    for i in range(4):
        ax = axs[i, j]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3)

        if j == 0:
            labels = [
                "Pressão, vento (1000 hPa) e temperatura 2m",
                "UR, vento e altura geopotencial em 850 hPa",
                "Vorticidade e vento (500 hPa) e espessura (500-1000)",
                "Vento e divergência (250 hPa)"
            ]
            ax.text(-0.15, 0.5, labels[i], fontsize=12,
                    va='center', rotation=90, transform=ax.transAxes)

# ==============================================================
#  COLORBARS - VERSÃO ROBUSTA (usa cf_first armazenado)
# ==============================================================
logger.info("Adicionando colorbars com referência explícita...")
labels_cb = ["Temperatura (°C)", "Umidade relativa (%)", "Vorticidade (x10?? s?¹)"]

for i, cf in enumerate(cf_first):
    if cf is not None:
        cb = fig.colorbar(cf, ax=axs[i, :], orientation='horizontal',
                          fraction=0.05, pad=0.02)
        cb.set_label(labels_cb[i], fontsize=10)
    else:
        logger.warning(f"Colorbar não adicionado para linha {i} (cf ausente).")

# ==============================================================
#  PRÓXIMOS PASSOS (já estavam no script anterior)
# ==============================================================

# 1. Legenda do ponto vermelho
axs[1, 0].plot([], [], 'o', color='red', markersize=7,
               markeredgecolor='black', label='Centro da baixa')
axs[1, 0].legend(loc='upper right', fontsize=10)

# 2. Salvar em PDF
pdf_file = "analise_vertical.pdf"
logger.info(f"Salvando figura em PDF: {pdf_file}")
plt.savefig(pdf_file, bbox_inches='tight')

# 3. Trajetória da baixa
if len(centros_lon) == len(rodadas):
    logger.info("Plotando trajetória da baixa...")
    for col in range(len(rodadas)):
        axs[1, col].plot(centros_lon, centros_lat, 'k--',
                         linewidth=1.5, transform=ccrs.PlateCarree())

# ==============================================================
#  FINALIZAÇÃO E SALVAMENTO (PNG)
# ==============================================================
plt.suptitle("Evolução do Furacão Julia (2022) - Previsões MONAN/GFS",
             fontsize=16, y=0.98)
logger.info("Salvando figura PNG...")
plt.savefig("analise_vertical.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("Processamento concluído. Figuras salvas como "
            "'analise_vertical.png' e 'analise_vertical.pdf'.")
