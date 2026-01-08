#!/usr/bin/python
# -*- coding: latin-1 -*-
# -*- coding: iso-8859-15 -*-
# -*- coding: ascii -*-

#========================

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from pathlib import Path

import shapely.geometry as sgeom
from shapely.prepared import prep
from cartopy.feature import LAND

# ================================
# CONFIGURA��ES (OBRIGAT�RIAS!)
# ================================
#base_path = "/share/bam/dist/paulo.kubota/externo/Curso_da_OMM_2025_estudos_de_casos/Central_America_Hurricane_Julia" #area antiga
base_path = "/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia"
rodadas_str = ["2022100600", "2022100700", "2022100800", "2022100900", "2022101000"]
rodadas_dt = [datetime.strptime(r, "%Y%m%d%H") for r in rodadas_str]

# Esta lista agora � usada S� para as COLUNAS do plot
previsoes_unicas = ["2022100700", "2022100800", "2022100900", "2022101000", "2022101100"]

# --- BEST TRACK NHC COM EST�GIOS ---
# (Omitido por brevidade - sem altera��es)
nhc_track_full = [
    (datetime(2022,10,6,12), 11.4, -66.5, 1006, 30, "disturbance"),
    (datetime(2022,10,6,18), 11.6, -67.8, 1005, 30, "disturbance"),
    (datetime(2022,10,7,0), 11.9, -69.2, 1004, 30, "tropical depression"),
    (datetime(2022,10,7,3), 12.0, -69.8, 1004, 30, "tropical depression"),
    (datetime(2022,10,7,6), 12.1, -70.6, 1004, 30, "tropical depression"),
    (datetime(2022,10,7,9), 12.2, -71.2, 1004, 35, "tropical depression"),
    (datetime(2022,10,7,12), 12.5, -72.1, 1002, 35, "tropical storm"),
    (datetime(2022,10,7,18), 12.8, -73.8, 1002, 40, "tropical storm"),
    (datetime(2022,10,8,0), 12.8, -75.5, 999, 45, "tropical storm"),
    (datetime(2022,10,8,6), 12.7, -77.2, 999, 50, "tropical storm"),
    (datetime(2022,10,8,12), 12.7, -78.9, 994, 55, "tropical storm"),
    (datetime(2022,10,8,18), 12.6, -80.5, 993, 60, "tropical storm"),
    (datetime(2022,10,9,0), 12.5, -82.0, 989, 65, "hurricane"),
    (datetime(2022,10,9,6), 12.4, -83.3, 982, 75, "hurricane"),  # M�XIMO
    (datetime(2022,10,9,7,15), 12.4, -83.6, 982, 75, "hurricane"),
    (datetime(2022,10,9,12), 12.3, -84.7, 985, 65, "tropical storm"),
    (datetime(2022,10,9,18), 12.3, -86.2, 988, 45, "tropical storm"),
    (datetime(2022,10,10,0), 12.6, -87.6, 993, 40, "tropical storm"),
    (datetime(2022,10,10,6), 13.1, -88.8, 998, 35, "tropical depression"),
    (datetime(2022,10,10,10), 13.6, -88.7, 1001, 35, "tropical depression"),
    (datetime(2022,10,10,11), 13.7, -88.9, 1001, 30, "tropical depression"),
    (datetime(2022,10,10,12), 13.7, -89.9, 1002, 30, "tropical depression"),
    (datetime(2022,10,10,18), 13.0, -89.0, 1002, 30, "dissipated")
]
nhc_dates = [item[0] for item in nhc_track_full]
nhc_lats = [item[1] for item in nhc_track_full]
nhc_lons = [item[2] for item in nhc_track_full]
nhc_stages = [item[5] for item in nhc_track_full]

stage_styles = {
    "disturbance": {"marker": "o", "color": "#1f77b4", "size": 6, "label": "Dist�rbio"},
    "tropical depression": {"marker": "s", "color": "#2ca02c", "size": 6, "label": "Depress�o Tropical"},
    "tropical storm": {"marker": "^", "color": "#ff7f0e", "size": 7, "label": "Tempestade Tropical"},
    "hurricane": {"marker": "*", "color": "#d62728", "size": 10, "label": "Furac�o"},
    "dissipated": {"marker": "x", "color": "#7f7f7f", "size": 6, "label": "Dissipado"}
}

# --- C�LCULO DE COLUNAS/LINHAS ---
# (Omitido por brevidade - sem altera��es)
primeira_coluna_com_dado = [5] * 5
ultima_linha_com_dado = [-1] * 5
for i, rod in enumerate(rodadas_str):
    rod_dt = rodadas_dt[i]
    for j, prev in enumerate(previsoes_unicas):
        prev_dt = datetime.strptime(prev, "%Y%m%d%H")
        if (prev_dt - rod_dt).days in [1, 2, 3, 4, 5]:
            if primeira_coluna_com_dado[i] == 5:
                primeira_coluna_com_dado[i] = j
            ultima_linha_com_dado[j] = max(ultima_linha_com_dado[j], i)


### ALTERA��O ###
# ================================
# GERAR LISTA DE PREVIS�ES DE 6H PARA O TRACKING
# ================================
print("Gerando lista de previs�es de 6h para o tracker...")
all_previsoes_6h_str = []

# Data inicial: 6h ap�s a primeira rodada
# (Assumindo que a previs�o mais curta � de 6h)
start_dt = rodadas_dt[0] + timedelta(hours=6) 

# Data final: a �ltima previs�o da nossa grade original
end_dt = datetime.strptime(previsoes_unicas[-1], "%Y%m%d%H")

current_dt = start_dt
while current_dt <= end_dt:
    all_previsoes_6h_str.append(current_dt.strftime("%Y%m%d%H"))
    current_dt += timedelta(hours=6)
    
print(f"Total de {len(all_previsoes_6h_str)} passos de 6h para verificar (de {start_dt} a {end_dt}).")
# Esta lista ser�: ["2022100606", "2022100612", ..., "2022101100"]

# ================================
# PR�-C�LCULO DO BEST TRACK MONAN
# ================================
print("=== Iniciando pr�-c�lculo dos tracks do MONAN (Min SLP) ===")
monan_tracks = {rod: [] for rod in rodadas_str}
land_mask = None

for rodada in rodadas_str:
    rod_dt = datetime.strptime(rodada, "%Y%m%d%H")
    
    ### ALTERA��O ###
    # Trocamos 'previsoes_unicas' pela lista completa de 6h
    for previsao in all_previsoes_6h_str: 
        prev_dt = datetime.strptime(previsao, "%Y%m%d%H")
        
        # Otimiza��o: Se a previs�o � ANTES da rodada, pular
        if prev_dt <= rod_dt:
            continue
            
        # Otimiza��o: Se a previs�o for > 5 dias, pular
        # (Seus arquivos parecem ser de no m�x 5 dias / 120h)
        if (prev_dt - rod_dt).days > 5:
            continue
            
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
        caminho = Path(base_path) / rodada / arquivo
        
        if not caminho.exists():
            continue # Pula se o arquivo de 6h n�o existir
            
        try:
            dado = Dataset(caminho, 'r')
            lons = dado.variables['longitude'][:]
            lats = dado.variables['latitude'][:]
            mslp_raw = dado.variables['mslp'][0, :, :]
            mslp = mslp_raw / 100.0

            if land_mask is None:
                print("--- Gerando m�scara de terra (ocorre apenas 1 vez)...")
                land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m')
                land_geoms = list(land_feature.geometries())
                land_polygon = sgeom.MultiPolygon(land_geoms)
                prepared_land = prep(land_polygon)
                xx, yy = np.meshgrid(lons, lats)
                points_flat = [sgeom.Point(x, y) for x, y in zip(xx.flat, yy.flat)]
                is_land_flat = [prepared_land.contains(p) for p in points_flat]
                land_mask = np.array(is_land_flat).reshape(mslp.shape)
                print("--- M�scara de terra gerada com sucesso! ---")

            mslp_ocean_only = np.ma.masked_array(mslp, mask=land_mask)
            min_idx_flat = np.argmin(mslp_ocean_only)
            
            if min_idx_flat is np.ma.masked:
                dado.close()
                continue
                
            min_idx_2d = np.unravel_index(min_idx_flat, mslp_ocean_only.shape)
            min_lat = lats[min_idx_2d[0]]
            min_lon = lons[min_idx_2d[1]]
            min_pressure = mslp_ocean_only.min() 
            
            monan_tracks[rodada].append((prev_dt, min_lat, min_lon, min_pressure))
            dado.close()
        except Exception as e:
            # Imprime menos para n�o poluir
            # print(f"Erro no pr�-c�lculo {caminho}: {e}")
            pass # Ignora erros de arquivos, ex. corrompidos

print("=== Pr�-c�lculo dos tracks do MONAN conclu�do ===")


# ================================
# FIGURA E LEGENDA
# ================================
# (Nenhuma altera��o nesta se��o)

fig, axs = plt.subplots(5, 5, figsize=(40, 20),
                        subplot_kw=dict(projection=ccrs.PlateCarree()),
                        gridspec_kw={'hspace': 0.15, 'wspace': -0.75})

legend_elements = []
seen_stages = set()
for stage in nhc_stages:
    if stage not in seen_stages:
        style = stage_styles.get(stage, stage_styles["disturbance"])
        legend_elements.append(plt.Line2D([0], [0], marker=style["marker"], color='w',
                                          markerfacecolor=style["color"], markeredgecolor='k',
                                          markersize=18, label=style["label"]))
        seen_stages.add(stage)

legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                  markerfacecolor='blue', markeredgecolor='k',
                                  markersize=14,
                                  label='Track MONAN (Min SLP)'))
legend_ax = axs[2, 0]

# ================================
# LOOP PRINCIPAL (Sem altera��es)
# ================================
# Este loop AINDA USA 'previsoes_unicas' (de 24h) para definir as colunas
# Isso est� CORRETO.
# ================================
for i, rodada in enumerate(rodadas_str):
    rod_dt = rodadas_dt[i]
    primeiro_mapa_j = primeira_coluna_com_dado[i]
    
    # Loop das colunas (mapas de 24h)
    for j, previsao in enumerate(previsoes_unicas): 
        ax = axs[i, j]
        
        if i == 2 and j == 0:
            ax.set_visible(False)
            continue
            
        ax.set_extent([-95, -60, 0, 25], crs=ccrs.PlateCarree())
        if i == 0:
            # T�tulo da coluna (ex: 2022100700)
            ax.set_title(previsao, fontsize=13, pad=12, weight='bold')
            
        prev_dt = datetime.strptime(previsao, "%Y%m%d%H")
        
        # Pular c�lulas vazias da matriz
        if (prev_dt - rod_dt).days not in [1, 2, 3, 4, 5]:
            ax.set_visible(False)
            continue
            
        # Nome do arquivo (o de 24h)
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
        caminho = Path(base_path) / rodada / arquivo
        
        if not caminho.exists():
            print(f"N�o encontrado (plot): {caminho}")
            ax.set_visible(False)
            continue
            
        print(f"Plotando Contorno: {caminho}")
        try:
            dado = Dataset(caminho, 'r')
            lons = dado.variables['longitude'][:]
            lats = dado.variables['latitude'][:]
            mslp = dado.variables['mslp'][0, :, :] / 100
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            levels = np.arange(960, 1031, 2)
            cs = ax.contour(lons, lats, mslp, levels=levels, colors='black', linewidths=0.8, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
            
            # --- TRAJET�RIA NHC---
            start_time = rod_dt
            end_time = prev_dt + timedelta(hours=6)
            track_lons, track_lats = [], []
            for idx, dt in enumerate(nhc_dates):
                if start_time <= dt <= end_time:
                    lat, lon, stage = nhc_lats[idx], nhc_lons[idx], nhc_stages[idx]
                    style = stage_styles.get(stage, stage_styles["disturbance"])
                    ms = style["size"] * 1.8 if dt == datetime(2022,10,9,6) else style["size"]
                    ax.plot(lon, lat, marker=style["marker"], color=style["color"],
                            markersize=ms, transform=ccrs.PlateCarree(), markeredgecolor='k')
                    track_lons.append(lon)
                    track_lats.append(lat)
            if track_lons:
                ax.plot(track_lons, track_lats, color='k', linewidth=1.2, alpha=0.7, transform=ccrs.PlateCarree())

            # --- TRAJET�RIA MONAN ---
            # (Nenhuma altera��o aqui, ele vai usar o track pr�-calculado
            # que agora tem pontos de 6h)
            monan_track_data = monan_tracks[rodada]
            monan_track_lons = []
            monan_track_lats = []
            
            for (dt, lat, lon, pres) in monan_track_data:
                # Plota todos os pontos do track AT� o tempo de previs�o deste subplot
                if dt <= prev_dt: 
                    monan_track_lons.append(lon)
                    monan_track_lats.append(lat)
                    ax.plot(lon, lat, marker='D', color='blue',
                            markersize=5, transform=ccrs.PlateCarree(), 
                            markeredgecolor='white', markeredgewidth=0.5, zorder=10)

            if monan_track_lons:
                ax.plot(monan_track_lons, monan_track_lats, color='blue', 
                        linestyle='--', linewidth=1.2, alpha=0.8, 
                        transform=ccrs.PlateCarree(), zorder=9)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='darkgray', alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            gl.left_labels = (j == primeira_coluna_com_dado[i])
            gl.bottom_labels = (i == ultima_linha_com_dado[j])
            
            if j == primeiro_mapa_j:
                ax.text(-0.15, 0.5, f"Rodada {rodada[:8]}",
                        fontsize=13, fontweight='bold', color='black',
                        ha='center', va='center', rotation='vertical',
                        transform=ax.transAxes)
            
            dado.close()
            
        except Exception as e:
            print(f"Erro no plot {caminho}: {e}")
            ax.set_visible(False)

# ================================
# LEGENDA E SALVAMENTO
# ================================
# (Nenhuma altera��o nesta se��o)

legend_ax.set_visible(True)
legend_ax.axis('off')
legend_ax.legend(handles=legend_elements,
                 loc='upper center',
                 bbox_to_anchor=(0.5, 0.78),
                 fontsize=19,
                 title="Est�gios NHC",
                 title_fontsize=21,
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 ncol=1,
                 borderpad=1.5,
                 labelspacing=1.5,
                 handletextpad=1.5,
                 handlelength=3.0)

fig.suptitle("Furac�o Julia (2022) - Previs�es MONAN/GFS", fontsize=18, y=0.97, weight='bold')
fig.text(0.5, 0.945, "Press�o ao N�vel M�dio do Mar (hPa) e Trajet�ria NHC com Est�gios", ha='center', fontsize=14)

plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.08, hspace=0.15, wspace=0.000)

output_file = "furacao_julia_mslp_track_stages_MONAN.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figura gerada: {output_file}")
print("TRACK DO MONAN TENTADO A CADA 6H.")
