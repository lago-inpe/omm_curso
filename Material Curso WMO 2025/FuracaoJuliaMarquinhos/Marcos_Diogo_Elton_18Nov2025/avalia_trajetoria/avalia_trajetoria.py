#!/usr/bin/python
# -*- coding: latin-1 -*-
# -*- coding: iso-8859-15 -*-
# -*- coding: ascii -*-

#========================

"""
previsoes_ciclone_track_error.py

Fun��o:
Rastreamento do centro (m�nimo de MSLP) para v�rias rodadas e lead times,
comparando com centros "observados" (an�lises do modelo) E com o "Best Track"
oficial do NHC.

Implementa��o:
- Adicionada m�scara de oceano (landmask == 0) para evitar a busca em terra.
- Adicionada a trajet�ria do "Best Track" do NHC ao gr�fico principal.
- Adicionado c�lculo de "erro_km_nhc" (Previs�o vs. NHC Best Track).
- Adicionados PLOT 4 e PLOT 5 para visualizar o erro oficial vs. NHC.

Sa�das:
 - tracks_csv/track_<rodada>.csv   : centros previstos por lead
 - tracks_csv/track_obs_all.csv    : centros 'observados' (an�lises)
 - track_error_summary.csv         : erros consolidados (incluindo erro_km_nhc)
 - tracks_vs_obs.png               : figura com trajet�rias previstas e observadas
 - erro_posicao_vs_lead.png        : erro (vs. AN�LISE) por lead time
 - erro_posicao_vs_data.png        : erro (vs. AN�LISE) por data da previs
 - erro_posicao_vs_lead_NHC.png    : NOVO: erro (vs. NHC) por lead time
 - erro_posicao_vs_data_NHC.png    : NOVO: erro (vs. NHC) por data da previs
"""

import os
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import math
import pandas as pd # ?? Importa��o necess�ria para o plot robusto
from matplotlib import colormaps # ?? Importa��o necess�ria para o plot robusto

# -----------------------
# CONFIGURA��O
# -----------------------

# <--- Defini��o do caminho base dos dados
#base_data_dir = "/share/bam/dist/paulo.kubota/externo/Curso_da_OMM_2025_estudos_de_casos/Central_America_Hurricane_Julia/" # area antiga
# base_data_dir = "/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia"
base_data_dir = "/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia"

# <--- Datas das rodadas ATUALIZADAS para o Furac�o Julia (EXEMPLO)
rodadas = [
    "2022100600", "2022100700", "2022100800",
    "2022100900", "2022101000", "2022101100"
]

# <--- Ajuste dos passos de previs�o (se necess�rio)
forecast_steps_h = list(range(0, 121, 24))

file_template = "MONAN_DIAG_R_POS_GFS_{rodada}_{forecast}.00.00.x1.5898242L55.nc"

# <--- Bounding Box ATUALIZADO para o Furac�o Julia (Caribe)
lon_min, lon_max = -90, -65  # Longitude Oeste
lat_min, lat_max = 8, 20      # Latitude Norte

R_earth = 6371.0
radii_km = [250, 500, 800, 1500]
out_dir = "tracks_csv"
os.makedirs(out_dir, exist_ok=True)

# ?? DEFINI��O DAS COLUNAS PARA O PANDAS PLOT (Nomes de coluna usados no CSV)
CSV_DATE_COLUMN = 'forecast_time'
CSV_RUN_COLUMN = 'init'

# Obtendo cores
try:
    cmap = colormaps['hsv']
except AttributeError:
    cmap = plt.cm.get_cmap('hsv')
colors_plot = cmap(np.linspace(0, 1, 15))
colors = colors_plot # Usado nos plots

# --- IN�CIO DA ADI��O DO BEST TRACK (NHC) ---
# Movido para cima para ser usado nos c�lculos de erro
nhc_track_full = [
    (datetime(2022,10,6,12), 11.4, -66.5, 1006, 30, "disturbance"),
    (datetime(2022,10,6,18), 11.6, -67.8, 1005, 30, "disturbance"),
    (datetime(2022,10,7,0), 11.9, -69.2, 1004, 30, "tropical depression"), # <--- Ponto 24h
    (datetime(2022,10,7,3), 12.0, -69.8, 1004, 30, "tropical depression"),
    (datetime(2022,10,7,6), 12.1, -70.6, 1004, 30, "tropical depression"),
    (datetime(2022,10,7,9), 12.2, -71.2, 1004, 35, "tropical depression"),
    (datetime(2022,10,7,12), 12.5, -72.1, 1002, 35, "tropical storm"),
    (datetime(2022,10,7,18), 12.8, -73.8, 1002, 40, "tropical storm"),
    (datetime(2022,10,8,0), 12.8, -75.5, 999, 45, "tropical storm"),     # <--- Ponto 24h
    (datetime(2022,10,8,6), 12.7, -77.2, 999, 50, "tropical storm"),
    (datetime(2022,10,8,12), 12.7, -78.9, 994, 55, "tropical storm"),
    (datetime(2022,10,8,18), 12.6, -80.5, 993, 60, "tropical storm"),
    (datetime(2022,10,9,0), 12.5, -82.0, 989, 65, "hurricane"),          # <--- Ponto 24h
    (datetime(2022,10,9,6), 12.4, -83.3, 982, 75, "hurricane"),
    (datetime(2022,10,9,7,15), 12.4, -83.6, 982, 75, "hurricane"),
    (datetime(2022,10,9,12), 12.3, -84.7, 985, 65, "tropical storm"),
    (datetime(2022,10,9,18), 12.3, -86.2, 988, 45, "tropical storm"),
    (datetime(2022,10,10,0), 12.6, -87.6, 993, 40, "tropical storm"),    # <--- Ponto 24h
    (datetime(2022,10,10,6), 13.1, -88.8, 998, 35, "tropical depression"),
    (datetime(2022,10,10,10), 13.6, -88.7, 1001, 35, "tropical depression"),
    (datetime(2022,10,10,11), 13.7, -88.9, 1001, 30, "tropical depression"),
    (datetime(2022,10,10,12), 13.7, -89.9, 1002, 30, "tropical depression"),
    (datetime(2022,10,11,00), 13.0, -89.0, 1002, 30, "dissipated")      # <--- Ponto 24h
]

# Converte a lista do NHC para um dicion�rio para buscas r�pidas (datetime -> dados)
nhc_track_dict = {item[0]: item for item in nhc_track_full}
# --- FIM DA ADI��O DO BEST TRACK (NHC) ---


# -----------------------
# FUN��ES AUXILIARES
# -----------------------
# (Sem altera��es aqui)
def lon_to_minus180_180(lon_arr):
    lon = np.array(lon_arr)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon

def ensure_2d_lonlat(lons, lats):
    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    else:
        return lons, lats

def haversine_km(lon1, lat1, lon2, lat2):
    lon1r, lat1r = math.radians(lon1), math.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R_earth * c

# -----------------------
# 1) Extrai centros das an�lises (usadas como "observadas")
# -----------------------
# (Sem altera��es aqui)
obs_centers = {}
print("Extraindo centros das an�lises (usadas como observadas)...")
for rodada in rodadas:
    analise_path = os.path.join(base_data_dir, rodada, file_template.format(rodada=rodada, forecast=rodada))
    
    if not os.path.exists(analise_path):
        print(f"??  An�lise ausente: {analise_path}")
        continue
    try:
        ds = Dataset(analise_path, "r")
        lons = lon_to_minus180_180(ds.variables["longitude"][:])
        lats = ds.variables["latitude"][:]
        mslp = ds.variables["mslp"][:]
        landmask = ds.variables["landmask"][:] 
        
        if mslp.ndim == 3:
            mslp = mslp[0, :, :]
        if np.nanmean(mslp) > 2000:
            mslp /= 100.0
            
        lons2d, lats2d = ensure_2d_lonlat(lons, lats)
        
	# <--- COMBINA M�SCARA DE BBOX E OCEANO
        mask_bbox = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
        mask_ocean = (landmask == 0)  # Apenas oceano (0 = oceano, 1 = terra)
        mask = mask_bbox & mask_ocean # Onde � oceano E est� dentro da caixa
        
        mslp_masked = np.where(mask, mslp, np.nan)
        
        if np.all(np.isnan(mslp_masked)):
            print(f"  -> Aviso: Nenhum dado na caixa de busca (OCEANO) para {rodada}. Usando m�nimo global.")
            mslp_masked_ocean_global = np.where(mask_ocean, mslp, np.nan)
            if np.all(np.isnan(mslp_masked_ocean_global)):
                idx = np.nanargmin(mslp) 
            else:
                idx = np.nanargmin(mslp_masked_ocean_global)
        else:
            idx = np.nanargmin(mslp_masked)
            
        iy, ix = np.unravel_index(idx, mslp.shape)
        
        lon_c, lat_c, p_c = float(lons2d[iy, ix]), float(lats2d[iy, ix]), float(mslp[iy, ix])
        obs_centers[rodada] = (lon_c, lat_c, p_c)
        ds.close()
    except Exception as e:
        print(f"  -> ERRO ao processar {analise_path}: {e}")
        continue

obs_csv = os.path.join(out_dir, "track_obs_all.csv")
with open(obs_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "lon_obs", "lat_obs", "mslp_hPa_obs"])
    for k in sorted(obs_centers.keys()):
        lon_o, lat_o, p_o = obs_centers[k]
        w.writerow([k, f"{lon_o:.4f}", f"{lat_o:.4f}", f"{p_o:.2f}"])
print(f"? Centros observados salvos em: {obs_csv}")

# -----------------------
# 2) Loop das rodadas: detectar centros previstos e calcular erro
# -----------------------
summary_rows = []

for rodada in rodadas:
    print(f"\nProcessando rodada: {rodada}")
    rodada_dt = datetime.strptime(rodada, "%Y%m%d%H")
    csvname = os.path.join(out_dir, f"track_{rodada}.csv")

    prev_lon, prev_lat = None, None
    found_first = False 

    with open(csvname, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        # <--- MUDAN�A: Adiciona colunas de erro NHC ao CSV da rodada
        w.writerow([CSV_RUN_COLUMN, "lead_h", CSV_DATE_COLUMN,
                    "lon_fc", "lat_fc", "mslp_fc",
                    "lon_obs", "lat_obs", "mslp_obs", "erro_km",
                    "lon_nhc", "lat_nhc", "mslp_nhc", "erro_km_nhc"])

        for fh in forecast_steps_h:
            fc_dt = rodada_dt + timedelta(hours=fh) 
            fc_str = fc_dt.strftime("%Y%m%d%H")
            
            fpath = os.path.join(base_data_dir, rodada, file_template.format(rodada=rodada, forecast=fc_str))
            
            if not os.path.exists(fpath):
                print(f"  -> Arquivo de previs�o ausente: {fpath}")
                continue
            
            try:
                ds = Dataset(fpath, "r")
                lons = lon_to_minus180_180(ds.variables["longitude"][:])
                lats = ds.variables["latitude"][:]
                mslp = ds.variables["mslp"][:]
                landmask = ds.variables["landmask"][:] 
                
                if mslp.ndim == 3:
                    mslp = mslp[0, :, :]
                if np.nanmean(mslp) > 2000:
                    mslp /= 100.0
                    
                lons2d, lats2d = ensure_2d_lonlat(lons, lats)
                lons_flat, lats_flat, mslp_flat = lons2d.ravel(), lats2d.ravel(), mslp.ravel()

                # L�gica de rastreamento (Busca do centro)
                # (Sem altera��es aqui)
                if not found_first:
		    # No primeiro passo (fh=0), busca na caixa inteira
                    
                    # <--- COMBINA M�SCARA DE BBOX E OCEANO
                    mask_bbox = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
                    mask_ocean = (landmask == 0)  # Apenas oceano
                    mask = mask_bbox & mask_ocean # Onde � oceano E est� dentro da caixa
                    mask_flat = mask.ravel()
                    
                    if np.any(mask_flat) and not np.all(np.isnan(np.where(mask_flat, mslp_flat, np.nan))):
                        vals = np.where(mask_flat, mslp_flat, np.nan)
                        idx = np.nanargmin(vals)
                    else:
                        print(f"  -> Aviso: Nenhum dado na caixa de busca (OCEANO) para {rodada} fh={fh}. Usando m�nimo global.")
                        mask_ocean_flat = mask_ocean.ravel()
                        mslp_masked_ocean_global = np.where(mask_ocean_flat, mslp_flat, np.nan)
                        if np.all(np.isnan(mslp_masked_ocean_global)):
                            idx = np.nanargmin(mslp_flat) 
                        else:
                            idx = np.nanargmin(mslp_masked_ocean_global)

                    lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                    found_first, prev_lon, prev_lat = True, lon_c, lat_c
                else:
		    # Nos passos seguintes, busca perto do ponto anterior
                    dists = haversine_km(prev_lon, prev_lat, lons_flat, lats_flat)
		    # <--- CRIA M�SCARA DE OCEANO 'FLAT' (3)
                    landmask_flat = landmask.ravel()
                    found = False
                    for rkm in radii_km:
		        # <--- COMBINA M�SCARA DE RAIO E OCEANO
                        mask_r = (dists <= rkm)  # M�scara de Raio
                        mask_ocean = (landmask_flat == 0) # M�scara de Oceano
                        mask = mask_r & mask_ocean # Onde � oceano E est� dentro do raio
                        
                        if np.any(mask): # Usa 'mask' final
                            vals = np.where(mask, mslp_flat, np.nan) # Usa 'mask' final
                            if not np.all(np.isnan(vals)):
                                idx = np.nanargmin(vals)
                                lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                                prev_lon, prev_lat = lon_c, lat_c
                                found = True
                                break 
                    
                    if not found:
		        # Se n�o achou em nenhum raio (ciclone dissipou ou pulou), pega o m�nimo global
                        print(f"  -> Aviso: Ciclone perdido em {rodada} fh={fh}. Usando m�nimo global (OCEANO).")
                        mask_ocean_flat = landmask_flat 
                        mslp_masked_ocean_global = np.where(mask_ocean_flat == 0, mslp_flat, np.nan)
                        if np.all(np.isnan(mslp_masked_ocean_global)):
                            idx = np.nanargmin(mslp_flat)  # Fallback final
                        else:
                            idx = np.nanargmin(mslp_masked_ocean_global)
                        
                        lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                        prev_lon, prev_lat = lon_c, lat_c

                # --- C�lculo de Erro (vs. AN�LISE) ---
                lon_obs, lat_obs, p_obs = obs_centers.get(fc_str, (np.nan, np.nan, np.nan))
                erro_km = np.nan
                if not np.isnan(lon_obs) and not np.isnan(lat_obs):
                    erro_km = haversine_km(lon_c, lat_c, lon_obs, lat_obs)
                
                # <--- IN�CIO DA MUDAN�A: C�lculo de Erro OFICIAL (vs. NHC) ---
                erro_km_nhc = np.nan
                lon_nhc, lat_nhc, p_nhc = np.nan, np.nan, np.nan
                
                # Busca a posi��o do NHC para esta data/hora exata
                nhc_data = nhc_track_dict.get(fc_dt) # fc_dt � um objeto datetime
                
                if nhc_data:
                    # nhc_data = (datetime, lat, lon, p, v, stage)
                    lat_nhc = nhc_data[1]
                    lon_nhc = nhc_data[2]
                    p_nhc = nhc_data[3]
                    # Calcula o erro oficial
                    erro_km_nhc = haversine_km(lon_c, lat_c, lon_nhc, lat_nhc)
                # <--- FIM DA MUDAN�A ---

                # <--- MUDAN�A: Adiciona dados NHC ao CSV da rodada
                w.writerow([rodada, fh, fc_str,
                            f"{lon_c:.4f}", f"{lat_c:.4f}", f"{p_c:.2f}",
                            f"{lon_obs:.4f}" if not np.isnan(lon_obs) else "",
                            f"{lat_obs:.4f}" if not np.isnan(lat_obs) else "",
                            f"{p_obs:.2f}" if not np.isnan(p_obs) else "",
                            f"{erro_km:.2f}" if not np.isnan(erro_km) else "",
                            f"{lon_nhc:.4f}" if not np.isnan(lon_nhc) else "",
                            f"{lat_nhc:.4f}" if not np.isnan(lat_nhc) else "",
                            f"{p_nhc:.2f}" if not np.isnan(p_nhc) else "",
                            f"{erro_km_nhc:.2f}" if not np.isnan(erro_km_nhc) else ""])
                
                # <--- MUDAN�A: Adiciona dados NHC ao resumo
                summary_rows.append([rodada, fh, fc_str, lon_c, lat_c, p_c,
                                     lon_obs, lat_obs, p_obs, erro_km,
                                     lon_nhc, lat_nhc, p_nhc, erro_km_nhc])
                ds.close()
            except Exception as e:
                print(f"  -> ERRO ao processar {fpath}: {e}")
                continue
    print(f"  -> CSV da rodada salvo: {csvname}")

# -----------------------
# 3) Salva resumo e plota
# -----------------------
summary_csv = "track_error_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    # <--- MUDAN�A: Adiciona colunas NHC ao cabe�alho do resumo
    w.writerow([CSV_RUN_COLUMN, "lead_h", CSV_DATE_COLUMN,
                "lon_fc", "lat_fc", "mslp_fc",
                "lon_obs", "lat_obs", "mslp_obs", "erro_km",
                "lon_nhc", "lat_nhc", "mslp_nhc", "erro_km_nhc"])
    
    # <--- MUDAN�A: L�gica de escrita de linha atualizada
    for row in summary_rows:
        row_out = list(row)
        # Formata erro_km (�ndice -5)
        if isinstance(row_out[-5], float):
            if np.isnan(row_out[-5]):
                row_out[-5] = ""
            else:
                row_out[-5] = f"{row_out[-5]:.2f}"
        # Formata erro_km_nhc (�ndice -1)
        if isinstance(row_out[-1], float):
            if np.isnan(row_out[-1]):
                row_out[-1] = ""
            else:
                row_out[-1] = f"{row_out[-1]:.2f}"
        w.writerow(row_out)
print(f"\n? Resumo salvo: {summary_csv}")

# -----------------------
# PLOT 1: TRAJET�RIAS
# -----------------------
# (Sem altera��es, j� inclui o NHC)
print("Gerando gr�fico de trajet�rias (tracks_vs_obs.png)...")
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent([-91, -65, 7, 21]) 
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
gl.top_labels, gl.right_labels = False, False

# Extrai lats/lons para a linha completa
nhc_lons = [item[2] for item in nhc_track_full]
nhc_lats = [item[1] for item in nhc_track_full]
ax.plot(nhc_lons, nhc_lats, 
        label='NHC Best Track', 
        color='#0000FF', 
        linewidth=1.5, 
        linestyle='-', 
        zorder=10) 

target_dates_24h = [
    datetime(2022, 10, 6, 0), 
    datetime(2022, 10, 7, 0),
    datetime(2022, 10, 8, 0),
    datetime(2022, 10, 9, 0),
    datetime(2022, 10, 10, 0),
    datetime(2022, 10, 11, 0)
]

for item in nhc_track_full:
    item_date = item[0]
    item_lat = item[1]
    item_lon = item[2]
    
    if item_date in target_dates_24h:
        ax.plot(item_lon, item_lat, 
                marker='s', 
                color='blue', 
                markersize=5, 
                fillstyle='none',
                mew=2, 
                zorder=11)
        
        label_txt = f"{item_date.day}/{item_date.hour:02d}Z"
        ax.text(item_lon + 0.1, item_lat + 0.1, label_txt, 
                color='blue', 
                fontsize=10, 
                weight='bold',
                zorder=12)

if obs_centers:
    sorted_items = sorted(obs_centers.items())
    obs_times = [item[0] for item in sorted_items]
    obs_lons =  [item[1][0] for item in sorted_items] 
    obs_lats =  [item[1][1] for item in sorted_items] 
    
    ax.plot(obs_lons, obs_lats, 'k*--', label='An�lise (modelo)', markersize=8, zorder=5)
    
    for i, key in enumerate(obs_times):
        label_obs = f"{key[6:8]}/{key[8:10]}Z"
        ax.text(obs_lons[i] + 0.05, obs_lats[i] + 0.05, label_obs, fontsize=9, color='k', zorder=6)

for i, rodada in enumerate(rodadas):
    csvfile = os.path.join(out_dir, f"track_{rodada}.csv")
    if not os.path.exists(csvfile):
        continue
    
    try:
        data = pd.read_csv(csvfile)
        if data.empty:
            continue
    except pd.errors.EmptyDataError:
        continue
        
    lons, lats, times = data['lon_fc'], data['lat_fc'], data['forecast_time']
    
    ax.plot(lons, lats, '-o', color=colors[i % len(colors)], label=rodada, markersize=5, zorder=8)
    
    for lonv, latv, tstr in zip(lons, lats, times):
        if pd.isna(lonv) or pd.isna(latv):
            continue
        tstr = str(tstr)
        label = f"{tstr[6:8]}/{tstr[8:10]}Z" if len(tstr) >= 10 else f"{tstr}Z"
        ax.text(lonv + 0.05, latv + 0.05, label, fontsize=8, color=colors[i % len(colors)], zorder=9)

ax.legend(fontsize=9, loc='upper right')
ax.set_title("Trajet�rias previstas (linhas) vs NHC Best Track (Azul)", fontsize=11)
plt.savefig("tracks_vs_obs.png", dpi=300, bbox_inches='tight')
plt.close()
print("? Figura salva: tracks_vs_obs.png")

# -----------------------
# PLOT 2 & 3: GR�FICOS DE ERRO (vs. AN�LISE)
# -----------------------

# === 1. LEITURA E PREPARA��O DOS DADOS (Robusta) ===
try:
    summary_df = pd.read_csv(summary_csv, na_values=[''])
except pd.errors.EmptyDataError:
    print("??  Arquivo de resumo 'track_error_summary.csv' est� vazio. Pulando gr�ficos de erro.")
    print("Processamento finalizado (com erros).")
    exit() 

summary_df = summary_df.rename(columns={CSV_RUN_COLUMN: 'rodada', CSV_DATE_COLUMN: 'forecast_time'})
summary_df['erro_km'] = pd.to_numeric(summary_df['erro_km'], errors='coerce') 
# <--- MUDAN�A: Carrega tamb�m a nova coluna de erro
summary_df['erro_km_nhc'] = pd.to_numeric(summary_df['erro_km_nhc'], errors='coerce') 

rodadas_plot = summary_df['rodada'].unique()


# === 2. PLOTAGEM DO ERRO DE POSI��O (vs. AN�LISE) VS. LEAD TIME ===
print("Gerando gr�fico 'erro_posicao_vs_lead.png' (Erro vs An�lise)...")
plt.figure(figsize=(12, 6))
plt.title("Erro de Posi��o (vs. An�lise do Modelo) vs. Tempo de Previs�o (Lead Time)")
plt.xlabel("Tempo de Previs�o (h)")
plt.ylabel("Erro de Posi��o (km)")

for i, rodada in enumerate(rodadas_plot):
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km']) # Plota erro_km
    if subset.empty:
        continue
    
    plt.plot(subset['lead_h'], subset['erro_km'], '-o', # Plota erro_km
             color=colors[i % len(colors)], label=rodada)

erro_medio_lead = summary_df.groupby('lead_h')['erro_km'].mean().reset_index() # Plota erro_km
lead_vals = erro_medio_lead['lead_h']
erro_medio = erro_medio_lead['erro_km'] # Plota erro_km

if not lead_vals.empty:
    plt.plot(lead_vals, erro_medio, 'k*--', linewidth=2, label='M�dia')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2, fontsize=9)
if not lead_vals.empty:
    plt.xticks(lead_vals)
plt.savefig("erro_posicao_vs_lead.png", dpi=300, bbox_inches='tight')
plt.close()
print("? Gr�fico 'erro_posicao_vs_lead.png' salvo.")

# ---
# === 3. GR�FICO ERRO (vs. AN�LISE) VS. DATA DA PREVIS�O ===
# ---
print("Gerando gr�fico 'erro_posicao_vs_data.png' (Erro vs An�lise)...")
summary_df['valid_date'] = pd.to_datetime(summary_df['forecast_time'], format='%Y%m%d%H', errors='coerce')

plt.figure(figsize=(12, 6))
plt.title("Erro de Posi��o (vs. An�lise do Modelo) vs. Data V�lida da Previs�o")
plt.xlabel("Data da Previs�o (Valid Date)")
plt.ylabel("Erro de Posi��o (km)")

for i, rodada in enumerate(rodadas_plot):
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km']) # Plota erro_km
    if subset.empty:
        continue
    
    plt.plot(subset['valid_date'], subset['erro_km'], '-o', # Plota erro_km
             color=colors[i % len(colors)], label=rodada)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2, fontsize=9)
plt.gcf().autofmt_xdate()
plt.savefig("erro_posicao_vs_data.png", dpi=300, bbox_inches='tight')
plt.close()
print("? NOVO Gr�fico 'erro_posicao_vs_data.png' salvo.")


# <--- IN�CIO DA MUDAN�A: Bloco de plotagem duplicado para o Erro Oficial vs. NHC ---

# -----------------------
# PLOT 4 & 5: GR�FICOS DE ERRO OFICIAL (vs. NHC)
# -----------------------

# === 4. PLOTAGEM DO ERRO DE POSI��O (vs. NHC) VS. LEAD TIME ===
print("Gerando gr�fico 'erro_posicao_vs_lead_NHC.png' (Erro vs NHC)...")
plt.figure(figsize=(12, 6))
plt.title("Erro de Posi��o OFICIAL (vs. NHC) vs. Tempo de Previs�o (Lead Time)")
plt.xlabel("Tempo de Previs�o (h)")
plt.ylabel("Erro de Posi��o (km)")

for i, rodada in enumerate(rodadas_plot):
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km_nhc']) # Plota erro_km_nhc
    if subset.empty:
        continue
    
    plt.plot(subset['lead_h'], subset['erro_km_nhc'], '-o', # Plota erro_km_nhc
             color=colors[i % len(colors)], label=rodada)

erro_medio_lead_nhc = summary_df.groupby('lead_h')['erro_km_nhc'].mean().reset_index() # Plota erro_km_nhc
lead_vals_nhc = erro_medio_lead_nhc['lead_h']
erro_medio_nhc = erro_medio_lead_nhc['erro_km_nhc'] # Plota erro_km_nhc

if not lead_vals_nhc.empty:
    plt.plot(lead_vals_nhc, erro_medio_nhc, 'k*--', linewidth=2, label='M�dia')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2, fontsize=9)
if not lead_vals_nhc.empty:
    plt.xticks(lead_vals_nhc)
plt.savefig("erro_posicao_vs_lead_NHC.png", dpi=300, bbox_inches='tight')
plt.close()
print("? Gr�fico 'erro_posicao_vs_lead_NHC.png' salvo.")

# ---
# === 5. GR�FICO ERRO (vs. NHC) VS. DATA DA PREVIS�O ===
# ---
print("Gerando gr�fico 'erro_posicao_vs_data_NHC.png' (Erro vs NHC)...")
# a coluna 'valid_date' j� foi criada, n�o precisa recriar

plt.figure(figsize=(12, 6))
plt.title("Erro de Posi��o OFICIAL (vs. NHC) vs. Data V�lida da Previs�o")
plt.xlabel("Data da Previs�o (Valid Date)")
plt.ylabel("Erro de Posi��o (km)")

for i, rodada in enumerate(rodadas_plot):
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km_nhc']) # Plota erro_km_nhc
    if subset.empty:
        continue
    
    plt.plot(subset['valid_date'], subset['erro_km_nhc'], '-o', # Plota erro_km_nhc
             color=colors[i % len(colors)], label=rodada)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2, fontsize=9)
plt.gcf().autofmt_xdate()
plt.savefig("erro_posicao_vs_data_NHC.png", dpi=300, bbox_inches='tight')
plt.close()
print("? NOVO Gr�fico 'erro_posicao_vs_data_NHC.png' salvo.")

# <--- FIM DA MUDAN�A ---


print("\nProcessamento finalizado com sucesso.")
