#!/usr/bin/env python3
"""
previsoes_ciclone_track_error.py

Função:
Rastreamento do centro (mínimo de MSLP) para várias rodadas e lead times (12h ou 24h, salvo com 24h),
comparando com centros "observados" estimados a partir das análises (arquivos de análise
no próprio modelo) e cálculo do erro de posição (km).

Saídas:
 - tracks_csv/track_<rodada>.csv  : centros previstos por lead
 - tracks_csv/track_obs_all.csv   : centros 'observados' (análises)
 - track_error_summary.csv        : erros consolidados
 - tracks_vs_obs.png              : figura com trajetórias previstas e observadas
 - erro_posicao_vs_lead.png       : erro de posição por rodada x lead time
 - erro_posicao_vs_data.png       : NOVO: erro de posição por data da previsão x rodada
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
import pandas as pd # 🚨 Importação necessária para o plot robusto
from matplotlib import colormaps # 🚨 Importação necessária para o plot robusto

# -----------------------
# CONFIGURAÇÃO
# -----------------------
rodadas = [
    "2023030700", "2023030800", "2023030900",
    "2023031000", "2023031100", "2023031200"
]
forecast_steps_h = list(range(0, 121, 6))
file_template = "MONAN_DIAG_R_POS_GFS_{rodada}_{forecast}.00.00.x1.5898242L55.nc"
pasta = "/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/"
lon_min, lon_max = -95, -80 
lat_min, lat_max = -20, 5 
R_earth = 6371.0
radii_km = [100, 250, 500, 1000]
out_dir = "tracks_csv"
os.makedirs(out_dir, exist_ok=True)

# 🚨 DEFINIÇÃO DAS COLUNAS PARA O PANDAS PLOT (Nomes de coluna usados no CSV)
CSV_DATE_COLUMN = 'forecast_time' 
CSV_RUN_COLUMN = 'init' 

# Obtendo cores 
try:
    cmap = colormaps['hsv']
except AttributeError:
    cmap = plt.cm.get_cmap('hsv')
colors_plot = cmap(np.linspace(0, 1, 10))
colors = colors_plot # Usado nos plots

# -----------------------
# FUNÇÕES AUXILIARES 
# -----------------------

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
# 1) Extrai centros das análises (usadas como "observadas")
# -----------------------
obs_centers = {}
print("Extraindo centros das análises (usadas como observadas)...")
for rodada in rodadas:
    analise_path = os.path.join(pasta, rodada, file_template.format(rodada=rodada, forecast=rodada))
    
    if not os.path.exists(analise_path):
        print(f"⚠️  Análise ausente: {analise_path}")
        continue
    try:
        ds = Dataset(analise_path, "r")
        lons = lon_to_minus180_180(ds.variables["longitude"][:])
        lats = ds.variables["latitude"][:]
        mslp = ds.variables["mslp"][:]
        
        if mslp.ndim == 3:
            mslp = mslp[0, :, :]
        if np.nanmean(mslp) > 2000:
            mslp /= 100.0
            
        lons2d, lats2d = ensure_2d_lonlat(lons, lats)
        mask = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
        
        mslp_masked = np.where(mask, mslp, np.nan)
        
        if np.all(np.isnan(mslp_masked)):
            idx = np.nanargmin(mslp)
        else:
            idx = np.nanargmin(mslp_masked)
            
        iy, ix = np.unravel_index(idx, mslp.shape)
        
        lon_c, lat_c, p_c = float(lons2d[iy, ix]), float(lats2d[iy, ix]), float(mslp[iy, ix])
        obs_centers[rodada] = (lon_c, lat_c, p_c)
        ds.close()
    except Exception as e:
        continue

obs_csv = os.path.join(out_dir, "track_obs_all.csv")
with open(obs_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "lon_obs", "lat_obs", "mslp_hPa_obs"])
    for k in sorted(obs_centers.keys()):
        lon_o, lat_o, p_o = obs_centers[k]
        w.writerow([k, f"{lon_o:.4f}", f"{lat_o:.4f}", f"{p_o:.2f}"])
print(f"✅ Centros observados salvos em: {obs_csv}")

# -----------------------
# 2) Loop das rodadas: detectar centros previstos e calcular erro
# -----------------------
summary_rows = []

for rodada in rodadas:
    rodada_dt = datetime.strptime(rodada, "%Y%m%d%H")
    csvname = os.path.join(out_dir, f"track_{rodada}.csv")

    prev_lon, prev_lat = None, None
    found_first = False 

    with open(csvname, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([CSV_RUN_COLUMN, "lead_h", CSV_DATE_COLUMN,
                    "lon_fc", "lat_fc", "mslp_fc",
                    "lon_obs", "lat_obs", "mslp_obs", "erro_km"])

        for fh in forecast_steps_h:
            fc_dt = rodada_dt + timedelta(hours=fh) 
            fc_str = fc_dt.strftime("%Y%m%d%H")
            fpath = os.path.join(pasta, rodada, file_template.format(rodada=rodada, forecast=fc_str))
            
            if not os.path.exists(fpath):
                continue
            
            try:
                ds = Dataset(fpath, "r")
                lons = lon_to_minus180_180(ds.variables["longitude"][:])
                lats = ds.variables["latitude"][:]
                mslp = ds.variables["mslp"][:]
                
                if mslp.ndim == 3:
                    mslp = mslp[0, :, :]
                if np.nanmean(mslp) > 2000:
                    mslp /= 100.0
                    
                lons2d, lats2d = ensure_2d_lonlat(lons, lats)
                lons_flat, lats_flat, mslp_flat = lons2d.ravel(), lats2d.ravel(), mslp.ravel()

                # Lógica de rastreamento (Busca do centro)
                if not found_first:
                    mask = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
                    mask_flat = mask.ravel()
                    if np.any(mask_flat):
                        vals = np.where(mask_flat, mslp_flat, np.nan)
                        idx = np.nanargmin(vals)
                    else:
                        idx = np.nanargmin(mslp_flat)
                        
                    lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                    found_first, prev_lon, prev_lat = True, lon_c, lat_c
                else:
                    dists = haversine_km(prev_lon, prev_lat, lons_flat, lats_flat)
                    found = False
                    for rkm in radii_km:
                        mask_r = (dists <= rkm)
                        if np.any(mask_r):
                            vals = np.where(mask_r, mslp_flat, np.nan)
                            idx = np.nanargmin(vals)
                            lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                            prev_lon, prev_lat = lon_c, lat_c
                            found = True
                            break
                    if not found:
                        idx = np.nanargmin(mslp_flat)
                        lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                        prev_lon, prev_lat = lon_c, lat_c

                # Cálculo de Erro
                lon_obs, lat_obs, p_obs = obs_centers.get(fc_str[:10], (np.nan, np.nan, np.nan))
                erro_km = np.nan
                if not np.isnan(lon_obs) and not np.isnan(lat_obs):
                    erro_km = haversine_km(lon_c, lat_c, lon_obs, lat_obs)

                # 🚨 CRÍTICO: Se o erro_km for NaN, salva a string vazia "" no CSV
                w.writerow([rodada, fh, fc_str,
                            f"{lon_c:.4f}", f"{lat_c:.4f}", f"{p_c:.2f}",
                            f"{lon_obs:.4f}" if not np.isnan(lon_obs) else "",
                            f"{lat_obs:.4f}" if not np.isnan(lat_obs) else "",
                            f"{p_obs:.2f}" if not np.isnan(p_obs) else "",
                            f"{erro_km:.2f}" if not np.isnan(erro_km) else ""])
                
                summary_rows.append([rodada, fh, fc_str, lon_c, lat_c, p_c, lon_obs, lat_obs, p_obs, erro_km])
                ds.close()
            except Exception as e:
                continue

# -----------------------
# 3) Salva resumo e plota
# -----------------------
summary_csv = "track_error_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([CSV_RUN_COLUMN, "lead_h", CSV_DATE_COLUMN,
                "lon_fc", "lat_fc", "mslp_fc",
                "lon_obs", "lat_obs", "mslp_obs", "erro_km"])
    for row in summary_rows:
        row_out = list(row)
        if isinstance(row_out[-1], float) and np.isnan(row_out[-1]):
            row_out[-1] = ""
        elif isinstance(row_out[-1], float):
            row_out[-1] = f"{row_out[-1]:.2f}"
            
        w.writerow(row_out)
print(f"✅ Resumo salvo: {summary_csv}")

# -----------------------
# PLOT 1: TRAJETÓRIAS (Usando numpy.genfromtxt do original)
# -----------------------
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent([-91, -80, -13, -3])
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.STATES, linewidth=0.3)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
gl.top_labels, gl.right_labels = False, False

if obs_centers:
    for key, (lon_o, lat_o, _) in obs_centers.items():
        ax.plot(lon_o, lat_o, 'k*', markersize=8)
        ax.text(lon_o + 0.01, lat_o + 0.01, f"{key[6:8]}/{key[8:10]}Z", fontsize=10, color='k')
    obs_lons = [v[0] for v in obs_centers.values()]
    obs_lats = [v[1] for v in obs_centers.values()]
    ax.plot(obs_lons, obs_lats, 'k--', label='Análise (observada)')

for i, rodada in enumerate(rodadas):
    csvfile = os.path.join(out_dir, f"track_{rodada}.csv")
    if not os.path.exists(csvfile):
        continue
    
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    
    if data.size == 0:
        continue
    if data.ndim == 0 and data.size == 1:
        data = np.array([data.item()])
        
    lons, lats, times = data['lon_fc'], data['lat_fc'], data['forecast_time']
    ax.plot(lons, lats, '-o', color=colors[i % len(colors)], label=f"{rodada}")
    
    # Para aparecer a data e horário junto a trajetória
    #for lonv, latv, tstr in zip(lons, lats, times):
        #if np.isnan(lonv) or np.isnan(latv):
            #continue
        #tstr = str(tstr)
        #label = f"{tstr[6:8]}/{tstr[8:10]}Z" if len(tstr) >= 10 else f"{tstr}Z"
        #ax.text(lonv + 0.01, latv + 0.01, label, fontsize=10, color=colors[i % len(colors)])

ax.legend(fontsize=10, loc='upper left')
ax.set_title("Trajetórias previstas vs observada", fontsize=11)
plt.savefig("tracks_vs_obs.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva: tracks_vs_obs.png")

# -----------------------
# PLOT 2 & 3: GRÁFICOS DE ERRO (Pandas Robusto)
# -----------------------

# === 1. LEITURA E PREPARAÇÃO DOS DADOS (Robusta) ===
# 🔑 CRÍTICO: na_values=[''] garante que as strings vazias (salvas para erros NaN) virem NaN no DataFrame
summary_df = pd.read_csv(summary_csv, na_values=[''])
summary_df = summary_df.rename(columns={CSV_RUN_COLUMN: 'rodada', CSV_DATE_COLUMN: 'forecast_time'})
# Converte a coluna de erro para numérico, tratando quaisquer erros remanescentes como NaN
summary_df['erro_km'] = pd.to_numeric(summary_df['erro_km'], errors='coerce') 

# Filtra rodadas únicas presentes no arquivo
rodadas_plot = summary_df['rodada'].unique()


# === 2. PLOTAGEM DO ERRO DE POSIÇÃO VS. LEAD TIME (COM MÉDIA) ===

plt.figure(figsize=(12, 6))
plt.title("Erro de Posição vs. Tempo de Previsão (Lead Time)")
plt.xlabel("Tempo de Previsão (h)")
plt.ylabel("Erro de Posição (km)")

# Plota a série de erros por rodada
for i, rodada in enumerate(rodadas_plot):
    # Filtra os dados da rodada atual e remove NaNs
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km'])
    
    if subset.empty:
        continue

    plt.plot(subset['lead_h'], subset['erro_km'], '-o', 
             color=colors[i % len(colors)], label=rodada)

# Média agregada: calcula a média dos erros para cada lead time
erro_medio_lead = summary_df.groupby('lead_h')['erro_km'].mean().reset_index()
lead_vals = erro_medio_lead['lead_h']
erro_medio = erro_medio_lead['erro_km']

# Plota a linha da média (tracejada preta)
#if not lead_vals.empty:
#    plt.plot(lead_vals, erro_medio, 'k--', linewidth=2, label='Média')

# Configurações finais do gráfico de erro vs. lead time
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2)
if not lead_vals.empty:
    plt.xticks(lead_vals)
plt.savefig("erro_posicao_vs_lead.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gráfico 'erro_posicao_vs_lead.png' salvo.")

# ---
# === 3. NOVO GRÁFICO: ERRO DE POSIÇÃO VS. DATA DA PREVISÃO (Sem Média) ===
# ---
# Cria a coluna 'valid_date'
summary_df['valid_date'] = pd.to_datetime(summary_df['forecast_time'], format='%Y%m%d%H', errors='coerce')

plt.figure(figsize=(12, 6))
plt.title("Erro de Posição vs. Data Válida da Previsão")
plt.xlabel("Data da Previsão (Valid Date)")
plt.ylabel("Erro de Posição (km)")

# Plota a série de erros por rodada, em função da data
for i, rodada in enumerate(rodadas_plot):
    # Filtra os dados da rodada atual e remove NaNs
    subset = summary_df[summary_df['rodada'] == rodada].dropna(subset=['erro_km'])
    
    if subset.empty:
        continue

    # Plota o erro de posição para a rodada atual
    plt.plot(subset['valid_date'], subset['erro_km'], '-o', 
             color=colors[i % len(colors)], label=rodada)

# Configurações finais do NOVO gráfico
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Rodada", loc='upper left', ncol=2)
# Formato das datas no eixo X
plt.gcf().autofmt_xdate()
plt.savefig("erro_posicao_vs_data.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ NOVO Gráfico 'erro_posicao_vs_data.png' salvo.")

print("Processamento finalizado com sucesso.")
