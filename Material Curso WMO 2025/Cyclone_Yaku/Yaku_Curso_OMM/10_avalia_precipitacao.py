#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma 
from netCDF4 import Dataset
import pygrib
from scipy.ndimage import uniform_filter 
from scipy.interpolate import RegularGridInterpolator 
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib import colors 
from matplotlib import gridspec

#
# DADOS DE ENTRADA:
# 1. Modelo (Previsão): Precipitação acumulada em 24h, calculada pela diferença 
#    entre dois campos de precipitação total acumulada (rainc + rainnc) do modelo 
#    em tempos sequenciais (t e t-24h).
# 2. Observação (Referência): Precipitação acumulada em 24h do produto MERGE, que 
#    serve como "verdade" observada. Os dados do modelo são interpolados para a 
#    grade do MERGE antes da comparação.
#
# -----------------------------------------------------------------------------
# 📊 MÉTRICAS DE AVALIAÇÃO CALCULADAS
# -----------------------------------------------------------------------------
# 1. BIAS (Viés):
#    - Cálculo: Média (Previsão - Observação).
#    - Indica a tendência sistemática do modelo de superestimar (Viés positivo) ou 
#      subestimar (Viés negativo) a precipitação, em mm.
#
# 2. RMSE (Root Mean Square Error - Erro Quadrático Médio): 
#    - Métrica de erro mais comum. Indica a magnitude média típica das diferenças 
#      entre o modelo e a observação, em mm. Penaliza erros maiores.
# 
# 3. CSI (Critical Success Index - Índice de Sucesso Crítico):
#    - Cálculo: Hits / (Hits + Misses + False Alarms).
#    - Métrica de acerto binário (previsão/observação acima de um limiar). Varia 
#      entre 0 (sem habilidade) e 1 (perfeito). Indica a habilidade do modelo em 
#      prever eventos que realmente ocorreram, penalizando eventos falsos previstos
#      (False Alarms) e omissões (Misses). Calculado para os limiares de **10 mm, 
#      50 mm e 100 mm**.
#
# 4. FSS (Fraction Skill Score - Índice de Habilidade por Fração):
#    - Métrica de acerto em escala de vizinhança. Avalia o desempenho do modelo 
#      em prever a fração de área com precipitação acima de um limiar em diferentes 
#      escalas de vizinhança. O FSS é calculado para o limiar de 10 mm/24h nas 
#      escalas de **3 km, 9 km e 25 km**. Varia de 0 (sem habilidade) a 1 (perfeito).
#
# 5. ERRO DE POSIÇÃO (Precipitation Track Error - COG):
#    - Cálculo: Distância Haversine (em km) entre os Centros de Gravidade (COG) da 
#      precipitação observada e a prevista (para eventos acima de 10 mm).
#    - Mede o erro de deslocamento horizontal do sistema de precipitação.
#
# -----------------------------------------------------------------------------
# 📈 SAÍDAS E VISUALIZAÇÕES
# -----------------------------------------------------------------------------
# 1. Painel de Mapas (panel_all.png):
#    - Apresenta o mapa de **Viés (Modelo - MERGE)** para cada célula válida da matriz 
#      (Rodada vs. Previsão). É o principal produto visual do script.
#
# 2. Mapa de Viés Agregado (bias_mean_aggregated_[rodada].png):
#    - Para cada rodada, é gerado um mapa do **Viés Médio Agregado** (média temporal 
#      do Bias sobre todos os lead times válidos daquela rodada), destacando o 
#      erro sistemático da inicialização.
#
# 3. Gráficos de Evolução Temporal (Séries Temporais):
#    - Três gráficos de linha são gerados, mostrando a evolução das métricas médias 
#      (RMSE, CSI 10mm, Erro de Posição COG) em função do tempo de liderança (lead time), 
#      com linhas separadas para cada Rodada de inicialização.
#
# 4. Arquivos CSV (metrics_[rodada].csv):
#    - Arquivos de tabela contendo todas as métricas calculadas (incluindo todos os 
#      limiares do CSI e todas as escalas do FSS) para cada par Rodada/Previsão.
# -----------------------------------------------------------------------------

# ---------------------------
# CONFIGURAÇÕES (PARÂMETROS DE ENTRADA)
# ---------------------------
# Rodadas (inicialização) e Previsões (target)
rodadas = ["2023030700", "2023030800", "2023030900", "2023031000", "2023031100"]
previsoes = ["2023030800", "2023030900", "2023031000", "2023031100", "2023031200"]

# Domínio geográfico de interesse: [lon_min, lon_max, lat_min, lat_max]
extent = [-90, -70, -20, 5]

# Pastas / templates dos arquivos
model_file_template = "MONAN_DIAG_R_POS_GFS_{rodada}_{forecast}.00.00.x1.5898242L55.nc"
merge_dir = "merge" 
out_dir = "avaliacao_precipitacao" 
os.makedirs(out_dir, exist_ok=True) 

pasta_merge = "/oper/share/ioper/tempo/MERGE/GPM/DAILY/"
pasta_modelo = "/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/"

# Limiares (thresholds) para CSI e FSS (em mm/24h)
thresholds = [10, 50, 100]            
fss_scales_km = [3, 9, 25]            

# Configuração da paleta de cores para o Viés (Modelo - Observação)
colors_vies = ['#9c0720', '#dc143c', '#f1666d', '#ff9ea2', '#f0c6f0', '#ffffff',
          '#87CEEB', '#00BFFF', '#1E90FF', '#4169E1', '#0000FF']
cmap_vies = colors.ListedColormap(colors_vies)
cmap_vies.set_over('#081d58') 
cmap_vies.set_under('#610000') 

data_min = -60
data_max = 70
interval = 10
levels_vies = np.arange(data_min, data_max, interval)

# ---------------------------
# FUNÇÕES AUXILIARES DE CÁLCULO 
# ---------------------------
def lon_to_minus180_180(lon_array):
    lon = np.array(lon_array)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_cog(field, lats, lons, threshold):
    field_data = field.data
    mask = np.isfinite(field_data) & (field.mask == False) & (field_data >= threshold)
    if not np.any(mask):
        return np.nan, np.nan
    weights = field_data[mask]
    lats_thr = lats[mask]
    lons_thr = lons[mask]
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.nan, np.nan
    cog_lat = np.sum(lats_thr * weights) / total_weight
    cog_lon = np.sum(lons_thr * weights) / total_weight
    return cog_lat, cog_lon

def precip_track_error_cog(obs, mod, lats, lons, threshold):
    cog_obs_lat, cog_obs_lon = calculate_cog(obs, lats, lons, threshold)
    cog_mod_lat, cog_mod_lon = calculate_cog(mod, lats, lons, threshold)
    if np.isnan(cog_obs_lat) or np.isnan(cog_mod_lat):
        return np.nan
    return haversine_distance(cog_obs_lat, cog_obs_lon, cog_mod_lat, cog_mod_lon)

def grid_spacing_km(lat_1d, lon_1d):
    dy = np.nanmean(np.abs(np.diff(lat_1d))) * 111.0 
    dx = np.nanmean(np.abs(np.diff(lon_1d))) * 111.0 * math.cos(math.radians(np.nanmean(lat_1d))) 
    return dy, dx

def compute_fss(obs, mod, thr_mm, scale_km, lat_1d, lon_1d):
    mask = np.isfinite(obs) & np.isfinite(mod)
    ob_b = np.full(obs.shape, np.nan)
    fc_b = np.full(mod.shape, np.nan)
    ob_b[mask] = (obs[mask] >= thr_mm).astype(float)
    fc_b[mask] = (mod[mask] >= thr_mm).astype(float)
    dy_km, dx_km = grid_spacing_km(lat_1d, lon_1d)
    box = max(1, int(round(scale_km / max(dx_km, dy_km))))
    ob_frac_temp = uniform_filter(np.nan_to_num(ob_b, nan=0.0), size=box, mode='constant')
    fc_frac_temp = uniform_filter(np.nan_to_num(fc_b, nan=0.0), size=box, mode='constant')
    ob_frac = np.full(obs.shape, np.nan)
    fc_frac = np.full(mod.shape, np.nan)
    ob_frac[mask] = ob_frac_temp[mask]
    fc_frac[mask] = fc_frac_temp[mask]
    mse = np.nanmean((fc_frac - ob_frac) ** 2)
    mse_ref = np.nanmean(ob_frac ** 2 + fc_frac ** 2)
    if mse_ref == 0:
        return 1.0 
    return 1.0 - mse / mse_ref

def pooled_contingency(obs, mod, thr):
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs_valid = obs[mask]
    mod_valid = mod[mask]
    ob_b = (obs_valid >= thr)
    fc_b = (mod_valid >= thr)
    hits = int(np.sum(ob_b & fc_b))             
    misses = int(np.sum(ob_b & (~fc_b)))        
    falses = int(np.sum((~ob_b) & fc_b))        
    cneg = int(np.sum((~ob_b) & (~fc_b)))       
    return hits, misses, falses, cneg

def csi_from_counts(h, m, f):
    denom = h + m + f
    if denom == 0:
        return np.nan
    return h / denom

def rmse_and_bias(ob, fc):
    mask = np.isfinite(ob) & np.isfinite(fc)
    if not np.any(mask):
        return np.nan, np.nan
    diff = fc[mask] - ob[mask]
    rmse = float(np.sqrt(np.mean(diff**2))) 
    bias = float(np.mean(diff)) 
    return rmse, bias


# ---------------------------
# BLOCO PRINCIPAL
# ---------------------------

all_metrics_by_rodada = {} 
nrows = len(rodadas)
ncols = len(previsoes)

# Inicialização da figura do Painel (mapas)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(4 * ncols, 3.5 * nrows),
                        subplot_kw=dict(projection=ccrs.PlateCarree()),
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
axs = np.atleast_2d(axs)

# *************************************************************************
# === MODIFICAÇÃO 1: CORREÇÃO DA LÓGICA DE CÁLCULO DA PRIMEIRA COLUNA VÁLIDA ===
# A lógica de identificação de células válidas é reescrita com base nas datas.
primeira_coluna_com_dado = [None] * len(rodadas)
ultima_linha_com_dado = [None] * len(previsoes)

for i, rodada in enumerate(rodadas):
    rodada_dt = datetime.strptime(rodada, '%Y%m%d%H')
    for j, previsao in enumerate(previsoes):
        previsao_dt = datetime.strptime(previsao, '%Y%m%d%H')
        
        # A previsão é válida se a data da previsão é estritamente depois da data da rodada.
        if previsao_dt > rodada_dt:
            # Encontra a primeira coluna com dado
            if primeira_coluna_com_dado[i] is None:
                primeira_coluna_com_dado[i] = j
            # Encontra a última linha com dado para cada coluna
            ultima_linha_com_dado[j] = i 

# *************************************************************************

last_valid_pcm = None 
lons_merge_grid = None  
lats_merge_grid = None 

# LOOP EXTERNO: Itera sobre as RODADAS
for i, rodada in enumerate(rodadas):
    metrics_rows = []
    bias_fields_list = [] 
    metrics_for_plots = {'lead_h': [], 'rmse_val': [], 'csi10_val': [], 'precip_track_error10km_val': []} 

    # LOOP INTERNO: Itera sobre as PREVISÕES
    for j, previsao in enumerate(previsoes):
        ax = axs[i, j] 
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())

        # ✅ TÍTULO DA COLUNA: Aplicado na primeira linha (i=0)
        if i == 0:
            ax.set_title(previsao, fontsize=10)

        # 🛑 Lógica de pular células (Diagonal e antes do primeiro dado válido)
        if rodada == previsao or not (primeira_coluna_com_dado[i] is not None and j >= primeira_coluna_com_dado[i]):
            # AÇÃO CORRIGIDA 1: Se for a primeira linha (onde o título é importante), usa ax.axis('off') para limpar sem ocultar o título.
            if i == 0:
                ax.axis('off')
            else:
                ax.set_visible(False)
            continue
            
        # *************************************************************************
        # === MODIFICAÇÃO 2: INSERÇÃO DO RÓTULO DE RODADA NO EIXO Y (Primeira Coluna Válida) ===
        if j == primeira_coluna_com_dado[i]:
            ax.text(-0.3, 0.5, f"Rodada: {rodada}", 
                    fontsize=12, 
                    va='center', 
                    rotation=90, 
                    transform=ax.transAxes)
        # *************************************************************************

        # Determinação das datas e caminhos dos arquivos
        merge_date_str = previsao[:8]
        merge_date = datetime.strptime(merge_date_str, '%Y%m%d')
        target_current = merge_date.strftime('%Y%m%d') + '12'
        prev_date = (merge_date - timedelta(days=1))
        target_prev = prev_date.strftime('%Y%m%d') + '12'

        arquivo_atual = model_file_template.format(rodada=rodada, forecast=target_current)
        arquivo_ant = model_file_template.format(rodada=rodada, forecast=target_prev)
        caminho_atual = os.path.join(pasta_modelo, rodada, arquivo_atual)
        caminho_ant = os.path.join(pasta_modelo, rodada, arquivo_ant)
        
        # 📢 PRINTS SOLICITADOS
        print("\n-----------------------------------------------")
        print(f"Rodada {rodada} — MERGE {merge_date_str}")
        print(f"Atual: {caminho_atual}")
        print(f"Anterior: {caminho_ant}")
        # -----------------------------------------------

        # Verifica se arquivos do modelo existem.
        if not os.path.exists(caminho_atual) or not os.path.exists(caminho_ant):
            print("⚠️ Arquivos do modelo ausentes -> painel vazio")
            # AÇÃO CORRIGIDA 2: Se faltar arquivo, limpa ou oculta o Axes
            if i == 0:
                ax.axis('off') # Mantém o título
            else:
                ax.set_visible(False)
            continue

        # Abre e lê dados do modelo.
        ds_atual = Dataset(caminho_atual, 'r')
        ds_ant = Dataset(caminho_ant, 'r')

        # Extrai coordenadas (tratando 1D ou 2D).
        lons_var = ds_atual.variables['longitude'][:]
        lats_var = ds_atual.variables['latitude'][:]
        if lons_var.ndim == 2: lons_model = lons_var[0, :].copy()
        else: lons_model = lons_var.copy()
        if lats_var.ndim == 2: lats_model = lats_var[:, 0].copy()
        else: lats_model = lats_var.copy()
        lons_model = lon_to_minus180_180(lons_model)

        # Calcula precipitação acumulada em 24h
        rainc_curr = ds_atual.variables['rainc'][0, :, :].astype(np.float64)
        rainnc_curr = ds_atual.variables['rainnc'][0, :, :].astype(np.float64)
        rainc_prev = ds_ant.variables['rainc'][0, :, :].astype(np.float64)
        rainnc_prev = ds_ant.variables['rainnc'][0, :, :].astype(np.float64)         
        rain_model = (rainc_curr + rainnc_curr) - (rainc_prev + rainnc_prev)                        
        ds_atual.close()
        ds_ant.close()
        max_rain_plausible = 1000.0 
        mask_anomalous = (~np.isfinite(rain_model)) | (rain_model < 0) | (rain_model > max_rain_plausible)
        rain_model[mask_anomalous] = np.nan

        # Lê dados observados (MERGE).
        arquivo_merge = f"{pasta_merge}/{merge_date_str[0:4]}/{merge_date_str[4:6]}/MERGE_CPTEC_{merge_date_str}.grib2"
        if not os.path.exists(arquivo_merge):
            print(f"⚠️ MERGE não encontrado -> painel vazio")
            if i == 0:
                ax.axis('off') # Mantém o título
            else:
                ax.set_visible(False)
            continue

        grib = pygrib.open(arquivo_merge)
        #grb = grib.select(name='Precipitation')[0]
        grb = grib.select(shortName='rdp')[0]
        prec_merge, lats_merge, lons_merge = grb.data(lat1=extent[2], lat2=extent[3],
                                                      lon1=extent[0]+360, lon2=extent[1]+360)
        grib.close()
        lons_merge = lon_to_minus180_180(lons_merge)

        # Ajusta eixos e interpola modelo para grade do MERGE
        lat_axis = lats_model.copy(); lon_axis = lons_model.copy()
        mask_model_grid = np.isfinite(rain_model).astype(float)
        if lat_axis[0] > lat_axis[-1]: lat_axis = lat_axis[::-1]; rain_model = rain_model[::-1, :]; mask_model_grid = mask_model_grid[::-1, :]
        if lon_axis[0] > lon_axis[-1]: lon_axis = lon_axis[::-1]; rain_model = rain_model[:, ::-1]; mask_model_grid = mask_model_grid[:, ::-1]
        interp_mask_nn = RegularGridInterpolator((lat_axis, lon_axis), mask_model_grid, method='nearest', bounds_error=False, fill_value=0.0)
        pts_merge = np.vstack((lats_merge.ravel(), lons_merge.ravel())).T
        mask_points = interp_mask_nn(pts_merge).reshape(lats_merge.shape)
        interp_func = RegularGridInterpolator((lat_axis, lon_axis), rain_model, method='linear', bounds_error=False, fill_value=np.nan)
        rain_interp = interp_func(pts_merge).reshape(lats_merge.shape)
        mask_invalid = (mask_points == 0) | np.isnan(rain_interp) | np.isnan(prec_merge)
        
        rain_model_masked = np.ma.masked_where(mask_invalid, rain_interp)
        prec_merge_masked = np.ma.masked_where(mask_invalid, prec_merge)
        
        if lons_merge_grid is None: lons_merge_grid = lons_merge; lats_merge_grid = lats_merge

        # --- CÁLCULO DE MÉTRICAS ---
        rmse_val, bias_val = rmse_and_bias(prec_merge_masked, rain_model_masked)
        precip_track_error10km = precip_track_error_cog(prec_merge_masked, rain_model_masked, lats_merge, lons_merge, threshold=10.0)
        
        lat_1d = lats_merge[:,0]; lon_1d = lons_merge[0,:]
        fss_vals = {s: compute_fss(prec_merge_masked, rain_model_masked, 10.0, s, lat_1d, lon_1d) for s in fss_scales_km}
        csi_vals = {thr: csi_from_counts(*pooled_contingency(prec_merge_masked, rain_model_masked, thr)[:3]) for thr in thresholds}
        csi10_val = csi_vals.get(10, np.nan) 
        lead_h = int((datetime.strptime(previsao, '%Y%m%d%H') - datetime.strptime(rodada, '%Y%m%d%H')).total_seconds() / 3600)

        metrics_for_plots['lead_h'].append(lead_h)
        metrics_for_plots['rmse_val'].append(rmse_val)
        metrics_for_plots['csi10_val'].append(csi10_val) 
        metrics_for_plots['precip_track_error10km_val'].append(precip_track_error10km)

        bias_field = rain_model_masked - prec_merge_masked
        bias_fields_list.append(bias_field) 

        metrics_rows.append({
            'rodada': rodada, 'merge_date': merge_date_str, 'previsao': previsao,
            'lead_h': lead_h, 'rmse_mm': rmse_val, 'bias_mm': bias_val,
            'precip_track_error10km': precip_track_error10km,
            **{f'fss_{s}km': fss_vals[s] for s in fss_scales_km},
            **{f'csi_{thr}mm': csi_vals[thr] for thr in thresholds}
        })

        # --- PLOTAGEM DO PAINEL  ---
        
        ax.axis('on') 
        ax.set_aspect('auto') 

        plot_field_masked = ma.masked_invalid(bias_field)

        pcm = ax.contourf(lons_merge, lats_merge, plot_field_masked, 
                          cmap=cmap_vies, levels=levels_vies, extend='both',
                          transform=ccrs.PlateCarree())
        last_valid_pcm = pcm 

        ax.coastlines(resolution='10m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.4)

        # Configuração de gridlines básica
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}
            
        # LÓGICA DE RÓTULOS (MANTIDA)
        is_bottom_border = (i == ultima_linha_com_dado[j])
        is_specific_lon = (i == 0 and j == 0) or (i == 1 and j == 1) or (i == 2 and j == 2)
        if is_bottom_border or is_specific_lon: gl.bottom_labels = True
        else: gl.bottom_labels = False
        is_left_border = (j == primeira_coluna_com_dado[i])
        is_specific_lat = (i == 2 and j == 2) or (i == 3 and j == 3) or (i == 4 and j == 4)
        if is_left_border or is_specific_lat: gl.left_labels = True
        else: gl.left_labels = False
    
    # ----------------------------------------------------
    # Processamento de métricas para plotagem final e CSV
    # ----------------------------------------------------
    if metrics_for_plots['lead_h']:
        df_plots = pd.DataFrame(metrics_for_plots)
        df_plots_grouped = df_plots.groupby('lead_h').agg(
            rmse_mean=('rmse_val', 'mean'), csi10_mean=('csi10_val', 'mean'),
            precip_track_error10km_mean=('precip_track_error10km_val', 'mean') 
        ).reset_index()
        all_metrics_by_rodada[rodada] = df_plots_grouped

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        csv_out = os.path.join(out_dir, f"metrics_{rodada}.csv")
        df.to_csv(csv_out, index=False)
    
    # ---------- Bias médio agregado (mapa) (POR RODADA) ----------
    if bias_fields_list and lons_merge_grid is not None:
        bias_sum = np.nansum(np.dstack(bias_fields_list), axis=2) 
        bias_count = np.sum(~np.isnan(np.dstack(bias_fields_list)), axis=2) 
        bias_mean = np.full(bias_sum.shape, np.nan)
        mask_valid_mean = bias_count > 0 
        bias_mean[mask_valid_mean] = bias_sum[mask_valid_mean] / bias_count[mask_valid_mean]
        
        fig3 = plt.figure(figsize=(8,6))
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        ax3.set_extent([extent[0], extent[1], extent[2], extent[3]])
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.4)
        gl3 = ax3.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4, linestyle='--')
        gl3.top_labels = False; gl3.right_labels = False
        gl3.xlabel_style = {'size': 8}; gl3.ylabel_style = {'size': 8}
        gl3.bottom_labels = True; gl3.left_labels = True
        pcm_agg = ax3.contourf(lons_merge_grid, lats_merge_grid, ma.masked_invalid(bias_mean), 
                                transform=ccrs.PlateCarree(), cmap=cmap_vies, levels=levels_vies, extend='both')
        plt.colorbar(pcm_agg, ax=ax3, label='Bias (mm)', extend='both', ticks=levels_vies) 
        ax3.set_title(f'Bias médio agregado (rodada {rodada})')
        fpath3 = os.path.join(out_dir, f"bias_mean_aggregated_{rodada}.png")
        fig3.savefig(fpath3, dpi=200, bbox_inches='tight')
        plt.close(fig3)


# --- PLOTAGEM FINAL DAS MÉTRICAS (GRÁFICOS DE LINHA SIMPLES) ---
if all_metrics_by_rodada:
    # RMSE
    fig_rmse, ax_rmse = plt.subplots(figsize=(9, 5))
    ax_rmse.set_title('RMSE vs lead (Separado por Rodada)'); ax_rmse.set_xlabel('Lead (h)'); ax_rmse.set_ylabel('RMSE (mm)'); ax_rmse.grid(True, linestyle='--', alpha=0.6)
    for rodada, data in all_metrics_by_rodada.items(): ax_rmse.plot(data['lead_h'], data['rmse_mean'], '-o', label=f'Rodada {rodada}')
    ax_rmse.legend(loc='upper right')
    fig_rmse.savefig(os.path.join(out_dir, "rmse_vs_lead_all_rodadas.png"), dpi=200, bbox_inches='tight'); plt.close(fig_rmse)
    # CSI
    fig_csi, ax_csi = plt.subplots(figsize=(9, 5))
    ax_csi.set_title('CSI (>10mm) vs lead (Separado por Rodada)'); ax_csi.set_xlabel('Lead (h)'); ax_csi.set_ylabel('CSI (>10mm)'); ax_csi.grid(True, linestyle='--', alpha=0.6)
    for rodada, data in all_metrics_by_rodada.items(): ax_csi.plot(data['lead_h'], data['csi10_mean'], '-s', label=f'Rodada {rodada}')
    ax_csi.legend(loc='upper right'); ax_csi.set_ylim(0, 1.1) 
    fig_csi.savefig(os.path.join(out_dir, "csi10_vs_lead_all_rodadas.png"), dpi=200, bbox_inches='tight'); plt.close(fig_csi)
    # Erro de Posição
    fig_track_new, ax_track_new = plt.subplots(figsize=(9, 5))
    ax_track_new.set_title('Erro de Posição da Precipitação (COG > 10mm) vs lead'); ax_track_new.set_xlabel('Lead (h)'); ax_track_new.set_ylabel('Erro Posição COG (km)'); ax_track_new.grid(True, linestyle='--', alpha=0.6)
    for rodada, data in all_metrics_by_rodada.items(): ax_track_new.plot(data['lead_h'], data['precip_track_error10km_mean'], '-o', label=f'Rodada {rodada}')
    ax_track_new.legend(loc='upper left')
    fig_track_new.savefig(os.path.join(out_dir, "precip_track_error_cog_vs_lead.png"), dpi=200, bbox_inches='tight'); plt.close(fig_track_new)


# --- SALVAMENTO FINAL DO PAINEL GERAL (APENAS VIÉS) ---
if last_valid_pcm is not None:
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.015])
    fig.colorbar(last_valid_pcm, cax=cbar_ax, orientation='horizontal', ticks=levels_vies,
                 label='Viés da precipitação acumulada em 24h (mm) — Modelo - MERGE', extend='both')

    panel_out = os.path.join(out_dir, "panel_all.png")
    fig.savefig(panel_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
print("\nProcessamento finalizado.")
