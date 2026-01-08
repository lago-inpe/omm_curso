#!/usr/bin/env python3
import os
from datetime import datetime, timedelta  # Para manipular datas e calcular intervalos de tempo.
import numpy as np  # Para cálculos com matrizes.
import matplotlib.pyplot as plt  # Para criar gráficos.
import matplotlib  # Biblioteca de plotagem.
import cartopy.crs as ccrs  # Projeções de mapas.
import cartopy.feature as cfeature  # Elementos geográficos.
from netCDF4 import Dataset  # Leitura de arquivos NetCDF (modelo).
import pygrib  # Leitura de arquivos GRIB (observações MERGE).
from scipy.interpolate import RegularGridInterpolator  # Para interpolar dados entre grades diferentes.

# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
# Datas das rodadas (início das simulações) e previsões (futuro previsto).
rodadas = ["2022100500", "2022100600", "2022100700", "2022100800", "2022100900"]
previsoes = ["2022100600", "2022100700", "2022100800", "2022100900", "2022101000"]

# Limites geográficos do mapa: longitude e latitude.
extent = [-95, -60, 0, 35]

# Se True, plota viés (diferença entre modelo e observação). Se False, plota só precipitação do modelo.
plot_vies = True

# Níveis e cores para precipitação ou viés.
rain_levels = [0, 1, 5, 10, 20, 30, 40, 50]
colors = ['#9c0720', '#dc143c', '#f1666d', '#ff9ea2', '#f0c6f0', '#ffffff',
          '#87CEEB', '#00BFFF', '#1E90FF', '#4169E1', '#0000FF']
cmap2 = matplotlib.colors.ListedColormap(colors)  # Paleta de cores para viés.
cmap2.set_over('#081d58')  # Cor para valores acima do máximo.
cmap2.set_under('#610000')  # Cor para valores abaixo do mínimo.
data_min = -60
data_max = 70
interval = 10
levels2 = np.arange(data_min, data_max, interval)  # Intervalos para viés.

# ---------------------------
# FUNÇÃO AUX - ajusta longitudes
# ---------------------------
# Converte longitudes de 0-360 para -180 a 180, padrão usado em mapas.
def lon_to_minus180_180(lon_array):
    """Converte longitudes 0..360 para -180..180 (se necessário)."""
    lon = np.array(lon_array)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon

# ---------------------------
# PREPARA FIGURA EM PAINEL
# ---------------------------
# Cria grade de painéis para os mapas.
fig, axs = plt.subplots(len(rodadas), len(previsoes),
                        figsize=(16, 16),
                        subplot_kw=dict(projection=ccrs.PlateCarree()),
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
axs = np.atleast_2d(axs)  # Garante que axs seja uma matriz 2D.

# Detecta quais painéis têm dados para rotular corretamente.
primeira_coluna_com_dado = [None] * len(rodadas)
ultima_linha_com_dado = [None] * len(previsoes)
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        if rodada == previsao:
            continue
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
        caminho = os.path.join(rodada, arquivo)
        if os.path.exists(caminho):
            if primeira_coluna_com_dado[i] is None:
                primeira_coluna_com_dado[i] = j
            ultima_linha_com_dado[j] = i

# ---------------------------
# LOOP PRINCIPAL
# ---------------------------
# Variável para guardar o último contorno válido (para a colorbar).
last_valid_cf = None
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        ax = axs[i, j]
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())

        # Adiciona rótulo da rodada na primeira coluna visível.
        if j == primeira_coluna_com_dado[i]:
            ax.text(-0.3, 0.5, f"Rodada: {rodada}", fontsize=12, va='center',
                    rotation=90, transform=ax.transAxes)

        # try:
        if rodada == previsao:
            ax.set_visible(False)
            continue

        # Calcula datas para os arquivos MERGE (observações fecham às 12Z).
        merge_date_str = previsao[:8]
        merge_date = datetime.strptime(merge_date_str, '%Y%m%d')
        target_current = merge_date.strftime('%Y%m%d') + '12'
        prev_date = (merge_date - timedelta(days=1))
        target_prev = prev_date.strftime('%Y%m%d') + '12'

        # Define caminhos dos arquivos do modelo (atual e anterior para acumulado).
        arquivo_atual = f"MONAN_DIAG_R_POS_GFS_{rodada}_{target_current}.00.00.x1.5898242L55.nc"
        arquivo_ant = f"MONAN_DIAG_R_POS_GFS_{rodada}_{target_prev}.00.00.x1.5898242L55.nc"
        caminho_atual = ('/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia/' + rodada + '/' + arquivo_atual)
        caminho_ant = ('/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia/' + rodada + '/' + arquivo_ant)

        print("\n-----------------------------------------------")
        print(f"Rodada {rodada} — MERGE {merge_date_str}")
        print(f"Atual: {caminho_atual}")
        print(f"Anterior: {caminho_ant}")

        # Verifica se arquivos do modelo existem.
        if not os.path.exists(caminho_atual) or not os.path.exists(caminho_ant):
            print("⚠️ Arquivos do modelo ausentes -> painel vazio")
            ax.set_visible(False)
            continue

        # Abre e lê dados do modelo.
        ds_atual = Dataset(caminho_atual, 'r')
        ds_ant = Dataset(caminho_ant, 'r')

        # Extrai coordenadas (tratando 1D ou 2D).
        lons_var = ds_atual.variables['longitude'][:]
        lats_var = ds_atual.variables['latitude'][:]
        if lons_var.ndim == 2:
            lons_model = lons_var[0, :].copy()
        else:
            lons_model = lons_var.copy()
        if lats_var.ndim == 2:
            lats_model = lats_var[:, 0].copy()
        else:
            lats_model = lats_var.copy()
        lons_model = lon_to_minus180_180(lons_model)

        # Calcula precipitação acumulada em 24h (subtrai dia anterior do atual).
        rainc_curr = ds_atual.variables['rainc'][0, :, :].astype(float)
        rainnc_curr = ds_atual.variables['rainnc'][0, :, :].astype(float)
        rainc_prev = ds_ant.variables['rainc'][0, :, :].astype(float)
        rainnc_prev = ds_ant.variables['rainnc'][0, :, :].astype(float)         
        rain_curr = rainc_curr + rainnc_curr
        rain_prev = rainc_prev + rainnc_prev                        
        ds_atual.close()
        ds_ant.close()
        rain_model = rain_curr - rain_prev

        # MODIFICAÇÃO AQUI: Adapta o caminho do arquivo MERGE para a estrutura /ano/mes
        merge_year = merge_date_str[:4]
        merge_month = merge_date_str[4:6]
        
        # Constrói o caminho completo: 'merge/ano/mes/MERGE_CPTEC_YYYYMMDD.grib2'
        merge_dir = os.path.join('/oper/share/ioper/tempo/MERGE/GPM/DAILY/', merge_year, merge_month)
        arquivo_merge_nome = f"MERGE_CPTEC_{merge_date_str}.grib2"
        arquivo_merge = os.path.join(merge_dir, arquivo_merge_nome)
        
        print(f"MERGE esperado em: {arquivo_merge}")

        if not os.path.exists(arquivo_merge):
            print(f"⚠️ MERGE não encontrado em {arquivo_merge} -> painel vazio")
            ax.set_visible(False)
            continue

        grib = pygrib.open(arquivo_merge)
        grb = grib.select(shortName='rdp')[0]
        # grb = grib.select(name='Precipitation')[0]
        prec_merge, lats_merge, lons_merge = grb.data(lat1=extent[2], lat2=extent[3],
                                                        lon1=extent[0]+360, lon2=extent[1]+360)
        grib.close()
        lons_merge = lon_to_minus180_180(lons_merge)
        # FIM DA MODIFICAÇÃO

        # Ajusta eixos do modelo e cria máscara de dados válidos.
        lat_axis = lats_model.copy()
        lon_axis = lons_model.copy()
        mask_model_grid = np.isfinite(rain_model).astype(float)
        if lat_axis[0] > lat_axis[-1]:
            lat_axis = lat_axis[::-1]
            rain_model = rain_model[::-1, :]
            mask_model_grid = mask_model_grid[::-1, :]
        if lon_axis[0] > lon_axis[-1]:
            lon_axis = lon_axis[::-1]
            rain_model = rain_model[:, ::-1]
            mask_model_grid = mask_model_grid[:, ::-1]

        # Interpola máscara (método nearest neighbor).
        interp_mask_nn = RegularGridInterpolator((lat_axis, lon_axis), mask_model_grid,
                                                method='nearest', bounds_error=False, fill_value=0.0)
        pts_merge = np.vstack((lats_merge.ravel(), lons_merge.ravel())).T
        mask_points = interp_mask_nn(pts_merge).reshape(lats_merge.shape)

        # Interpola precipitação do modelo para a grade do MERGE (método linear).
        interp_func = RegularGridInterpolator((lat_axis, lon_axis), rain_model,
                                                method='linear', bounds_error=False, fill_value=np.nan)
        rain_interp_flat = interp_func(pts_merge)
        rain_interp = rain_interp_flat.reshape(lats_merge.shape)

        # Cria máscara final e calcula viés (modelo - MERGE).
        mask_invalid = (mask_points == 0) | np.isnan(rain_interp) | np.isnan(prec_merge)
        rain_interp_masked = np.ma.masked_where(mask_invalid, rain_interp)
        prec_merge_masked = np.ma.masked_where(mask_invalid, prec_merge)
        vies = np.ma.masked_where(mask_invalid, rain_interp - prec_merge)

        # Adiciona elementos geográficos ao mapa.
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, linewidth=0.2)

        # Plota contorno: viés ou precipitação, conforme configuração.
        if plot_vies:
            cf = ax.contourf(lons_merge, lats_merge, vies, levels=levels2,
                                cmap=cmap2, extend='both', transform=ccrs.PlateCarree())
        else:
            cf = ax.contourf(lons_merge, lats_merge, rain_interp_masked, levels=rain_levels,
                                cmap='Blues', extend='max', transform=ccrs.PlateCarree())
        last_valid_cf = cf

        # Adiciona linhas de grade e rótulos.
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        if j != primeira_coluna_com_dado[i]:
            gl.left_labels = False
        if i != ultima_linha_com_dado[j]:
            gl.bottom_labels = False
        if i == 0:
            ax.set_title(previsao, fontsize=13)

        # except Exception as e:
        #     print(f"❌ Erro rodada {rodada}, previsão {previsao}: {e}")
        #     ax.set_visible(False)
        #     continue

# ---------------------------
# COLORBAR E SALVAMENTO
# ---------------------------
print("\n💾 Salvando figura final...")
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015])
if last_valid_cf is not None:
    if plot_vies:
        fig.colorbar(last_valid_cf, cax=cbar_ax, orientation='horizontal', ticks=levels2,
                     label='Viés da precipitação acumulada em 24h (mm) — Modelo - MERGE')
    else:
        fig.colorbar(last_valid_cf, cax=cbar_ax, orientation='horizontal', ticks=rain_levels,
                     label='Precipitação acumulada em 24h (mm) — Modelo')

plt.suptitle("Furacão Julia (2022). Viés da Precipitação Acumulada (24h) — Modelo vs MERGE", fontsize=15, y=0.95)
outfn = "furacao_julia_vies_precipitacao_jul.png"
plt.savefig(outfn, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Figura salva: {os.path.abspath(outfn)}")
