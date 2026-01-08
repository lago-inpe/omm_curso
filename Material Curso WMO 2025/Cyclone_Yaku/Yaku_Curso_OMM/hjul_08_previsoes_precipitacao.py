import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# --- CONFIGURAÇÕES ---
rodadas = ["2023030700", "2023030800", "2023030900", "2023031000", "2023031100"]
previsoes = ["2023030800", "2023030900", "2023031000", "2023031100", "2023031200"]

lon_min, lon_max = -95, -70
lat_min, lat_max = -20, 0

# Limiares de precipitação: Começa em 5 mm
rain_levels = [5, 10, 20, 30, 40, 50, 100]

# Criação de um Colormap customizado para maior contraste
# 1. Baseado no colormap 'Blues'
base_cmap = cm.get_cmap('Blues', 256) 
# 2. Usar a metade mais intensa do Blues
new_colors = base_cmap(np.linspace(0.5, 1, 256))
custom_cmap = ListedColormap(new_colors)

# --- FIGURA ---
fig, axs = plt.subplots(
    len(rodadas), len(previsoes),
    figsize=(16, 14),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    gridspec_kw={'hspace':0.05, 'wspace':0.05}
)

# --- DETECTAR PRIMEIRA COLUNA E ÚLTIMA LINHA COM DADO ---
primeira_coluna_com_dado = [None]*len(rodadas)
ultima_linha_com_dado = [None]*len(previsoes)

for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        if rodada == previsao:
            continue
        pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
        caminho = os.path.join(pasta, arquivo)
        if os.path.exists(caminho):
            if primeira_coluna_com_dado[i] is None:
                primeira_coluna_com_dado[i] = j
            ultima_linha_com_dado[j] = i

# --- LOOP DE PLOTAGEM ---
cf = None 
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        ax = axs[i,j]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        if j == primeira_coluna_com_dado[i]:
            ax.set_visible(True)
            ax.text(-0.2, 0.5, f"Rodada: {rodada}", fontsize=12, va='center', rotation=90, transform=ax.transAxes)

        try:
            if rodada == previsao:
                ax.set_visible(False)
                continue

            pasta = pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')
            arquivo_atual = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
            caminho_atual = os.path.join(pasta, arquivo_atual)
            
            if not os.path.exists(caminho_atual):
                # Mantido o print de arquivo não encontrado
                print(f"⚠️ Arquivo não encontrado: {caminho_atual}")
                ax.set_visible(False)
                continue
            
            # Mantido o print de arquivo sendo lido
            print(f"🔹 Lendo: {caminho_atual}")
            dado_atual = Dataset(caminho_atual, 'r')

            lons = dado_atual.variables['longitude'][:]
            lats = dado_atual.variables['latitude'][:]
            rain_atual = dado_atual.variables['rainnc'][0,:,:]

            # --- PRECIPITAÇÃO ACUMULADA 24H ---
            if j == primeira_coluna_com_dado[i]:
                rain = rain_atual
            else:
                previsao_anterior = previsoes[j-1]
                arquivo_ant = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao_anterior}.00.00.x1.5898242L55.nc"
                caminho_ant = os.path.join(pasta, arquivo_ant)
                
                if os.path.exists(caminho_ant):
                    dado_ant = Dataset(caminho_ant, 'r')
                    rain_ant = dado_ant.variables['rainnc'][0,:,:]
                    rain = rain_atual - rain_ant
                    dado_ant.close()
                else:
                    # Mantido o print de arquivo anterior não encontrado
                    print(f"⚠️ Arquivo anterior não encontrado: {caminho_ant}, painel vazio")
                    ax.set_visible(False)
                    dado_atual.close()
                    continue

            # --- PLOT ---
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.STATES, linewidth=0.2)

            # Precipitação: extend='both' para ter cores acima e abaixo
            cf = ax.contourf(lons, lats, rain, levels=rain_levels,
                             cmap=custom_cmap, extend='both', transform=ccrs.PlateCarree())

            # Define a cor para a faixa abaixo (0-5 mm) como branco
            cf.cmap.set_under('white')
            
            # Define a cor para a faixa acima ( > 150 mm) como magenta
            cf.cmap.set_over('magenta') 

            # --- GRIDLINES ---
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size':6}
            gl.ylabel_style = {'size':6}

            if j == primeira_coluna_com_dado[i]:
                gl.left_labels = True
            else:
                gl.left_labels = False

            if i != ultima_linha_com_dado[j]:
                gl.bottom_labels = False

            if i==0:
                ax.set_title(previsao, fontsize=12)

            dado_atual.close()

        except Exception as e:
            print(f"❌ Erro rodada {rodada}, previsão {previsao}: {e}")
            ax.set_visible(False)
            continue

# --- COLORBAR E SALVAMENTO ---
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015])

# Ticks são os limites de precipitação
cbar_ticks = rain_levels
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', 
                    ticks=cbar_ticks,
                    # Adiciona 'max' para mostrar o triângulo da cor 'over' na colorbar
                    extend='max', 
                    label='Precipitação acumulada (mm)')

# Rótulos ajustados: Começa em 5 e não inclui o 0
cbar.set_ticklabels(['5', '10', '25', '50', '75', '100', '150']) 

plt.suptitle("Precipitação acumulada (24h)", fontsize=15, y=0.94)
plt.savefig("previsoes_precip.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva como previsoes_precip.png")
