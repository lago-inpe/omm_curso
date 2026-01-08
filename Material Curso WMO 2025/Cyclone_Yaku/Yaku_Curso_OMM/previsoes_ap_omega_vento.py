import os
from netCDF4 import Dataset  # Para ler arquivos NetCDF.
import numpy as np  # Para cálculos com matrizes.
import matplotlib.pyplot as plt  # Para gráficos e mapas.
import cartopy.crs as ccrs  # Projeções de mapas.
import cartopy.feature as cfeature  # Elementos geográficos.
from scipy.ndimage import gaussian_filter  # Para suavizar dados (usado em omega).
import matplotlib  # Biblioteca de plotagem.

# --- CONFIGURAÇÕES ---
# Datas das rodadas e previsões.
rodadas = ["2023030700", "2023030800", "2023030900", "2023031000", "2023031100"]
previsoes = ["2023030800", "2023030900", "2023031000", "2023031100", "2023031200"]

# Limites geográficos do mapa.
lon_min, lon_max = -90, -70
lat_min, lat_max = -20, 5

# Limiares para precipitação (mantido, mas não usado aqui).
rain_levels = [0, 1, 5, 10, 20, 30, 40, 50]

# --- FIGURA ---
# Cria grade de painéis para os mapas.
fig, axs = plt.subplots(
    len(rodadas), len(previsoes),
    figsize=(16, 16),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    gridspec_kw={'hspace':0.05, 'wspace':0.05}
)

# --- DETECTAR PRIMEIRA COLUNA E ÚLTIMA LINHA COM DADO ---
# Identifica painéis com dados para rotular corretamente.
primeira_coluna_com_dado = [None]*len(rodadas)
ultima_linha_com_dado = [None]*len(previsoes)

for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        if rodada == previsao:
            continue
        pasta = rodada
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
        caminho = os.path.join(pasta, arquivo)
        if os.path.exists(caminho):
            if primeira_coluna_com_dado[i] is None:
                primeira_coluna_com_dado[i] = j
            ultima_linha_com_dado[j] = i

# --- LOOP DE PLOTAGEM ---
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        ax = axs[i,j]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Rótulo da rodada na primeira coluna visível.
        if j == primeira_coluna_com_dado[i]:
            ax.set_visible(True)
            ax.text(-0.3, 0.5, f"Rodada: {rodada}", fontsize=12, va='center', rotation=90, transform=ax.transAxes)

        try:
            if rodada == previsao:
                ax.set_visible(False)
                continue

            pasta = rodada
            arquivo_atual = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
            caminho_atual = os.path.join(pasta, arquivo_atual)
            if not os.path.exists(caminho_atual):
                print(f"⚠️ Arquivo não encontrado: {caminho_atual}")
                ax.set_visible(False)
                continue

            print(f"🔹 Lendo: {caminho_atual}")
            dado_atual = Dataset(caminho_atual, 'r')

            # Extrai variáveis: água precipitável, vento em 850 hPa, omega em 500 hPa.
            lons = dado_atual.variables['longitude'][:]
            lats = dado_atual.variables['latitude'][:]
            ap = dado_atual.variables['precipw'][0,:,:]  # Quantidade de água na atmosfera (mm).
            u850 = dado_atual.variables['uzonal'][0,2,:,:]  # Vento leste-oeste a 850 hPa.
            v850 = dado_atual.variables['umeridional'][0,2,:,:]  # Vento norte-sul a 850 hPa.
            omega = dado_atual.variables['omega'][0,5,:,:]  # Movimento vertical do ar a 500 hPa (Pa/s).

            # --- PLOT ---
            # Adiciona elementos geográficos.
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.STATES, linewidth=0.2)

            # Define intervalos para água precipitável.
            data_min = 30
            data_max = 75 
            interval = 5
            levels_ap = np.arange(data_min, data_max, interval)

            # Paleta de cores para água precipitável.
            colors = ["#b4f0f0", "#96d2fa", "#78b9fa", "#3c95f5", "#1e6deb", "#1463d2", 
                      "#0fa00f", "#28be28", "#50f050", "#72f06e", "#b3faaa", "#fff9aa", 
                      "#ffe978", "#ffc13c", "#ffa200", "#ff6200", "#ff3300", "#ff1500", 
                      "#c00100", "#a50200", "#870000", "#653b32"]
            cmap = matplotlib.colors.ListedColormap(colors)
            cmap.set_over('#000000')
            cmap.set_under('#ffffff')

            # Plota contorno preenchido de água precipitável.
            img1 = ax.contourf(lons, lats, ap, cmap=cmap, levels=levels_ap, extend='max', alpha=0.85)

            # Ticks para a colorbar.
            ticks = [30, 35, 40, 45, 50, 55, 60, 65, 70]

            # Barbelas de vento a 850 hPa.
            flip = np.zeros((u850.shape[0], u850.shape[1]))
            flip[lats < 0] = 1
            img_barbs = ax.barbs(
                lons[::20], lats[::20],
                u850[::20, ::20], v850[::20, ::20],
                length=5.0,
                sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
                linewidth=0.8,
                pivot='middle',
                barbcolor='dimgray',
                flip_barb=flip[::20, ::20],
                transform=ccrs.PlateCarree()
            )

            # Intervalos para omega (movimento vertical).
            data_min = -5
            data_max = -0.1 
            interval = 0.2
            levels = np.arange(data_min, data_max, interval)

            # Plota contorno de omega (suavizado).
            img5 = ax.contour(lons, lats, gaussian_filter(omega, 2), cmap='seismic', linestyles='dashed', linewidths=1.0, levels=levels, zorder=2)

            # --- GRIDLINES ---
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='darkgray', alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size':9}
            gl.ylabel_style = {'size':9}

            if j == primeira_coluna_com_dado[i]:
                gl.left_labels = True
            else:
                gl.left_labels = False

            if i != ultima_linha_com_dado[j]:
                gl.bottom_labels = False

            if i==0:
                ax.set_title(previsao, fontsize=13)

            dado_atual.close()

        except Exception as e:
            print(f"❌ Erro rodada {rodada}, previsão {previsao}: {e}")
            ax.set_visible(False)
            continue

# --- COLORBAR E SALVAMENTO ---
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015])
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='horizontal', ticks=ticks,
                    label='Água Precipitável na Atmosfera (mm)')

plt.suptitle("Água Precipitável (mm), Vento em 850 hPa e Omega (Pa/s) em 500 hPa", fontsize=15, y=0.95)
plt.savefig("previsao_ap_omega_vento.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva como previsao_ap_omega_vento.png")
