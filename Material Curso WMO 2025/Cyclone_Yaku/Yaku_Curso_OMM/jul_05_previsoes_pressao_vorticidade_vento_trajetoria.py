import os
from netCDF4 import Dataset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter 
import matplotlib
import metpy.calc as mpcalc
from metpy.units import units

# --- CONFIGURAÇÕES ---
# Datas das rodadas (início das simulações) e previsões (datas futuras previstas).
#rodadas = ["2023030700", "2023030800", "2023030900", "2023031000", "2023031100"]
#previsoes = ["2023030800", "2023030900", "2023031000", "2023031100", "2023031200"]
rodadas = ["2022100500", "2022100600", "2022100700", "2022100800", "2022100900", "2022101000", "2022101100"]
previsoes = ["2022100600", "2022100700", "2022100800", "2022100900", "2022101000", "2022101100", "2022101200"]


# Template do nome do arquivo
file_template = "MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
out_file = "previsoes_vorticidade_pressao_vento_850hpa.png"
pasta = "/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia/"

# Limites geográficos do mapa (ajustados para a área de interesse)
#lon_min, lon_max = -104, -76
#lat_min, lat_max = -24, 4


lon_min, lon_max = -90, -60
lat_min, lat_max = 5, 25
# --- DADOS DE OBSERVAÇÃO ---
# time, lon_obs, lat_obs
OBS_DATA = {
    "2023030700": {"lon": -87.6995, "lat": -7.7700},
    "2023030800": {"lon": -85.7371, "lat": -6.5211},
    "2023030900": {"lon": -85.5587, "lat": -6.4319},
    "2023031000": {"lon": -85.7371, "lat": -7.2348},
    "2023031100": {"lon": -84.0423, "lat": -8.9296},
    "2023031200": {"lon": -84.1315, "lat": -10.5352}
}

# --- PARÂMETROS DE VISUALIZAÇÃO ---
# Nível de pressão 850 hPa (Índice vertical para o seu dataset)
NIVEL_PRESSAO = 2 

# Suavização da Vorticidade (sigma em pontos de grade)
SMOOTH_SIGMA = 3.0

# Contornos de Pressão (MSLP)
pressao_levels = np.arange(990, 1024, 2) # Isolinas de 2 em 2 hPa

# Escala de Vorticidade (s^-1)
vort_min = -10e-5 # Valor mínimo (azul escuro)
vort_max = 10e-5  # Valor máximo (vermelho escuro)
vort_interval = 1e-5
vort_levels = np.arange(vort_min, vort_max + vort_interval, vort_interval)

# --- FUNÇÃO AUXILIAR ---
# Função para converter longitudes para o intervalo padrão de -180 a 180 graus.
def lon_to_minus180_180(lon_arr):
    lon = np.array(lon_arr)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon

# --- FIGURA ---
# Cria uma grade de painéis (subgráficos)
n_linhas, n_colunas = len(rodadas), len(previsoes)
fig, axs = plt.subplots(
    n_linhas, n_colunas,
    figsize=(16,16),
    subplot_kw=dict(projection=ccrs.PlateCarree())
)

# Ajusta o espaçamento para as barbelas e títulos
fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.9, bottom=0.15)

# --- DETECÇÃO DE DADOS VÁLIDOS ---
# Arrays para ajudar a definir quais painéis precisam de rótulos de latitude/longitude
tem_dado = np.zeros((n_linhas, n_colunas), dtype=bool)

# Loop principal
for i, rodada in enumerate(rodadas):
    rodada_dt = datetime.strptime(rodada, "%Y%m%d%H")
    
    for j, previsao in enumerate(previsoes):
        previsao_dt = datetime.strptime(previsao, "%Y%m%d%H")
        
        # Garante que 'ax' seja um objeto de eixo único
        if n_linhas == 1 and n_colunas == 1:
            ax = axs
        elif n_linhas == 1:
            ax = axs[j]
        elif n_colunas == 1:
            ax = axs[i]
        else:
            ax = axs[i, j]
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        # Lógica para omitir painel da análise (rodada == previsao) 
        if rodada == previsao:
            ax.set_visible(False)
            continue
        
        # Pula a iteração se a data da previsão for anterior à rodada
        if previsao_dt < rodada_dt:
            ax.set_visible(False)
            continue
        
        try:
            # 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
            fpath = os.path.join(pasta, rodada, file_template.format(rodada=rodada, previsao=previsao))
            dado_atual = Dataset(fpath, 'r')
            
            lons = lon_to_minus180_180(dado_atual.variables["longitude"][:])
            lats = dado_atual.variables["latitude"][:]
            
            # Pressão ao Nível Médio do Mar (MSLP)
            mslp = dado_atual.variables["mslp"][0, :, :]
            if np.nanmean(mslp) > 2000: mslp /= 100.0 # Conversão de Pa para hPa
            
            # Vento em 850 hPa
            u850 = dado_atual.variables["uzonal"][0, NIVEL_PRESSAO, :, :]
            v850 = dado_atual.variables["umeridional"][0, NIVEL_PRESSAO, :, :]

            # 2. CÁLCULO DA VORTICIDADE EM 850 hPa
            
            u_q = units.Quantity(u850, "m/s")
            v_q = units.Quantity(v850, "m/s")
            
            dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
            vort_q = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy)
            vort = vort_q.to('1/s').magnitude
            
            # Suaviza a vorticidade (Usando SciPy para robustez)
            vort_temp = np.nan_to_num(vort, nan=0.0) 
            vort_smoothed = gaussian_filter(vort_temp, sigma=SMOOTH_SIGMA, mode='nearest', truncate=3.0)

            # 3. PLOTAGEM DOS CAMPOS
            
            # A) Vorticidade em Cores (Contorno Preenchido)
            img1 = ax.contourf(lons, lats, vort_smoothed, 
                               levels=vort_levels, 
                               cmap='RdBu_r', 
                               extend='both', 
                               transform=ccrs.PlateCarree())

            # B) Pressão ao Nível Médio do Mar (MSLP) em Isolinhas
            img2 = ax.contour(lons, lats, mslp, 
                              levels=pressao_levels, 
                              colors='black', 
                              linewidths=1, 
                              transform=ccrs.PlateCarree(),
                              zorder=3) 

            ax.clabel(img2, inline=1, fontsize=10, fmt='%d', colors='k')

            # C) Vento em 850 hPa em Barbelas (Quiver)
            skip = 20 # Subamostragem
            ax.barbs(lons[::skip], lats[::skip], u850[::skip, ::skip], v850[::skip, ::skip], 
                     length=3, color='k', linewidth=0.5, zorder=4)
            
            # D) Ponto de Observação (adicionado)
            if previsao in OBS_DATA:
                lon_obs = OBS_DATA[previsao]["lon"]
                lat_obs = OBS_DATA[previsao]["lat"]
                
                # Plota o ponto de observação na data de previsão
                ax.plot(lon_obs, lat_obs, 
                        marker='o',         # Marcador de círculo
                        color='red',        # Cor vermelha
                        markersize=8,       # Tamanho maior para destaque
                        transform=ccrs.PlateCarree(), # TRANSFORMAÇÃO CRUCIAL
                        zorder=5,           # Garante que o ponto esteja no topo
                        label=f"Obs {previsao}")

            # 4. ELEMENTOS DO MAPA E TÍTULOS
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            tem_dado[i, j] = True

            dado_atual.close()

        except Exception as e:
            # O painel é desativado em caso de erro (ex: arquivo corrompido ou outro problema)
            print(f"❌ Erro rodada {rodada}, previsão {previsao}: {e}")
            ax.set_visible(False)
            continue

# --- RÓTULOS E GRIDLINES ---

# Ajuste para lidar com 1 linha ou 1 coluna
if n_linhas == 1 and n_colunas > 1:
    axs = np.expand_dims(axs, axis=0)
elif n_colunas == 1 and n_linhas > 1:
    axs = np.expand_dims(axs, axis=1)
elif n_linhas == 1 and n_colunas == 1:
     axs = np.array([[axs]])
     
primeira_coluna_com_dado = np.argmax(tem_dado, axis=1)
ultima_linha_com_dado = (n_linhas - 1) - np.argmax(tem_dado[::-1, :], axis=0) 

# Loop para configurar gridlines e títulos de linha/coluna
for i in range(n_linhas):
    for j in range(n_colunas):
        ax = axs[i, j]

        if not tem_dado[i, j]:
            continue

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='darkgray', alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size':9}
        gl.ylabel_style = {'size':9}

        if j == primeira_coluna_com_dado[i]:
            gl.left_labels = True
            # Adiciona rótulo vertical da Rodada - REINCORPORADO
            ax.text(-0.25, 0.5, f"Rodada: {rodadas[i]}", fontsize=12, 
                    va='center', rotation=90, transform=ax.transAxes)
        else:
            gl.left_labels = False
            # Remove o set_ylabel que estava duplicando a função

        if i == ultima_linha_com_dado[j]:
            gl.bottom_labels = True
        else:
            gl.bottom_labels = False
        
        # Título da data da previsão
        if i == 0:
            ax.set_title(previsoes[j], fontsize=13)


# --- COLORBAR E SALVAMENTO ---

# Adiciona uma barra de cores na parte inferior para a Vorticidade
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.015]) # [esquerda, baixo, largura, altura]
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='horizontal', 
                    label=r'Vorticidade Relativa em 850 hPa ($\mathrm{x} 10^{-5} \mathrm{s}^{-1}$)',
                    format='%.1f')

# Correção para o espaçamento dos rótulos
cbar_ticks_unidades = np.array([-10, -5, 0, 5, 10]) 
cbar_ticks_s = cbar_ticks_unidades / 1e5

cbar.set_ticks(cbar_ticks_s)
cbar.set_ticklabels([f"{t:.0f}" for t in cbar_ticks_unidades])


# Título geral da figura
plt.suptitle("Vorticidade (Cores) e Vento (Barbelas) em 850 hPa; Pressão ao Nível Médio do Mar (Isolinhas)\nLocalização de ciclone extraída das análises (Círculo Vermelho)", 
             fontsize=15, y=0.98)
             
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figura salva como: {out_file}")
