#!/usr/bin/python
# -*- coding: latin-1 -*-
# -*- coding: iso-8859-15 -*-
# -*- coding: ascii -*-

#========================
# INPE: Vento M�ximo em 850hPa para periodos de 24, 48, 72, 96 e 120 horas
# Autor: Diogo Arsego
# Adaptado por: Gemini (para o modelo MONAN NetCDF)
#
# L�gica:
# - Usa netCDF4 para ler os dados
# - Encontra a magnitude m�xima do vento em 850hPa (uzonal, umeridional)
# - No ponto de m�ximo, armazena o vento de 10m (u10, v10) para as barbelas
#========================

import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
import matplotlib
from datetime import timedelta, datetime
from matplotlib.offsetbox import AnchoredText
import warnings

# Ignora avisos de "invalid value" e "overflow"
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in power')

# --- Constantes de Configura��o ---

# Caminho para o shapefile
SHAPEFILE_PATH = '/pesq/scripts/doppesq/shapefile/BR_UF_2019.shp'

# Caminho para os dados MONAN NetCDF
DATA_PATH_TEMPLATE = '/pesq/share/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Julia/{year}{month}{day}{run}/'

# Template de nome de arquivo MONAN
FILE_NAME_TEMPLATE = 'MONAN_DIAG_R_POS_GFS_{date_run_str}_{valid_time_str}.00.00.x1.5898242L55.nc'

# Template para o diret�rio de sa�da
#OUTPUT_DIR_TEMPLATE = '/share/doppesq/dist/avaliacao/vento/{year}{month}/{data}/'
# OUTPUT_DIR_TEMPLATE = '/scripts/avaliacao/casos_curso_omm/furacao_julia'
OUTPUT_DIR_TEMPLATE = '.'


# Extent para a �rea do Furac�o Julia
EXTENT_JULIA = [-95.0, -65.0, 5.0, 25.0]

# Configura��es de plotagem (N�veis em m/s)
CMAP_COLORS = ['#ffffcc','#fed976','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026','#980043','#e7298a','#c994c7']
CONTOUR_LEVELS = np.arange(10, 45, 5)

# Per�odos de previs�o (hora inicial, hora final)
PERIODOS_PREVISAO = [
    (1, 24),    # 24h
    (25, 48),   # 48h
    (49, 72),   # 72h
    (73, 96),   # 96h
    (97, 120)   # 120h
]

# --- Fun��es Auxiliares ---

### CORRE��O: A fun��o agora recebe os �ndices pr�-calculados ###
def processar_periodo(date_run, base_path_template, file_template,
                      hour_ini, hour_end,
                      lat_indices, lon_indices, level_idx_850):
    '''
    Processa arquivos NetCDF (usando netCDF4) para um dado per�odo,
    usando �ndices pr�-calculados.
    '''
    print(f"Processando per�odo: {hour_ini}h - {hour_end}h")
    
    # Vari�veis para armazenar os m�ximos
    wind_mag_max = None
    u_at_max, v_at_max = None, None
    
    # Extrai informa��es de data da 'date_run'
    year = date_run.strftime('%Y')
    month = date_run.strftime('%m')
    day = date_run.strftime('%d')
    run = date_run.strftime('%H')
    date_run_string = date_run.strftime('%Y%m%d%H')
    
    path = base_path_template.format(year=year, month=month, day=day, run=run)

    for hour in range(hour_ini, hour_end + 1):
        
        date_forecast = date_run + timedelta(hours=hour)
        valid_time_str = date_forecast.strftime('%Y%m%d%H')
            
        file_name = file_template.format(
            date_run_str=date_run_string, 
            valid_time_str=valid_time_str
        )
        file_path = os.path.join(path, file_name)

        if not os.path.isfile(file_path):
            continue
        
        print(f"  Analisando arquivo: {file_name}")

        try:
            nc = Dataset(file_path, 'r')
            
            ### CORRE��O: Bloco de c�lculo de �ndice foi REMOVIDO daqui ###
            
            # --- Leitura dos Dados (usando os �ndices passados) ---
            time_idx = 0
            
            # 1. Pega U e V em 850hPa
            u850_data = nc.variables['uzonal'][time_idx, level_idx_850, lat_indices, lon_indices]
            v850_data = nc.variables['umeridional'][time_idx, level_idx_850, lat_indices, lon_indices]
            
            # 2. Pega U e V a 10m
            u10_data = nc.variables['u10'][time_idx, lat_indices, lon_indices]
            v10_data = nc.variables['v10'][time_idx, lat_indices, lon_indices]
            
            nc.close()

            # Calcula a magnitude do vento 850hPa
            wind_mag_850 = np.sqrt(u850_data**2 + v850_data**2)

            # Inicializa��o na primeira leitura bem-sucedida
            if wind_mag_max is None:
                wind_mag_max = np.full_like(wind_mag_850, -9999.0)
                u_at_max = np.full_like(u10_data, np.nan)
                v_at_max = np.full_like(v10_data, np.nan)

            # M�scara onde a magnitude atual � maior que o m�ximo anterior
            mask = wind_mag_850 > wind_mag_max
            
            # Atualiza m�ximos
            wind_mag_max[mask] = wind_mag_850[mask]
            
            # Atualiza U10 e V10
            u_at_max[mask] = u10_data[mask]
            v_at_max[mask] = v10_data[mask]

        except Exception as e:
            print(f"  Erro ao processar arquivo {file_name}: {e}")
            if 'nc' in locals() and nc.isopen():
                nc.close()

    gust_max = wind_mag_max 
    
    ### CORRE��O: N�o retorna mais lats/lons (j� est�o em 'main') ###
    return gust_max, u_at_max, v_at_max
# --- Fim da Fun��o Adaptada ---


def plotar_mapa(ax, lons, lats, shapefile, extent, gust_data, u_data, v_data, title_str, levels, cmap, show_labels=None):
    """
    Desenha um �nico subplot de mapa.
    'lons' e 'lats' DEVEM ser vetores 1D.
    """
    if show_labels is None:
        show_labels = {'left': False, 'bottom': False, 'top': False, 'right': False}

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Adiciona fei��es geogr�ficas
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    ax.add_feature(cartopy.feature.OCEAN, edgecolor='white', facecolor='#D3D3D3')
    ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.3)

    # Plota os contornos (aceita 1D lons/lats)
    if gust_data is not None:
        img_contour = ax.contourf(lons, lats, gust_data, cmap=cmap, levels=levels, extend="both", transform=ccrs.PlateCarree())
    else:
        img_contour = None
    
    
    # Plota as barbelas de vento
    if u_data is not None and v_data is not None:
        
        ### CORRE��O: Cria grades 2D para a fun��o 'barbs' ###
        # lons (1D) e lats (1D) s�o usados para criar grades 2D
        lons_2d, lats_2d = np.meshgrid(lons, lats)
        
        fatia_y = 15
        fatia_x = 15

        # Fatiamos as grades 2D
        ax.barbs(lons_2d[::fatia_y, ::fatia_x], 
                 lats_2d[::fatia_y, ::fatia_x], 
                 u_data[::fatia_y, ::fatia_x], 
                 v_data[::fatia_y, ::fatia_x],
                 length=5,
                 linewidth=0.9,
                 color='black',
                 transform=ccrs.PlateCarree(),
                 zorder=10)
    
    
    # Configura as gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = show_labels.get('top', False)
    gl.right_labels = show_labels.get('right', False)
    gl.bottom_labels = show_labels.get('bottom', False)
    gl.left_labels = show_labels.get('left', False)
    
    # Adiciona t�tulo e texto
    ax.set_title(title_str, fontweight='bold', fontsize=10, loc='left')
    text = AnchoredText("INPE/CGCT/DIPTC", loc=3, prop={'size': 9}, frameon=True)
    text.patch.set_boxstyle("round4,pad=0.,rounding_size=0.2")
    ax.add_artist(text)

    return img_contour

# --- Script Principal ---
def main():
    
    # --- 1. Configura��o de Datas ---
    
    date_today = datetime(2022, 10, 10, 0)
    
    year = date_today.strftime('%Y')
    month = date_today.strftime('%m')
    day = date_today.strftime('%d')
    run = '00'
    
    dataa = datetime.strptime(year + month + day, '%Y%m%d')
    data = dataa.strftime('%Y%m%d')

    date_run = datetime.strptime(year + month + day + run, '%Y%m%d%H')
    date_run_string = date_run.strftime('%Y%m%d%H')
    
    doppesq_path = OUTPUT_DIR_TEMPLATE.format(year=year, month=month, data=data)
    os.makedirs(doppesq_path, mode=0o777, exist_ok=True)
    print(f"Salvando imagens em: {doppesq_path}")

    # Gera datas para os t�tulos
    datas_titulos = [(date_today + timedelta(days=i)).strftime('%d/%m') for i in range(6)]
    titulos = []
    for i in range(5):
        titulos.append(f'Validade: 00:00 UTC - {datas_titulos[i]} - 00:00 UTC - {datas_titulos[i+1]}')
    titulos.append(f'Validade: 00:00 UTC - {datas_titulos[0]} - 00:00 UTC - {datas_titulos[5]}')

    # --- 2. Carregamento do Shapefile ---
    print("Carregando shapefile...")
    try:
        shapefile = list(shpreader.Reader(SHAPEFILE_PATH).geometries())
    except Exception as e:
        print(f"ERRO: N�o foi poss�vel carregar o shapefile em {SHAPEFILE_PATH}")
        print(f"Detalhes: {e}")
        shapefile = []

    # --- 3. Pr�-c�lculo de �ndices (CORRE��O) ---
    print("Calculando �ndices da grade (uma �nica vez)...")
    
    # Tenta encontrar o primeiro arquivo (h=3) para ler a grade
    try:
        date_forecast_h3 = date_run + timedelta(hours=3)
        valid_time_str_h3 = date_forecast_h3.strftime('%Y%m%d%H')
        first_file_name = FILE_NAME_TEMPLATE.format(
            date_run_str=date_run_string, 
            valid_time_str=valid_time_str_h3
        )
        path = DATA_PATH_TEMPLATE.format(year=year, month=month, day=day, run=run)
        first_file_path = os.path.join(path, first_file_name)
        
        nc = Dataset(first_file_path, 'r')
        lats_all = nc.variables['latitude'][:]
        lons_all = nc.variables['longitude'][:]
        levels_all = nc.variables['level'][:]
        nc.close()

        # Encontra o �ndice do n�vel de press�o
        level_pa = 85000
        level_idx_850 = np.argmin(np.abs(levels_all - level_pa))
        print(f"  N�vel 850hPa encontrado no �ndice: {level_idx_850} (Valor: {levels_all[level_idx_850]} Pa)")

        # Encontra os �ndices para o 'extent' (lat/lon)
        lat_min, lat_max = EXTENT_JULIA[1], EXTENT_JULIA[3]
        lon_min, lon_max = EXTENT_JULIA[0], EXTENT_JULIA[2]
        
        lat_indices = np.where((lats_all >= lat_min) & (lats_all <= lat_max))[0]
        lon_indices = np.where((lons_all >= lon_min) & (lons_all <= lon_max))[0]
        
        # Salva os vetores 1D de lats/lons FATIADOS para plotagem
        lats = lats_all[lat_indices]
        lons = lons_all[lon_indices]
        
        if lats.size == 0 or lons.size == 0:
            raise ValueError("Nenhum ponto de grade encontrado no 'extent' definido.")

    except Exception as e:
        print(f"ERRO FATAL: N�o foi poss�vel ler a grade do primeiro arquivo NetCDF.")
        print(f"Tentativa de abrir: {first_file_path}")
        print(f"Erro: {e}")
        return

    # --- 4. Processamento dos Dados ---
    
    all_gust_data = [] # (wind_mag_850)
    all_u_data = []    # (u10)
    all_v_data = []    # (v10)
    
    for (h_ini, h_end) in PERIODOS_PREVISAO:
        # Processa para o extent do Furac�o Julia
        gust, u, v = processar_periodo(date_run, DATA_PATH_TEMPLATE, FILE_NAME_TEMPLATE,
                                         h_ini, h_end,
                                         lat_indices, lon_indices, level_idx_850)
        
        all_gust_data.append(gust)
        all_u_data.append(u)
        all_v_data.append(v)

    # Verifica se algum dado foi carregado (checa o primeiro per�odo)
    if all_gust_data[0] is None:
        print("ERRO FATAL: Nenhum dado foi processado. Verifique os caminhos e nomes de arquivo.")
        return

    # Calcula o m�ximo total
    valid_gust_data = [g for g in all_gust_data if g is not None]
    if valid_gust_data:
        gust_max_total = np.maximum.reduce(valid_gust_data)
        gust_max_total[gust_max_total == -9999.0] = np.nan
    else:
        gust_max_total = np.full_like(lats, np.nan) # Usa 'lats' (1D)
    
    # Limpa os dados de -9999 para NaN
    for i in range(len(all_gust_data)):
        if all_gust_data[i] is not None:
            all_gust_data[i][all_gust_data[i] == -9999.0] = np.nan

    # --- 5. Plotagem (Figura �nica - Furac�o Julia) ---
    print("Gerando Figura (Furac�o Julia)...")
    
    cmap = matplotlib.colors.ListedColormap(CMAP_COLORS)
    cmap.set_over('#e7e1ef')
    cmap.set_under('#ffffff') # Branco para NaN

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), 
                            gridspec_kw={'left':0, 'bottom':0, 'right':1, 'top':1, 'hspace':0.05, 'wspace':0.05},
                            subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    axs_flat = axs.flatten()
    img_contour_main = None

    # Plota os 5 per�odos
    for i in range(5):
        # Passa lons e lats (1D)
        img_contour = plotar_mapa(axs_flat[i], lons, lats, shapefile, EXTENT_JULIA,
                                  all_gust_data[i], all_u_data[i], all_v_data[i],
                                  titulos[i], CONTOUR_LEVELS, cmap,
                                  show_labels={'left': (i % 3 == 0), 'bottom': (i >= 3)})
        if img_contour_main is None:
             img_contour_main = img_contour

    # Plota o 6� mapa (Total)
    plotar_mapa(axs_flat[5], lons, lats, shapefile, EXTENT_JULIA,
                gust_max_total, None, None, # Sem barbelas
                titulos[5], CONTOUR_LEVELS, cmap,
                show_labels={'left': False, 'bottom': True})

    # Adiciona Colorbar e T�tulo
    if img_contour_main:
        fig.colorbar(img_contour_main, label='Vento M�x. 850hPa (m/s)', orientation='vertical', pad=0.02, fraction=0.03, ax=axs.ravel().tolist())
    
    plt.suptitle('MONAN - Vento M�ximo 850hPa (Furac�o Julia)', fontweight='bold', fontsize=12, y=1.02)
    output_filename = f'monan_vento_850_max_julia_{data}.png'
    plt.savefig(os.path.join(doppesq_path, output_filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("Processamento conclu�do.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()
