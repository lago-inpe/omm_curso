# -*- coding: utf-8 -*-
"""
Created on May 2025

@author: maicon
"""
import os
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import numpy.ma as ma
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import datetime
from metpy.units import units
import cartopy.io.shapereader as shpreader                      
from datetime import timedelta, date, datetime                 
from metpy.calc import reduce_point_density                         
import xarray as xr                  
from cartopy.feature import ShapelyFeature
import datetime
import metpy.calc as mpcalc
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker
#import cmaps
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_barbs_auto(ax, lons, lats, u, v, max_points=30, factor=1.9438444924406, **kwargs):
    """
    Plota barbelas subamostrando automaticamente o campo para não sobrecarregar o mapa.
    """
    n_lat, n_lon = lons.shape
    step_lat = max(1, n_lat // max_points)
    step_lon = max(1, n_lon // max_points)

    lons_sub = lons[::step_lat, ::step_lon]
    lats_sub = lats[::step_lat, ::step_lon]
    u_sub = u[::step_lat, ::step_lon] * factor
    v_sub = v[::step_lat, ::step_lon] * factor

    ax.barbs(
        lons_sub,
        lats_sub,
        u_sub,
        v_sub,
        transform=ccrs.PlateCarree(),
        **kwargs
    )

 
# Variavel contadora para o numero sequencial
contador = 0 

caminho_arquivo = "/pesq/share/monan/curso_OMM_INPE_2025/caso_RS/2024050200_2024050300/MONAN_DIAG_R_POS_GFS_2024050200"
output_dir = "figuras/rodada_2025050200/superficie"

# Extração da data do caminho do arquivo
data_base_str = caminho_arquivo.split("/")[-2]  # Extrai o diretório "2024113012"
data_base = datetime.datetime(
    year=int(data_base_str[:4]),   # Ano (2024)
    month=int(data_base_str[4:6]), # Mês (11)
    day=int(data_base_str[6:8])    # Dia (30)
)


# Definir o intervalo de dias para processar
dia_inicial = 0
dia_final = 2


# Inicializar a variável de inicialização
hora_inicializacao = None


# Loop pelos dias
for dia in range(dia_inicial, dia_final + 1):
    # Ajustar a data base para o dia atual do loop
    data_atual = data_base + datetime.timedelta(days=dia)

    # Definir o intervalo de horas com base na condição
    if dia == dia_inicial:
        hora_inicial = 0  # Primeiro dia inicia às 12 horas
    else:
        hora_inicial = 0  # Demais dias iniciam às 00 horas
    hora_final = 23  # Todos os dias terminam às 23 horas
    
    # Loop pelas horas
    for hora in range(hora_inicial, hora_final + 1):
        # Atualizar a hora em data_atual
        data_hora = data_atual.replace(hour=hora)
    
         # Formatar a data e hora no formato desejado
        caminho_arquivo_1 = (
            caminho_arquivo +
            "_{}{}{}{}.mm.x20.835586L55.nc".format(
                data_hora.strftime("%Y"),
                data_hora.strftime("%m"),
                data_hora.strftime("%d"),
                data_hora.strftime("%H"),
            )
        )


        
        # Salvar o horário de inicialização (apenas na primeira iteração)
        if hora_inicializacao is None:
            hora_inicializacao = data_hora.strftime("%Y-%m-%d")
    
        # Formatar a data e hora no formato desejado
        caminho_arquivo_1 = (caminho_arquivo) + "_{}{}{}{}.mm.x20.835586L55.nc".format(
           data_hora.strftime("%Y"),
           data_hora.strftime("%m"),
           data_hora.strftime("%d"),
           data_hora.strftime("%H"),
           )

        # Abre o arquivo
        if(os.path.isfile(caminho_arquivo_1)):
            ds = xr.open_dataset(caminho_arquivo_1)
        else:
            print(f"O arquivo '{caminho_arquivo_1}' nao existe no diretorio.")
            continue
            
        # Matrizes de latitude e longitude
        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

        prec_h = ds.prec[0, :, :]  
        pressao_nivel_mar = ds.mslp[0, :, :] / 100
        u10 = ds.u10[0, :, :]
        v10 = ds.v10[0, :, :]  

        zgeo_1000 = ds.zgeo_isobaric.sel(level=100000)[0, :, :]
        zgeo_500 = ds.zgeo_isobaric.sel(level=50000)[0, :, :]
        Esp_500_1000 = zgeo_500 - zgeo_1000

        
    
        # Definindo os pontos de cor com mais tons intermediários e ajustando as cores para tons mais foscos
        colors = [
    (0.7, 1, 0.7),  # Verde claro fosco
    (0.5, 0.8, 0.5),  # Verde médio-claro fosco
    (0.3, 0.6, 0.3),  # Verde médio fosco
    (0.2, 0.4, 0.2),  # Verde médio-escuro fosco
    (0, 0.3, 0),    # Verde escuro fosco
    (0.2, 0.4, 0.7),  # Azul fosco
    (0.3, 0.5, 0.8),  # Azul mais claro fosco
    (0.4, 0.6, 0.9),  # Azul claro fosco
    (0.5, 0.7, 1),  # Azul mais claro ainda fosco
    (0.6, 0.8, 1),  # Azul muito claro fosco
    (0.5, 0, 0.5),  # Roxo claro
    (0.4, 0, 0.4),  # Roxo médio-claro
    (0.3, 0, 0.3),  # Roxo médio
    (0.2, 0, 0.2),  # Roxo escuro
    (0.55, 0, 0),   # darkred
    (0.7, 0.2, 0),  # Red-orange intermediate
    (0.85, 0.4, 0), # darkorange
    (0.9, 0.55, 0), # Orange-yellow intermediate
    (0.8, 0.7, 0),  # darkyellow
    (1, 1, 0)       # yellow
        ]

        # Criando o mapa de cores
        cmap = LinearSegmentedColormap.from_list("GreenBluePurpleRedOrangeYellow", colors)
    
        # Select the extent [min. lon, min. lat, max. lon, max. lat]
        extent = [-70, -40, -45, -20]
        
        plt.figure(figsize=(12, 16))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=extent[0] - extent[2]))
        ax.set_extent([extent[0], extent[2], extent[1], extent[3]], ccrs.PlateCarree())
        
        PREC = ax.contourf(lons, lats, prec_h, levels=np.arange(1, 51, 0.5), cmap=cmap, extend='neither', transform=ccrs.PlateCarree())
   
        # Criar inset axes para a colorbar no canto superior direito
        axins = inset_axes(ax, 
                   width="80%",   # largura da barra relativa ao eixo principal
                   height="2%",   # altura da barra relativa ao eixo principal
                   loc='lower center',  # posição no canto superior direito
                   borderpad=-2.7)   # espaço de margem

        # Criar a colorbar dentro do inset axes, orientada horizontalmente
        cbar = plt.colorbar(PREC, cax=axins, orientation='horizontal')
    
        # Definindo os marcadores da barra de cores para inteiros usando FixedLocator
        cbar.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(5, 51, 5)))

        # Criar box branco ao redor da colorbar
        for spine in axins.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)

        axins.set_facecolor('white')  # fundo branco
    
        PNM = ax.contour(lons, lats, pressao_nivel_mar, levels = np.arange(960, 1048, 4), colors = 'k', linewidths=2, transform=ccrs.PlateCarree())
        ax.clabel(PNM, inline = True)

        Esp_500_1000_1 = ax.contour(lons, lats, Esp_500_1000/10, levels = np.arange(500, 600, 6), linewidths=2.0, linestyles='dashed', colors = 'blue',     transform=ccrs.PlateCarree())
        ax.clabel(Esp_500_1000_1, inline = True)

        plot_barbs_auto(ax,
        lons,
        lats,
        u10.values,
        v10.values,
        max_points=30,  # Pode ajustar, ex.: 20, 40, conforme a densidade desejada
        length=5.5,
        flip_barb=True,
        zorder=4,
        color='k'
        )

        # Adicionado contorno dos continentes e paises
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor = 'k', zorder = 5)
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor = 'k', linewidth = 0.5, zorder = 5)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor = 'none', edgecolor = 'k', zorder = 5)

   
        # Legenda na parte inferior central
        ax.text(0.5, 0.03, f'PNMM, Espessura 500-1000hPa, Precipitação horária (mm), Vento em 10m(Kt) - MONAN/INPE - Data Inicialiazação: 2025-05-02 00Z - Validade:{data_hora.strftime("%Y-%m-%d %H")}Z ', 
        transform=ax.transAxes, fontsize=7, ha='center', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'))
        
        # Carimbo no canto superior direito
        carimbo_text = f'MONAN - INPE\n{data_hora.strftime("%Y-%m-%d %H")}Z'
        ax.text(
        0.185, 0.97, carimbo_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12,  # AUMENTA AQUI! Por exemplo, de 12 para 18 ou mais.
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='round,pad=0.3'), zorder=7)

        # Adicionando linhas de grade
        g1 = ax.gridlines(crs = ccrs.PlateCarree(), linestyle = '--', color = 'gray', draw_labels = True)

        # Removendo os labels do topo e da direita
        g1.right_labels = False
        g1.top_labels = False
        g1.yformatter = LATITUDE_FORMATTER
        g1.xformatter = LONGITUDE_FORMATTER

        # Salvando o mapa como uma figura
        os.makedirs(output_dir, exist_ok=True)
        #plt.savefig('/dados/nowcasting/maicon.veber/caso_RS/rodada_2025050200/superficie/superficie_MONAN_{:03d}.png'.format(contador), dpi=100, bbox_inches='tight')
        #plt.savefig('/oper/share/nowcasting/ftp/MONAN_figuras/MONAN/superficie/superficie_MONAN_{:03d}.png'.format(contador), dpi=100, bbox_inches='tight')
        plt.savefig(output_dir + '/superficie_MONAN_{:03d}.png'.format(contador), dpi=100, bbox_inches='tight')
    
        plt.close()
    
        # Incrementar o contador
        contador += 1
