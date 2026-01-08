# -*- coding: utf-8 -*-
"""
Created on Feb 2024

@author: maicon
"""
import xarray as xr
import datetime
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
#import cmaps
#from pyart.graph import cm
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Caminho do arquivo
#caminho_arquivo = "/scripts/nowcasting/maicon.veber/dados/2024120512/MONAN_DIAG_R_POS_GFS_2024120512"
#caminho_arquivo = "/dados/nowcasting/maicon.veber/MONAN/UFSC/2023081100_ERA5/MONAN_DIAG_R_POS_ERA5_2023081100"
caminho_arquivo = "/pesq/share/monan/curso_OMM_INPE_2025/caso_RS/2024050200_2024050300/MONAN_DIAG_R_POS_GFS_2024050200"
output_dir = "figuras/rodada_2025050200/cape"

# Extração da data do caminho do arquivo
data_base_str = caminho_arquivo.split("/")[-2]  # Extrai o diretório "2024113012"
data_base = datetime.datetime(
    year=int(data_base_str[:4]),   # Ano (2024)
    month=int(data_base_str[4:6]), # Mês (11)
    day=int(data_base_str[6:8])    # Dia (30)
)


# Definir o intervalo de dias para processar
dia_inicial = 0
dia_final = 0

# Variável contadora para o número sequencial
contador = 0  

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

        # Abrir o dataset com xarray
#        ds = xr.open_dataset(caminho_arquivo_1)
        # Abre o arquivo
        if(os.path.isfile(caminho_arquivo_1)):
            ds = xr.open_dataset(caminho_arquivo_1)
        else:
            print(f"O arquivo '{caminho_arquivo_1}' nao existe no diretorio.")
            continue
        
        # Selecionar o plano de dados desejado (por exemplo, índice 0)
        pnm = ds.mslp[0, :, :] / 100
        
        cape = ds.cape[0, :, :]
            
        # Configurar a figura e o eixo
        plt.figure(figsize=(10, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
            
        # Criar matrizes de latitude e longitude
        lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)
            
        # Configurar a extensão do mapa
        #ax.set_extent([-65, -42, -29, -11]) #CentroOeste
        ax.set_extent([-64, -39, -20, -40]) #Sul
        #ax.set_extent([-57, -38, -27.5, -12]) #Sudeste
        
        # Níveis
        levels = np.array([
    100, 250, 500, 750, 1000,
    1250, 1500, 1750, 2000,
    2500, 3000, 3500, 4000,
    4500, 5000, 5500, 6000
        ])

        # 16 cores exatas (1 para cada intervalo)
        cores = [
    (0.9, 0.9, 0.9),  
    (0.75, 0.75, 0.75),  
    (0.6, 0.6, 0.6),  
    (0.45, 0.45, 0.45),  
    (0.7, 0.8, 1.0),  
    (0.5, 0.7, 1.0),  
    (0.3, 0.6, 1.0),  
    (0.1, 0.5, 1.0),  
    (0.7, 1.0, 0.7),  
    (0.5, 0.9, 0.5),  
    (0.3, 0.8, 0.3),  
    (0.1, 0.7, 0.1),  
    (1.0, 1.0, 0.4),  
    (1.0, 0.7, 0.0),  
    (1.0, 0.0, 0.0),  
    (0.5, 0.0, 0.0)
        ]

        # Criar o colormap e a normalização
        cmap = ListedColormap(cores)
        norm = BoundaryNorm(boundaries=levels, ncolors=len(cores))

        # Criando o gráfico de contorno com os novos níveis e colormap
        cp = ax.contourf(lons, lats, cape, levels=levels, cmap=cmap, norm=norm, extend='neither', transform=ccrs.PlateCarree())
        
        # Criar inset axes para a colorbar no canto superior direito
        axins = inset_axes(ax, 
                   width="80%",   # largura da barra relativa ao eixo principal
                   height="2%",   # altura da barra relativa ao eixo principal
                   loc='lower center',  # posição no canto superior direito
                   borderpad=-2.3)   # espaço de margem

        # Criar a colorbar dentro do inset axes, orientada horizontalmente
        cbar = plt.colorbar(cp, cax=axins, orientation='horizontal')

        # Definindo os marcadores da barra de cores para inteiros usando FixedLocator
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(5, 51, 5)))
        
         # Criar box branco ao redor da colorbar
        for spine in axins.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)

        axins.set_facecolor('white')  # fundo branco
        
        PNM = ax.contour(lons, lats, pnm, levels = np.arange(950, 1050, 3), colors = 'k', linewidths=0.75, transform=ccrs.PlateCarree())
        ax.clabel(PNM, inline = True)

        # Adicionado contorno dos continentes e paises
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor = 'k')
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor = 'gray', linewidth = 0.5)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor = 'none', edgecolor = 'k', zorder = 2)

        # Adicionando título com a hora de inicialização
        ax.set_title('CAPE[J/Kg], PNMM [hPa] - MONAN Regional_060-003Km\nInicializado: {}-00Z - Validade: {}'.format(hora_inicializacao,data_hora.strftime("%Y-%m-%d-%HZ")), fontsize=10)

        # Adicionando linhas de grade
        g1 = ax.gridlines(crs = ccrs.PlateCarree(), linestyle = '--', color = 'gray', linewidth=0.25, draw_labels = True)

        # Removendo os labels do topo e da direita
        g1.right_labels = False
        g1.top_labels = False

        # Formatando ps labels como latitude e longitude
        g1.yformatter = LATITUDE_FORMATTER
        g1.xformatter = LONGITUDE_FORMATTER
    
        # Salvando o mapa como uma figura
        os.makedirs(output_dir, exist_ok=True)
        #plt.savefig('/scripts/nowcasting/maicon.veber/figuras/prec/prec_MONAN_{:03d}.png'.format(contador), dpi=300, bbox_inches='tight')
        #plt.savefig('/dados/nowcasting/maicon.veber/Figuras/cape_MONAN_{:03d}.png'.format(contador), dpi=300, bbox_inches='tight')
        plt.savefig(output_dir + '/cape_MONAN_{:03d}.png'.format(contador), dpi=300, bbox_inches='tight')

        #plt.show()
        plt.close()
    
        # Incrementar o contador
        contador += 1

print("Mapas de indice CAPE gerados com sucesso!")

