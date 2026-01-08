# -*- coding: utf-8 -*-
"""
Created on apr 2025

@author: maicon
"""
import os
import xarray as xr
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

# Caminho do arquivo
caminho_arquivo =  "/pesq/share/monan/curso_OMM_INPE_2025/caso_RS/2024050200_2024050300/MONAN_DIAG_R_POS_GFS_2024050200"
#caminho_arquivo = "/pesq/bam/paulo.kubota/monan/MPAS_Model_Global/pos/runs/GFS/2025040300/MONAN_DIAG_G_POS_GFS_2025040300_"

output_dir = "figuras/rodada_2025050200/prec_acumulada"

# Extração da data do caminho do arquivo
data_base_str = caminho_arquivo.split("/")[-2]  # Extrai "2024042700"
data_base = datetime.datetime(
    year=int(data_base_str[:4]),   # Ano (2024)
    month=int(data_base_str[4:6]), # Mês (04)
    day=int(data_base_str[6:8])    # Dia (27)
)

# Definir o intervalo de dias e horas para acumular precipitação
dia_inicial = 0
quantidade_dias = 1  # Pode ajustar conforme necessário
hora_inicial = 0  # Pode iniciar em qualquer hora do primeiro dia
hora_final = 0  # Definir a hora final do último dia

# Criar uma variável para acumular precipitação
prec_acumulada = None
hora_finalizacao = None

# Loop pelos dias
for dia in range(dia_inicial, quantidade_dias + 1):
    data_atual = data_base + datetime.timedelta(days=dia)
    
    # Definir a hora inicial do primeiro dia e 00Z para os demais
    hora_inicio = hora_inicial if dia == dia_inicial else 0
    hora_limite = hora_final if dia == quantidade_dias else 23  # Define a hora final corretamente
    
    # Loop pelas horas
    for hora in range(hora_inicio, hora_limite + 1):  # De hora_inicio a hora_limite
        data_hora = data_atual.replace(hour=hora)
        
        # Construção do caminho do arquivo
        caminho_arquivo_1 = (
            caminho_arquivo +
            "_{}{}{}{}.mm.x20.835586L55.nc".format(
            #"{}{}{}{}.mm.x1.1024002L55.nc".format(

                data_hora.strftime("%Y"),
                data_hora.strftime("%m"),
                data_hora.strftime("%d"),
                data_hora.strftime("%H"),
            )
        )
        
        try:
            ds = xr.open_dataset(caminho_arquivo_1)
            prec_h = ds.prec[0, :, :]
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {caminho_arquivo_1}")
            continue
        
        # Inicializa o acumulado na primeira iteração
        if prec_acumulada is None:
            prec_acumulada = np.zeros_like(prec_h)
            hora_inicializacao = data_hora.strftime("%Y-%m-%d %HZ")
        
        # Soma a precipitação horária ao acumulado
        prec_acumulada += prec_h
        
        # Atualiza a hora de finalização
        hora_finalizacao = data_hora.strftime("%Y-%m-%d %HZ")

# Criar o mapa de precipitação acumulada
fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()})
#ax.set_extent([-57, -38, -27.5, -12])  # Sudeste
ax.set_extent([-70, -40, -45, -20])  # Região Sul
#ax.set_extent([315, 327, -16, -2]) #Região Nordeste
#ax.set_extent([-70, -50, -25, -45])  # Prata
#ax.set_extent([-64, -42, -18, -42])  # Região Sul2

# Criar o colormap
colors = [
    (0.7, 1, 0.7), (0.5, 0.8, 0.5), (0.3, 0.6, 0.3),
    (0.2, 0.4, 0.2), (0, 0.3, 0), (0.2, 0.4, 0.7),
    (0.3, 0.5, 0.8), (0.4, 0.6, 0.9), (0.5, 0.7, 1),
    (0.6, 0.8, 1), (0.5, 0, 0.5), (0.4, 0, 0.4),
    (0.3, 0, 0.3), (0.2, 0, 0.2), (0.55, 0, 0),
    (0.7, 0.2, 0), (0.85, 0.4, 0), (0.9, 0.55, 0),
    (0.8, 0.7, 0), (1, 1, 0)
]
cmap = LinearSegmentedColormap.from_list("GreenBluePurpleRedOrangeYellow", colors)

# Criando matrizes de latitude e longitude
lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

# Criando o gráfico
PREC = ax.contourf(lons, lats, prec_acumulada, levels=np.arange(0, 501, 5), cmap=cmap, extend='neither', transform=ccrs.PlateCarree())

# Inserindo uma colorbar
cbar = plt.colorbar(PREC, ax=ax, pad=0.03, fraction=0.023)
cbar.set_label(label='Precipitação Acumulada [mm]', size=10)
cbar.ax.tick_params(labelsize=12)

# Definindo os marcadores da barra de cores para inteiros usando FixedLocator
cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(50, 501, 25)))

# Adicionando contornos e elementos do mapa
ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='k')
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='none', edgecolor='k', zorder=2)

g1 = ax.gridlines(crs = ccrs.PlateCarree(), linestyle = '--', color = 'gray', linewidth=0.25, draw_labels = True)
g1.right_labels = False
g1.top_labels = False
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER


# Adicionar valores de temperatura em pontos específicos
capitais = [
    #(-9.974, -67.824),  # Rio Branco (AC)
    #(-9.649, -35.708),  # Maceió (AL)
    #(0.035, -51.070),   # Macapá (AP)
    #(-3.119, -60.021),  # Manaus (AM)
    #(-12.971, -38.501), # Salvador (BA)
    #(-3.717, -38.543),  # Fortaleza (CE)
    #(-15.780, -47.929), # Brasília (DF)
    #(-20.315, -40.312), # Vitória (ES)
    #(-16.680, -49.256), # Goiânia (GO)
    #(-2.529, -44.304),  # São Luís (MA)
    #(-15.601, -56.097), # Cuiabá (MT)
    #(-20.469, -54.620), # Campo Grande (MS)
    #(-19.922, -43.941), # Belo Horizonte (MG)
    #(-1.455, -48.490),  # Belém (PA)
    #(-7.119, -34.845),  # João Pessoa (PB)
    #(-25.428, -49.273), # Curitiba (PR)
    #(-8.047, -34.877),  # Recife (PE)
    #(-5.092, -42.803),  # Teresina (PI)
    #(-22.906, -43.173), # Rio de Janeiro (RJ)
    #(-5.794, -35.211),  # Natal (RN)
    #(-30.034, -51.230), # Porto Alegre (RS)
    #(-8.760, -63.903),  # Porto Velho (RO)
    #(2.819, -60.671),   # Boa Vista (RR)
    #(-27.595, -48.548), # Florianópolis (SC)
    #(-23.550, -46.633), # São Paulo (SP)
    #(-10.947, -37.074), # Aracaju (SE)
    #(-10.167, -48.327), # Palmas (TO)
    #(-23.433, -45.083), # Ubatuba (SP)
    #(-23.006, -44.318), # Angra dos Reis (RJ)
    #(-22.520, -43.192), # Petrópolis (RJ)
    #(-21.759, -43.350), # Juiz de Fora (MG)
    #(-23.964, -46.328)  # Santos (SP)
    #(-29.7561, -57.448), #Uruguaiana
    #(-30.89, -55.533), #Santana do Livramento
    #(-31.3297, -54.107), #Bagé
    #(-31.7713, -52.343), #Pelotas
    #(-29.6868, -53.815), #Santa Maria
    
]

for lat, lon in capitais:
    # Verificar se o ponto está dentro dos limites de latitude e longitude
    if (lats.min() <= lat <= lats.max()) and (lons.min() <= lon <= lons.max()):
       # Encontrar o índice mais próximo
       lat_idx = (np.abs(lats[:, 0] - lat)).argmin()
       lon_idx = (np.abs(lons[0, :] - lon)).argmin()
       prc_ponto = prec_acumulada[lat_idx, lon_idx]
       ax.text(lon, lat, '{:.1f}'.format(prc_ponto), color='k', fontweight='bold', fontsize=12, ha='left', va='center', transform=ccrs.PlateCarree(), zorder=8)
    else:
        print(f"Ponto fora dos limites: lat={lat}, lon={lon}")


ax.set_title(f'Precipitação Acumulada [mm] - MONAN Regional_060-003Km\n Período: {hora_inicializacao} - {hora_finalizacao}',fontweight='bold', fontsize=12)

# Salvando a figura
os.makedirs(output_dir, exist_ok=True)
#plt.savefig(f'/dados/nowcasting/maicon.veber/caso_RS/rodada_2025050200/prec_acumulada/prec_acumulada_MONAN_Regional_{hora_inicializacao}_{hora_finalizacao}.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir + f'/prec_acumulada_MONAN_Regional_{hora_inicializacao}_{hora_finalizacao}.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Mapa de precipitação acumulada gerado com sucesso!")

