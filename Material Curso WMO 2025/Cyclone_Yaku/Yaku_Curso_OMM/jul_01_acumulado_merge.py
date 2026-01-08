#!/usr/bin/env python3
# ============================================================
# PAINEL DE PRECIPITAÇÃO OBSERVADA (MERGE)
# ============================================================
# Este script cria um painel 2x3 com precipitação diária de 6 dias
# e um painel maior à direita com o acumulado total.
# Usa dados observados de arquivos GRIB (MERGE).
# ============================================================

import os
import pygrib  # Para ler arquivos GRIB com dados de precipitação observada.
import numpy as np  # Para cálculos com matrizes.
import matplotlib.pyplot as plt  # Para gráficos e mapas.
import matplotlib.colors as mcolors  # Para criar paletas de cores personalizadas.
import cartopy.crs as ccrs  # Projeções de mapas.
import cartopy.feature as cfeature  # Elementos geográficos.

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
# Datas dos 6 dias consecutivos a serem plotados.
#datas = [
#    "20230308",
#    "20230309",
#    "20230310",
#    "20230311",
#    "20230312",
#    "20230313"
#]


datas = [
    "20221005",
    "20221006",
    "20221007",
    "20221008",
    "20221009",
    "20221010",
]

#rodadas = ["2022100500", "2022100600", "2022100700", "2022100800", "2022100900", "2022101000", "2022101100"]
#previsoes = ["2022100600", "2022100700", "2022100800", "2022100900", "2022101000", "2022101100", "2022101200"]
# Limites geográficos do mapa.
#extent = [-90, -70, -20, 5]
extent = [-110, -75, 5, 25]

# dir_merge = "merge"  # Pasta onde estão os arquivos GRIB.
dir_merge = "/oper/share/ioper/tempo/MERGE/GPM/DAILY/"


# -------------------------------
# DEFINIÇÃO DAS ESCALAS DE CORES
# -------------------------------
# Intervalos de precipitação para os dias individuais.
boundaries = [0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
# Intervalos para o acumulado dos 6 dias.
boundaries2 = [0, 50, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]

# Paleta de cores para os mapas (de branco a azul escuro).
colors = [
    (1.0, 1.0, 1.0),             # Branco (<10)
    (255/255, 160/255, 0/255),   # Laranja escuro (10–50)
    (255/255, 192/255, 0/255),   # Laranja médio (50–100)
    (255/255, 232/255, 100/255), # Amarelo escuro (100–150)
    (255/255, 250/255, 170/255), # Amarelo claro (150–200)
    (192/255, 255/255, 160/255), # Verde amarelado (200–250)
    (16/255, 192/255, 32/255),   # Verde médio (250–300)
    (0/255, 160/255, 0/255),     # Verde intermediário (300–350)
    (0/255, 128/255, 0/255),     # Verde forte (350–400)
    (0/255, 96/255, 0/255),      # Verde escuro (400–450)
    (0/255, 64/255, 0/255),      # Verde mais escuro (450–500)
    (0/255, 20/255, 160/255)     # Azul escuro (>500)
]

# Cria colormaps e normalizações para mapear valores às cores.
cmap_daily = mcolors.ListedColormap(colors)
norm_daily = mcolors.BoundaryNorm(boundaries, len(boundaries) - 1)

cmap_sum = mcolors.ListedColormap(colors)
norm_sum = mcolors.BoundaryNorm(boundaries2, len(boundaries2) - 1)

# -------------------------------
# LEITURA DOS ARQUIVOS MERGE
# -------------------------------
# Lista para armazenar dados de precipitação de cada dia.
prec_list = []
lats_merge = None
lons_merge = None

# Loop para ler cada arquivo GRIB.
for data in datas:
    year = data[:4]
    month = data[4:6]
    arq_merge = f"{dir_merge}/{year}/{month}/MERGE_CPTEC_{data}.grib2"
    print(f"🔹 Lendo arquivo MERGE: {arq_merge}")

    if not os.path.exists(arq_merge):
        print(f"⚠️ Arquivo {arq_merge} não encontrado. Pulando...")
        continue

    grib = pygrib.open(arq_merge)
    # grb = grib.select(name='Precipitation')[0]
    grb = grib.select(shortName='rdp')[0]
    prec, lats, lons = grb.data(lat1=extent[2], lat2=extent[3],
                                lon1=extent[0]+360, lon2=extent[1]+360)
    grib.close()

    prec_list.append(prec)
    if lats_merge is None:
        lats_merge = lats
        lons_merge = lons

# -------------------------------
# VERIFICAÇÃO E ACUMULADO
# -------------------------------
if len(prec_list) == 0:
    raise RuntimeError("Nenhum arquivo MERGE encontrado!")

# Calcula o acumulado somando as precipitações diárias.
prec_all = np.stack(prec_list)
prec_sum = np.sum(prec_all, axis=0)

# -------------------------------
# CONFIGURAÇÃO DO PAINEL
# -------------------------------
# Cria figura com ajustes de espaçamento.
fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(left=0.05, right=0.65, top=0.95, bottom=0.10, hspace=0.15, wspace=0.1)

# Cria painéis 2x3 para dias individuais e um maior para o acumulado.
axes = [fig.add_subplot(2, 3, i+1, projection=ccrs.PlateCarree()) for i in range(6)]
ax_total = fig.add_axes([0.70, 0.15, 0.25, 0.7], projection=ccrs.PlateCarree())

# -------------------------------
# PLOTAGEM DOS DIAS INDIVIDUAIS
# -------------------------------
for i, (ax, data) in enumerate(zip(axes, datas)):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.STATES, linewidth=0.2)

    # Plota precipitação diária.
    cf = ax.contourf(lons_merge, lats_merge, prec_list[i],
                     levels=boundaries, cmap=cmap_daily, norm=norm_daily,
                     extend='max', transform=ccrs.PlateCarree())

    ax.set_title(f"{data}", fontsize=10)
    ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.4)

# -------------------------------
# PLOTAGEM DO ACUMULADO
# -------------------------------
ax_total.set_extent(extent, crs=ccrs.PlateCarree())
ax_total.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax_total.add_feature(cfeature.BORDERS, linewidth=0.4)
ax_total.add_feature(cfeature.STATES, linewidth=0.3)

# Plota acumulado dos 6 dias.
cf_total = ax_total.contourf(lons_merge, lats_merge, prec_sum,
                             levels=boundaries2, cmap=cmap_sum, norm=norm_sum,
                             extend='max', transform=ccrs.PlateCarree())

ax_total.set_title("Acumulado dos 6 dias", fontsize=12, fontweight='bold')

# -------------------------------
# BARRAS DE CORES E SALVAMENTO
# -------------------------------
# Barra de cores para os dias individuais.
cbar_ax1 = fig.add_axes([0.20, 0.06, 0.3, 0.02])
fig.colorbar(cf, cax=cbar_ax1, orientation='horizontal',
             ticks=boundaries, label='Precipitação diária observada (mm)')

# Barra de cores para o acumulado.
cbar_ax2 = fig.add_axes([0.72, 0.06, 0.22, 0.02])
fig.colorbar(cf_total, cax=cbar_ax2, orientation='horizontal',
             ticks=boundaries2, label='Precipitação acumulada em 6 dias (mm)')

plt.suptitle("Precipitação observada (MERGE) — 6 dias e acumulado", fontsize=14, y=0.98)
plt.savefig("precip_merge.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva como precip_merge.png")
