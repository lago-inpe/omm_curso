# -*- coding: utf-8 -*-
"""
Comparacao de precipitacao acumulada - MONAN vs MERGE
Vies Espacial: MONAN - MERGE
"""

import xarray as xr
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import os

# ================================
# Parametros de entrada
# ================================

data_base = datetime.datetime(2024, 5, 2, 0)

# Intervalo de analise
dia_inicial = 0
quantidade_dias = 2
hora_inicial = 0
hora_final = 0

# Diretorios
dir_monan = "/pesq/share/monan/curso_OMM_INPE_2025/caso_RS/2024050200_2024050300/"
#saida_figuras = "/dados/nowcasting/maicon.veber/caso_RS/rodada_2025050200/vies"
saida_figuras = "figuras/rodada_2025050200/vies"

# ================================
# Gerar lista de datas
# ================================

def gerar_datas():
    datas = []
    for delta_dia in range(quantidade_dias):
        for hora in range(24):
            if delta_dia == 0 and hora < hora_inicial:
                continue
            if delta_dia == (quantidade_dias - 1) and hora > hora_final:
                continue
            data_hora = data_base + datetime.timedelta(days=dia_inicial + delta_dia, hours=hora)
            datas.append(data_hora)
    return datas

datas = gerar_datas()

print("Horários acumulados MONAN:")
for dt in datas:
    print(dt.strftime("%Y-%m-%d %H:%MZ"))


# ================================
# MONAN - Acumulado
# ================================

monan_acumulada = None
for dt in datas:
    nome = f"MONAN_DIAG_R_POS_GFS_2024050200_{dt.strftime('%Y%m%d%H')}.mm.x20.835586L55.nc"
    caminho = os.path.join(dir_monan, nome)
    try:
        ds = xr.open_dataset(caminho)
        prec = ds["prec"].squeeze()
    except Exception as e:
        print(f"[MONAN] Erro com {caminho}: {e}")
        continue
    if monan_acumulada is None:
        monan_acumulada = prec.copy(deep=True)
    else:
        monan_acumulada += prec

if monan_acumulada is None:
    print("[MONAN] Nenhum dado carregado. Encerrando script.")
    exit()

# ================================
# MERGE - Acumulado (nova abordagem)
# ================================

merge_acumulada = None
hora_inicializacao = None
hora_finalizacao = None

for dt in datas:
    ano = dt.strftime("%Y")
    mes = dt.strftime("%m")
    dia = dt.strftime("%d")
    hora = dt.strftime("%H")

    caminho_arquivo = f"/oper/share/ioper/tempo/MERGE/GPM/HOURLY/{ano}/{mes}/{dia}/"
    nome_arquivo = f"MERGE_CPTEC_{dt.strftime('%Y%m%d%H')}.grib2"
    caminho_completo = os.path.join(caminho_arquivo, nome_arquivo)

    try:
        ds = xr.open_dataset(caminho_completo, engine="cfgrib", backend_kwargs={"indexpath": ""})
#        var_precip = [v for v in ds.data_vars if 'prec' in v.lower() or 'tp' in v.lower()]
        var_precip = [v for v in ds.data_vars if 'prec' in v.lower() or 'tp' in v.lower() or 'rdp' in v.lower()]
#        print(ds)
        if not var_precip:
            print(f"[MERGE] Variável de precipitação não encontrada em {caminho_completo}")
            continue
        prec = ds[var_precip[0]].squeeze()

    except FileNotFoundError:
        print(f"[MERGE] Arquivo não encontrado: {caminho_completo}")
        continue
    except Exception as e:
        print(f"[MERGE] Erro ao abrir {caminho_completo}: {e}")
        continue

    if merge_acumulada is None:
        merge_acumulada = prec.copy(deep=True)
        hora_inicializacao = dt.strftime("%Y-%m-%d %HZ")
    else:
        merge_acumulada += prec

    hora_finalizacao = dt.strftime("%Y-%m-%d %HZ")

if merge_acumulada is None:
    print("[MERGE] Nenhum dado carregado. Encerrando script.")
    exit()

# ================================
# Ajuste de longitudes do MERGE
# ================================

merge_acumulada = merge_acumulada.assign_coords({"longitude": (((merge_acumulada.longitude + 180) % 360) - 180)})
merge_acumulada = merge_acumulada.sortby("longitude")

# ================================
# Recorte MONAN para area do MERGE
# ================================

lat_min = float(merge_acumulada.latitude.min())
lat_max = float(merge_acumulada.latitude.max())
lon_min = float(merge_acumulada.longitude.min())
lon_max = float(merge_acumulada.longitude.max())

lat_monan = monan_acumulada.latitude
lat_crescente = lat_monan[1] > lat_monan[0]
lat_slice = slice(lat_min, lat_max) if lat_crescente else slice(lat_max, lat_min)

monan_recorte = monan_acumulada.sel(
    latitude=lat_slice,
    longitude=slice(lon_min, lon_max)
)

# ================================
# Interpolacao MONAN para grade do MERGE
# ================================

monan_interp = monan_recorte.interp(
    latitude=merge_acumulada.latitude,
    longitude=merge_acumulada.longitude,
    method="nearest"
)

# ================================
# Calculo do Vies e métricas
# ================================

mask_monan = ~np.isnan(monan_interp)
mask_merge = ~np.isnan(merge_acumulada)
mask_valida = mask_monan & mask_merge

vies_valido = (monan_interp - merge_acumulada).where(mask_valida)
pontos_validos = np.count_nonzero(mask_valida)
print(f"Pontos válidos em comum: {pontos_validos}")

vies_medio = vies_valido.mean().item()
print(f"? Viés médio (MONAN - MERGE): {vies_medio:.2f} mm")

dif = vies_valido
mae = np.abs(dif).mean().item()
rmse = np.sqrt((dif ** 2).mean()).item()
correlacao = np.corrcoef(monan_interp.values[mask_valida], merge_acumulada.values[mask_valida])[0, 1]
nse = 1 - ((dif ** 2).sum() / ((merge_acumulada - merge_acumulada.mean()).where(mask_valida) ** 2).sum()).item()

# ================================
# Plot do mapa de viés
# ================================

fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()})
#ax.set_extent([-57, -38, -27.5, -12])  # Sudeste
#ax.set_extent([-64, -39, -20, -40])  # Região Sul
#ax.set_extent([315, 327, -16, -2]) #Região Nordeste
#ax.set_extent([-70, -50, -25, -45])  # Prata
ax.set_extent([-70, -40, -45, -20])  # Região Sul2


lons, lats = np.meshgrid(merge_acumulada.longitude.values, merge_acumulada.latitude.values)

# Máscara para valores absolutos abaixo de 5 mm
vies_mascarado = vies_valido.where(np.abs(vies_valido) >= 5)

# Níveis de contorno de 5 em 5 mm (excluindo -5 a +5)
levels = np.concatenate((
    np.arange(-100, -4.9, 5),
    np.arange(5.1, 105, 5)
))

# Normalização centrada no zero
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)

# Contourf
cf = ax.contourf(lons, lats, vies_mascarado.values,
                 levels=levels, cmap="RdBu", norm=norm,
                 extend="both", transform=ccrs.PlateCarree())

# Barra de cores com ticks de 20 em 20 mm
cbar = plt.colorbar(cf, ax=ax, pad=0.03, fraction=0.023,
                    ticks=np.arange(-100, 101, 20))
cbar.set_label('Viés (MONAN - MERGE) [mm]', fontsize=12)

# Mapa
ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor='k')
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)

# Gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.3)
gl.right_labels = gl.top_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Título
ax.set_title(f'Viés de Precipitação Acumulada\nMONAN - MERGE\n{hora_inicializacao} até {hora_finalizacao}\nRodada do dia {data_base.strftime("%d/%m/%Y")}', fontsize=12, fontweight='bold')

# Salvar figura
os.makedirs(saida_figuras, exist_ok=True)
nome_figura = f'vies_monan_merge_{hora_inicializacao}_{hora_finalizacao}.png'
plt.savefig(os.path.join(saida_figuras, nome_figura), dpi=300, bbox_inches='tight')
plt.close()

print("? Mapa de viés gerado com sucesso!")

