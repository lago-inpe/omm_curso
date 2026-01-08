import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import ticker
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter
import numpy.ma as ma

# === CONFIGURAÇÕES GERAIS ===

# Rodadas (início da simulação) a serem analisadas
# Queremos as previsões dos dias 7, 8, 9 e 10
rodadas = ["2023030700", "2023030800", "2023030900", "2023031000"]

# O dia alvo da previsão (coluna única)
previsao_alvo = "2023031100" 
previsoes = [previsao_alvo] * len(rodadas)

# Limites da área de interesse para PLOTAGEM E CÁLCULO DO CENTRO (Mantidos)
lon_min, lon_max = -90, -80
lat_min, lat_max = -15, 0

# Fator de suavização (Sigma) para Vorticidade e Divergência
SIGMA_VORT_DIV = 1.5 

# === ESTRUTURA DA FIGURA ===
# 4 linhas (painéis: 1000, 850, 500, 250 hPa) x 4 colunas (rodadas 07 a 10)
fig, axs = plt.subplots(
    4, len(rodadas),
    figsize=(14, 26),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
)

# Níveis explícitos para o plot de Divergência
div_levels = np.arange(1, 11, 1) # Divergência de 1 a 10 (x10⁻⁵ s⁻¹)

for j, rodada in enumerate(rodadas):
    previsao = previsao_alvo
    
    pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')
    arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
    caminho = os.path.join(pasta, arquivo)

    if not os.path.exists(caminho):
        for i in range(4): axs[i, j].set_visible(False)
        print(f"⚠️ Arquivo não encontrado para Rodada {rodada[:8]} / Previsão {previsao[:8]}. Coluna {j+1} pulada.")
        continue

    print(f"🔹 Lendo: Rodada {rodada} | Previsão: {previsao}")
    dado = Dataset(caminho, 'r')
    lons_all = dado.variables['longitude'][:]
    lats_all = dado.variables['latitude'][:]
    
    # === 1. RESTRIÇÃO DO CÁLCULO DO CENTRO À ÁREA DE INTERESSE ===
    if "mslp" in dado.variables:
        mslp_all = dado.variables['mslp'][0, :, :] / 100.0
    elif "pres_sf" in dado.variables:
        mslp_all = dado.variables['pres_sf'][0, :, :] / 100.0
    else:
        # Se não há mslp ou pres_sf, desativa a coluna
        for i in range(4): axs[i, j].set_visible(False)
        print(f"⚠️ Variável de pressão de superfície não encontrada. Coluna {j+1} pulada.")
        continue

    # Máscara para restringir a busca do mínimo de pressão
    lat2d, lon2d = np.meshgrid(lats_all, lons_all, indexing='ij')
    area_mask = (lon2d >= lon_min) & (lon2d <= lon_max) & \
                (lat2d >= lat_min) & (lat2d <= lat_max)
    
    mslp_masked = np.ma.masked_where(~area_mask, mslp_all)
    
    idx_min_flat = np.argmin(mslp_masked)
    centro_idx = np.unravel_index(idx_min_flat, mslp_all.shape)
    
    lat_centro_sf = lats_all[centro_idx[0]]
    lon_centro_sf = lons_all[centro_idx[1]]
    
    print(f"   Centro de Superfície (Restrito): Lon={lon_centro_sf:.2f}, Lat={lat_centro_sf:.2f}")

    # === FUNÇÃO PARA PLOTAR O PONTO DE REFERÊNCIA ===
    def plot_center_point(ax):
        ax.plot(lon_centro_sf, lat_centro_sf, 'o', color='red', markersize=7, 
                markeredgecolor='black', zorder=10, transform=ccrs.PlateCarree())


    lons = lons_all
    lats = lats_all
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    flip = np.zeros((lons.shape[0], lats.shape[0])).T # Ajuste para flip no HS
    flip[lats < 0] = 1


    # ========== 1️⃣ Superfície (1000 hPa) ==========
    ax = axs[0, j]
    temp2m = dado.variables['t2m'][0, :, :] - 273.15
    u10 = dado.variables['uzonal'][0, 0, :, :]
    v10 = dado.variables['umeridional'][0, 0, :, :]

    cf = ax.contourf(lons, lats, temp2m, levels=np.arange(20, 33, 1),
                     cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, mslp_all, levels=np.arange(980, 1020, 2),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())

    ax.clabel(cs, fmt='%1.0f', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u10[::20, ::20], v10[::20, ::20],
             length=5.5, linewidth=0.8, pivot='middle', color='dimgray', flip_barb=flip[::20, ::20],
             transform=ccrs.PlateCarree())


    # ========== 2️⃣ 850 hPa ==========
    ax = axs[1, j]
    rh850 = dado.variables['relhum'][0, 2, :, :]
    z850 = dado.variables['zgeo'][0, 2, :, :] 
    u850 = dado.variables['uzonal'][0, 2, :, :]
    v850 = dado.variables['umeridional'][0, 2, :, :]

    cf_850 = ax.contourf(lons, lats, rh850, levels=np.arange(40, 101, 5),
                     cmap='YlGnBu', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, z850, levels=np.arange(1400, 1600, 10),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u850[::20, ::20], v850[::20, ::20],
             length=5.0, linewidth=0.8, color='black', flip_barb=flip[::20, ::20], transform=ccrs.PlateCarree())
    plot_center_point(ax)


    # ========== 3️⃣ 500 hPa ==========
    ax = axs[2, j]
    z500 = dado.variables['zgeo'][0, 5, :, :] 
    z1000 = dado.variables['zgeo'][0, 0, :, :] 
    thickness = z500 - z1000
    u500 = dado.variables['uzonal'][0, 5, :, :]
    v500 = dado.variables['umeridional'][0, 5, :, :]

    u_q = units.Quantity(u500, "m/s")
    v_q = units.Quantity(v500, "m/s")
    vort = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy).to('1/s').magnitude * 1e5
    
    vort_smooth = gaussian_filter(vort, sigma=SIGMA_VORT_DIV)

    cf_500 = ax.contourf(lons, lats, vort_smooth, levels=np.arange(-10, 11, 1),
                     cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, gaussian_filter(thickness,8), cmap='seismic', linestyles='dashed', linewidths=1.0, levels=np.arange(4900, 5900, 20), transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u500[::20, ::20], v500[::20, ::20],
             length=5.0, linewidth=0.8, color='dimgray', flip_barb=flip[::20, ::20], transform=ccrs.PlateCarree())
    plot_center_point(ax)


    # ========== 4️⃣ 250 hPa ==========
    ax = axs[3, j]
    u250 = dado.variables['uzonal'][0, 8, :, :]
    v250 = dado.variables['umeridional'][0, 8, :, :]

    # 🚨 CORREÇÃO DA UNIDADE: Envolve os vetores em Quantity(m/s) para resolver a DimensionalityError
    u250_q = units.Quantity(u250, "m/s")
    v250_q = units.Quantity(v250, "m/s")

    div_nivel = mpcalc.divergence(u250,v250, dx=dx, dy=dy)
    div_nivel = gaussian_filter(div_nivel, 1)
    #Quero somente os valores positivos da divergencia
    mask_div = ma.masked_less_equal(div_nivel, 0).mask
    div_nivel[mask_div] = np.nan

    # Altera para contourf e usa os níveis definidos
    cf = ax.contour(lons, lats, div_nivel*1e5, cmap='gnuplot', linewidths=1.0,linestyles='solid', extend='both', transform=ccrs.PlateCarree())
    ax.streamplot(lons, lats, u250, v250, density=[2, 2], linewidth=1, color='gray', transform=ccrs.PlateCarree())
    plot_center_point(ax)

    # --- elementos comuns e títulos de coluna ---
    for i in range(4):
        ax = axs[i, j]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3)
        
        # Título da coluna
        if i == 0:
            ax.set_title(f"Rodada: {rodada[:8]}", fontsize=12)
        
        # Rótulos da linha
        if j == 0:
            ax.text(-0.15, 0.5, ["Pressão, vento (1000 hPa) e Temp. 2m", "UR, Vento e altura geopotencial em 850 hPa", "Vorticidade e vento (500 hPa) e Espessura 500-1000", "Vento e divergência (250 hPa)"][i],
                    fontsize=12, va='center', rotation=90, transform=ax.transAxes)

# === COLORBARS ===
labels = ["Temperatura (°C)", "Umidade relativa (%)",
          "Vorticidade (x10⁻⁵ s⁻¹)"]

# Cbars para as 3 primeiras linhas (Temperatura, UR, Vorticidade)
im_list = [axs[0, 0].collections[0], axs[1, 0].collections[0], axs[2, 0].collections[0]]

for i, im in enumerate(im_list):
    cb = fig.colorbar(im, ax=axs[i, :], orientation='horizontal', fraction=0.05, pad=0.02)
    cb.set_label(labels[i], fontsize=10)

plt.suptitle(f"Previsão para {previsao_alvo[:8]} 00 UTC", fontsize=15, y=0.88)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("previsoes_analise_vertical.png", dpi=300, bbox_inches='tight')
print("✅ Figura salva como analise_vertical.png")
