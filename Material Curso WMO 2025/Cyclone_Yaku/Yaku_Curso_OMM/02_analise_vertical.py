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
rodadas = ["2023030900", "2023031000", "2023031100", "2023031200"]
previsoes = ["2023030900", "2023031000", "2023031100", "2023031200"]

# Limites da área de interesse para PLOTAGEM E CÁLCULO DO CENTRO
lon_min, lon_max = -90, -80
lat_min, lat_max = -15, 0

# Fator de suavização (Sigma) para Vorticidade e Divergência
SIGMA_VORT_DIV = 1.5 

fig, axs = plt.subplots(
    4, len(previsoes),
    figsize=(14, 26),
    subplot_kw=dict(projection=ccrs.PlateCarree()),
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
)

for j, previsao in enumerate(previsoes):
    rodada = previsao
    pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')
    arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
    caminho = os.path.join(pasta, arquivo)

    if not os.path.exists(caminho):
        for i in range(4): axs[i, j].set_visible(False)
        print(f"⚠️ Arquivo não encontrado para {previsao[:8]}. Coluna {j+1} pulada.")
        continue

    print(f"🔹 Lendo: {caminho}")
    dado = Dataset(caminho, 'r')
    lons_all = dado.variables['longitude'][:]
    lats_all = dado.variables['latitude'][:]
    
    # === 1. RESTRIÇÃO DO CÁLCULO DO CENTRO À ÁREA DE INTERESSE ===
    mslp_all = dado.variables['mslp'][0, :, :] / 100.0
    
    # Máscara para restringir a busca do mínimo de pressão
    lat2d, lon2d = np.meshgrid(lats_all, lons_all, indexing='ij')
    area_mask = (lon2d >= lon_min) & (lon2d <= lon_max) & \
                (lat2d >= lat_min) & (lat2d <= lat_max)
    
    mslp_masked = np.ma.masked_where(~area_mask, mslp_all)
    
    idx_min_flat = np.argmin(mslp_masked)
    centro_idx = np.unravel_index(idx_min_flat, mslp_all.shape)
    
    lat_centro_sf = lats_all[centro_idx[0]]
    lon_centro_sf = lons_all[centro_idx[1]]
    
    print(f"   Centro de Superfície {previsao[:8]} (Restrito): Lon={lon_centro_sf:.2f}, Lat={lat_centro_sf:.2f}")

    # === PLOTAGEM DO PONTO DE REFERÊNCIA ===
    # Esta função será chamada apenas a partir da 2ª linha (850 hPa)
    def plot_center_point(ax):
        ax.plot(lon_centro_sf, lat_centro_sf, 'o', color='red', markersize=7, 
                markeredgecolor='black', zorder=10, transform=ccrs.PlateCarree())


    lons = lons_all
    lats = lats_all

    # ========== 1️⃣ Superfície  ==========
    ax = axs[0, j]
    temp2m = dado.variables['t2m'][0, :, :] - 273.15
    u10 = dado.variables['uzonal'][0, 0, :, :]
    v10 = dado.variables['umeridional'][0, 0, :, :]

    cf = ax.contourf(lons, lats, temp2m, levels=np.arange(20, 33, 1),
                     cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, mslp_all, levels=np.arange(980, 1020, 2),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())

    flip = np.zeros((u10.shape[0], u10.shape[1]))
    flip[lats < 0] = 1
    ax.clabel(cs, fmt='%1.0f', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u10[::20, ::20], v10[::20, ::20],
             length=5.5, linewidth=0.8, pivot='middle', color='dimgray', flip_barb=flip[::20, ::20],
             transform=ccrs.PlateCarree())
    ax.set_title(f"{previsao[:8]}", fontsize=12)


    # ========== 2️⃣ 850 hPa ==========
    ax = axs[1, j]
    rh850 = dado.variables['relhum'][0, 2, :, :]
    z850 = dado.variables['zgeo'][0, 2, :, :] 
    u850 = dado.variables['uzonal'][0, 2, :, :]
    v850 = dado.variables['umeridional'][0, 2, :, :]

    cf = ax.contourf(lons, lats, rh850, levels=np.arange(40, 101, 5),
                     cmap='YlGnBu', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, z850, levels=np.arange(1400, 1600, 10),
                    colors='black', linewidths=1, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u850[::20, ::20], v850[::20, ::20],
             length=5.0, linewidth=0.8, color='black', flip_barb=flip[::20, ::20], transform=ccrs.PlateCarree())
    ax.set_title(f"{previsao[:8]}", fontsize=12)
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
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    vort = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy).to('1/s').magnitude * 1e5
    
    vort_smooth = gaussian_filter(vort, sigma=SIGMA_VORT_DIV)

    cf = ax.contourf(lons, lats, vort_smooth, levels=np.arange(-10, 11, 1),
                     cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
    cs = ax.contour(lons, lats, gaussian_filter(thickness,8), cmap='seismic', linestyles='dashed', linewidths=1.0, levels=np.arange(4900, 5900, 20), transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt='%d', inline=0, fontsize=10)
    ax.barbs(lons[::20], lats[::20], u500[::20, ::20], v500[::20, ::20],
             length=5.0, linewidth=0.8, color='dimgray', flip_barb=flip[::20, ::20], transform=ccrs.PlateCarree())
    ax.set_title(f"{previsao[:8]}", fontsize=12)
    plot_center_point(ax)

    # ========== 4️⃣ 250 hPa ==========
    ax = axs[3, j]
    u250 = dado.variables['uzonal'][0, 8, :, :]
    v250 = dado.variables['umeridional'][0, 8, :, :]

    div_nivel = mpcalc.divergence(u250,v250, dx=dx, dy=dy)
    div_nivel = gaussian_filter(div_nivel, 1)
    #Quero somente os valores positivos da divergencia
    mask_div = ma.masked_less_equal(div_nivel, 0).mask
    div_nivel[mask_div] = np.nan

    cf = ax.contour(lons, lats, div_nivel*1e5, cmap='gnuplot', linewidths=1.0,linestyles='solid', extend='both', transform=ccrs.PlateCarree())
    ax.streamplot(lons, lats, u250, v250, density=[2, 2], linewidth=1, color='gray', transform=ccrs.PlateCarree())
    ax.set_title(f"{previsao[:8]}", fontsize=12)
    plot_center_point(ax)

    # --- elementos comuns ---
    for i in range(4):
        ax = axs[i, j]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3)
        if j == 0:
            ax.text(-0.15, 0.5, ["Pressão, vento (1000 hPa) e Temp. 2m", "UR, vento e altura geopotencial em 850 hPa", "Vorticidade e vento (500 hPa) e espessura (500-1000)", "Vento e divergência (250 hPa)"][i],
                    fontsize=12, va='center', rotation=90, transform=ax.transAxes)

# === COLORBARS ===
labels = ["Temperatura (°C)", "Umidade relativa (%)",
          "Vorticidade (x10⁻⁵ s⁻¹)"]

for i in range(3):
    im = axs[i, 0].collections[0]
    cb = fig.colorbar(im, ax=axs[i, :], orientation='horizontal', fraction=0.05, pad=0.02)
    cb.set_label(labels[i], fontsize=10)

plt.suptitle("Evolução do Ciclone Tropical – 9 a 12 de março de 2023", fontsize=15, y=0.88)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("analise_vertical.png", dpi=300, bbox_inches='tight')
