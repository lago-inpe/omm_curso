# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 20:46:27 2025

@author: fedeo
"""

# -*- coding: utf-8 -*-
"""
cross_section_l2_profiles_monan.py

Versión MONAN del script WRF:
- Cross-section U + θ
- l² < 0 (ducting)
- Perfiles verticales: N, l², Ri, Fr (Ri y Fr basados solo en U)
- Escala log firmada para ver todos los índices juntos
- Recorte vertical uniforme a 12 km

@author: fede
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import gaussian_filter1d

# =============================================
# CONFIGURACIÓN
# =============================================

#monan_path = (
#    "D:/disco_EFI_WRF/datos_MONAN/nuevo_recorte/"
#    "2023071900/MONAN_DIAG_R_POS_GFS_2023071900_2023072123.00.00.x1.65536002L55_mendoza.nc"
#)

monan_path = ( 
	"https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/MendozaAR_ZondaWind_3km_1h/"
	"2023071900/MONAN_DIAG_R_POS_GFS_2023071900_2023072123.00.00.x1.65536002L55.nc#mode=bytes"
)

# Corte Chile → Mendoza
start = (-32.9, -71.0)
end   = (-32.9, -68.0)

profile_lon_sotavento = -68.86
vertical_smooth_sigma = 1.0

zmax_cross = 18000.0   # límite vertical común

#outdir = "C:/Users/Usuario/Desktop/proyecto_SMN_ERA/figuras/cross_l2_profiles_monan"
outdir = "figuras_cross_l2_profiles_monan"
os.makedirs(outdir, exist_ok=True)

# =============================================
# ABRIR MONAN
# =============================================

ds = xr.open_dataset(monan_path).metpy.parse_cf()

temp  = ds["temperature"]
qv    = ds["spechum"]
ucomp = ds["uzonal"]
zgeo  = ds["zgeo"]
topo  = ds["ter"]

# =============================================
# SECCIÓN TRANSVERSAL
# =============================================

cross = cross_section(
    ds[["temperature", "spechum", "uzonal", "zgeo", "ter"]],
    start,
    end
).squeeze()

lons = cross["longitude"].values
z = cross["zgeo"].values             # (nz, nx)
U = cross["uzonal"].values
T = cross["temperature"].values
topo_line = cross["ter"].values

nz, nx = U.shape

# ==========================
# RECORTAR TODO A 12 km
# ==========================
mask_z = z[:, 0] <= zmax_cross
z = z[mask_z, :]
U = U[mask_z, :]
T = T[mask_z, :]
theta_p_z = z[:, 0]   # eje vertical final

# =============================================
# THETA
# =============================================
p = mpcalc.height_to_pressure_std(z * units.m)
theta = mpcalc.potential_temperature(p, T * units.K).magnitude

# =============================================
# BRUNT–VAISÄLÄ
# =============================================
def calc_N(theta_prof, z_prof):
    g = 9.81
    dθdz = np.gradient(theta_prof, z_prof)
    N2 = (g / theta_prof) * dθdz
    N = np.sqrt(np.maximum(N2, 0))
    return N, N2, dθdz

# =============================================
# L²
# =============================================
def compute_l2(N2_field, U_field, z_levels):
    nz, nx = U_field.shape
    l2 = np.full_like(U_field, np.nan)

    for ix in range(nx):
        Ucol = U_field[:, ix]
        mask = np.isfinite(Ucol)
        if mask.sum() < 3:
            continue

        # rellenar huecos
        first = np.argmax(mask)
        last  = len(mask) - np.argmax(mask[::-1]) - 1
        Ucol[:first] = Ucol[first]
        Ucol[last+1:] = Ucol[last]

        # suavizar
        U_sm = gaussian_filter1d(Ucol, sigma=vertical_smooth_sigma)

        d2U = np.gradient(np.gradient(U_sm, z_levels), z_levels)
        U_safe = np.where(np.abs(U_sm) < 1e-6, 1e-6, U_sm)

        l2[:, ix] = (N2_field[:, ix] / (U_safe**2)) - (d2U / U_safe)

    return l2

# =============================================
# COMPUTAR N² EN TODO EL CORTE
# =============================================
N2_field = np.full_like(U, np.nan)

for ix in range(nx):
    th_col = theta[:, ix]
    mask = np.isfinite(th_col)
    if mask.sum() < 3:
        continue

    first = np.argmax(mask)
    last  = len(mask) - np.argmax(mask[::-1]) - 1
    th_col[:first] = th_col[first]
    th_col[last+1:] = th_col[last]

    _, N2_col, _ = calc_N(th_col, z[:, ix])
    N2_field[:, ix] = N2_col

# =============================================
# L²
# =============================================
l2_field = compute_l2(N2_field, U, z[:, 0])

# =============================================
# PUNTOS DE PERFIL
# =============================================
crest_idx = np.argmax(topo_line)
crest_lon = lons[crest_idx]
bar_idx   = max(0, crest_idx - int(0.2 * nx))
sot_idx   = np.argmin(np.abs(lons - profile_lon_sotavento))

idxs  = [bar_idx, crest_idx, sot_idx]
names = ["barlovento", "cresta", "sotavento"]

# =============================================
# FIGURA TOTAL
# =============================================
fig = plt.figure(figsize=(16, 6))

# =============================================
# PANEL A: CROSS SECTION
# =============================================
ax1 = fig.add_subplot(1, 2, 1)

cf = ax1.contourf(
    lons, z[:, 0], U,
    levels=40, cmap="viridis"
)

thetalevs = np.arange(np.nanmin(theta), np.nanmax(theta), 2)
ax1.contour(lons, z[:, 0], theta, levels=thetalevs,
            colors="k", linewidths=0.4)

# l² < 0
mask_l2 = l2_field < 0
ax1.contourf(lons, z[:, 0],
             np.where(mask_l2, -1, np.nan),
             levels=[-1, -0.5, 0],
             colors=("red",), alpha=0.25)

# terreno
ax1.fill_between(lons, 0, topo_line, color="saddlebrown")
ax1.plot(lons, topo_line, "k", lw=0.7)

ax1.axvline(crest_lon,     color="white",   ls="--")
ax1.axvline(lons[bar_idx], color="cyan",    ls="--")
ax1.axvline(lons[sot_idx], color="magenta", ls="--")

ax1.set_xlabel("Longitude")
ax1.set_ylabel("Height (m)")
ax1.set_title("Cross-section MONAN: U + θ + l²<0")

fig.colorbar(cf, ax=ax1, pad=0.02, label="U (m/s)")

# =============================================
# PANEL B: PERFILES
# =============================================
ax2 = fig.add_subplot(1, 2, 2)

cols = ["tab:blue", "tab:green", "tab:red"]
H_mountain = max(np.max(topo_line) - np.min(topo_line), 1)

# escala log firmada
def signed_log(x):
    return np.sign(x) * np.log10(1 + np.abs(x))

for color, ix, name in zip(cols, idxs, names):

    θ = theta[:, ix]
    Uc = U[:, ix]
    zc = z[:, ix]

    N, N2, _ = calc_N(θ, zc)
    dUdz = np.gradient(Uc, zc)
    shear2 = np.maximum(dUdz**2, 1e-6)
    Ri = N2 / shear2
    Fr = Uc / (np.maximum(N, 1e-4) * H_mountain)
    l2 = l2_field[:, ix]

    # transformar
    ax2.plot(signed_log(N),  zc, color=color, lw=1.8, label=f"N — {name}")
    ax2.plot(signed_log(l2), zc, color=color, ls="--", lw=1.4, label=f"l² — {name}")
    ax2.plot(signed_log(Ri), zc, color=color, ls="-.", lw=1.7, label=f"Ri — {name}")
    ax2.plot(signed_log(Fr), zc, color=color, ls=":",  lw=1.5, label=f"Fr — {name}")

# referencias
ax2.axvline(signed_log(0.25), color="grey", ls=":", lw=0.8)
ax2.axvline(signed_log(1.0),  color="grey", ls="--", lw=0.8)

ax2.set_xlim(-3, 3)
ax2.set_xlabel("Índices (escala log firmada)")
ax2.set_ylabel("Altura (m)")
ax2.set_title("Perfiles MONAN: N, l², Ri, Fr")
ax2.grid(True)
ax2.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "cross_monan_full.png"), dpi=130)

print("\n>>> COMPLETADO CON ÉXITO PARA MONAN <<<")
