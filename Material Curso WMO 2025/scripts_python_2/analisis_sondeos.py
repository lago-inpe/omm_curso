# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:02:52 2025

@author: fedeo
"""

# -*- coding: utf-8 -*-
"""
diagnostics_foehn_master.py
Versión extendida con:
- Descarga Wyoming (Siphon)
- Scorer parameter con régimen de onda
- Brunt-Vaisala
- Richardson (Ri)
- Froude con zonas supercríticas
- Probabilidad dinámica de foehn
- Gráficos de diagnóstico completos (PNG)

Autor: Fede Otero (adaptado por ChatGPT)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

g = 9.81
R_div_cp = 0.286

# =========================================================
# FISICA
# =========================================================

def calc_theta(T_K, p_hPa):
    return T_K * (1000.0 / p_hPa) ** R_div_cp

def brunt_vaisala(theta, z):
    dtheta_dz = np.gradient(theta, z)
    N2 = (g / theta) * dtheta_dz
    N = np.sqrt(np.maximum(N2, 0.0))
    return N, N2, dtheta_dz

def scorer_parameter(N2, U, z):
    d2U_dz2 = np.gradient(np.gradient(U, z), z)
    return (N2 / (U**2 + 1e-6)) - (1.0 / (U + 1e-6)) * d2U_dz2

def froude_number(U_char, N_char, H):
    if N_char <= 0 or H <= 0:
        return np.nan
    return U_char / (N_char * H)

def richardson_number(theta, u, v, z):
    dtheta_dz = np.gradient(theta, z)
    du_dz = np.gradient(u, z)
    dv_dz = np.gradient(v, z)
    shear2 = du_dz**2 + dv_dz**2 + 1e-6
    return (g / theta) * dtheta_dz / shear2

# =========================================================
# REGIMEN DE ONDAS
# =========================================================

def classify_wave_regime(l2):
    if np.all(l2 > 0):
        return "Onda propagante (no atrapada)"
    elif np.any(l2 < 0):
        return "Onda atrapada (trapped wave)"
    else:
        return "Indeterminado"

# =========================================================
# PROBABILIDAD DINÁMICA DE FOEHN
# =========================================================

def foehn_probability(Fr, l2, Ri):
    score = 0

    # 1. Froude crítico
    if Fr > 1.2:
        score += 2
    elif Fr > 0.8:
        score += 1

    # 2. Onda atrapada (clásico en Zonda fuerte)
    if np.any(l2 < 0):
        score += 2

    # 3. Turbulencia/rotor (Ri < 0.25)
    if np.any(Ri < 0.25):
        score += 2
    elif np.any(Ri < 0.5):
        score += 1

    # 4. Estabilidad fuerte en cima (inversión)
    # (este punto puede añadirse si se desea medir dtheta/dz grande)
    
    # Clasificación
    if score >= 5:
        return "Alta"
    elif score >= 3:
        return "Moderada"
    else:
        return "Baja"

# =========================================================
# DESCARGA WYOMING
# =========================================================

def download_sound_wyoming(station, year, month, day, hour):
    dt = datetime(year, month, day, hour)
    print(f"Descargando sondeo WYOMING: {station} {dt}")

    df = WyomingUpperAir.request_data(dt, station)

    # Normalización
    df = df[['pressure', 'height', 'temperature', 'u_wind', 'v_wind']]
    df.rename(columns={
        'pressure': 'p',
        'height': 'z',
        'temperature': 'temp',
        'u_wind': 'u',
        'v_wind': 'v'
    }, inplace=True)

    df.dropna(subset=['z'], inplace=True)
    return df.reset_index(drop=True)

# =========================================================
# PROCESAMIENTO COMPLETO
# =========================================================

def process_profile(df, outdir="diag_out"):

    os.makedirs(outdir, exist_ok=True)

    z = df['z'].values.astype(float)
    T = df['temp'].values.astype(float)
    T_K = T + 273.15
    p = df['p'].values.astype(float)
    u = df['u'].values.astype(float)
    v = df['v'].values.astype(float)
    U = np.sqrt(u**2 + v**2)

    theta = calc_theta(T_K, p)
    N, N2, dtheta_dz = brunt_vaisala(theta, z)
    l2 = scorer_parameter(N2, U, z)
    Ri = richardson_number(theta, u, v, z)

    H = np.max(z) - np.min(z)
    U_char = np.nanmax(U)
    N_char = np.nanmean(N[N > 0])
    Fr = froude_number(U_char, N_char, H)

    regime = classify_wave_regime(l2)
    foehn_prob = foehn_probability(Fr, l2, Ri)

    # =========================================================
    # GUARDAR DIAGNÓSTICOS
    # =========================================================
    out = pd.DataFrame({
        'z': z, 'T': T, 'p': p,
        'u': u, 'v': v, 'U': U,
        'theta': theta,
        'N': N, 'l2': l2, 'Ri': Ri
    })
    out.to_csv(os.path.join(outdir, "diagnostics_profile.csv"), index=False)

    # =========================================================
    # GRAFICOS
    # =========================================================

    # 1) Viento
    plt.figure()
    plt.plot(U, z)
    plt.xlabel("U (m/s)"); plt.ylabel("z (m)")
    plt.title("Perfil de viento U")
    plt.grid(); plt.savefig(os.path.join(outdir,"U_profile.png"))

    # 2) Temperatura
    plt.figure()
    plt.plot(T, z)
    plt.xlabel("T (°C)"); plt.ylabel("z (m)")
    plt.title("Perfil de temperatura")
    plt.grid(); plt.savefig(os.path.join(outdir,"T_profile.png"))

    # 3) Brunt-Vaisala
    plt.figure()
    plt.plot(N, z)
    plt.xlabel("N (1/s)"); plt.ylabel("z (m)")
    plt.title("Brunt-Väisälä Frequency")
    plt.grid(); plt.savefig(os.path.join(outdir,"N_profile.png"))

    # 4) Scorer con zonas atrapadas
    plt.figure()
    plt.plot(l2, z, label="l²")
    plt.fill_betweenx(z, l2, 0, where=(l2<0), color='red', alpha=0.3,
                      label="l² < 0 (onda atrapada)")
    plt.xlabel("l² (1/m²)"); plt.ylabel("z (m)")
    plt.title("Scorer Parameter (Durran 1990)")
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(outdir,"scorer_regime.png"))

    # 5) Richardson Number
    plt.figure()
    plt.plot(Ri, z)
    plt.axvline(0.25, color='r', linestyle='--', label="Ri=0.25 (Rotor)")
    plt.xlabel("Ri"); plt.ylabel("z (m)")
    plt.xlim(-1, 1)
    plt.title("Richardson Number")
    plt.grid(); plt.legend()
    plt.savefig(os.path.join(outdir,"Ri_profile.png"))

    # 6) Froude diagnostic
    plt.figure()
    plt.axhline(Fr, color='b')
    plt.title(f"Froude characteristic: Fr={Fr:.2f}")
    plt.text(0.1,0.8,f"Fr={Fr:.2f}", transform=plt.gca().transAxes)
    plt.savefig(os.path.join(outdir,"Froude.png"))

    # =========================================================
    # IMPRIMIR RESULTADOS
    # =========================================================
    print("\n==== DIAGNÓSTICO COMPLETO ====")
    print(f"Régimen de onda: {regime}")
    print(f"Froude: {Fr:.2f}")
    print(f"Probabilidad dinámica de foehn: {foehn_prob}")
    print(f"Archivos generados en: {outdir}\n")

    # -----------------------------------------
    # CÁLCULO DEL NÚMERO DE FROUDE – MÚLTIPLES VERSIÓNES
    # -----------------------------------------

    # Froude clásico (mantengo tu criterio)
    H = np.max(z) - np.min(z)
    U_char = float(np.nanmax(U))
    N_char = float(np.nanmean(N[np.isfinite(N)]))
    Fr_classic = froude_number(U_char, N_char, H)

    # -----------------------------------------
    # Froude vertical: Fr(z) = U(z)/(N(z)*H)
    # -----------------------------------------
    Fr_z = np.zeros_like(z) * np.nan
    valid = (N > 0)
    Fr_z[valid] = U[valid] / (N[valid] * H)

    # -----------------------------------------
    # Froude de capa (0–5000 m o top disponible)
    # -----------------------------------------
    z_top = 5000.0
    layer_mask = z <= min(z_top, np.max(z))

    if np.sum(layer_mask) > 5:
        U_layer = np.nanmean(U[layer_mask])
        N_layer = np.nanmean(N[layer_mask][N[layer_mask] > 0])
        H_layer = np.max(z[layer_mask]) - np.min(z[layer_mask])
        Fr_layer = U_layer / (N_layer * H_layer)
    else:
        Fr_layer = np.nan

    # -----------------------------------------
    # Froude efectivo (promedio ponderado)
    # Durran (1990) y Klemp & Lilly (1975)
    # -----------------------------------------
    dz = np.gradient(z)
    U_int = np.trapz(U, z)
    N_int = np.trapz(N, z)

    if N_int > 0:
        Fr_eff = U_int / (H * N_int)
    else:
        Fr_eff = np.nan

    # Guardamos todos
    out['Fr_z'] = Fr_z

    summary = {
        'U_max_m_s': float(np.nanmax(U)),
        'N_mean_1_s': float(N_char),
        'H_m': float(H),
        'Fr_classic': float(Fr_classic),
        'Fr_layer_0_3km': float(Fr_layer),
        'Fr_effective': float(Fr_eff)
    }

    # Gráficos del Froude
    plt.figure()
    plt.plot(Fr_z, z, label='Fr(z)', marker='o')
    plt.axvline(1, color='red', linestyle='--', label='Fr = 1')
    plt.xlabel('Fr')
    plt.ylabel('Altura (m)')
    plt.title('Perfil vertical del número de Froude')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, 'Fr_vertical.png'))
    plt.close()

    plt.figure()
    labels = ['Fr clásico', 'Fr capa 0–3 km', 'Fr efectivo']
    values = [Fr_classic, Fr_layer, Fr_eff]
    plt.bar(labels, values)
    plt.ylabel('Froude')
    plt.title('Comparación de variantes de Froude')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(outdir, 'Fr_comparison.png'))
    plt.close()

    # Guardar summary
    pd.DataFrame([summary]).to_csv(os.path.join(outdir, 'summary_froude.csv'),
                                   index=False)


# =========================================================
# EJECUCION PARA SPYDER
# =========================================================

if __name__ == "__main__":
    # Cambiá estos valores para probar distintos sondeos
    station = "85586"  #  "85586"   # SCSN
    year, month, day, hour = 2023, 7, 22, 12

    df = download_sound_wyoming(station, year, month, day, hour)
    process_profile(df, outdir="foehn_diag_SCSN_20230722")

################################################################

