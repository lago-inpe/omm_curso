import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import datetime
import metpy.calc as mpcalc
from metpy.units import units

# === CONFIGURAÇÕES GERAIS ===
# Lista de datas/rodadas para processar: 10, 11 e 12 de Março de 2023 às 00Z.
rodadas = ["2023031000", "2023031100", "2023031200"] 
Re = 6371.0   # Raio médio da Terra (km)
dr = 10       # Largura dos anéis (km)
r_max = 500   # Raio máximo (km)

# LIMITES GEOGRÁFICOS DE BUSCA (São mantidos os mesmos do script original)
lon_min, lon_max = -95, -80 # Longitude (Oeste a Leste)
lat_min, lat_max = -20, 5   # Latitude (Sul a Norte)

# OBTÉM NÍVEIS DE PRESSÃO (definidos aqui para evitar repetição no loop, a menos que sejam dinâmicos)
# Serão ajustados dentro do loop se a variável 'level' estiver presente no arquivo.
niveis_padrao = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200])

# Variáveis de interesse
temp_var_name = "temperature" 
u_var_name = "uzonal"
v_var_name = "umeridional" 

# === LOOP PRINCIPAL SOBRE AS DATAS ===
for rodada_str in rodadas:
    
    # 1. CONSTRUÇÃO DO NOME DO ARQUIVO
    arquivo = f"/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/{rodada_str}/MONAN_DIAG_R_POS_GFS_{rodada_str}_{rodada_str}.00.00.x1.5898242L55.nc"
    nome_saida = f"umidade_temperatura_vorticidade_{rodada_str[:8]}.png"

    print(f"\n--- Processando rodada: {rodada_str} ---")

    # === EXTRAI DATA DO ARQUIVO ===
    try:
        data_rodada = datetime.datetime.strptime(rodada_str, "%Y%m%d%H")
        data_fmt = data_rodada.strftime("%d %b %Y %H UTC")
    except:
        data_fmt = rodada_str  # fallback
    
    # Verifica a existência do arquivo antes de prosseguir
    if not os.path.exists(arquivo):
        print(f"Aviso: Arquivo '{arquivo}' não encontrado. Pulando para a próxima rodada.")
        continue

    # === LEITURA DO ARQUIVO ===
    try:
        dado = Dataset(arquivo)
    except Exception as e:
        print(f"Erro ao abrir o arquivo {arquivo}: {e}. Pulando.")
        continue

    lons = dado.variables['longitude'][:]
    lats = dado.variables['latitude'][:]
    
    # Tentativa de obter pressão (mslp ou pres_sf)
    pres_raw = None
    if "mslp" in dado.variables:
        pres_raw = dado.variables['mslp'][0, :, :] / 100.0  # Pa → hPa
    elif "pres_sf" in dado.variables:
        pres_raw = dado.variables['pres_sf'][0, :, :] / 100.0  # Pa → hPa
    else:
         print(f"Erro: Variável de pressão de superfície (mslp ou pres_sf) não encontrada em {arquivo}. Pulando.")
         dado.close()
         continue

    # === IDENTIFICAR CENTRO DO CICLONE (mínimo de pressão) ===
    # 1. Cria a máscara para a área de busca
    lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]

    if len(lon_indices) == 0 or len(lat_indices) == 0:
        print("Erro: A área de busca especificada não contém pontos no domínio do arquivo. Pulando.")
        dado.close()
        continue

    # 2. Restringe a matriz de pressão e as coordenadas para a área de busca
    lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

    pres_search = pres_raw[lat_start:lat_end, lon_start:lon_end]
    lats_search = lats[lat_start:lat_end]
    lons_search = lons[lon_start:lon_end]

    # 3. Encontra o mínimo de pressão (índices relativos à matriz 'pres_search')
    if np.all(np.isnan(pres_search)):
        print("Erro: A matriz de pressão na área de busca é toda NaN. Pulando.")
        dado.close()
        continue
        
    centro_relativo = np.unravel_index(np.nanargmin(pres_search), pres_search.shape)

    # 4. Converte os índices relativos para coordenadas (lat/lon)
    lat_centro, lon_centro = float(lats_search[centro_relativo[0]]), float(lons_search[centro_relativo[1]])
    print(f"Centro do ciclone (Busca em [{lat_min}/{lat_max}] x [{lon_min}/{lon_max}]): {lat_centro:.2f}°S, {lon_centro:.2f}°W")

    pres = pres_raw 

    # === CALCULAR DISTÂNCIA RADIAL ===
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    # Fórmula para cálculo de distância em uma esfera (grande círculo)
    dist = Re * np.arccos(
        np.sin(np.deg2rad(lat_centro)) * np.sin(np.deg2rad(lat2d)) +
        np.cos(np.deg2rad(lat_centro)) * np.cos(np.deg2rad(lat2d)) *
        np.cos(np.deg2rad(lon2d - lon_centro))
    )
    raios = np.arange(0, r_max + dr, dr)
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)


    # === OBTÉM NÍVEIS DE PRESSÃO (corrigindo unidade e limitando até 200 hPa) ===
    niveis_orig = niveis_padrao # Começa com o padrão
    if "level" in dado.variables:
        niveis_orig_data = dado.variables["level"][:]
        # Tenta corrigir a unidade se os valores forem muito grandes (assumindo Pa)
        if np.nanmax(niveis_orig_data) > 2000:
            niveis_orig_data = niveis_orig_data / 100.0
        niveis_orig = niveis_orig_data
        
    mask_niveis = niveis_orig >= 200
    niveis = niveis_orig[mask_niveis]

    if len(niveis) == 0:
        print("Erro: Nenhum nível de pressão acima de 200 hPa encontrado. Pulando.")
        dado.close()
        continue

    # === MATRIZES PARA PERFIS VERTICAIS MÉDIOS ===
    rh_vertical = np.zeros((len(niveis), len(raios)))
    temp_vertical = np.zeros((len(niveis), len(raios)))
    vort_vertical = np.zeros((len(niveis), len(raios))) 

    # Verifica avisos sobre variáveis de interesse
    if temp_var_name not in dado.variables:
        print(f"Aviso: Variável de Temperatura '{temp_var_name}' não encontrada.")
    if u_var_name not in dado.variables or v_var_name not in dado.variables:
        print(f"Aviso: Variáveis de Vento ('{u_var_name}' ou '{v_var_name}') não encontradas.")
    
    # Variável 'relhum' (Umidade Relativa) é essencial, se não existir, pula o processamento do loop principal.
    if "relhum" not in dado.variables:
        print(f"Erro: Variável de Umidade Relativa ('relhum') não encontrada. Pulando.")
        dado.close()
        continue


    # === LOOP EM NÍVEIS PARA RH, TEMPERATURA E VORT ===
    for ki, k in enumerate(np.where(mask_niveis)[0]):
        
        # --- Umidade Relativa ---
        try:
            rh = dado.variables["relhum"][0, k, :, :]
            for i, r in enumerate(raios[:-1]):
                mask = (dist >= r) & (dist < r + dr)
                if np.any(mask):
                    rh_vertical[ki, i] = np.nanmean(rh[mask])
        except Exception as e:
            print(f"Erro ao processar Umidade Relativa no nível {niveis[ki]} hPa: {e}")
            
        # --- Temperatura ---
        if temp_var_name in dado.variables:
            try:
                temp = dado.variables[temp_var_name][0, k, :, :] - 273.15 # K -> °C
                for i, r in enumerate(raios[:-1]):
                    mask = (dist >= r) & (dist < r + dr)
                    if np.any(mask):
                        temp_vertical[ki, i] = np.nanmean(temp[mask])
            except Exception as e:
                print(f"Erro ao processar Temperatura no nível {niveis[ki]} hPa: {e}")
                
        # --- CÁLCULO DA VORTICIDADE RELATIVA (VORT) ---
        if u_var_name in dado.variables and v_var_name in dado.variables:
            try:
                u_comp = dado.variables[u_var_name][0, k, :, :]
                v_comp = dado.variables[v_var_name][0, k, :, :]
                
                u_q = units.Quantity(u_comp, "m/s")
                v_q = units.Quantity(v_comp, "m/s")
                
                # Calcula a vorticidade e converte para a unidade desejada (1e-5 s^-1)
                vort_calc = mpcalc.vorticity(u_q, v_q, dx=dx, dy=dy).to('1/s').magnitude * 1e5
                
                for i, r in enumerate(raios[:-1]):
                    mask = (dist >= r) & (dist < r + dr)
                    if np.any(mask):
                        vort_vertical[ki, i] = np.nanmean(vort_calc[mask])
            except Exception as e:
                print(f"Erro ao calcular e processar Vorticidade no nível {niveis[ki]} hPa: {e}")
                
    # Fechar o arquivo NetCDF antes de iniciar a próxima iteração
    dado.close()

    # Suavização leve
    rh_vertical = gaussian_filter(rh_vertical, sigma=1)
    temp_vertical = gaussian_filter(temp_vertical, sigma=1)
    # A vorticidade não foi suavizada no script original, mas pode ser útil:
    vort_vertical = gaussian_filter(vort_vertical, sigma=1)

    # === PLOTAGEM DO PAINEL ===
    fig, (ax_rh, ax_temp, ax_vort) = plt.subplots(1, 3, figsize=(18, 6), sharey=True) 
    X, Y = np.meshgrid(raios, niveis)

    # --- GRÁFICO 1: UMIDADE RELATIVA (Esquerda) ---
    cf_rh = ax_rh.contourf(X, Y, rh_vertical, levels=np.arange(40, 101, 5),
                           cmap="YlGnBu", extend="both")
    cs_rh = ax_rh.contour(X, Y, rh_vertical, levels=[70, 80, 90, 95],
                          colors="k", linewidths=0.5)

    ax_rh.invert_yaxis()
    ax_rh.set_xlabel("Distância radial ao centro (km)")
    ax_rh.set_ylabel("Nível de pressão (hPa)")
    ax_rh.set_yticks(niveis)
    ax_rh.set_yticklabels([f"{int(n)}" for n in niveis])

    ax_rh.set_title("Umidade relativa (%)", fontsize=12)
    plt.colorbar(cf_rh, ax=ax_rh, label="Umidade relativa (%)", pad=0.02)


    # --- GRÁFICO 2: TEMPERATURA (Centro) ---
    levels_temp = np.arange(-30, 31, 5) 
    cf_temp = ax_temp.contourf(X, Y, temp_vertical, levels=levels_temp,
                               cmap="RdYlBu_r", extend="both")
    cs_temp = ax_temp.contour(X, Y, temp_vertical, levels=levels_temp[::2],
                              colors="k", linewidths=0.5)

    ax_temp.set_xlabel("Distância radial ao centro (km)")
    ax_temp.set_title(f"Temperatura (°C)", fontsize=12)
    plt.colorbar(cf_temp, ax=ax_temp, label="Temperatura (°C)", pad=0.02)


    # --- GRÁFICO 3: VORTICIDADE RELATIVA (Direita) ---
    levels_vort = np.arange(-10, 11, 1) 
    cf_vort = ax_vort.contourf(X, Y, vort_vertical, levels=levels_vort,
                               cmap="RdBu_r", extend="both")
                               
    cs_vort = ax_vort.contour(X, Y, vort_vertical, levels=np.arange(1, 10, 2), # Linhas positivas
                              colors="k", linewidths=0.5)
    ax_vort.clabel(cs_vort, inline=True, fontsize=8, fmt='%1.0f') 

    cs_vort_neg = ax_vort.contour(X, Y, vort_vertical, levels=np.arange(-9, 0, 2), # Linhas negativas
                              colors="k", linestyles='dashed', linewidths=0.5)
    ax_vort.clabel(cs_vort_neg, inline=True, fontsize=8, fmt='%1.0f') 

    ax_vort.set_xlabel("Distância radial ao centro (km)")
    ax_vort.set_title(r"Vorticidade Relativa ($\times 10^{-5} s^{-1}$)", fontsize=12)
    plt.colorbar(cf_vort, ax=ax_vort, label=r"Vorticidade ($\times 10^{-5} s^{-1}$)", pad=0.02)


    # --- TÍTULO GERAL ---
    fig.suptitle(
        f"Seção Vertical Média Radial (RH, T, VORT)\n"
        f"Centro: {lat_centro:.2f}°S, {lon_centro:.2f}°W — {data_fmt}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(nome_saida, dpi=300, bbox_inches='tight')
    plt.close(fig) # Fecha a figura para liberar memória

print("\nProcessamento concluído para todas as rodadas.")
