#!/usr/bin/env python3
"""
previsoes_ciclone_track_error.py

Função:
Rastreamento do centro (mínimo de MSLP) para várias rodadas e lead times (12h ou 24h, salvo com 24h),
comparando com centros "observados" estimados a partir das análises (arquivos de análise
no próprio modelo) e cálculo do erro de posição (km).

Saídas:
 - tracks_csv/track_<rodada>.csv  : centros previstos por lead
 - tracks_csv/track_obs_all.csv   : centros 'observados' (análises)
 - track_error_summary.csv        : erros consolidados
 - tracks_vs_obs.png              : figura com trajetórias previstas e observadas
 - erro_posicao_vs_lead.png       : erro de posição por rodada x lead time
"""

import os # Para interagir com o sistema operacional (caminhos e diretórios)
from datetime import datetime, timedelta # Para manipulação de datas e tempos de previsão
import numpy as np # Biblioteca essencial para operações numéricas e arrays (matrizes)
from netCDF4 import Dataset # Para ler e manipular arquivos no formato NetCDF (dados do modelo)
import matplotlib.pyplot as plt # Para gerar gráficos (as trajetórias e os erros)
import cartopy.crs as ccrs # Para definir e manipular projeções cartográficas
import cartopy.feature as cfeature # Para adicionar elementos geográficos (costa, fronteiras)
import csv # Para ler e escrever arquivos CSV (onde os resultados são salvos)
import math # Para funções matemáticas (usado no cálculo da distância Haversine)

# -----------------------
# CONFIGURAÇÃO
# -----------------------
# Lista das datas e horas de inicialização (rodadas) das previsões a serem analisadas
rodadas = [
    "2023030700", "2023030800", "2023030900",
    "2023031000", "2023031100", "2023031200"
]

# Intervalos de tempo de previsão (lead times) em horas. 
# Ex: 0h (análise), 24h, 48h, ..., 120h.
forecast_steps_h = list(range(0, 121, 24))

# Padrão de nome dos arquivos de dados. {rodada} é a inicialização, {forecast} é o tempo de previsão.
file_template = "MONAN_DIAG_R_POS_GFS_{rodada}_{forecast}.00.00.x1.5898242L55.nc"

# Limites geográficos (Longitudes e Latitudes) da "caixa" onde o ciclone é procurado.
lon_min, lon_max = -95, -80 # Longitude (Oeste a Leste)
lat_min, lat_max = -20, 5 # Latitude (Sul a Norte)

# Raio da Terra em quilômetros (usado para o cálculo preciso de distâncias geográficas)
R_earth = 6371.0

# Raios de busca local (em km) que serão usados no rastreamento para refinar a posição do centro.
radii_km = [100, 250, 500, 800, 1500]

# Cria o diretório de saída para salvar os resultados, se ele não existir
out_dir = "tracks_csv"
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# FUNÇÕES AUXILIARES
# -----------------------

# Função para converter longitudes para o intervalo padrão de -180 a 180 graus.
def lon_to_minus180_180(lon_arr):
    lon = np.array(lon_arr)
    # Se a longitude for maior que 180, subtrai 360 para cair no intervalo negativo.
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon

# Função para garantir que as coordenadas de longitude e latitude sejam arrays 2D (grade).
# Cria uma grade se receber arrays 1D.
def ensure_2d_lonlat(lons, lats):
    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    else:
        return lons, lats

# Função que calcula a distância entre dois pontos (lon1, lat1) e (lon2, lat2) 
# usando a fórmula de Haversine, que considera a curvatura da Terra (em km).
def haversine_km(lon1, lat1, lon2, lat2):
    lon1r, lat1r = math.radians(lon1), math.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R_earth * c

# -----------------------
# 1) Extrai centros das análises (usadas como "observadas")
# -----------------------
# Dicionário para armazenar a posição central (lon, lat, pressão) das análises.
# As análises (lead time 0h) são usadas como "trajetória observada" de referência.
obs_centers = {}

print("Extraindo centros das análises (usadas como observadas)...")
for rodada in rodadas:
    # Constrói o caminho completo para o arquivo de análise (onde forecast=rodada)
    analise_path = os.path.join(rodada, file_template.format(rodada=rodada, forecast=rodada))
    
    if not os.path.exists(analise_path):
        print(f"⚠️  Análise ausente: {analise_path}")
        continue
    try:
        ds = Dataset(analise_path, "r")
        lons = lon_to_minus180_180(ds.variables["longitude"][:])
        lats = ds.variables["latitude"][:]
        mslp = ds.variables["mslp"][:]
        
        # Ajusta a dimensão da pressão se necessário (remove a dimensão de tempo se presente)
        if mslp.ndim == 3:
            mslp = mslp[0, :, :]
        # Converte de Pascal (Pa) para HectoPascal (hPa) se os valores forem muito altos
        if np.nanmean(mslp) > 2000:
            mslp /= 100.0
            
        lons2d, lats2d = ensure_2d_lonlat(lons, lats)
        # Cria uma máscara para selecionar apenas os dados dentro da região de busca configurada
        mask = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
        
        # Aplica a máscara: fora da região de busca, o valor é NaN (Not a Number)
        mslp_masked = np.where(mask, mslp, np.nan)
        
        # Encontra o índice do menor valor (mínimo de MSLP = centro do ciclone)
        if np.all(np.isnan(mslp_masked)):
            # Se toda a área mascarada for NaN (ciclone fora da caixa), usa o mínimo global
            idx = np.nanargmin(mslp)
        else:
            # Caso contrário, usa o mínimo dentro da caixa de busca
            idx = np.nanargmin(mslp_masked)
            
        # Converte o índice 1D (flat) para coordenadas 2D (linha/coluna)
        iy, ix = np.unravel_index(idx, mslp.shape)
        
        # Armazena a posição (lon/lat) e a pressão central (p_c)
        lon_c, lat_c, p_c = float(lons2d[iy, ix]), float(lats2d[iy, ix]), float(mslp[iy, ix])
        obs_centers[rodada] = (lon_c, lat_c, p_c)
        print(f"  {rodada}: lon={lon_c:.2f}, lat={lat_c:.2f}, mslp={p_c:.2f}")
        ds.close()
    except Exception as e:
        print(f"❌ Erro lendo análise {analise_path}: {e}")
        continue

# Salva os centros das análises em um arquivo CSV para referência
obs_csv = os.path.join(out_dir, "track_obs_all.csv")
with open(obs_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "lon_obs", "lat_obs", "mslp_hPa_obs"])
    for k in sorted(obs_centers.keys()):
        lon_o, lat_o, p_o = obs_centers[k]
        w.writerow([k, f"{lon_o:.4f}", f"{lat_o:.4f}", f"{p_o:.2f}"])
print(f"✅ Centros observados salvos em: {obs_csv}")

# -----------------------
# 2) Loop das rodadas: detectar centros previstos e calcular erro
# -----------------------
# Lista para acumular todos os resultados de erro de todas as rodadas
summary_rows = []

for rodada in rodadas:
    print("\n===================================")
    print(f"Rodada {rodada}")
    rodada_dt = datetime.strptime(rodada, "%Y%m%d%H")
    csvname = os.path.join(out_dir, f"track_{rodada}.csv")

    # Variáveis para rastrear a última posição encontrada (usadas para a busca local)
    prev_lon, prev_lat = None, None
    found_first = False # Flag para saber se o primeiro ponto da track foi encontrado

    with open(csvname, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        # Cabeçalho do arquivo CSV da trajetória
        w.writerow(["init", "lead_h", "forecast_time",
                    "lon_fc", "lat_fc", "mslp_fc",
                    "lon_obs", "lat_obs", "mslp_obs", "erro_km"])

        # Loop sobre os tempos de previsão (lead times)
        for fh in forecast_steps_h:
            fc_dt = rodada_dt + timedelta(hours=fh) # Calcula o tempo de previsão real
            fc_str = fc_dt.strftime("%Y%m%d%H")
            fpath = os.path.join(rodada, file_template.format(rodada=rodada, forecast=fc_str))
            
            if not os.path.exists(fpath):
                continue
            
            try:
                ds = Dataset(fpath, "r")
                lons = lon_to_minus180_180(ds.variables["longitude"][:])
                lats = ds.variables["latitude"][:]
                mslp = ds.variables["mslp"][:]
                
                # Normalização e conversão da pressão
                if mslp.ndim == 3:
                    mslp = mslp[0, :, :]
                if np.nanmean(mslp) > 2000:
                    mslp /= 100.0
                    
                lons2d, lats2d = ensure_2d_lonlat(lons, lats)
                # "Achata" os arrays para facilitar a busca de índice
                lons_flat, lats_flat, mslp_flat = lons2d.ravel(), lats2d.ravel(), mslp.ravel()

                if not found_first:
                    # Rastreamento: Passo 1 (Busca do primeiro ponto da track)
                    
                    # Máscara do domínio global (para a busca inicial)
                    mask = (lons2d >= lon_min) & (lons2d <= lon_max) & (lats2d >= lat_min) & (lats2d <= lat_max)
                    mask_flat = mask.ravel()
                    
                    if np.any(mask_flat):
                        # Se houver pontos válidos, busca o mínimo de MSLP dentro da caixa
                        vals = np.where(mask_flat, mslp_flat, np.nan)
                        idx = np.nanargmin(vals)
                    else:
                        # Se não houver, busca o mínimo em todo o domínio (fallback)
                        idx = np.nanargmin(mslp_flat)
                        
                    # Armazena a primeira posição e inicializa a posição anterior
                    lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                    found_first, prev_lon, prev_lat = True, lon_c, lat_c
                else:
                    # Rastreamento: Passo 2 (Busca local)
                    
                    # Calcula a distância de cada ponto da grade em relação à posição anterior
                    dists = haversine_km(prev_lon, prev_lat, lons_flat, lats_flat)
                    found = False
                    
                    # Tenta encontrar o mínimo dentro de raios circulares sucessivamente maiores
                    for rkm in radii_km:
                        # Máscara circular (pontos dentro do raio atual)
                        mask_r = (dists <= rkm)
                        
                        if np.any(mask_r):
                            # Pega os valores de MSLP apenas dentro do raio atual
                            vals = np.where(mask_r, mslp_flat, np.nan)
                            idx = np.nanargmin(vals)
                            
                            # Atualiza a posição encontrada e a posição anterior para o próximo passo
                            lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                            prev_lon, prev_lat = lon_c, lat_c
                            found = True
                            break # Ponto encontrado, sai do loop de raios
                            
                    if not found:
                        # Se a busca local em todos os raios falhar (o que é raro para MSLP),
                        # volta a buscar o mínimo em todo o domínio global (fallback)
                        idx = np.nanargmin(mslp_flat)
                        lon_c, lat_c, p_c = float(lons_flat[idx]), float(lats_flat[idx]), float(mslp_flat[idx])
                        prev_lon, prev_lat = lon_c, lat_c

                # --- Cálculo de Erro e Saída ---
                
                # Pega a posição "observada" (análise) correspondente ao tempo de previsão (fc_str)
                # O fatiamento "fc_str[:10]" garante que apenas a data/hora da análise seja usada (ex: 2023030700)
                lon_obs, lat_obs, p_obs = obs_centers.get(fc_str[:10], (np.nan, np.nan, np.nan))
                erro_km = np.nan
                
                # Calcula o erro de posição (distância Haversine) se o ponto observado for válido
                if not np.isnan(lon_obs) and not np.isnan(lat_obs):
                    erro_km = haversine_km(lon_c, lat_c, lon_obs, lat_obs)

                # Escreve a linha de resultados no arquivo CSV da rodada
                w.writerow([rodada, fh, fc_str,
                            f"{lon_c:.4f}", f"{lat_c:.4f}", f"{p_c:.2f}",
                            f"{lon_obs:.4f}" if not np.isnan(lon_obs) else "",
                            f"{lat_obs:.4f}" if not np.isnan(lat_obs) else "",
                            f"{p_obs:.2f}" if not np.isnan(p_obs) else "",
                            f"{erro_km:.2f}" if not np.isnan(erro_km) else ""])
                
                # Armazena os dados para o resumo e plot final
                summary_rows.append([rodada, fh, fc_str, lon_c, lat_c, p_c, lon_obs, lat_obs, p_obs, erro_km])
                ds.close()
            except Exception as e:
                print(f"Erro em {fpath}: {e}")
                continue
    print(f"✅ Track salvo: {csvname}")

# -----------------------
# 3) Salva resumo e plota
# -----------------------
# Salva o resumo consolidado de todas as rodadas
summary_csv = "track_error_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    # Cabeçalho do arquivo de resumo
    w.writerow(["init", "lead_h", "forecast_time",
                "lon_fc", "lat_fc", "mslp_fc",
                "lon_obs", "lat_obs", "mslp_obs", "erro_km"])
    for row in summary_rows:
        w.writerow(row)
print(f"✅ Resumo salvo: {summary_csv}")

# -----------------------
# PLOT 1: TRAJETÓRIAS
# -----------------------
# Cria a figura e o eixo do mapa com a projeção Plate Carree (Lat/Lon simples)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
# Define os limites do mapa (atenção: estes limites são menores que os de busca)
ax.set_extent([-88, -80, -11, -5])
# Adiciona elementos geográficos (necessário o cartopy)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.STATES, linewidth=0.3)
# Adiciona linhas de grade e etiquetas
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
gl.top_labels, gl.right_labels = False, False

# Desenha a trajetória "observada" (análises)
if obs_centers:
    # Plota os marcadores das análises (estrelas pretas) e o rótulo da data
    for key, (lon_o, lat_o, _) in obs_centers.items():
        ax.plot(lon_o, lat_o, 'k*', markersize=8)
        ax.text(lon_o + 0.01, lat_o + 0.01, f"{key[6:8]}/{key[8:10]}Z", fontsize=10, color='k')
    # Plota a linha tracejada das análises
    obs_lons = [v[0] for v in obs_centers.values()]
    obs_lats = [v[1] for v in obs_centers.values()]
    ax.plot(obs_lons, obs_lats, 'k--', label='Análise (observada)')

# Cores para diferenciar cada rodada de previsão
colors = ['b', 'g', 'r', 'm', 'y', 'c']

# Loop para desenhar cada trajetória prevista
for i, rodada in enumerate(rodadas):
    csvfile = os.path.join(out_dir, f"track_{rodada}.csv")
    if not os.path.exists(csvfile):
        continue
    # Carrega os dados da trajetória
    data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    if data.size == 0:
        continue
    lons, lats, times = data['lon_fc'], data['lat_fc'], data['forecast_time']
    # Plota a linha e os marcadores da previsão
    ax.plot(lons, lats, '-o', color=colors[i % len(colors)], label=f"{rodada}")
    # Adiciona as etiquetas de tempo de previsão (data/hora)
    for lonv, latv, tstr in zip(lons, lats, times):
        if np.isnan(lonv) or np.isnan(latv):
            continue
        tstr = str(tstr)
        label = f"{tstr[6:8]}/{tstr[8:10]}Z" if len(tstr) >= 10 else f"{tstr}Z"
        ax.text(lonv + 0.01, latv + 0.01, label, fontsize=10, color=colors[i % len(colors)])

ax.legend(fontsize=10, loc='upper left')
ax.set_title("Trajetórias previstas vs observada", fontsize=11)
plt.savefig("tracks_vs_obs.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva: tracks_vs_obs.png")

# -----------------------
# PLOT 2: ERRO DE POSIÇÃO POR RODADA + MÉDIA
# -----------------------
# Carrega os dados do arquivo de resumo (incluindo o erro em km)
summary = np.genfromtxt(summary_csv, delimiter=',', names=True, dtype=None, encoding=None)

plt.figure(figsize=(8, 5))

colors = ['b', 'g', 'r', 'm', 'y', 'c']
erro_agregado = {} # Dicionário para somar os erros por lead time para o cálculo da média

for i, rodada in enumerate(rodadas):
    # Seleciona os dados apenas para a rodada atual
    mask = np.array([str(x) == rodada for x in summary['init']])
    if np.sum(mask) == 0:
        continue

    leads = summary['lead_h'][mask]
    erros = summary['erro_km'][mask]

    # Converte a string de erro para float (lidando com strings vazias como NaN)
    erros = np.array([float(e) if e != '' else np.nan for e in erros])
    # Filtra os dados de erro que são válidos (não NaN)
    leads_valid, erros_valid = leads[~np.isnan(erros)], erros[~np.isnan(erros)]
    if len(leads_valid) == 0:
        continue

    # Plota o erro de posição para a rodada atual
    plt.plot(leads_valid, erros_valid, '-o', color=colors[i % len(colors)], label=rodada)

    # Acumula os erros para o cálculo da média por lead time
    for l, e in zip(leads_valid, erros_valid):
        if l not in erro_agregado:
            erro_agregado[l] = []
        erro_agregado[l].append(e)

# Média agregada: calcula a média dos erros para cada lead time
lead_vals = sorted(erro_agregado.keys())
erro_medio = [np.nanmean(erro_agregado[l]) for l in lead_vals]
# Plota a linha da média (tracejada preta)
plt.plot(lead_vals, erro_medio, 'k--', linewidth=2, label='Média')

# Configurações finais do gráfico de erro
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Tempo de previsão (h)")
plt.ylabel("Erro de posição (km)")
plt.title("Erro de posição x Tempo de previsão (por rodada e média)")
plt.legend(fontsize=8)
plt.savefig("erro_posicao_vs_lead.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figura salva: erro_posicao_vs_lead.png")

print("Processamento finalizado com sucesso.")
