import os
from netCDF4 import Dataset  # Biblioteca para ler arquivos NetCDF, que contêm dados climáticos como pressão e vento.
import numpy as np  # Para cálculos matemáticos com matrizes de números.
import matplotlib.pyplot as plt  # Para criar gráficos e mapas.
import cartopy.crs as ccrs  # Para projeções de mapas (ex.: mapas geográficos da Terra).
import cartopy.feature as cfeature  # Para adicionar elementos como costas e fronteiras nos mapas.
import matplotlib  # Biblioteca principal de plotagem.

# --- CONFIGURAÇÕES ---
# Definimos as datas das rodadas (início das simulações) e previsões (datas futuras previstas).
rodadas = ["2023030700", "2023030800", "2023030900", "2023031000", "2023031100"]
previsoes = ["2023030800", "2023030900", "2023031000", "2023031100", "2023031200"]

# Limites geográficos do mapa: longitude (oeste a leste) e latitude (sul a norte).
lon_min, lon_max = -110, -70
lat_min, lat_max = -30, 10

# --- FIGURA ---
# Cria uma grade de painéis (subgráficos) para mostrar mapas de várias rodadas e previsões.
fig, axs = plt.subplots(
    len(rodadas), len(previsoes),  # Linhas = número de rodadas, colunas = número de previsões.
    figsize=(16, 16),  # Tamanho da figura em polegadas (grande para caber todos os painéis).
    subplot_kw=dict(projection=ccrs.PlateCarree()),  # Usa projeção de mapa simples.
    gridspec_kw={'hspace':0.05, 'wspace':0.05}  # Espaçamento pequeno entre painéis.
)

# --- DETECTAR PRIMEIRA COLUNA E ÚLTIMA LINHA COM DADO ---
# Essas listas ajudam a identificar quais painéis têm dados válidos, para colocar rótulos corretamente.
primeira_coluna_com_dado = [None]*len(rodadas)
ultima_linha_com_dado = [None]*len(previsoes)

# Verifica quais arquivos existem para marcar onde há dados.
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        if rodada == previsao:  # Ignora casos onde rodada e previsão são a mesma data (sem dados).
            continue
        pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')  # Pasta onde o arquivo está armazenado.
        arquivo = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"  # Nome do arquivo NetCDF.
        caminho = os.path.join(pasta, arquivo)  # Caminho completo do arquivo.
        if os.path.exists(caminho):  # Se o arquivo existe...
            if primeira_coluna_com_dado[i] is None:
                primeira_coluna_com_dado[i] = j  # Marca a primeira coluna com dados para essa rodada.
            ultima_linha_com_dado[j] = i  # Marca a última linha com dados para essa previsão.

# --- LOOP DE PLOTAGEM ---
# Para cada combinação de rodada e previsão, cria um mapa no painel correspondente.
for i, rodada in enumerate(rodadas):
    for j, previsao in enumerate(previsoes):
        ax = axs[i,j]  # Seleciona o painel atual.
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # Define os limites do mapa.

        # Adiciona rótulo com a data da rodada na primeira coluna visível.
        if j == primeira_coluna_com_dado[i]:
            ax.set_visible(True)
            ax.text(-0.3, 0.5, f"Rodada: {rodada}", fontsize=12, va='center', rotation=90, transform=ax.transAxes)

        try:
            if rodada == previsao:  # Se datas iguais, painel fica vazio.
                ax.set_visible(False)
                continue

            pasta = ('/pesq/share/monan/curso_OMM_INPE_2025/Galapagos_YAKU/' + rodada + '/')
            arquivo_atual = f"MONAN_DIAG_R_POS_GFS_{rodada}_{previsao}.00.00.x1.5898242L55.nc"
            caminho_atual = os.path.join(pasta, arquivo_atual)
            if not os.path.exists(caminho_atual):  # Se arquivo não existe, esconde painel.
                print(f"⚠️ Arquivo não encontrado: {caminho_atual}")
                ax.set_visible(False)
                continue

            print(f"🔹 Lendo: {caminho_atual}")  # Informa qual arquivo está sendo lido.
            dado_atual = Dataset(caminho_atual, 'r')  # Abre o arquivo NetCDF.

            # Extrai variáveis: longitudes, latitudes, pressão e componentes do vento.
            lons = dado_atual.variables['longitude'][:]
            lats = dado_atual.variables['latitude'][:]
            pres = dado_atual.variables['mslp'][0,:,:]/100.0  # Pressão ao nível do mar (em hPa).
            u10 = dado_atual.variables['u10'][0,:,:]  # Componente leste-oeste do vento a 10m.
            v10 = dado_atual.variables['v10'][0,:,:]  # Componente norte-sul do vento a 10m.
            ws = np.sqrt(u10**2 + v10**2)  # Calcula a velocidade do vento (magnitude).

            # --- PLOT ---
            # Adiciona elementos geográficos ao mapa: linhas de costa, fronteiras, estados.
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.STATES, linewidth=0.2)

            # Define intervalos para a velocidade do vento (em m/s).
            data_min = 2
            data_max = 10 
            interval = 1
            levels = np.arange(data_min, data_max, interval)

            # Define paleta de cores para o mapa de vento.
            colors = ["#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"]
            cmap = matplotlib.colors.ListedColormap(colors)
            cmap.set_over('#800026')  # Cor para valores acima do máximo.
            cmap.set_under('#ffffff')  # Cor para valores abaixo do mínimo.

            # Plota o mapa preenchido com a velocidade do vento.
            img1 = ax.contourf(lons, lats, ws, cmap=cmap, levels=levels, extend='both')    

            # --- PRESSÃO ---
            # Plota linhas de contorno para pressão (isóbaras).
            levels_pres = np.arange(500, 1050, 2)
            img_pres = ax.contour(lons, lats, pres, colors='black', linewidths=1, levels=levels_pres, transform=ccrs.PlateCarree())
            ax.clabel(img_pres, inline=1, fontsize=10, fmt='%1.0f', colors='black')  # Adiciona valores nas linhas.

            # --- VENTO (BARBELAS) ---
            # Adiciona setas (barbelas) para mostrar direção e força do vento.
            flip = np.zeros((u10.shape[0], u10.shape[1]))
            flip[lats < 0] = 1  # Inverte setas no hemisfério sul para correta orientação.
            img_barbs = ax.barbs(
                lons[::20], lats[::20],  # Espaçamento para evitar excesso de setas.
                u10[::20, ::20], v10[::20, ::20],
                length=5.0,
                sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
                linewidth=0.8,
                pivot='middle',
                barbcolor='gray',
                flip_barb=flip[::20, ::20],
                transform=ccrs.PlateCarree()
            )

            # --- GRIDLINES ---
            # Adiciona linhas de grade com rótulos de latitude e longitude.
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='dimgray', alpha=0.4)
            gl.top_labels = False  # Sem rótulos no topo.
            gl.right_labels = False  # Sem rótulos à direita.
            gl.xlabel_style = {'size':9}  # Tamanho da fonte dos rótulos.
            gl.ylabel_style = {'size':9}

            # Rótulos de latitude só na primeira coluna com dados.
            if j == primeira_coluna_com_dado[i]:
                gl.left_labels = True
            else:
                gl.left_labels = False

            # Rótulos de longitude só na última linha com dados.
            if i != ultima_linha_com_dado[j]:
                gl.bottom_labels = False

            # Adiciona título com a data da previsão na primeira linha.
            if i==0:
                ax.set_title(previsao, fontsize=13)

            dado_atual.close()  # Fecha o arquivo para liberar memória.

        except Exception as e:  # Se houver erro (ex.: arquivo corrompido), esconde painel.
            print(f"❌ Erro rodada {rodada}, previsão {previsao}: {e}")
            ax.set_visible(False)
            continue

# --- COLORBAR E SALVAMENTO ---
# Adiciona uma barra de cores na parte inferior para explicar a escala de velocidade do vento.
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015])
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='horizontal', ticks=levels,
                    label='Velocidade do vento a 10m (m/s)')

# Título geral da figura.
plt.suptitle("Velocidade e direção do vento (10 m) e Pressão ao Nível Médio do Mar (hPa)", fontsize=15, y=0.95)
plt.savefig("previsao_pressao_vento.png", dpi=300, bbox_inches='tight')  # Salva a imagem.
plt.close()  # Fecha a figura para liberar memória.
print("✅ Figura salva como previsao_pressao_vento.png")
