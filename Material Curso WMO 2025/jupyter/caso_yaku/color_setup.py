
import matplotlib.colors as mcolors
import numpy as np 

def color_setup(type='daily'):
    """
    Configura as escalas de cores para os mapas de precipitação diária e acumulada.
    Retorna os colormaps e normalizações para uso em visualizações.
    Returns:
        cmap_daily (ListedColormap): Colormap para precipitação diária.
        norm_daily (BoundaryNorm): Normalização para precipitação diária.
        cmap_sum (ListedColormap): Colormap para precipitação acumulada.
        norm_sum (BoundaryNorm): Normalização para precipitação acumulada.
        type (str): Tipo de escala ('daily' ou 'sum').
    """
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
    if type == 'daily':
        cmap_daily = mcolors.ListedColormap(colors)
        norm_daily = mcolors.BoundaryNorm(boundaries, len(boundaries) - 1)
        return cmap_daily, norm_daily
    else:
        cmap_sum = mcolors.ListedColormap(colors)
        norm_sum = mcolors.BoundaryNorm(boundaries2, len(boundaries2) - 1)
        return cmap_sum, norm_sum
    
def cores_precip():
# Níveis e cores para precipitação ou viés.
    rain_levels = [0, 1, 5, 10, 20, 30, 40, 50]
    colors = ['#9c0720', '#dc143c', '#f1666d', '#ff9ea2', '#f0c6f0', '#ffffff',
            '#87CEEB', '#00BFFF', '#1E90FF', '#4169E1', '#0000FF']
    cmap2 = mcolors.ListedColormap(colors)  # Paleta de cores para viés.
    cmap2.set_over('#081d58')  # Cor para valores acima do máximo.
    cmap2.set_under('#610000')  # Cor para valores abaixo do mínimo.
    data_min = -60
    data_max = 70
    interval = 10
    levels2 = np.arange(data_min, data_max, interval)  # Intervalos para viés.

    return cmap2, data_min, data_max, rain_levels, levels2

def cores_wind():
    
    # Define paleta de cores para o mapa de vento.
    colors = ["#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"]
    cmap = mcolors.ListedColormap(colors)
    cmap.set_over('#800026')  # Cor para valores acima do máximo.
    cmap.set_under('#ffffff')  # Cor para valores abaixo do mínimo.
    
    return cmap

def cores_pw():

    # Paleta de cores para água precipitável.
    colors = ["#b4f0f0", "#96d2fa", "#78b9fa", "#3c95f5", "#1e6deb", "#1463d2", 
              "#0fa00f", "#28be28", "#50f050", "#72f06e", "#b3faaa", "#fff9aa", 
              "#ffe978", "#ffc13c", "#ffa200", "#ff6200", "#ff3300", "#ff1500", 
              "#c00100", "#a50200", "#870000", "#653b32"]
    cmap = mcolors.ListedColormap(colors)
    cmap.set_over('#000000')
    cmap.set_under('#ffffff')
    
    return cmap

