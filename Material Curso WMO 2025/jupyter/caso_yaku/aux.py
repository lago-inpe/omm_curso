#!/usr/bin/env python3

import numpy as np  # Biblioteca essencial para operações numéricas e arrays (matrizes)
import math # Para funções matemáticas (usado no cálculo  da distância Haversine)


# ---------------------------
# FUNÇÃO AUX - ajusta longitudes
# ---------------------------
# Converte longitudes de 0-360 para -180 a 180, padrão usado em mapas.
def lon_to_minus180_180(lon_array):
    """Converte longitudes 0..360 para -180..180 (se necessário)."""
    lon = np.array(lon_array)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    return lon


# Função para garantir que as coordenadas de longitude e latitude sejam arrays 2D (grade).
# Cria uma grade se receber arrays 1D.
def ensure_2d_lonlat(lons, lats):
    # Convert to numpy arrays first
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    
    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    else:
        return lons, lats

# Função que calcula a distância entre dois pontos (lon1, lat1) e (lon2, lat2) 
def haversine_km(lon1, lat1, lon2, lat2):
    # Raio da Terra em quilômetros (usado para o cálculo preciso de distâncias geográficas)
    R_earth = 6371.0
    
    lon1r, lat1r = math.radians(lon1), math.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R_earth * c