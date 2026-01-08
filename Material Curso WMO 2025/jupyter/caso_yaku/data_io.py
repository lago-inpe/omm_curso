#!/usr/bin/env python3

import pandas as pd
import numpy as np

#For loading and handling netCDF data
import xarray as xr
import os

# Create cache directory for GRIB index files
cache_dir = os.path.expanduser('~/.cfgrib_cache')
os.makedirs(cache_dir, exist_ok=True)

def load_merge(date):
    """Load GPM MERGE data for a specific date.
    Parameters
    date : str
        Date in 'YYYYMMDD' format."""
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    base_dir = "/oper/share/ioper/tempo/MERGE/GPM/DAILY/"

    file_path = f"{base_dir}/{year}/{month}/MERGE_CPTEC_{date}.grib2"
    
    # Check if an index file already exists in the source directory
    source_idx = file_path + ".idx"
    cache_idx = os.path.join(cache_dir, f"MERGE_CPTEC_{date}.grib2.idx")
    
    # If index exists in source but not in cache, copy it
    if os.path.exists(source_idx) and not os.path.exists(cache_idx):
        print(f"Copying existing index file to cache: {source_idx}")
        import shutil
        shutil.copy2(source_idx, cache_idx)
    elif os.path.exists(cache_idx):
        print(f"Using existing cached index: {cache_idx}")
    else:
        print(f"Will create new index in cache: {cache_idx}")
    
    # Use the cache directory for the index file
    ds = xr.open_dataset(
        file_path, 
        engine='cfgrib',
        backend_kwargs={'indexpath': cache_idx}
    )
    return ds

def check_merge(date):
    """Check existence of MONAN data for a specific date.
    Parameters
    date : str
        Date in 'YYYYMMDD' format."""
    
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    base_dir = "/oper/share/ioper/tempo/MERGE/GPM/DAILY/"

    file_path = f"{base_dir}/{year}/{month}/MERGE_CPTEC_{date}.grib2"

    # Checking
    if os.path.exists(file_path):
        return True
    else:
        return False

def load_monan(date_run, date_prev):
    """Load MONAN data for a specific date.
    Parameters
    date : str
        Date in 'YYYYMMDD' format."""
    
    year = date_run[:4]
    month = date_run[4:6]
    day = date_run[6:8]
    base_dir = "/pesq/share/bam/dist/paulo.kubota/externo/Curso_da_OMM_2025_estudos_de_casos/Galapagos_YAKU/"

    file_path=f"{base_dir}/{date_run}/MONAN_DIAG_R_POS_GFS_{date_run}_{date_prev}.00.00.x1.5898242L55.nc"
#     print('Opening ', file_path)

    # Openning
    ds = xr.open_dataset(
        file_path, 
        engine='netcdf4'
    )
    return ds

def check_monan(date_run, date_prev):
    """Check existence of MONAN data for a specific date.
    Parameters
    date : str
        Date in 'YYYYMMDD' format."""
    
    year = date_run[:4]
    day = date_run[6:8]
    base_dir = "/pesq/share/bam/dist/paulo.kubota/externo/Curso_da_OMM_2025_estudos_de_casos/Galapagos_YAKU/"

    file_path=f"{base_dir}/{date_run}/MONAN_DIAG_R_POS_GFS_{date_run}_{date_prev}.00.00.x1.5898242L55.nc"

    # Checking
    if os.path.exists(file_path):
        return True
    else:
        return False
