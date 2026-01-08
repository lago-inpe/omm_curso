# A simple program to test the remote access to .nc files available from HTTP server
import xarray as xr
 
# Modern NetCDF client libraries (including C, Java, and Python netcdf4 libraries) support remote access to data without requiring specialized data servers like THREDDS or OPeNDAP,
# provided the standard web server supports HTTP Range Requests (getRange header and functinality). These libraries have built-in capability to open files via a standard HTTP URL 
# using HTTP Range Requests, which allows the client to request only specific byte ranges of the file instead of downloading the entire file.

# We can test if a HTTP dataserver supports Range Requests. This can be done using "curl -I  URL". If successfull the HTTP response includes the Accept-Ranges header with any value
# other than none (e.g.: bytes), and the server supports range requests. If responses omit the Accept-Ranges header, it indicates the server doesn't support partial requests.
# e.g.: curl -I https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Erin/2025081000/MONAN_DIAG_R_POS_GFS_2025081000_2025081000.00.00.x1.5898242L55.nc

# Newer versions of the netCDF-C library can read data over HTTP directly using byte-range requests, if this option was enabled when the library was compiled. The latest versions 
# available on conda-forge (a packaging channel for the conda/Anaconda Python ecosystem) now have this enabled, thus you can directly open the url with #mode=bytes 

# HTTP dataserver at CPTEC supports byte ranges requests. However,  the first URL bellow does not work properly, while the second is working fine using the tag: #mode=bytes 
#url = "https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Erin/2025081000/MONAN_DIAG_R_POS_GFS_2025081000_2025081000.00.00.x1.5898242L55.nc"
url = "https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/curso_OMM_INPE_2025/Central_America_Hurricane_Erin/2025081000/MONAN_DIAG_R_POS_GFS_2025081000_2025081000.00.00.x1.5898242L55.nc#mode=bytes"

try:
    # Open the dataset directly from the URL
    ds = xr.open_dataset(url)

    # Now you can work with the dataset (ds)
    print(ds)

    # Access a specific variable, for example, 'temperature'
    #temperature_data = ds['temperature']
    #print(temperature_data)

except Exception as e:
    print(f"Error opening or reading NetCDF file from URL: {e}")
