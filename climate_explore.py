### Exploring the climate data

## Import libraries
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


def mapformat():
    m = Basemap(projection='robin', lon_0=0, resolution='c')
    # resolution c, l, i, h, f in that order

    m.drawmapboundary(fill_color='white', zorder=-1)
    m.fillcontinents(color='0.8', lake_color='white', zorder=0)

    m.drawcoastlines(color='0.6', linewidth=0.5)
    m.drawcountries(color='0.6', linewidth=0.5)

    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 1], dashes=[1, 1], linewidth=0.25, color='0.5')
    m.drawmeridians(np.arange(0., 360., 60.), labels=[1, 0, 0, 1], dashes=[1, 1], linewidth=0.25, color='0.5')

    return m

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap,shiftgrid
import matplotlib as mpl
from netCDF4 import Dataset


# get climate data.
my_example_nc_file = './CMIP6_data/tasmax_day_GFDL-ESM4_ssp126_r1i1p1f1_gr1_20150101-20341231.nc'  # Your filename

nc = Dataset(my_example_nc_file, mode='r')

tasmax = nc.variables['tasmax']
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

m = mapformat()
array = np.empty((180,360))
array[:] = np.NAN;
x = np.arange(0.5, 360.5, 1.0)
y = np.arange(-90, 91, 1.0)
for i in range(0, lat.size):
    ilat = int(lat[i] + 90.0)
    ilon = int(lon[i] - 0.5)
    array[ilat, ilon] = tasmax[i, ilat, ilon]
    if ilon < 179 or ilon > 181: # HACK: to avoid date-line wraparound problem
        array[ilat,ilon] = tasmax[i]

array, x = shiftgrid(180, array, x)
array_mask = np.ma.masked_invalid(array)
x,y = np.meshgrid(x,y)
x,y = m(x,y)
# rbb = np.loadtxt('cmaps/runoff-brownblue.txt')/255;
# rbb = mpl.colors.ListedColormap(rbb)
m.pcolormesh(x,y,array_mask, rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.solids.set_edgecolor("face")
# cbar.set_ticks([0,100])
plt.title("Max temperaturre")
plt.show()







###### -----------------------------------

# calculate degree days from the base temperatures and list of climate data.
def calc_degree_days(HDD_base_temp, CDD_base_temp, climate):
    ''' Function to calculate the number of heating and cooling degree days for a year
            given the daily maximum and minimum temperatures according the UK Meteorological Office equations
            [REF]
        climate: is a dataframe with three columns: (1) day, (2) maximum daily temp, and (3) minimum daily temp
        HDD_base_temp: is the base temperature for calculating heating degree days
        CDD_base_temp: is the base temperature for calculating cooling degree days

        Method used for degree day calculation is the UK Met Office Approach:
            CIBSE. (2006). Degree-Days - Theory and Application - TM41: 2006. The Chartered Institution of Building Services Engineers.
    '''
    for index, row in climate.iterrows():
        theta_max = row['max_daily_temp']
        theta_min = row['min_daily_temp']

        # calculate heating degree days
        if theta_max <= HDD_base_temp:
            HDD_day_i = HDD_base_temp - 0.5 * (theta_max + theta_min)
        elif theta_min < HDD_base_temp and (
                (theta_max - HDD_base_temp) < (HDD_base_temp - theta_min)) and theta_max > HDD_base_temp:
            HDD_day_i = 0.5 * (HDD_base_temp - theta_min) - 0.25 * (theta_max - HDD_base_temp)
        elif theta_max > HDD_base_temp and (
                (theta_max - HDD_base_temp) > (HDD_base_temp - theta_min)) and theta_min < HDD_base_temp:
            HDD_day_i = 0.25 * (HDD_base_temp - theta_min)
        elif theta_min >= HDD_base_temp:
            HDD_day_i = 0
        else:
            HDD_day_i = 'NA'

        climate.at[index, 'HDD'] = HDD_day_i

        # calculate cooling degree days
        if theta_min >= CDD_base_temp:
            CDD_day_i = 0.5 * (theta_max + theta_min) - CDD_base_temp
        elif theta_max > CDD_base_temp and (
                (theta_max - CDD_base_temp) > (CDD_base_temp - theta_min)) and theta_min < CDD_base_temp:
            CDD_day_i = 0.5 * (theta_max - CDD_base_temp) - 0.25 * (CDD_base_temp - theta_min)
        elif theta_min < CDD_base_temp and (
                (theta_max - CDD_base_temp) < (CDD_base_temp - theta_min)) and theta_max > CDD_base_temp:
            CDD_day_i = 0.25 * (theta_max - CDD_base_temp)
        elif theta_max <= CDD_base_temp:
            CDD_day_i = 0
        else:
            CDD_day_i = 'NA'
        climate.at[index, 'CDD'] = CDD_day_i

    return climate

# dummy dataframe for climate data:
climate = pd.DataFrame({
    'day': [1,2,3,4,5,6,7],
    'min_daily_temp': [10,15,20,25,30,35,40],
    'max_daily_temp': [15,20,25,30,35,40,45]
    # 'day': [1],
    # 'min_daily_temp': [15],
    # 'max_daily_temp': [20]
})

HDD_base_temp = 18
CDD_base_temp = 15

df = calc_degree_days(HDD_base_temp, CDD_base_temp, climate)
