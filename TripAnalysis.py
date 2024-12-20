# %% [markdown]
# # Uber NYC TLC data project | Exploratory Data Analysis
#

# %% [markdown]
# ## Description
# Author: Shuheng Mo
# Contact: shuheng_mo@outlook.com, shuheng_mo_mail@163.com
# Notifications:

# %% [markdown]
# ## Business Understanding
# In Newyork City, all taxi vehicles are managed by TLC (Taxi and Limousine Commission), here is a brief description about TLC:
# ```
# The New York City Taxi and Limousine Commission (TLC), created in 1971, is the agency
# responsible for licensing and regulating New York City's Medallion (Yellow) taxi cabs, for-hire
# vehicles (community-based liveries, black cars and luxury limousines), commuter vans, and
# paratransit vehicles. The Commission's Board consists of nine members, eight of whom are
# unsalaried Commissioners. The salaried Chair/ Commissioner presides over regularly
# scheduled public commission meetings and is the head of the agency, which maintains a
# staff of approximately 600 TLC employees.
# Over 200,000 TLC licensees complete approximately 1,000,000 trips each day. To operate for
# hire, drivers must first undergo a background check, have a safe driving record, and
# complete 24 hours of driver training. TLC-licensed vehicles are inspected for safety and
# emissions at TLC's Woodside Inspection Facility.
# ```
# Now NYC TLC has released its Trip Record data to public for research and study purposes. There are three main taxi types in NYC:
# * Yellow taxis are traditionally hailed by signaling to a driver who is on duty and seeking a
# passenger (street hail), but now they may also be hailed using an e-hail app like Curb or Arro.
# Yellow taxis are the only vehicles permitted to respond to a street hail from a passenger in all
# five boroughs.
# * Green taxis, also known as boro taxis and street-hail liveries, were introduced in August of
# 2013 to improve taxi service and availability in the boroughs. Green taxis may respond to
# street hails, but only in the areas indicated in green on the map (i.e. above W 110 St/E 96th St
# in Manhattan and in the boroughs).
# * FHV data includes trip data from high-volume for-hire vehicle bases (bases for companies
# dispatching 10,000+ trip per day, meaning Uber, Lyft, Via, and Juno), community livery bases,
# luxury limousine bases, and black car bases.
#
# Uber as one of the biggest ride-hailing services providers, its trip records are collected in `High Volume For-Hire Vehicle Trip
# Records` as well. In this project, there are three business goals we want to achieve to improve Uber's ride-hailing service:
# 1. Exploratory data analysis, research data `fhvhv_tripdata_2021-01` and figure out underlying trip patterns in Jan,2021.
# 2. Based on `fhvhv_tripdata_2021-01` and weather data, build predict model to predict the peak footfall.
# 3. Try explore Uber's user portrait and study the user's wishes when the price was increased. (which orders are urgent and what kind of users should be given higher priorities?)

# %% [markdown]
# ## Data requirements
# Because of the privacy policy, some user data has been encrypted or masked.

# %% [markdown]
# ## Data collection

# %% [markdown]
# TLC provided the data and downloaded as `NYC TLC.zip`, the file structure of decompressed folder `NYC TLC`:
# ```
# .
# ├── data_dictionary_trip_records_hvfhs.pdf
# ├── fhvhv_tripdata_2021-01.parquet
# ├── fhvhv_tripdata_2021-02.parquet
# ├── fhvhv_tripdata_2021-03.parquet
# ├── fhvhv_tripdata_2021-04.parquet
# ├── fhvhv_tripdata_2021-05.parquet
# ├── fhvhv_tripdata_2021-06.parquet
# ├── fhvhv_tripdata_2021-07.parquet
# ├── fhvhv_tripdata_2021-08.parquet
# ├── fhvhv_tripdata_2021-09.parquet
# ├── fhvhv_tripdata_2021-10.parquet
# ├── fhvhv_tripdata_2021-11.parquet
# ├── fhvhv_tripdata_2021-12.parquet
# ├── nyc 2021-01-01 to 2021-12-31.csv
# ├── taxi+_zone_lookup.csv
# ├── taxi_zones
# │   ├── taxi_zones.dbf
# │   ├── taxi_zones.prj
# │   ├── taxi_zones.sbn
# │   ├── taxi_zones.sbx
# │   ├── taxi_zones.shp
# │   ├── taxi_zones.shp.xml
# │   └── taxi_zones.shx
# ├── taxi_zones.zip
# └── working_parquet_format.pdf
# ```
# which `nyc 2021-01-01 to 2021-12-31.csv` record the weather data of year 2021,`taxi+_zone_lookup.csv` stored the zone information of all taxi, data file end with `.parquet` will be processed with pyarrow.

# %% [markdown]
# ## Data understanding
# The data dictionary of Trip Record data please refer to official site https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
# or just read the summary in `data_dictionary_trip_records_hvfhs.pdf`. Datasets that will be used are explained:
#
# Dataset 1: fhvhv_tripdata_2021-01.parquet, 11908468 rows, 24 columns
# - Features: (see 'data_dictionary_trip_records_hvfhs.pdf')
# Multiple dataset end with `.parquet` could be used according to requirements.
#
# Dataset 2: nyc 2021-01-01 to 2021-12-31.csv, 365 rows, 21 columns
# - Features:
#     - 'name':
#     - 'address':
#     - 'resolvedAddress':
#     - 'datetime':
#     - 'temp':
#     - 'feelslike':
#     - 'dew':
#     - 'humidity':
#     - 'precip':
#     - 'precipprob':
#     - 'preciptype':
#     - 'snow':
#     - 'snowdepth':
#     - 'windgust':
#     - 'windspeed':
#     - 'winddir':
#     - 'sealevelpressure':
#     - 'cloudcover':
#     - 'visibility':
#     - 'uvindex':
#     - 'severerisk':
#
# Dataset 3: taxi+_zone_lookup.csv, 265 rows, 4 columns
# - Features:
#     - LocationID:
#     - Borough:
#     - Zone:
#     - service_zone:

# %% [markdown]
# ## Pre-requisites & Auxiliary Functions

# %%
# ! pip install -r requirements.txt
## imports
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import folium
import os
from datetime import time

dataset_dir = os.path.join(os.environ["BIG_DATA"], "input")

# auxiliary functions
## data processing methods
def check_missing_val(data):
    print("Checking attributes that have missing values...")
    for col in data.columns:
        if data[col].isnull().sum() != 0:
            print("{} : {}".format(col, data[col].isnull().sum()))


def check_uniqueness(data):
    print("Checking uniqueness of each attributes ...")
    for col in data.columns:
        print("{} : {} -> {}".format(col, data[col].is_unique, len(data[col].unique())))


def get_quantile(data, col_name, q_val, inter):
    return np.percentile(data[col_name], q_val, method=inter)


def iqr_test(data, col_name, inter):
    qu = get_quantile(data, 'quantity', 75, inter)
    ql = get_quantile(data, 'quantity', 25, inter)
    diff = qu - ql
    U = qu + 1.5 * diff
    L = ql - 1.5 * diff
    return U, L


def remove_outliers(data, col_name, inter):
    """filter the outliers of the numerical data

    Args:
        data (_type_): _description_
        col_name (_type_): _description_
        inter (_type_): _description_

    Returns:
        _type_: _description_
    """
    U, L = iqr_test(data, col_name, inter)
    return data[data[col_name] >= L | data[col_name] <= U]


def get_kmeans_categories(centroid, centers):
    """Returns the correct class when centroid is given

    Args:
        centroid (float): centroid assigned to given order
        centers (float): all centroids given by KNN

    Returns:
        str: specified distance range
    """
    if centroid == centers[0]:
        return 'short_range'
    elif centroid == centers[1]:
        return 'mid_range'
    else:
        return 'long_range'


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %% [markdown]
# ## Data Wrangling

# %% [markdown]
# ### Data Loading and pre-processing

# %%
fhvhv_tripdata_2021_01 = pq.read_table(
    f'{dataset_dir}\\fhvhv_tripdata_2021-01.parquet')
fhvhv_tripdata_2021_01 = fhvhv_tripdata_2021_01.to_pandas()

# %%
fhvhv_tripdata_2021_01.info()

# %%
# we see those flags are boolean values, can be represented as 0 and 1
fhvhv_tripdata_2021_01['shared_request_flag'] = fhvhv_tripdata_2021_01['shared_request_flag'].apply(
    lambda x: 1 if x == 'Y' else 0)
fhvhv_tripdata_2021_01['shared_match_flag'] = fhvhv_tripdata_2021_01['shared_match_flag'].apply(
    lambda x: 1 if x == 'Y' else 0)
fhvhv_tripdata_2021_01['access_a_ride_flag'] = fhvhv_tripdata_2021_01['access_a_ride_flag'].apply(
    lambda x: 1 if x == 'Y' else 0)
fhvhv_tripdata_2021_01['wav_request_flag'] = fhvhv_tripdata_2021_01['wav_request_flag'].apply(
    lambda x: 1 if x == 'Y' else 0)
fhvhv_tripdata_2021_01['wav_match_flag'] = fhvhv_tripdata_2021_01['wav_match_flag'].apply(
    lambda x: 1 if x == 'Y' else 0)

# %%
nyc_weather_2021 = pd.read_csv(
    f'{dataset_dir}\\nyc 2021-01-01 to 2021-12-31.csv')
nyc_weather_2021.info()

# %%
taxi_zone_lookup = pd.read_csv(f'{dataset_dir}\\taxi_zone_lookup.csv')
taxi_zone_lookup.info()

# %% [markdown]
# #### Missing values

# %%
check_missing_val(fhvhv_tripdata_2021_01)

# %%
check_missing_val(nyc_weather_2021)

# %%
check_missing_val(taxi_zone_lookup)

# %%
for idx in taxi_zone_lookup.Zone.value_counts().index.to_list():
    if idx.find('Airport') != -1:
        print(idx)

# %%
taxi_zone_lookup[taxi_zone_lookup.Zone.isin(
    ['Newark Airport', 'LaGuardia Airport', 'JFK Airport'])]  # airport_fee has sth to do with location ID 1,132,138

# %%
airport_ids = [1, 132, 138]

# %%
print(len(fhvhv_tripdata_2021_01[
              fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & fhvhv_tripdata_2021_01.DOLocationID.isin(
                  airport_ids)]))
print(len(fhvhv_tripdata_2021_01[
              fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & ~fhvhv_tripdata_2021_01.DOLocationID.isin(
                  airport_ids)]))
print(len(fhvhv_tripdata_2021_01[
              ~fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & fhvhv_tripdata_2021_01.DOLocationID.isin(
                  airport_ids)]))

# %%
fhvhv_tripdata_2021_01.loc[
    fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & fhvhv_tripdata_2021_01.DOLocationID.isin(
        airport_ids), 'airport_fee'] = 6852 * [5]
fhvhv_tripdata_2021_01.loc[
    fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & ~fhvhv_tripdata_2021_01.DOLocationID.isin(
        airport_ids), 'airport_fee'] = 168695 * [2.5]
fhvhv_tripdata_2021_01.loc[
    ~fhvhv_tripdata_2021_01.PULocationID.isin(airport_ids) & fhvhv_tripdata_2021_01.DOLocationID.isin(
        airport_ids), 'airport_fee'] = 190456 * [2.5]

# %%
fhvhv_tripdata_2021_01['airport_fee'].fillna(value=0, inplace=True)
fhvhv_tripdata_2021_01.dropna(inplace=True)  # drop the other missing values

# %%
nyc_weather_2021['windgust'].fillna(-1, inplace=True)
windgust_data = nyc_weather_2021['windgust'].to_list()

# %%
# nyc_weather_2021['windgust'].fillna(method='ffill',inplace=True)
for idx, num in enumerate(windgust_data):
    if num == -1:
        if idx == 0:
            windgust_data[idx] = windgust_data[idx + 1]
        elif idx == len(windgust_data) - 1:
            windgust_data[idx] = windgust_data[idx - 1]
        else:
            if windgust_data[idx - 1] != -1 and windgust_data[idx + 1] != -1:
                windgust_data[idx] = (windgust_data[idx - 1] + windgust_data[idx + 1]) / 2
            else:
                windgust_data[idx] = windgust_data[idx - 1] if windgust_data[idx - 1] != -1 else windgust_data[idx + 1]

nyc_weather_2021['windgust'] = windgust_data

# %%
nyc_weather_2021['preciptype'].fillna(value='rain', inplace=True)  # fill with mode, which is 'rain'

# %%
nyc_weather_2021.drop(columns=['severerisk'], inplace=True)  # drop the useless column

# %%
taxi_zone_lookup[taxi_zone_lookup.Zone.isnull()]

# %%
taxi_zones = gpd.read_file(f'{dataset_dir}\\taxi_zones\\taxi_zones.shp')
taxi_zones.info()

# %%
taxi_zones[taxi_zones.LocationID == 265]  # no data retrieved, seems we have to drop the empty data

# %%
taxi_zone_lookup.dropna(inplace=True)
taxi_zones.dropna(inplace=True)

# %% [markdown]
# ##### Duplicated values

# %%
fhvhv_tripdata_2021_01.drop_duplicates(inplace=True)
nyc_weather_2021.drop_duplicates(inplace=True)
taxi_zone_lookup.drop_duplicates(inplace=True)
taxi_zones.drop_duplicates(inplace=True)

# %% [markdown]
# #### Outliers

# %%
# do we still need to find and filter outliers for this project? suppose all the data collected well
trip_data_outliers = fhvhv_tripdata_2021_01[['trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 'bcf',
                                             'sales_tax', 'congestion_surcharge', 'airport_fee', 'tips',
                                             'driver_pay']].boxplot(rot=90)

trip_data_outliers
# Image.open("img/trip_data_outliers.png")

# %%
# trip_data_outliers.figure.savefig("img/trip_data_outliers.png",bbox_inches="tight")

# %% [markdown]
# No abnormal extreme values detected except for `trip_time`. However, the trip can take very long time in reality. No operation applied here.

# %%
# weather_attrs = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'snow', 'snowdepth', 'windgust',
#                  'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility']
# weather_data_outliers = nyc_weather_2021[weather_attrs].boxplot(rot=60)
# weather_data_outliers
# Image.open("img/weather_data_outliers.png")

# %%
# weather_data_outliers.figure.savefig("img/weather_data_outliers.png",bbox_inches="tight")

# %% [markdown]
# No abnormal extreme values detected except for `precip`. Also, we will `windgust` with our prediction and there could be further improvement.

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# The first thing we want to investigate is, where do the most trip occured? Looking at the pickup and drop-off locations of the taxi trips, we can see that
# the distribution of those locations are quiet similar, so is there really a region where most of the trip occured in NYC? What are they? When do the customers go there?

# %%
trip_locations = fhvhv_tripdata_2021_01[['PULocationID', 'DOLocationID']].boxplot()
# trip_locations.figure.savefig("img/trip_loc_distribution.png",bbox_inches="tight")
trip_locations
# Image.open("img/trip_loc_distribution.png")

# %%
## warning: takes about 20 secs
# Filter the data according to our need
# All pickup (originate) locations in Jan,2021
# All drop-off (destination) locations in Jan,2021
# now seperate the data based on day (6:00-18:00) and night (18:00-6:00)
day_start = time(6, 0, 0)
day_end = time(18, 0, 0)
# locations in day time
tripdata_2021_01_src_day = fhvhv_tripdata_2021_01.loc[
    (fhvhv_tripdata_2021_01['pickup_datetime'].dt.time >= day_start) & (
                fhvhv_tripdata_2021_01['pickup_datetime'].dt.time <= day_end)]
tripdata_2021_01_dst_day = fhvhv_tripdata_2021_01.loc[
    (fhvhv_tripdata_2021_01['dropoff_datetime'].dt.time >= day_start) & (
                fhvhv_tripdata_2021_01['dropoff_datetime'].dt.time <= day_end)]
# locations in night
tripdata_2021_01_src_night = fhvhv_tripdata_2021_01.loc[
    (fhvhv_tripdata_2021_01['pickup_datetime'].dt.time < day_start) | (
                fhvhv_tripdata_2021_01['pickup_datetime'].dt.time > day_end)]
tripdata_2021_01_dst_night = fhvhv_tripdata_2021_01.loc[
    (fhvhv_tripdata_2021_01['dropoff_datetime'].dt.time < day_start) | (
                fhvhv_tripdata_2021_01['dropoff_datetime'].dt.time > day_end)]

# %% [markdown]
# ### Geospatial Analysis

# %%
taxi_zones.plot(figsize=(6, 6))
print(taxi_zones.columns.to_list())
taxi_zones.crs

# %%
taxi_zones = taxi_zones.to_crs(2263)
taxi_zones['centroid'] = taxi_zones.centroid

# %%
taxi_zones = taxi_zones.to_crs(epsg=4326)  # project the centroids' locations to another coordinate system
taxi_zones['centroid'] = taxi_zones['centroid'].to_crs(epsg=4326)
taxi_zones.head()

# %%
m1 = folium.Map(location=[40.70, -73.94], zoom_start=12, tiles="Stamen Toner")

# add centroid marker to each taxi zones
latitudes = []
longitudes = []
# markers_group = folium.FeatureGroup(name='Taxi Zones Centroids')
for _, r in taxi_zones.iterrows():
    lat = r['centroid'].y
    latitudes.append(lat)
    lon = r['centroid'].x
    longitudes.append(lon)
    folium.Marker(location=[lat, lon],
                  popup='LocationID:{}<br>Zone:{}<br>Borough:{}'.format(r['LocationID'], r['zone'], r['borough']),
                  icon=folium.Icon(icon='info-sign')).add_to(m1)

# project geometries on the map to locate precise regions of taxi zones
for _, r in taxi_zones.iterrows():
    sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.0001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'green'})
    folium.Popup(r['zone']).add_to(geo_j)
    geo_j.add_to(m1)

# markers_group.add_to(m)
taxi_zones['c_latitude'] = latitudes
taxi_zones['c_longitude'] = longitudes
# m1.save('maps/taxi_zones.html') # save as html
m1


