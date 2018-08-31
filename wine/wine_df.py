

# Import libaries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import geocoder
import time
import requests
from bs4 import BeautifulSoup
import jellyfish as jf  # https://github.com/jamesturk/jellyfish
import unidecode  # !pip3 install unidecode
import networkx as nx
from networkx.algorithms import bipartite

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText

print(mpl.rcParams['backend'])

# data set
file = 'winegrapes-detailed-regional-2000-and-2010-rev0714.xlsx'
sheet = 'Detailed_regional_2010'
df = pd. read_excel(file, sheet_name=sheet, skiprows=3)
dfc = pd. read_excel(file, sheet_name=sheet, skiprows=2)
# df.head()
# df.info()
# df.describe()

df_cols = df.columns.tolist()[1:]
dfc_cols = dfc.columns.tolist()[1:]
dfc2 = pd.melt(dfc, id_vars=['Country'], value_vars=dfc_cols)
df2 = pd.melt(df, id_vars=['Region of planting'], value_vars=df_cols)
df2['country'] = dfc2['variable']
df2.columns = ['Grape', 'Region', 'Area', 'Country']

# Fill forward country names
M = [x[:7]=='Unnamed' for x in df2['Country']]
df2.loc[M, 'Country'] = np.nan
df2.loc[:, 'Country'] = df2.loc[:, 'Country'].fillna(method='ffill')

# Remove nan values
df2 = df2[~pd.isnull(df2['Area'])]

# Remove country Total
M = [x[-5:]=='Total' for x in df2['Region']]
df2.loc[M, 'Region'] = ''

# Remove 'other, Other'
df2.loc[:, 'Region'] = [re.sub(r'-?.*[Oo]ther.*-?', '', x).strip() for x in df2.loc[:, 'Region']]
# df2.loc[:, 'Region'].unique()

# Create Address
df2['Address'] = ['{}, {}'.format(df2.loc[i, 'Region'], df2.loc[i, 'Country']) for i in df2.index]

df2.to_csv('wine_regions.csv')

# --------------------------------------------->  <-------------------------------------------- #
# Find region GIS, latlon
geolocator = Nominatim(scheme='http')

def get_location(address):
    try:
        location = geolocator.geocode(address, timeout=5)
        return location
    except GeocoderTimedOut as e:
        print("Error: geocode failed on input %s with message %s".format(e.msg))

region_dict2 = {name:get_location(name) for name in df2.loc[:, 'Address'].unique()}

#
location = geolocator.geocode('Netherlands')
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)


# --------------------------------------------->  <-------------------------------------------- #

# import requests
# url = 'https://maps.googleapis.com/maps/api/geocode/json'
# params = {'sensor': 'false', 'address': 'Mountain View, CA'}
# r = requests.get(url, params=params)
# results = r.json()['results']
# location = results[0]['geometry']['location']
# location['lat'], location['lng']

# ---------------------------------------------> Geocoder <-------------------------------------------- #
# !pip3 install geocoder

# 1st attempt
g2 = []
for name in df2.loc[:, 'Address'].unique():
    time.sleep(0.3)  # .n sec timeout
    g2.append(geocoder.google(name))

g22 = []
for g in g2:
    if g.status!='OK':
        time.sleep(0.3)  # .n sec timeout
        g22.append(geocoder.google(g.location))

g222 = []
for g in g22:
    if g.status!='OK':
        time.sleep(0.8)  # .n sec timeout
        g222.append(geocoder.google(g.location))

g2222 = []
for g in g222:
    if g.status!='OK':
        time.sleep(0.8)  # .n sec timeout
        g2222.append(geocoder.google(g.location))

g22222 = []
for g in g2222:
    if g.status!='OK':
        time.sleep(1.5)  # .n sec timeout
        g22222.append(geocoder.google(g.location))

g222222 = []
for g in g22222:
    if g.status=='OVER_QUERY_LIMIT':
        time.sleep(1.5)  # .n sec timeout
        g222222.append(geocoder.google(g.location))

g_d1 = {g.location:g for g in g2}
g_d2 = {g.location:g for g in g22}
g_d3 = {g.location:g for g in g222}
g_d4 = {g.location:g for g in g2222}
g_d5 = {g.location:g for g in g22222}
g_d6 = {g.location:g for g in g222222}
g_dict = {**g_d1, **g_d2, **g_d3, **g_d4, **g_d5, **g_d6}

# Create Bbox geo data set
geo_bbox_dict = {v.location:v.geojson['features'][0]['bbox'] for k, v in g_dict.items() if v.status=='OK'}
df_geo = pd.DataFrame(geo_bbox_dict).T
df_geo.info()
df_geo.to_csv('bbox_wine_regions.csv')

# Bbox to find
geo_bbox_todo = {v.location:None for k, v in g_dict.items() if v.status!='OK'}
for k, v in geo_bbox_todo.items():
    if not v or v.status!='OK':
        geo_bbox_todo[k] = (geocoder.google(k.split()[-1]))
geo_bbox_dict2 = {v.location:v.geojson['features'][0]['bbox'] for k, v in geo_bbox_todo.items() if v.status=='OK'}
df_geo2 = pd.DataFrame(geo_bbox_dict2).T
df_geo = pd.concat([df_geo, df_geo2])
df_geo.info()
df_geo.to_csv('bbox_wine_regions_2.csv')

# --------------------------------------------->  <-------------------------------------------- #

# import os
# # os.environ["GOOGLE_API_KEY"] = "api_key_from_google_cloud_platform"
# os.environ.get('GOOGLE_API_KEY')
# import geocoder
#
# geo = geocoder.google(address, key='API_KEY')
# latlong = geo.latlng
# geocoder.geonames('Mountain View, CA', maxRows=2)

# ---------------------------------------------> Merge GIS boundary box <-------------------------------------------- #

df_reg = pd.read_csv('wine_regions.csv')
df_geo = pd.read_csv('bbox_wine_regions_2.csv')
df_geo.columns = ['Address', 'min_long', 'min_lat', 'max_long', 'max_lat']

df = pd.merge(df_reg, df_geo, how='left', left_on='Address', right_on='Address')
df.head(-1)
df = df.loc[:, ['Grape', 'Region', 'Country', 'Address', 'Area',
       'min_long', 'min_lat', 'max_long', 'max_lat']]
df.to_csv('grape_region_area_bbox.csv')


# ---------------------------------------------> Scrape grape description <-------------------------------------------- #

url_base = 'https://www.jancisrobinson.com'
url_red = 'https://www.jancisrobinson.com/learn/grape-varieties/red/'
url_white = 'https://www.jancisrobinson.com/learn/grape-varieties/white/'

def get_hrefs(url_base, url_, tag, class_):
    response = requests.get(url_)
    if response.status_code == 200:
        pass
    else:
        print("Failure")
    results_page = BeautifulSoup(response.content, 'lxml')
    href_list = results_page.find_all(tag, class_=class_)[0].find_all('a')
    return ['{}{}'.format(url_base, href.get('href')) for href in href_list]

def get_grape_text(url_, tag, class_1, class_2, tag_1, color):
    response = requests.get(url_)
    if response.status_code == 200:
        pass
    else:
        print("Failure")
    results_page = BeautifulSoup(response.content, 'lxml')
    grape = results_page.find_all(tag, class_=class_1)[0].find_all(tag_1)[0].get_text()
    content = results_page.find_all(tag, class_=class_2)[0].get_text()
    return grape, color ,content

hrefs_red = get_hrefs(url_base, url_red, 'ul', 'info-table')
hrefs_white = get_hrefs(url_base, url_white, 'ul', 'info-table')

grape_text_list = []
for href in hrefs_red:
    grape_text_list.append(get_grape_text(href, 'div', 'learn-header', 'row', 'h1', 'red'))

for href in hrefs_white:
    grape_text_list.append(get_grape_text(href, 'div', 'learn-header', 'row', 'h1', 'white'))

df_grapes = pd.DataFrame(grape_text_list)
df_grapes.columns = ['Grape', 'Color', 'Description']

# Remove excessive spaces
df_grapes['Grape'] = [str(x.strip()) for x in df_grapes['Grape']]
# Encode to english, removing special chars
df_grapes['Grape_utf'] = [str(unidecode.unidecode(x).strip()) for x in df_grapes['Grape']]

# Save df_grapes
df_grapes.to_csv('grape_descr.csv')

# --------------------------------------------->  <-------------------------------------------- #

# Import df data
df = pd.read_csv('grape_region_area_bbox.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df_grapes = pd.read_csv('grape_descr.csv')
df_grapes.drop('Unnamed: 0', axis=1, inplace=True)

# Create best matching joint grape name
def get_best_match(lookup, lookin, dist_metric=jf.jaro_distance):
    grapes_join = []
    for g in lookup:
        grapes_join.append(
            sorted([(g2, dist_metric(g, g2)) for g2 in lookin.unique()], key=lambda i: i[1])[0][0])
    return grapes_join

# df_grapes['Grape_join'] = get_best_match(df_grapes['Grape'], df['Grape'], jf.jaro_distance)
# df_grapes['Grape_join'] = get_best_match(df_grapes['Grape'], df['Grape'], jf.levenshtein_distance)  # matching A->B
df['Grape_join'] = get_best_match(df['Grape'], df_grapes['Grape_utf'], jf.levenshtein_distance)         # matching B->A

# ---------------------------------------------> Merge grape description <-------------------------------------------- #

# Merging df's
df_merged = pd.merge(df, df_grapes, how='left', left_on='Grape_join', right_on='Grape_utf')
df_merged.drop(['Grape_join', 'Address', 'Grape_utf'], axis=1, inplace=True)
df_merged.columns = ['Grape', 'Region', 'Country', 'Area', 'min_long',
       'min_lat', 'max_long', 'max_lat', 'Grape_name', 'Color', 'Description']
df_merged.to_csv('grape_region_area_bbox_descr.csv')

# ---------------------------------------------> add Address <-------------------------------------------- #

df = pd.read_csv('grape_region_area_bbox_descr.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Create Address
df['Location'] = df['Region'] + ', ' + df['Country']
M = pd.isnull(df['Location'])
df.loc[M, 'Location'] = df.loc[M, 'Country']

# add coordinates
df['lat_long'] = list(zip(df.loc[:, 'min_long'], df.loc[:, 'min_lat']))

# Remove nan coordinates
M = ~pd.isnull(df.min_lat)
df = df[M]

df.to_csv('grape_region_area_bbox_descr.csv')


# ---------------------------------------------> networkx <-------------------------------------------- #


df = pd.read_csv('grape_region_area_bbox_descr.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# add coordinates
region_nodes = [(df.loc[i, 'Location'], {'Area': df.loc[i, 'Area'],
                                         'Coords': (df.loc[i, 'min_long'], df.loc[i, 'min_lat'])})
                for i in df.index]

B = nx.Graph()
# Add nodes with the node attribute "bipartite"
B.add_nodes_from(df['Grape'], bipartite=0)  # TODO  if df.loc[i, 'Grape']=='Merlot'
B.add_nodes_from(region_nodes, bipartite=1)
# Add edges only between nodes of opposite node sets
edges = list(zip(df['Grape'], df['Location']))
B.add_edges_from(edges)
B.name = 'Bipartite Graph Grapes - Region'
print(nx.info(B))
B.nodes(data=True)
nx.is_bipartite(B)
nx.is_connected(B)

grape_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
region_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}
region_sub_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
G = bipartite.projected_graph(B, region_nodes)
print(nx.info(G))
G.nodes(data=True)

# ---------------------------------------------> Plot on map <-------------------------------------------- #


def plot_map(G):
    # https://github.com/conda-forge/cartopy-feedstock/issues/36
    # downgrade to shapely 1.5.17
    proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw=dict(projection=proj))
    ax.set_global()
    ax.stock_img()
    # ax.add_feature(cfeature.LAND)     # fills land; covers overland edges
    ax.add_feature(cfeature.COASTLINE)  # draws borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    #
    pos = nx.get_node_attributes(G, 'Coords')
    node_color = [G.degree(v) for v in G]
    node_size = [0.001 * nx.get_node_attributes(G, 'Area')[v] for v in G]
    edge_width = 0.0005

    nx.draw_networkx(G, pos, node_size=node_size,
                     node_color=node_color, alpha=0.5, with_labels=False,
                     width=edge_width, edge_color='.1', cmap=plt.cm.Blues,
                     transform=ccrs.Geodetic()) #, ax=ax, zorder=0)

    plt.show()

plot_map(G)


# ---------------------------------------------> 42 mb! <-------------------------------------------- #
# Map
# !pip3 install mplleaflet
import mplleaflet

plt.figure(figsize=(12, 8))
fig, ax = plt.subplots()

pos = nx.get_node_attributes(G, 'Coords')
node_color = [G.degree(v) for v in G]
node_size = [0.001*nx.get_node_attributes(G, 'Area')[v] for v in G]
edge_width = 0.001

nx.draw_networkx(G, pos, node_size=node_size,
                 node_color=node_color, alpha=0.6, with_labels=False,
                 width=edge_width, edge_color='.4', cmap=plt.cm.Blues)

mplleaflet.show(fig=ax.figure)  # for.py
# mplleaflet.display(fig=ax.figure)  # For Jupyter

# ---------------------------------------------> pickle <-------------------------------------------- #
import pickle
G2 = nx.read_gpickle('major_us_cities.dms')
G2 = pickle.load(open('major_us_cities','rb'))  # alternative
print(nx.info(G2))

# pos = nx.get_node_attributes(G, 'gis')
# nx.draw_networkx(G, pos)

# G = nx.from_pandas_edgelist(df[~pd.isnull(df.min_lat)], 'Grape', 'Location')
# G.name = 'Graph Grapes around the globe'
# print(nx.info(G))
#
# # nx.from_pandas_dataframe(df, 0, 'b', ['weight', 'cost'])
# G.edges(data=True)
# G.nodes(data=True)
# pos = nx.get_node_attributes(G, 'gis')
# nx.draw_networkx(G, pos)
# # nx.draw_networkx(G)
#
# # Reverse lat long
# pos = {city:(long, lat) for (city, (lat,long)) in nx.get_node_attributes(G, 'pos').items()}
# nx.draw(G, pos, with_labels=True, node_size=0)

# ---------------------------------------------> jellyfish <-------------------------------------------- #

# String comparison
grape_1 = 'Ma'
grape_2 = 'Mariette'
jf.levenshtein_distance(grape_1, grape_2)
jf.jaro_distance(grape_1, grape_2)
jf.damerau_levenshtein_distance(grape_1, grape_2)

# Phonetic encoding
jf.metaphone(grape_1)
jf.soundex(grape_1)
jf.nysiis(grape_1)
jf.match_rating_codex(grape_1)
jf.match_rating_codex(grape_2)


# ---------------------------------------------> Udacity <-------------------------------------------- #

scores = [3.0, 1.0, 0.2]

scores2 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

# --------------------------------------------->  <-------------------------------------------- #
x = 1e9
y = 1000000
for i in range(0, y):
    x += 0.000001
x = x - 1e9
x
