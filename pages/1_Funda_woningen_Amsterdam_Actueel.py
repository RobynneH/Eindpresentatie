#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm import tqdm
from funda_scraper import FundaScraper
import geocoder
import folium
from folium import plugins
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px

# Load the split DataFrames into Python
coords1 = pd.read_csv('coords1.csv')
coords2 = pd.read_csv('coords2.csv')
# Merge them back into a single DataFrame
coords = pd.concat([coords1, coords2], ignore_index=True)

# Filter out rows with year_built = 0
coords_filtered = coords[coords['year_built'] > 0].copy()  # Use .copy() to avoid the warning
# Correct the energy_label value for 'A+'
coords_filtered.loc[:, 'energy_label'] = coords_filtered['energy_label'].replace({'>A+': 'A+'})


def boxplot_listing_price():
    # Create the box plot using Plotly Express
    fig = px.box(coords, x='house_type', y='price', hover_data=['address', 'living_area', 'room', 'bedroom', 'bathroom'],
                 labels={'house_type': 'Woningtype', 'price': 'Prijs (in Euros)'},
                 title='Interactieve boxplot verkoopprijzen woningen in Amsterdam.',
                 width=800, height=600)
    fig.update_traces(marker_color='rgba(7, 37, 190, 0.6)', marker_line_color='rgba(0, 0, 0, 0.6)',
                      hovertemplate='<b>%{y:.0f} Euros</b><br>%{customdata[0]}<br>%{customdata[1]} m²<br>%{customdata[2]} rooms<br>%{customdata[3]} bedrooms<br>%{customdata[4]} bathrooms')
    fig.update_layout(showlegend=False)
    # Display the box plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)

def energie_labels():
    # Energy label palette
    energy_label_palette = {
        'A+': 'rgb(0, 128, 0)',  # Green
        'A': 'rgb(0, 255, 0)',   # Lighter Green
        'B': 'rgb(255, 255, 0)',  # Yellow
        'C': 'rgb(255, 165, 0)',  # Orange
        'D': 'rgb(255, 69, 0)',   # Red-Orange
        'E': 'rgb(255, 0, 0)',    # Red
        'F': 'rgb(139, 0, 0)',    # Dark Red
        'G': 'rgb(139, 0, 0)',    # Dark Red
        'na': 'gray'
    }
    # Create the scatter plot using Plotly Express
    fig = px.scatter(coords_filtered, x='year_built', y='price', color='energy_label',
                     color_discrete_map=energy_label_palette,
                     hover_data=['address', 'living_area', 'room', 'bedroom', 'bathroom'],
                     labels={'year_built': 'Bouwjaar', 'price': 'Prijs (in Euros)'},
                     title='Scatter Plot van vraagprijs per bouwjaar woningen in Amsterdam',
                     width=800, height=600,
                     category_orders={'energy_label': ['A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'na']})
    fig.update_layout(legend_title='Energie Label')
    # Display the scatter plot using Streamlit
    st.plotly_chart(fig)
    
def map_amsterdam():
    # Create a Folium map centered around Amsterdam
    initial_location = [52.3676, 4.9041]
    map_amsterdam = folium.Map(location=initial_location, zoom_start=12)
    # Create a marker cluster group for better performance
    marker_cluster = MarkerCluster().add_to(map_amsterdam)
    # Energy label palette
    energy_label_palette = {
        'A+': 'rgb(0, 128, 0)',
        'A': 'rgb(0, 255, 0)',
        'B': 'rgb(255, 255, 0)',
        'C': 'rgb(255, 165, 0)',
        'D': 'rgb(255, 69, 0)',
        'E': 'rgb(255, 0, 0)',
        'F': 'rgb(139, 0, 0)',
        'G': 'rgb(139, 0, 0)',
        'na': 'gray'
    }
    # Create a FeatureGroup for markers
    marker_group = folium.FeatureGroup(name='Markers').add_to(map_amsterdam)
    # Add markers to the MarkerCluster
    for index, row in coords_filtered.iterrows():
        if not pd.isnull(row['Lon']) and not pd.isnull(row['Lat']):
            energy_label = row['energy_label']
            color = energy_label_palette.get(energy_label, 'gray')  # Default to gray for 'na'
            popup_html = f"""
                <strong>Address:</strong> {row['address']}<br>
                <strong>ZIP Code:</strong> {row['zip']}<br>
                <strong>Energy Label:</strong> {energy_label}<br>
                <strong>Year Built:</strong> {row['year_built']}<br>
                <strong>Price:</strong> {row['price']}<br>
                <strong>Living Area:</strong> {row['living_area']} sqm<br>
                <img src="{row['photo']}" alt="Property Photo" style="width:200px;height:150px;"><br>
            """
            iframe = folium.IFrame(html=popup_html, width=300, height=250)
            popup = folium.Popup(iframe, max_width=300)
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                popup=popup,
                icon=folium.Icon(color='cadetblue', icon_color=color),
            ).add_to(marker_cluster)
    # Display the Folium map using streamlit_folium
    folium_static(map_amsterdam, use_container_width=True)

st.set_page_config(page_title = "Koopwoningen Amsterdam", layout = "wide")
st.title("Koopwoningen Amsterdam")
st.sidebar.header("Koopwoningen Amsterdam")
image = Image.open('fundalogo.jpg')
st.image(image)

# Boxplot
col1, col2 = st.columns(2)
with col1:
    boxplot_listing_price()
with col2:
    st.write("Deze interactieve boxplot visualiseert de verdeling van de huidige verkoopprijzen voor verschillende woningtypen in Amsterdam. De vakken vertegenwoordigen het interkwartielbereik (IQR), met de mediaan aangegeven door de lijn binnen elk vak. De whiskers strekken zich uit tot de minimale en maximale waarden binnen 1,5 keer het IQR. Uitschieters, weergegeven als individuele punten, zijn opmerkelijke datapunten buiten dit bereik.")
    st.write("**Inzichten:**")
    st.write("- **Prijsverschil per Woningtype:** In het figuur is duidelijk te zien dat appartementen gemiddeld gezien goedkoper zijn dan huizen.")
    st.write("Appartementen in Amsterdam kosten gemiddeld 510K en huizen momenteel 760K.")
    st.write("- **Uitschieters boven de whiskers:** Interessant om te zien zijn de woningen buiten de whiskers, die opvallend enkel boven de bovengrens liggen.")

# energie labels
col3, col4 = st.columns(2)
with col3:
    st.write("Deze scatter plot biedt inzicht in de relatie tussen de bouwjaar en verkoopprijzen van vastgoed in Amsterdam. Elk punt vertegenwoordigt een woning en wordt gekleurd op basis van het energielabel. Opvallend is de overvloed aan 'N/A' waarden voor het energielabel, wat aangeeft dat deze informatie bij veel huizen niet beschikbaar is.")
    st.write("**Belangrijke Waarnemingen:**")
    st.write("- **Invloed van het Bouwjaar:** interessant om te zien is de verandering in verkoopprijzen in relatie tot het bouwjaar van de woning.")
    st.write("- **Toename van 'Groene' Huizen na 2000:** Merk op dat er een duidelijke toename is van woningen met een 'groen' energielabel, met name na het jaar 2000.")
with col4:
    energie_labels()
    
# map
col5, col6 = st.columns(2)
with col5:
    map_amsterdam()
with col6:
    st.write("Deze interactieve kaart visualiseert de locaties van vastgoed in Amsterdam, inclusief informatie zoals vraagprijs, energielabel, bouwjaar en meer. De functionaliteiten van de kaart zijn als volgt:")
    st.write("- **Marker Cluster Groepering:** Bij het uitzoomen worden markers geclusterd voor een overzichtelijke weergave van dicht opeengepakte woningen.")
    st.write("- **Individuele Markers:** Bij het inzoomen op een specifieke locatie worden markers individueel weergegeven, waardoor gedetailleerde informatie beschikbaar is.")
    st.write("- **Legenda voor Energielabels:** Een legenda aan de linkerkant geeft de verschillende energielabels weer met bijbehorende kleuren. Hierdoor kan de gebruiker eenvoudig de energielabel van woningen identificeren.")
    st.write("- **Pop-up Informatie:** Door op een marker te klikken, verschijnt gedetailleerde informatie zoals adres, vraagprijs, energielabel, en een foto van het vastgoed.")
    st.write("- **Zoom en Pan Functionaliteit:** De kaart biedt de mogelijkheid om in te zoomen en te pannen, waardoor gebruikers specifieke gebieden van Amsterdam kunnen verkennen.")
    st.write("De kaart is ontworpen om een visuele weergave te bieden van de geografische spreiding van vastgoed en is handig voor het verkennen van woningkenmerken in verschillende delen van de stad.")


# In[ ]:




