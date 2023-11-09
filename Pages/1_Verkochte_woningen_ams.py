#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats

from funda_scraper import FundaScraper

import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def plots_prijs():
    # laad alle dataset in en voeg deze samen tot één dataframe
    Funda1 = pd.read_csv("AMS_ges1.csv")
    Funda2 = pd.read_csv("AMS_ges2.csv")
    Funda3 = pd.read_csv("AMS_ges3.csv")
    Funda4 = pd.read_csv("AMS_ges4.csv")
    Funda5 = pd.read_csv("AMS_ges5.csv")
    Funda6 = pd.read_csv("AMS_ges6.csv")
    Funda7 = pd.read_csv("AMS_ges7.csv")
    Funda8 = pd.read_csv("AMS_ges8.csv")
    Funda_scraper_verkocht =  pd.concat([Funda1, Funda2, Funda3,Funda4,Funda5,Funda6,Funda7,Funda8])
    
    # vervang de string 'na' met een echte NonAvailableValue
    Funda_scraper_verkocht = Funda_scraper_verkocht.replace('na', np.nan)
    
    # verwijder kollommen die niet gebruikt worden
    Funda_scraper_verkocht_col = Funda_scraper_verkocht.drop(["price", "photo", "ownership", "log_id"], axis = 1)
    
    # Verander de datums zodat de computer ze kan lezen
    replace_values = {"januari":"1","februari":"2","maart":"3","april":"4","mei":"5","juni":"6","juli":"7","augustus":"8",
                      "september":"9","oktober":"10","november":"11","december":"12", " ":"/"}
    Funda_scraper_verkocht_col = Funda_scraper_verkocht_col.replace({'date_list': replace_values, 
                                                              'date_sold': replace_values}, regex = True)
    Funda_scraper_verkocht_col["date_list_dt"] = pd.to_datetime(Funda_scraper_verkocht_col['date_list'], format = "%d/%m/%Y")
    Funda_scraper_verkocht_col["date_sold_dt"] = pd.to_datetime(Funda_scraper_verkocht_col['date_sold'], format = "%d/%m/%Y")
    
    # pak alleen rijen waarvan de prijzen bekend zijn en selecteer alleen de prijs
    Funda_scraper_verkocht_pr = Funda_scraper_verkocht_col[Funda_scraper_verkocht_col["price_sold"].str.contains("mnd|aanvraag|inschrijving") 
                                             == False]
    replace_values_price = {" k.k.":"", " v.o.n.":"", "€ ": ""}
    Funda_scraper_verkocht_pr = Funda_scraper_verkocht_pr.replace({'price_sold': replace_values_price,
                                              'last_ask_price_m2': replace_values_price}, regex = True)
    Funda_scraper_verkocht_pr["price_sold"] = Funda_scraper_verkocht_pr["price_sold"].str.replace('.', '').astype(float)
    Funda_scraper_verkocht_pr["price_sold"].apply(pd.to_numeric)
    Funda_scraper_verkocht_pr["last_ask_price_m2"] = Funda_scraper_verkocht_pr["last_ask_price_m2"].str.replace('.', '').astype(float)
    Funda_scraper_verkocht_pr["last_ask_price_m2"].apply(pd.to_numeric)
    
    FS_eengezins = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('engezin', na=False)]
    FS_appartement = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('appartement', na=False)]
    FS_villa = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Villa', na=False)]
    FS_penthouse = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Penthouse', na=False)]
    FS_grachtenpand = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Grachtenpand', na=False)]
    FS_herenhuis = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Herenhuis', na=False)]
    FS_flat = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('flat|Flat', na=False)]
    FS_woonboot = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Woonboot', na=False)]
    FS_bungalow = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Bungalow', na=False)]
    FS_bovenwoning = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('oven', na=False)]
    FS_benedenwoning = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('eneden', na=False)]
    FS_maisonnette = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Maisonnette', na=False)]
    FS_portiek = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Portiek', na=False)]
    FS_tussen = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Tussen', na=False)]
    FS_landhuis = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Landhuis', na=False)]
    FS_boerderij = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('boerderij', na=False)]
    FS_bouw = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["kind_of_house"].str.contains('Bouw', na=False)]
    
    # plot1
    fig = go.Figure()
    fig.add_trace(go.Box(y = Funda_scraper_verkocht_pr["price_sold"], name = "Verkochte prijs per woning"))
    fig.add_trace(go.Box(y = Funda_scraper_verkocht_pr["last_ask_price_m2"], name = "Prijs per m2 per woning", visible = False))

    dropdown_buttons = [{'label':'Verkochte prijs per woning', 'method':'update','args':[{'visible':[True, False]},
                                                                             {'yaxis':{'title':'Prijs per huis'}}]},
                       {'label':'Laatst gevraagde prijs per m2', 'method':'update','args':[{'visible':[False, True]},
                                                                             {'yaxis':{'title':'Prijs per m2'}}]}
                       ]
    fig.update_layout({'updatemenus':[{'active':0, 'buttons':dropdown_buttons}]}, 
                       title = "Prijs per verkocht woning of laatst gevraagde prijs per vierkante meter",
                         yaxis_title = 'Prijs per huis')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.write("Deze boxplot laat de verdeling van de prijzen van verkochtte woningen zien. Met de button kan je de boxplot wijzigen naar de verdeling van laatst gevraagde prijs per vierkante meter. Hierbij is bij beide te zien dat de IQR-range niet heel breed is, en dat de prijzen dus best veel van elkaar verschillen. Ook is het duidelijk te zien dat de kwart met de duurste woning prijzen een groot verschil hebben. Hier zijn duidelijk een paar uitwijkers te zien.")

    # plot 2
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x = Funda_scraper_verkocht_pr["date_sold_dt"],
                               y = Funda_scraper_verkocht_pr["price_sold"],
                               mode = "markers"))
    fig.add_trace(go.Scattergl(x = Funda_scraper_verkocht_pr["date_sold_dt"],
                               y = Funda_scraper_verkocht_pr["last_ask_price_m2"],
                               mode = "markers",
                               visible = False))
    dropdown_buttons = [{'label':'Verkochte prijs per woning', 'method':'update','args':[{'visible':[True, False]},
                                                                             {'yaxis':{'title':'Prijs per huis'}}]},
                       {'label':'Laatst gevraagde prijs per m2', 'method':'update','args':[{'visible':[False, True]},
                                                                             {'yaxis':{'title':'Prijs per m2'}}]}
                       ]
    fig.update_layout({'updatemenus':[{'active':0, 'buttons':dropdown_buttons, 'y':1.1, 'x':0.22}]}, 
                       title = "Verkochte prijs per woning of laatst gevraagde prijs per vierkante meter per datum",
                         xaxis_title = 'Datum',
                         yaxis_title = 'Prijs per huis')
    col3, col4 = st.columns(2)
    with col3:
        st.write("In deze scatterplot is per verkochte woning in het afgelopen anderhalf jaar. Er is hier duidelijk te zien dat we voor juli 2022 niet veel data hebben, dit omdat er een limiet zit aan hoe veel de scraper van de Funda site kan afhalen.")
        st.write("Bij de scatterplot waarbij de datum tegenover de laatst gevraagde prijs staat is te zien dat de prijzen veel hoger verdeeld liggen.")
        st.write("Er is bij beide verder te zien dat veel woningen voor ongeveer dezelde prijs verkocht zijn.")
    with col4:
        st.plotly_chart(fig)
        
        
        
    # plot3
    fig = go.Figure()
    fig.add_trace(go.Box(y = FS_eengezins["price_sold"], name = "Verkochte prijs per eengezinswoning"))
    fig.add_trace(go.Box(y = FS_appartement["price_sold"], name = "Verkochte prijs per appartement"))
    fig.add_trace(go.Box(y = FS_villa["price_sold"], name = "Verkochte prijs per villa"))
    fig.add_trace(go.Box(y = FS_penthouse["price_sold"], name = "Verkochte prijs per penthouse"))
    fig.add_trace(go.Box(y = FS_grachtenpand["price_sold"], name = "Verkochte prijs per grachtenpand"))
    fig.add_trace(go.Box(y = FS_herenhuis["price_sold"], name = "Verkochte prijs per herenhuis"))
    fig.add_trace(go.Box(y = FS_flat["price_sold"], name = "Verkochte prijs per flat"))
    fig.add_trace(go.Box(y = FS_woonboot["price_sold"], name = "Verkochte prijs per woontboot"))
    fig.add_trace(go.Box(y = FS_bovenwoning["price_sold"], name = "Verkochte prijs per bovenwoning"))
    fig.add_trace(go.Box(y = FS_benedenwoning["price_sold"], name = "Verkochte prijs per bedenwoning"))
    fig.add_trace(go.Box(y = FS_maisonnette["price_sold"], name = "Verkochte prijs per maissonnette"))
    fig.add_trace(go.Box(y = FS_portiek["price_sold"], name = "Verkochte prijs per protiekhuis"))
    fig.add_trace(go.Box(y = FS_tussen["price_sold"], name = "Verkochte prijs per tussenverdieping"))
    fig.update_layout(title = "Prijs per soort verkochte woning",
                     showlegend = False,
                     xaxis_title = "Verkochte prijs",
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig)
    with col6:
        st.write("In de boxplots hiernaast zijn de prijzen per type woning te zien. Hier is een groot verschil te zien tussen de prijzen van een flat en de prijzen van een grachtenpand of herenhuis. Ook hier is te zien dat zelfs per catagory woning de prijzen groot verschillen. Dit kan zijn door bijvoorbeeld ligging of grootte van de woning.")

st.set_page_config(page_title = "Verkochte woningen Amsterdam", layout = "wide")
st.sidebar.header("Verkochte woningen Amsterdam")
st.title("Verkochte woningen Amsterdam")
st.subheader("Sten den Hartog & Robynne Hughes")

st.divider()
st.header("Prijzen van verkochte woningen")
plots_prijs()

data_frame_demo()

