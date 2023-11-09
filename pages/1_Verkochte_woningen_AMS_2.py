#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats

from funda_scraper import FundaScraper

import math
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


def plot1():
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
    st.plotly_chart(fig)


st.set_page_config(page_title = "Verkochte woningen Amsterdam", layout = "wide")
st.title("Verkochte woningen Amsterdam")
st.sidebar.header("Verkochte woningen Amsterdam")
st.header("Prijzen van verkochte woningen")

col1, col2 = st.columns(2)

with col1:
    plot1
with col2:
    st.write("Deze boxplot laat de verdeling van de prijzen van verkochtte woningen zien. Met de button kan je de boxplot wijzigen naar de verdeling van laatst gevraagde prijs per vierkante meter. Hierbij is bij beide te zien dat de IQR-range niet heel breed is, en dat de prijzen dus best veel van elkaar verschillen. Ook is het duidelijk te zien dat de kwart met de duurste woning prijzen een groot verschil hebben. Hier zijn duidelijk een paar uitwijkers te zien.")


# In[ ]:




