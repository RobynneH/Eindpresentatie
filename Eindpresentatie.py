#!/usr/bin/env python
# coding: utf-8

# # Eindpresentatie: Koopwoningen Amsterdam

# Sten den Hartog & Robynne Hughes

# In[1]:


import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funda_scraper import FundaScraper

pd.set_option("display.max_columns", None)


# In[2]:


st.set_page_config(layout = "wide")


# In[3]:


st.title("Einpresentatie: Koopwoningen Amsterdam")


# In[4]:


st.subheader("Sten den Hartog & Robynne Hughes")


# In[5]:


image = Image.open('Funda_logo.png')
st.image(image)


# ## Verkochte Huizen Ams

# ### Data loading and cleaning

# In[6]:


# scraper_ges = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=1, n_pages=500)
# Funda_scraper_ges = scraper_ges.run(raw_data=True, save=True, filepath="AMS_ges.csv")


# In[7]:


# Funda_scraper_hist = pd.read_csv("AMS_ges.csv")
# Funda_scraper_hist = Funda_scraper_hist.replace('na', np.nan)
# Funda_scraper_hist.isna().sum()


# In[8]:


# # check voor dubbele rijen
# Funda_scraper_hist.duplicated().sum()


# In[9]:


# # verwijder kollommen die niet gebruikt worden
# Funda_scraper_hist_col = Funda_scraper_hist.drop(["price", "photo", "ownership", "log_id"], axis = 1)
# Funda_scraper_hist_col.nunique()


# In[10]:


# # Verander de datums zodat de computer ze kan lezen en print begin en einddatum
# replace_values = {"januari":"1","februari":"2","maart":"3","april":"4","mei":"5","juni":"6","juli":"7","augustus":"8",
#                   "september":"9","oktober":"10","november":"11","december":"12", " ":"/"}
# Funda_scraper_hist_date = Funda_scraper_hist_col.replace({'date_list': replace_values, 
#                                                           'date_sold': replace_values}, regex = True)
# Funda_scraper_hist_date["date_list_dt"] = pd.to_datetime(Funda_scraper_hist_date['date_list'], format = "%d/%m/%Y")
# Funda_scraper_hist_date["date_sold_dt"] = pd.to_datetime(Funda_scraper_hist_date['date_sold'], format = "%d/%m/%Y")
# print(Funda_scraper_hist_date["date_sold_dt"].min())
# print(Funda_scraper_hist_date["date_sold_dt"].max())


# In[11]:


# # pak alleen rijen waarvan de prijzen bekend zijn en selecteer alleen de prijs
# FS_hist_date_p = Funda_scraper_hist_date[Funda_scraper_hist_date["price_sold"].str.contains("mnd|aanvraag|inschrijving") 
#                                          == False]
# replace_values_price = {" k.k.":"", " v.o.n.":"", "€ ": ""}
# FS_hist_date_p2 = FS_hist_date_p.replace({'price_sold': replace_values_price,
#                                           'last_ask_price_m2': replace_values_price}, regex = True)
# FS_hist_date_p2["price_sold"] = FS_hist_date_p2["price_sold"].str.replace('.', '').astype(float)
# FS_hist_date_p2["price_sold"].apply(pd.to_numeric)
# FS_hist_date_p2["last_ask_price_m2"] = FS_hist_date_p2["last_ask_price_m2"].str.replace('.', '').astype(float)
# FS_hist_date_p2["last_ask_price_m2"].apply(pd.to_numeric)


# In[ ]:




