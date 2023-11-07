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

from scipy import stats

from funda_scraper import FundaScraper

import math
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

# ### Data loading and cleaning Funda Scraper

# Laad de dataset in delen in ivm met GitHub max bestand grootte. Het maximum aantal pagina's met de Scraper is 579

# In[6]:


st.divider()
st.header("Verkochte huizen Amsterdam")


# In[7]:


# scraper_ges1 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=1, n_pages=80)
# Funda1 = scraper_ges1.run(raw_data=True, save=True, filepath="AMS_ges1.csv")
# scraper_ges2 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=80, n_pages=80)
# Funda2 = scraper_ges2.run(raw_data=True, save=True, filepath="AMS_ges2.csv")
# scraper_ges3 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=160, n_pages=80)
# Funda3 = scraper_ges3.run(raw_data=True, save=True, filepath="AMS_ges3.csv")
# scraper_ges4 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=240, n_pages=80)
# Funda4 = scraper_ges4.run(raw_data=True, save=True, filepath="AMS_ges4.csv")
# scraper_ges5 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=320, n_pages=80)
# Funda5 = scraper_ges5.run(raw_data=True, save=True, filepath="AMS_ges5.csv")
# scraper_ges6 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=400, n_pages=80)
# Funda6 = scraper_ges6.run(raw_data=True, save=True, filepath="AMS_ges6.csv")
# scraper_ges7 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=480, n_pages=80)
# Funda7 = scraper_ges7.run(raw_data=True, save=True, filepath="AMS_ges7.csv")
# scraper_ges8 = FundaScraper(area="amsterdam", want_to="buy", find_past=True, page_start=560, n_pages=80)
# Funda8 = scraper_ges8.run(raw_data=True, save=True, filepath="AMS_ges8.csv")

# ## last avaiable page is 579


# In[8]:


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
# Funda_scraper_verkocht.isna().sum()


# In[9]:


# check voor dubbele rijen
Funda_scraper_verkocht.duplicated().sum()


# In[10]:


# verwijder kollommen die niet gebruikt worden
Funda_scraper_verkocht_col = Funda_scraper_verkocht.drop(["price", "photo", "ownership", "log_id"], axis = 1)
# Funda_scraper_verkocht_col.nunique()


# In[11]:


# Verander de datums zodat de computer ze kan lezen
replace_values = {"januari":"1","februari":"2","maart":"3","april":"4","mei":"5","juni":"6","juli":"7","augustus":"8",
                  "september":"9","oktober":"10","november":"11","december":"12", " ":"/"}
Funda_scraper_verkocht_col = Funda_scraper_verkocht_col.replace({'date_list': replace_values, 
                                                          'date_sold': replace_values}, regex = True)
Funda_scraper_verkocht_col["date_list_dt"] = pd.to_datetime(Funda_scraper_verkocht_col['date_list'], format = "%d/%m/%Y")
Funda_scraper_verkocht_col["date_sold_dt"] = pd.to_datetime(Funda_scraper_verkocht_col['date_sold'], format = "%d/%m/%Y")
# print(Funda_scraper_verkocht_col["date_sold_dt"].min())
# print(Funda_scraper_verkocht_col["date_sold_dt"].max())


# In[12]:


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


# ### Lineaire regressie aanpassingen

# In[13]:


# selecteer alleen de verkochte huizen via Funda in het jaar 2022
Funda_2022 = Funda_scraper_verkocht_pr[Funda_scraper_verkocht_pr["date_sold_dt"].dt.year == 2022]

# selecteer van het dataframe de benodigde kollomen
Funda_reg_col = Funda_2022[["size", "year", "living_area", "num_of_rooms","num_of_bathrooms", "energy_label",
                                 "price_sold","last_ask_price_m2","date_sold_dt"]]

# filter kollomen met foute waardes uit de dataframe en zet de energielabel apart van de score
Funda_reg_col = Funda_reg_col[Funda_reg_col["num_of_bathrooms"].str.contains('badkamer|toilet', na = False)]
Funda_reg_col = Funda_reg_col[~Funda_reg_col["year"].str.contains('-|Na|Voor', na = False)]
Funda_reg_col["energy_label2"] = Funda_reg_col["energy_label"].str[0]

# zet alle values van de kollomen om in cijfers
Funda_reg_col["kamers"] = Funda_reg_col.num_of_rooms.apply(lambda x: 1 if "1 kamer" in x else 
                                                                  (2 if "2 kamers" in x else
                                                                   (3 if "3 kamers" in x else 
                                                                    (4 if "4 kamers" in x else
                                                                     (5 if "5 kamers" in x else
                                                                      (6 if "6 kamers" in x else
                                                                       (7 if "7 kamers" in x else
                                                                        (8 if "8 kamers" in x else
                                                                         (9 if "9 kamers" in x else 
                                                                          (10 if "10 kamers" in x else
                                                                           (11 if "11 kamers" in x else
                                                                            (12 if "12 kamers" in x else
                                                                             (13 if "13 kamers" in x else np.nan
                                                                             )))))))))))))
Funda_reg_col["slaapkamers"] = Funda_reg_col.num_of_rooms.apply(lambda x: 1 if "1 slaapkamer" in x else 
                                                                  (2 if "2 slaapkamers" in x else
                                                                   (3 if "3 slaapkamers" in x else 
                                                                    (4 if "4 slaapkamers" in x else
                                                                     (5 if "5 slaapkamers" in x else
                                                                      (6 if "6 slaapkamers" in x else
                                                                       (7 if "7 slaapkamers" in x else
                                                                        (8 if "8 slaapkamers" in x else
                                                                         (9 if "9 slaapkamers" in x else 
                                                                          (10 if "10 slaapkamers" in x else np.nan
                                                                             ))))))))))
Funda_reg_col["badkamers"] = Funda_reg_col.num_of_bathrooms.apply(lambda x: 1 if "1 badkamer" in x else 
                                                                  (2 if "2 badkamers" in x else
                                                                   (3 if "3 badkamers" in x else 
                                                                    (4 if "4 badkamers" in x else
                                                                     (5 if "5 badkamers" in x else 0
                                                                             )))))
Funda_reg_col["toiletten"] = Funda_reg_col.num_of_bathrooms.apply(lambda x: 1 if "1 apart toilet" in x else 
                                                                  (2 if "2 aparte toiletten" in x else
                                                                   (3 if "3 aparte toiletten" in x else 
                                                                    (4 if "4 aparte toiletten" in x else 0
                                                                             ))))
Funda_reg_col["energie_num"] = Funda_reg_col.energy_label2.apply(lambda x: 1 if "A" == x else 
                                                                  (2 if "B" == x else
                                                                   (3 if "C" == x else 
                                                                    (4 if "D" == x else
                                                                     (5 if "E" == x else
                                                                      (6 if "F" == x else
                                                                       (7 if "G" == x else 0)))))))
Funda_reg_col["size"] = Funda_reg_col["size"].str.replace('m²', '', regex = False)
Funda_reg_col["living_area"] = Funda_reg_col["living_area"].str.replace('m²', '', regex = False)


# In[14]:


# selecteer allen de kollomen met cijfers uit het dataframe
Funda_reg_col2 = Funda_reg_col[["size", "year", "kamers", "slaapkamers", "badkamers", "toiletten", 
                               "energie_num", "price_sold", "last_ask_price_m2", "date_sold_dt"]]

# en omdat we willen kijken naar de uitkomst van de bekende waarde, vervallen de values die niet bekend zijn
Funda_reg_na = Funda_reg_col2.dropna()

# verander het type van de prijs kolommen
Funda_reg_na["price_sold"] = Funda_reg_na["price_sold"].astype(int)
Funda_reg_na["size"] = Funda_reg_na["size"].astype(int)


# ### Data loading en cleaning Amsterdam data gemeente

# In[15]:


veiligheid = pd.read_excel('veiligheidsindex_2023.xlsx', sheet_name = 'index_volgorde_VMgebied', index_col = 0, 
                           skiprows = 1, header = [0,1,2])


# In[16]:


# selecteer alleen de veiligheidsindexen van het jaar 2022 en maak van de multiindex één index
veiligheid_2022 = veiligheid.drop(["Unnamed: 1_level_2", 2021, "2022_1", "2022_2", "2023_1", "2023_2", "2023_2.1"], 
                                  axis = 1, level = 2)
veiligheid_2022.columns = [c[0] + "_" + c[1] + "_"+str(c[2]) for c in veiligheid_2022.columns]

# slecteer alleen de basis indexen
veiligheid_2022_col = veiligheid_2022.drop(['Geregistreerde_High Volume_2022', 'Geregistreerde_High Impact_2022',
                                            'HIC_slachtofferschap _2022','HVC_slachtofferschap_2022',
                                            'Digitaal_slachtofferschap_2022', 'Digitaal_Risicoperceptie_2022', 
                                            'Digitaal_Onveiligheidsgevoelens_2022','Digitaal_Vermijding_2022'], axis = 1)

# wijzig de kolomnamen
veiligheid_2022_col.columns = ["Wijknaam", "Criminaliteitsindex", "Slachtofferschap", "Overlast", "Verloedering", 
                               "Personenoverlast", "Onveiligheidsbeleving"]


# In[17]:


# wijzig een aantal wijknamen in de Funda en veilighieds dataframes, zodat ze overeen komen
replace_values_Funda = {'Centrale Markt': 'Frederik Hendrikbuurt/Centrale Markt',
                  'Frederik Hendrikbuurt':'Frederik Hendrikbuurt/Centrale Markt',
                  'IJburg Oost':'IJburg Oost/IJburg Zuid',
                  'IJburg Zuid':'IJburg Oost/IJburg Zuid',
                  'Spaarndammer  en Zeeheldenbuurt':'Spaarndammerbuurt/Zeeheldenbuurt/Houthavens',
                  'Houthavens':'Spaarndammerbuurt/Zeeheldenbuurt/Houthavens',
                  'Zuidas':'Zuidas/Prinses Irenebuurt e.o.',
                  'Prinses Irenebuurt e.o.':'Zuidas/Prinses Irenebuurt e.o.',
                  '-':' '}
replace_values_Veilig = {'IJplein/Vogelbuurt(Noordelijke IJ oevers Oost)':'IJplein/Vogelbuurt',
                         '-':' '}

Funda_wijken = Funda_scraper_verkocht_pr.replace({'neighborhood_name': replace_values_Funda}, regex = True)
Veilig_wijken = veiligheid_2022_col.replace({'Wijknaam': replace_values_Veilig}, regex = True)

# maak een lijst met alle mogelijke wijken
Funda_wijk = Funda_wijken["neighborhood_name"].dropna().unique().tolist()
Veiligheid_wijk = Veilig_wijken['Wijknaam'].unique().tolist()

wijken = []
for wijk in Funda_wijk:
    if wijk in Veiligheid_wijk:
        wijken.append(wijk)
        
# selecteer van beide dataframes alleen de huizen in wijken die overeen komen
Funda_overeen = Funda_wijken[Funda_wijken["neighborhood_name"].isin(wijken)]
Veiligheid_overeen = Veilig_wijken.query('Wijknaam in @wijken')
Veiligheid_overeen_sort = Veiligheid_overeen.sort_values(by = ['Wijknaam'])


# ## Grafieken

# ### 1D/2D

# In[18]:


fig = go.Figure()

fig.add_trace(go.Box(y = Funda_scraper_verkocht_pr["price_sold"], name = "Verkochte prijs per huis"))
fig.add_trace(go.Box(y = Funda_scraper_verkocht_pr["last_ask_price_m2"], name = "Prijs per m2 per huis", visible = False))

dropdown_buttons = [{'label':'Verkochte prijs per huis', 'method':'update','args':[{'visible':[True, False]},
                                                                         {'yaxis':{'title':'Prijs per huis'}}]},
                   {'label':'Gevraagde prijs per m2', 'method':'update','args':[{'visible':[False, True]},
                                                                         {'yaxis':{'title':'Prijs per m2'}}]}
                   ]
fig.update_layout({'updatemenus':[{'active':0, 'buttons':dropdown_buttons}]}, 
                   title = "Prijs per verkocht huis of prijs per vierkante meter",
                     yaxis_title = 'Prijs per huis')

fig.show()


# In[19]:


fig = go.Figure()

fig.add_trace(go.Scattergl(x = Funda_scraper_verkocht_pr["date_sold_dt"],
                           y = Funda_scraper_verkocht_pr["price_sold"],
                           mode = "markers"))
fig.add_trace(go.Scattergl(x = Funda_scraper_verkocht_pr["date_sold_dt"],
                           y = Funda_scraper_verkocht_pr["last_ask_price_m2"],
                           mode = "markers",
                           visible = False))

dropdown_buttons = [{'label':'Verkochte prijs per huis', 'method':'update','args':[{'visible':[True, False]},
                                                                         {'yaxis':{'title':'Prijs per huis'}}]},
                   {'label':'Gevraagde prijs per m2', 'method':'update','args':[{'visible':[False, True]},
                                                                         {'yaxis':{'title':'Prijs per m2'}}]}
                   ]
fig.update_layout({'updatemenus':[{'active':0, 'buttons':dropdown_buttons, 'y':1.1, 'x':0.22}]}, 
                   title = "Verkochte prijs per huis of laats gevraagde prijs per vierkante meter per datum",
                     xaxis_title = 'Datum',
                     yaxis_title = 'Prijs per huis')
fig.show()


# ### Hulpbron

# In[20]:


fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Histogram(histfunc = "count", x = Funda_overeen["neighborhood_name"], name = "Aantal huizen"),
              secondary_y = False)

fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Criminaliteitsindex"], 
                         mode = 'lines+markers', name = "Criminaliteitsindex"), secondary_y = True)
fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Slachtofferschap"], 
                         mode = 'lines+markers', name = "Slachtofferschapsindex"), secondary_y = True)
fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Overlast"], 
                         mode = 'lines+markers', name = "Overlastindex"), secondary_y = True)
fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Verloedering"], 
                         mode = 'lines+markers', name = "Verloederingsindex"), secondary_y = True)
fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Personenoverlast"], 
                         mode = 'lines+markers', name = "Personenoverlastinex"), secondary_y = True)
fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Onveiligheidsbeleving"], 
                         mode = 'lines+markers', name = "Onveiligheidsbelevingsindex"), secondary_y = True)

fig.update_layout(title = "Aantal verkochte huizen per wijk in Amsterdam en de veiligheidsindexen per categorie", 
                  xaxis_title = "Wijknaam", height = 800, width = 1100)

# Set y-axes titles
fig.update_yaxes(title_text="Aantal huizen", secondary_y=False)
fig.update_yaxes(title_text="Index", secondary_y=True, range = [0,540])

fig.update_xaxes(tickangle = 45, categoryorder = "category ascending")
fig.show()


# ### Lineaire regressie

# In[21]:


# maak een correlatie matrix en heatmap van de correlaties tussen de prijzen en andere waardes
correlaties = Funda_reg_na.corr()
sns.heatmap(correlaties[['price_sold']].sort_values(by="price_sold", ascending = True), annot = True)


# In[22]:


# splits de dataframe op in de inputvariabelen en targetkolom
y = correlaties['price_sold']
X = correlaties.drop('price_sold', axis = 1)

# maak voor machine learne een traindata en testdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# maak nu een Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
predictions

# bereken de R-squared en de error
MSE = mean_squared_error(y_test,predictions)**(1/2)
R2 = r2_score(y_test,predictions)


# In[23]:


plt.figure(figsize=(15,10))
plt.scatter(Funda_reg_na['size'], Funda_reg_na['price_sold'])
m, b = np.polyfit(Funda_reg_na['size'], Funda_reg_na['price_sold'], 1)
plt.plot(Funda_reg_na['size'].values, m*Funda_reg_na['size'].values + b, c='k')
plt.ticklabel_format(style = 'plain', axis = 'y')
plt.xlabel('Grooter van het huis in m2')
plt.ylabel('Prijs per huis Amsterdam')
plt.title("Lineair Regressie model tussen de prijs en oppervlakte van een huis in Amsterdam")
plt.show()


# ## Bronnen

# In[24]:


with st.expander("Bronnen"):
    st.write("Funda Scraper: ", 
             "https://pypi.org/project/funda-scraper/")


# In[ ]:




