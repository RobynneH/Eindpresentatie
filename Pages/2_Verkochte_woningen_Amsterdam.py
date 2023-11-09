#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
# maak dataframes voor elk soort woning
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
# lineaire regressie
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
# selecteer allen de kollomen met cijfers uit het dataframe
Funda_reg_col2 = Funda_reg_col[["size", "year", "kamers", "slaapkamers", "badkamers", "toiletten", 
                               "energie_num", "price_sold", "last_ask_price_m2", "date_sold_dt"]]
# en omdat we willen kijken naar de uitkomst van de bekende waarde, vervallen de values die niet bekend zijn
Funda_reg_na = Funda_reg_col2.dropna()
# verander het type van de prijs kolommen
Funda_reg_na["price_sold"] = Funda_reg_na["price_sold"].astype(int)
Funda_reg_na["size"] = Funda_reg_na["size"].astype(int)
correlaties = Funda_reg_na.corr()

# Hulpbron
veiligheid = pd.read_excel('veiligheidsindex_2023.xlsx', sheet_name = 'index_volgorde_VMgebied', index_col = 0, 
                           skiprows = 1, header = [0,1,2])
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
# zet de veiligheidsindexen in dezelfde volgorde
Veiligheid_overeen_sort = Veiligheid_overeen.sort_values(by = ['Wijknaam'])
Veiligheid_overeen['Wijknaam'] = pd.Categorical(Veiligheid_overeen['Wijknaam'], 
                                                ['Omval/Overamstel','Hoofddorppleinbuurt','Staatsliedenbuurt','Jordaan',
                                                 'Oude Pijp','Buitenveldert West','Oostelijk Havengebied','Middenmeer',
                                                 'Frederik Hendrikbuurt/Centrale Markt','Indische Buurt West',
                                                 'Scheldebuurt','Nieuwe Pijp','Osdorp Oost','Museumkwartier','IJburg West',
                                                 'Oosterparkbuurt','Oostelijke Eilanden/Kadijken','Hoofdweg e.o.',
                                                 'Transvaalbuurt','Buikslotermeer','Geuzenbuurt','Erasmuspark',
                                                 'De Kolenkit','Van Lennepbuurt','Waterlandpleinbuurt','Overtoomse Sluis',
                                                 'Westindische Buurt','Banne Buiksloot','Overtoomse Veld',
                                                 'Weesperbuurt/Plantage','De Weteringschans','Stadionbuurt',
                                                 'Tuindorp Oostzaan','Dapperbuurt','Buitenveldert Oost','Rijnbuurt',
                                                 'Indische Buurt Oost','Apollobuurt','Van Galenbuurt','Weesperzijde',
                                                 'Noordelijke IJ oevers West','Chassébuurt','Haarlemmerbuurt','Gein',
                                                 'Osdorp Midden','Nieuwmarkt/Lastage','Da Costabuurt',
                                                 'Spaarndammerbuurt/Zeeheldenbuurt/Houthavens','Zuid Pijp',
                                                 'Grachtengordel West','IJburg Oost/IJburg Zuid',
                                                 'Zuidas/Prinses Irenebuurt e.o.','Geuzenveld','Willemspark',
                                                 'Slotervaart Noord','Slotermeer Noordoost','Slotervaart Zuid','Volewijck',
                                                 'Westlandgracht','Frankendael','De Punt','IJselbuurt',
                                                 'Grachtengordel Zuid','Schinkelbuurt','Nellestein','Betondorp',
                                                 'Burgwallen Nieuwe Zijde','Burgwallen Oude Zijde','Elzenhagen','Driemond',
                                                 'Waterland'])
Veiligheid_overeen_sort = Veiligheid_overeen.sort_values("Wijknaam")


def boxplot_prijs():
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
    
def scatter_prijs():
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
    st.plotly_chart(fig)
    
def  boxplot_soort():
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
                     yaxis_title = "Soort woning")
    st.plotly_chart(fig)
    
def hulpbron():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(histfunc = "count", x = Funda_overeen["neighborhood_name"], name = "Aantal huizen"),
                  secondary_y = False)
    fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Criminaliteitsindex"], 
                             mode = 'lines+markers', name = "Criminaliteitsindex"), secondary_y = True)
    fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Slachtofferschap"], 
                             mode = 'lines+markers', name = "Slachtofferschapsindex"), secondary_y = True)
    fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Overlast"], 
                             mode = 'lines+markers', name = "Overlastindex"), secondary_y = True)
    fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Personenoverlast"], 
                             mode = 'lines+markers', name = "Personenoverlastinex"), secondary_y = True)
    fig.add_trace(go.Scatter(x = Veiligheid_overeen_sort["Wijknaam"], y = Veiligheid_overeen_sort["Onveiligheidsbeleving"], 
                             mode = 'lines+markers', name = "Onveiligheidsbelevingsindex"), secondary_y = True)
    fig.update_layout(title = "Aantal verkochte huizen per wijk in Amsterdam en de veiligheidsindexen per categorie", 
                      xaxis_title = "Wijknaam", height = 800, width = 1100)
    fig.update_yaxes(title_text="Aantal huizen", secondary_y=False)
    fig.update_yaxes(title_text="Index", secondary_y=True, range = [0,540])
    fig.update_xaxes(tickangle = 45, categoryorder = "array", 
                     categoryarray = ['Omval/Overamstel','Hoofddorppleinbuurt','Staatsliedenbuurt','Jordaan','Oude Pijp',
                     'Buitenveldert West','Oostelijk Havengebied','Middenmeer','Frederik Hendrikbuurt/Centrale Markt',
                     'Indische Buurt West','Scheldebuurt','Nieuwe Pijp','Osdorp Oost','Museumkwartier','IJburg West',
                     'Oosterparkbuurt','Oostelijke Eilanden/Kadijken','Hoofdweg e.o.','Transvaalbuurt','Buikslotermeer',
                     'Geuzenbuurt','Erasmuspark','De Kolenkit','Van Lennepbuurt','Waterlandpleinbuurt','Overtoomse Sluis',
                     'Westindische Buurt','Banne Buiksloot','Overtoomse Veld','Weesperbuurt/Plantage','De Weteringschans',
                     'Stadionbuurt','Tuindorp Oostzaan','Dapperbuurt','Buitenveldert Oost','Rijnbuurt','Indische Buurt Oost',
                     'Apollobuurt','Van Galenbuurt','Weesperzijde','Noordelijke IJ oevers West','Chassébuurt','Haarlemmerbuurt',
                     'Gein','Osdorp Midden','Nieuwmarkt/Lastage','Da Costabuurt','Spaarndammerbuurt/Zeeheldenbuurt/Houthavens',
                     'Zuid Pijp','Grachtengordel West','IJburg Oost/IJburg Zuid','Zuidas/Prinses Irenebuurt e.o.','Geuzenveld',
                     'Willemspark','Slotervaart Noord','Slotermeer Noordoost','Slotervaart Zuid','Volewijck','Westlandgracht',
                     'Frankendael','De Punt','IJselbuurt','Grachtengordel Zuid','Schinkelbuurt','Nellestein','Betondorp',
                     'Burgwallen Nieuwe Zijde','Burgwallen Oude Zijde','Elzenhagen','Driemond','Waterland'])
    st.plotly_chart(fig)
    
def heatmap():
    # maak een correlatie matrix en heatmap van de correlaties tussen de prijzen en andere waardes
    fig, ax = plt.subplots()
    correlaties = Funda_reg_na.corr()
    sns.heatmap(correlaties[['price_sold']].sort_values(by="price_sold", ascending = True), annot = True, ax = ax)
    st.pyplot(fig)
    
def bereken_r2():
    # splits de dataframe op in de inputvariabelen en targetkolom
    y = correlaties['price_sold']
    X = correlaties.drop('price_sold', axis = 1)

    # maak voor machine learne een traindata en testdata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    # maak nu een Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    # bereken de R-squared en de error
    r2 = r2_score(y_test,predictions)
    return r2

def bereken_MSE():
    # splits de dataframe op in de inputvariabelen en targetkolom
    y = correlaties['price_sold']
    X = correlaties.drop('price_sold', axis = 1)

    # maak voor machine learne een traindata en testdata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    # maak nu een Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    # bereken de R-squared en de error
    MSE = mean_squared_error(y_test,predictions)**(1/2)
    return MSE

def lin_reg():
    plt.figure(figsize=(15,10))
    plt.scatter(Funda_reg_na['size'], Funda_reg_na['price_sold'])
    m, b = np.polyfit(Funda_reg_na['size'], Funda_reg_na['price_sold'], 1)
    plt.plot(Funda_reg_na['size'].values, m*Funda_reg_na['size'].values + b, c='k')
    plt.ticklabel_format(style = 'plain', axis = 'y')
    plt.xlabel('Oppervlakte van het huis in m2')
    plt.ylabel('Prijs per huis Amsterdam')
    plt.title("Lineair Regressie model tussen de prijs en oppervlakte van een huis in Amsterdam")
    st.pyplot(plt)


st.set_page_config(page_title = "Verkochte woningen Amsterdam", layout = "wide")
st.title("Verkochte woningen Amsterdam")
st.sidebar.header("Verkochte woningen Amsterdam")
image = Image.open('Funda-verkocht.jpg')
st.image(image)

# 1D/2D plots
st.header("Prijzen van verkochte woningen")
col1, col2 = st.columns(2)
with col1:
    boxplot_prijs()
with col2:
    st.write("Deze boxplot laat de verdeling van de prijzen van verkochtte woningen zien. Met de button kan je de boxplot wijzigen naar de verdeling van laatst gevraagde prijs per vierkante meter. Hierbij is bij beide te zien dat de IQR-range niet heel breed is, en dat de prijzen dus best veel van elkaar verschillen. Ook is het duidelijk te zien dat de kwart met de duurste woning prijzen een groot verschil hebben. Hier zijn duidelijk een paar uitwijkers te zien.")

col3, col4 = st.columns(2)
with col3:
    st.write("In deze scatterplot is per verkochte woning in het afgelopen anderhalf jaar. Er is hier duidelijk te zien dat we voor juli 2022 niet veel data hebben, dit omdat er een limiet zit aan hoe veel de scraper van de Funda site kan afhalen.")
    st.write("Bij de scatterplot waarbij de datum tegenover de laatst gevraagde prijs staat is te zien dat de prijzen veel hoger verdeeld liggen.")
    st.write("Er is bij beide verder te zien dat veel woningen voor ongeveer dezelde prijs verkocht zijn.")
with col4:
    scatter_prijs()
    
col5, col6 = st.columns(2)
with col5:
    boxplot_soort()
with col6:
    st.write("In de boxplots hiernaast zijn de prijzen per type woning te zien. Hier is een groot verschil te zien tussen de prijzen van een flat en de prijzen van een grachtenpand of herenhuis. Ook hier is te zien dat zelfs per catagory woning de prijzen groot verschillen. Dit kan zijn door bijvoorbeeld ligging of grootte van de woning.")

# hulprbon
st.divider()
st.header("Veiligheidsindex in de wijken in Amsterdam")

col7, col8 = st.columns(2)
with col7:
    st.write("De Gemeente Amsterdam heeft per wijk een aantal veiligheids-indexen opgesteld in een aantal categroieen. Dit zijn: ")
    st.write(f"- **Criminaliteitsindex.** Deze index is gebaseerd uit politiecijfers over o.a. overvallen, straatroof, zakkenrollerij en fietsendiefstal.")
    st.write(f"- **Slachtofferschap index.** Deze index is gebaseerd op vragen uit de Veiligheidsmonitor. Hierin zit zit high impact slachtofferschap, veelvoorkomend slachtofferschap en digitale criminaliteit.")
    st.write(f"- **Overlastindex.** Deze index is gebaseerd op overlastmeldingen van bewoners. Hieronder vallen jeugdoverlast, geluidsoverlast, rommel op straat en onderhoud straatverlichting.")
    st.write(f"- **Onveiligheidsbelevingsindex.** Deze index is gebaseerd op enquêtegegevens met vragen over onveiligheidsgevoelens, risicoperceptie en vermijdingsgedrag.")
    st.write("Bij alle indexen geld hoe lager de index hoe minder last er is van bijvoorbeeld de criminaliteit.")
    st.write("")
    st.write("In de grafiek hebben we de aantal verkochte woningen per wijk van hoog naar laag geplaatst om te kijken of de indexcijfers van laag naar hoog zouden komen te staan. Het is te zien dat de indexen aan de rechterkant van de grafiek duidelijk hoger liggen dan links. Dit betekent dat onze verwachting klopt en dat er minder woningen verkocht worden in wijken waar meer overlast en/of criminaliteits is.")
with col8:
    hulpbron()
    
# lineaire regressie
st.divider()
st.header("Lineaire regressie tussen de grootte van de woning en de prijs")

col9, col10 = st.columns(2)
with col9:
    heatmap()
with col10:
    st.write("Met de heatmap kunnen we de correlaties zien van een aantal variabelen op de prijs van een woning. Hierbij is te zien dat de oppervlakte van een woning de grootste correlatie met de prijs van een woning heeft. Dit wordt dan ook de variabele die wij gaan gebruiken voor ons lineair model.")

col11, col12 = st.columns(2)
with col11:
    lin_reg()
with col12:
    st.write("In de plot hiernaast is een simpele lineaire regressie te zien met een scatterplot tussen de prijs van een woning en de oppervlakte van een woning. Hieruit komt een logische conclusie dat hoe groter de oppervlakte van een woning, hoe duurder de woning is.")
    st.write("Met dit model komen we op een mean squared error(MSE) van ", bereken_MSE(), " en een R kwadraat van ", bereken_r2())


# In[ ]:




