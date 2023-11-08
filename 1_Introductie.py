#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
from PIL import Image


# In[19]:


st.set_page_config(page_title = "Eindpresentatie Funda", layout = "wide")


# In[21]:


st.sidebar.success("Kies een pagina.")
st.title("Einpresentatie: Koopwoningen Amsterdam")
st.subheader("Sten den Hartog & Robynne Hughes")
image = Image.open('fundalogo.jpg')
st.image(image)


# In[ ]:


st.write("In dit dashboard kijken we naar woningen in Amsterdam die momenteel te koop staan en verkocht zijn via Funda in het afgelopen anderhalf jaar. Deze informatie wordt met een Scraper direct van Funda afgehaald.")
st.write("De informatie die wij per woning ontvangen bevat onder andere de url naar de Funda site, de prijs, oppervlakte, aantal kamers etc.")
st.write("In de sidebar kun je kijken naar de data analyses die zijn uitgevoerd over de huidige woningen op Funda of de verkochte woningen op Funda.")


# In[16]:


st.write("Naast de informatie die we van Funda afhalen, kijken we ook naar veilighieds-indexen in Amsterdam. Deze informatie hebben we van de gemeente van Amsterdam. (Een link naar de website staat bij de bronnen). Wil je zelf naar de data kijken, dan kun je een document met de knop hieronder downloaden: ")
with open('veiligheidsindex_2023.xlsx', 'rb') as file:
    st.download_button(label = "Download veiligheidsindex bestand",
                       data = file,
                       file_name = "veiligheidsindex_2023.xlsx",
                       mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ## Bronnen

# In[17]:


st.header("Bronnen")
st.write("Funda Scraper: ", "https://pypi.org/project/funda-scraper/")
st.write("Data Amsterdam, veiligheidsindex: ", 
         "https://data.amsterdam.nl/datasets/bcy0MclnBpXyDQ/cijfers-veiligheidsindex/")
st.write('Logo banner: ', "https://vastgoedactueel.nl/wp-content/uploads/2018/06/Vastgoed-actueel-fundalogo.jpg.webp")
st.write("Lineaire regressie: ", 
         "https://datasciencepartners.nl/linear-regression-python/#wat-is-linear-regression-python")


# In[ ]:




