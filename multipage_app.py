#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st

from multipage import MultiPage

from pages.Verkochte_woningen_Amsterdam import Verkochte_woningen_Amsterdam
from pages.Introductie import Introductie

app = MultiPage()

# Add all your pages here
app.add_page("Introductie", Introductie)
app.add_page("Verkochte woningen Amsterdam", Verkochte_woningen_Amsterdam)

# Add the main content
st.title("Einpresentatie: Koopwoningen Amsterdam")
st.subheader("Sten den Hartog & Robynne Hughes")

app.run()


# In[ ]:




