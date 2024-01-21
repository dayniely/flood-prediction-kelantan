# home.py

import streamlit as st
import pandas as pd
import plotly.express as px
import calendar
import numpy as np
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.graph_objects as go
from PIL import Image


# Set page configuration - this should be the first Streamlit command
st.set_page_config(page_title="Flood Monitoring Dashboard", page_icon="Images/flood.ico", layout="wide", initial_sidebar_state="expanded")

def show_home():

    # Header
    st.markdown("<h1 style='text-align: center; color: #00C59F;'>Flood Prediction of Kelantan River Basin Using Machine Learning Model</h1>", unsafe_allow_html=True)
    st.write("-----")


    st.title('Kelantan River Basin')

    # You can adjust the width of the image in the sidebar if needed
    image = Image.open('Images/kelantan.png')  # Update the path to your image
    st.image(image, caption='Kelantan River', width=1100)

    st.write("""
    ## Kelantan flood
    The Kelantan River Basin in Malaysia, characterized by its intricate network of rivers, 
    faces recurring floods during the northeast monsoon season (November to March). The flat terrain of Kelantan amplifies the impact of heavy rainfall, resulting in inundated areas and substantial economic losses. Efforts to manage floods include infrastructure development and early warning systems. Despite these measures, challenges persist, prompting communities to adopt adaptive strategies. Government and non-governmental initiatives focus on enhancing resilience through outreach and education,
    aiming to better prepare and respond to the cyclic floods in the region.
    """)

    st.write("-----")


    st.title('Kelantan River Basin')

    # Displaying the image and bullet points side by side
    col1, col2 = st.columns(2)

    with col1:
        kelantan_map = Image.open('Images/kelantanmap.jpg')  # Update the path to your image
        st.image(kelantan_map, caption='Kelantan River Basin Map', width=400)

    with col2:
        st.markdown("""
        **Key Features of Kelantan River Basin:**
        - Prone to annual flooding during the monsoon season, affecting local communities and ecosystems.
                    
        - Encompasses a large area with diverse geographical features, including flat terrains that exacerbate flood impacts.
                    
        - The river system is crucial for the region's ecology, economy, agriculture, and fisheries.
    
        - Rich in cultural heritage, the region has historically significant sites often affected by flooding.

        - Recurring floods pose challenges to infrastructure, requiring resilient and adaptive urban planning.

        - Collaborative efforts between government agencies, NGOs, and local communities are essential for sustainable flood management.
        """)

    st.write("-----")



# Main execution
if __name__ == "__main__":
    
    show_home()
