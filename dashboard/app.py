import streamlit as st
import pandas as pd

st.title('Batlytics')


# Reading CSV file
def load_csv_data(filepath):
    data = pd.read_csv('../dataset/02_02_21_Scrimmage.csv', low_memory=False)
    return data

# Dropdown component to switch between tabs
dropdown = st.selectbox(
    'Click on tabs to navigate to pages',
    (
        'Home', 'About', 'OpenCV model', 'Visualizations'
    )
)

st.write(f'You selected : {dropdown}')

# Load part of tabs when user clicks on the tab
if dropdown == 'Home':
    st.write('Welcome to Home page')

if dropdown == 'About':
    st.write('Welcome to About page')

if dropdown == 'OpenCV model':
    st.write('Welcome to OpenCV')

if dropdown == 'Visualizations':
    st.write('Welcome to Viz.')