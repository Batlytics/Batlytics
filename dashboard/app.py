from plot import *
import pandas as pd
import streamlit as st
from streamlit_metrics import metric_row

# Setting page title and icon
st.set_page_config(
    page_title='Batlytics', 
    page_icon = '../asset/logo/batlytics_favicon.png'
)

# Reading CSV file
@st.cache
def load_data():
    df =pd.read_csv('../dataset/02_02_21_Scrimmage.csv', low_memory=False)
    return df

# Header image logo
st.image('../asset/logo/Batlytics_logo.png')

# Dropdown component to switch between tabs
dropdown = st.selectbox(
    'Click on tabs to navigate to pages',
    (
        'About', 'OpenCV model', 'Visualizations', 'Future work'
    )
)

# Load part of tabs when user clicks on the tab
if dropdown == 'About':
    
    # header 1
    st.header('Inspiration')
    st.write('''
        Compared to the professional level, college level baseball lacks the complex analytical tools to further improve training and performance of the players. 
        Given the ease of recording video data of practices and matches and the advances in the field of computer vision, it is a no-brainer to not use these techniques to analyze actions of players. 
        Furthermore, technical schools can leverage their high-performance clusters for a very detailed analysis, and also have a light-weight version ready to be used by coaches on their laptops or phones. 
        These insights provide a significant advantage over the other university teams.
    '''
    )
    
    # header 2
    st.header('What it does')
    st.write(
        '''
        Batlytics is a light-weight web-app running in cloud, that offers following baseball analytics features, that can be used for both hitters and pitchers:
        
        • Tracks limbs, joints positions in a given video recording - this data is then visualized in the original video
        
        • Tracks arm angles in a given recording - this data is then visualized in the original video
        
        • Tracks a baseball bat in a given recording - this data is then visualized in the original video
        
        • Tracks a baseball ball in a given recording - this data is then visualized in the original video
        
        • It uses past match data and generates interactive 3-d plots, where user can filter the data as per PitchCall, ExitSpeed, etc. It also displays the KPI metrics such as average relative speed, avg_spinAixs values. 

        As described above, the app takes an input video and produces an output video with detailed information visualized in the original video. 
        In addition, these analytical information can be accessed and used in other ways suitable for the coaches or other team members.
        '''
    )
    
if dropdown == 'OpenCV model':
    st.write('Welcome to OpenCV')

if dropdown == 'Visualizations':
    # KPI cards
    metric_row(
        {
            "Avg Relative speed(miles per hour)": speed_metric(df),
            "Avg SpinAxis": spinAngle_metric(df)
        }
    )
    # Adding css to make the radio options horizontal
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    # Radio button
    choose = st.radio(
        "",
        (
            "TaggedPitchType",
            "AutoPitchType",
            "PitchCall",
            'PlayResult',
            'RelSpeed',
            'SpinRate',
            'SpinAxis',
            'ExitSpeed'
        )
    )
    # Display the plots as per selected radio option
    if choose == 'TaggedPitchType':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'TaggedPitchType'))

    if choose == 'AutoPitchType':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'AutoPitchType'))

    if choose == 'PitchCall':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'PitchCall'))

    if choose == 'PlayResult':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'PlayResult'))

    if choose == 'RelSpeed':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'RelSpeed'))
    
    if choose == 'SpinRate':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'SpinRate'))
    
    if choose == 'SpinAxis':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'SpinAxis'))
    
    if choose == 'ExitSpeed':
        st.plotly_chart(overall_pitch_trajectory(load_data(), 'ExitSpeed'))

    # Plot for Xc0, Yc0, Zc0
    st.plotly_chart(trajectory_xc0_yc0_zc0(load_data()))

# Future work section
if dropdown == 'Future work':
    pass