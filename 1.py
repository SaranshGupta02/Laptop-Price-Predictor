import streamlit as st
import pickle
import numpy as np

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# Sidebar
st.sidebar.title("Input Parameters")

# Brand
company = st.sidebar.selectbox('Brand',df['Company'].unique())

# Type of laptop
laptop_type = st.sidebar.selectbox('Type',df['TypeName'].unique())

# RAM
ram = st.sidebar.selectbox('RAM (in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
weight = st.sidebar.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.sidebar.selectbox('IPS',['No','Yes'])

# Screen size
screen_size = st.sidebar.number_input('Screen Size')

# Resolution
resolution = st.sidebar.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.sidebar.selectbox('CPU',df['Cpu brand'].unique())

# HDD
hdd = st.sidebar.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

# SSD
ssd = st.sidebar.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

# GPU
gpu = st.sidebar.selectbox('GPU',df['Gpu brand'].unique())

# OS
os = st.sidebar.selectbox('OS',df['os'].unique())

if st.sidebar.button('Predict Price'):
    # Preprocess the input data
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    
    # Convert categorical variables to numerical representations
    company_idx = np.where(df['Company'].unique() == company)[0][0]
    laptop_type_idx = np.where(df['TypeName'].unique() == laptop_type)[0][0]
    cpu_idx = np.where(df['Cpu brand'].unique() == cpu)[0][0]
    gpu_idx = np.where(df['Gpu brand'].unique() == gpu)[0][0]
    os_idx = np.where(df['os'].unique() == os)[0][0]
    
    # Create the query array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    
    # Predict the price
    predicted_price = pipe.predict(query)
    
    st.title("The predicted price of this configuration is ₹" + str(int(np.exp(predicted_price))))
