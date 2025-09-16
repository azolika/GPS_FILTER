import streamlit as st
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import folium
from streamlit_folium import st_folium

# =========================
# Streamlit - Input params
# =========================
st.title("GPS Filtering")

st.sidebar.header("Parameters")
GPS_ERROR_THRESHOLD = st.sidebar.number_input("Kalman Filter Threshold (m)", 1, 1000, 100)
MIN_SATELLITES = st.sidebar.number_input("Min Satellites", 1, 12, 4)
P_INITIAL = 500
R_MEASUREMENT = 5
Q_PROCESS = 0.1
MIN_HDOP = st.sidebar.number_input("Min HDOP", 0.0, 10.0, 0.1)
MAX_HDOP = st.sidebar.number_input("Max HDOP", 0.0, 10.0, 2.0)
MAX_SPEED = st.sidebar.number_input("Max Speed", 0, 500, 130)
MIN_ALT = st.sidebar.number_input("Min Altitude", -100, 10000, 0)
MAX_ALT = st.sidebar.number_input("Max Altitude", 0, 10000, 2500)
SPEED_IGN = st.sidebar.number_input("Speed w Ignition", 0, 1, 1)

uploaded_file = st.file_uploader("upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Fixtime UTC"])
    df = df.sort_values("Fixtime UTC").reset_index(drop=True)

    # Conversie lat/lon -> x/y
    R = 6371000
    lat0 = np.radians(df["Latitude"].iloc[0])
    lon0 = np.radians(df["Longitude"].iloc[0])
    df["x"] = (np.radians(df["Longitude"]) - lon0) * np.cos(lat0) * R
    df["y"] = (np.radians(df["Latitude"]) - lat0) * R

    # Kalman Filter init
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    kf.P *= P_INITIAL
    kf.R *= R_MEASUREMENT
    kf.Q *= Q_PROCESS
    kf.x = np.array([df["x"].iloc[0], df["y"].iloc[0], 0, 0])

    valid = []
    x_filtered = []
    y_filtered = []
    last_time = df["Fixtime UTC"].iloc[0]

    for i, row in df.iterrows():
        dt = (row["Fixtime UTC"] - last_time).total_seconds()
        last_time = row["Fixtime UTC"]

        kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        z = np.array([row["x"], row["y"]])
        kf.predict()
        kf.update(z)

        x_filtered.append(kf.x[0])
        y_filtered.append(kf.x[1])

        residual = z - (kf.H @ kf.x)
        error = np.linalg.norm(residual)
        if row["Speed"] != 0 and row["Custom Ignition (io409)"] == 0 and SPEED_IGN == 1:
            speed_ing_error = True
        else:
            speed_ing_error = False

        HDOP = 0
        try:
            HDOP = row["HDOP raw (io300)"]
        except:
            HDOP = row["HDOP (hdop)"]

        valid.append(
            row["Satelites (sat)"] >= MIN_SATELLITES and
            error <= GPS_ERROR_THRESHOLD and
            MIN_HDOP < HDOP < MAX_HDOP and
            row["Speed"] < MAX_SPEED and
            MIN_ALT < row["Altitude"] < MAX_ALT and
            speed_ing_error is False
        )

    df["valid"] = valid
    df["x_filtered"] = x_filtered
    df["y_filtered"] = y_filtered
    df["lat_filtered"] = np.degrees(df["y_filtered"] / R + lat0)
    df["lon_filtered"] = np.degrees(df["x_filtered"] / (R * np.cos(lat0)) + lon0)

    # Folium map
    center_lat = df["Latitude"].iloc[0]
    center_lon = df["Longitude"].iloc[0]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Original route
    folium.PolyLine(df[["Latitude","Longitude"]].values.tolist(), color="red", weight=3, opacity=0.7, tooltip="Original").add_to(m)

    # Filtered route
    folium.PolyLine(df[df["valid"]][["lat_filtered","lon_filtered"]].values.tolist(), color="blue", weight=3, opacity=0.7, tooltip="Filtered").add_to(m)

    # Invalid points
    for i, row in df[~df["valid"]].iterrows():
        folium.CircleMarker([row["lat_filtered"], row["lon_filtered"]], radius=4, color="yellow", fill=True, fill_opacity=0.9).add_to(m)

    # Start & End
    folium.Marker(df[["Latitude","Longitude"]].values[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(df[["Latitude","Longitude"]].values[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    st.subheader("Maps")
    st_data = st_folium(m, width=700, height=500)
