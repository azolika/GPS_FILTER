import streamlit as st
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import folium
from streamlit_folium import st_folium

# =========================
# Streamlit - Input params
# =========================
st.title("GPS Filtering (Speed-based)")

st.sidebar.header("Parameters")

# Threshold (m/s) → float mindenhol
GPS_ERROR_THRESHOLD = st.sidebar.number_input(
    "Kalman Filter Threshold (m/s)",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

# Minimum satellites → int mindenhol
MIN_SATELLITES = st.sidebar.number_input(
    "Min Satellites",
    min_value=1,
    max_value=12,
    value=4,
    step=1
)

P_INITIAL = 500
R_MEASUREMENT = 5
Q_PROCESS = 0.1

# HDOP → float mindenhol
MIN_HDOP = st.sidebar.number_input(
    "Min HDOP",
    min_value=0.0,
    max_value=10.0,
    value=0.1,
    step=0.1
)
MAX_HDOP = st.sidebar.number_input(
    "Max HDOP",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

# Speed → int
MAX_SPEED = st.sidebar.number_input(
    "Max Speed",
    min_value=0,
    max_value=500,
    value=130,
    step=1
)

# Altitude → int
MIN_ALT = st.sidebar.number_input(
    "Min Altitude",
    min_value=-100,
    max_value=10000,
    value=0,
    step=1
)
MAX_ALT = st.sidebar.number_input(
    "Max Altitude",
    min_value=0,
    max_value=10000,
    value=2500,
    step=1
)

# Speed w Ignition → int
SPEED_IGN = st.sidebar.number_input(
    "Speed w Ignition",
    min_value=0,
    max_value=1,
    value=1,
    step=1
)

MAX_ALT_SPEED = st.sidebar.number_input(
    "Max Altitude Change Speed (m/s)",
    min_value=0.1,
    max_value=50.0,
    value=5.0,
    step=0.1
)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Fixtime UTC"])
    df = df.sort_values("Fixtime UTC").reset_index(drop=True)

    # Lat/Lon -> x/y
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

    last_alt = None

    for i, row in df.iterrows():
        dt = (row["Fixtime UTC"] - last_time).total_seconds()
        if dt <= 0:
            dt = 1  # elkerüljük a nulla osztást

        last_time = row["Fixtime UTC"]

        if last_alt is None:
            alt_speed = 0
        else:
            alt_speed = abs(row["Altitude"] - last_alt) / dt
        last_alt = row["Altitude"]

        kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        z = np.array([row["x"], row["y"]])
        kf.predict()
        kf.update(z)

        x_filtered.append(kf.x[0])
        y_filtered.append(kf.x[1])

        # Residual speed [m/s]
        residual = z - (kf.H @ kf.x)
        error_speed = np.linalg.norm(residual) / dt

        # Altitude check
        alt_speed_error = alt_speed > MAX_ALT_SPEED

        # Speed/ignition check
        speed_ing_error = row["Speed"] != 0 and row["Custom Ignition (io409)"] == 0 and SPEED_IGN == 1

        # HDOP
        try:
            HDOP = row["HDOP raw (io300)"]
        except:
            HDOP = row["HDOP (hdop)"]

        valid.append(
            row["Satelites (sat)"] >= MIN_SATELLITES and
            error_speed <= GPS_ERROR_THRESHOLD and
            MIN_HDOP < HDOP < MAX_HDOP and
            row["Speed"] < MAX_SPEED and
            MIN_ALT < row["Altitude"] < MAX_ALT and
            not speed_ing_error and
            not alt_speed_error
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
