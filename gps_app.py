import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import folium
from streamlit_folium import st_folium

# CSS: erőltetjük az iframe szélességét/magasságát
st.markdown(
    """
    <style>
    .stApp iframe {
        width: 100% !important;
        height: 700px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Streamlit - Input params
# =========================
st.title("GPS Filtering (Speed-based)")

st.sidebar.header("Parameters")

GPS_ERROR_THRESHOLD = st.sidebar.number_input(
    "Kalman Filter Threshold (m/s)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

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

MAX_SPEED = st.sidebar.number_input(
    "Max Speed",
    min_value=0,
    max_value=500,
    value=130,
    step=1
)

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

show_original = st.sidebar.checkbox("Original path", value=True)
show_filtered = st.sidebar.checkbox("Filtered path", value=True)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    # read CSV — ha a header rossz, megpróbáljuk skiprows=1-el
    try:
        df = pd.read_csv(
            uploaded_file,
            parse_dates=["Fixtime UTC"],
            skip_blank_lines=True
        )
        if "Fixtime UTC" not in df.columns:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, parse_dates=["Fixtime UTC"], skiprows=1)
    except Exception as e:
        st.error(f"Hiba a CSV beolvasásakor: {e}")
        st.stop()

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
            dt = 1
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

        residual = z - (kf.H @ kf.x)
        error_speed = np.linalg.norm(residual) / dt

        alt_speed_error = alt_speed > MAX_ALT_SPEED
        speed_ing_error = row["Speed"] != 0 and row.get("Custom Ignition (io409)", 0) == 0 and SPEED_IGN == 1

        try:
            HDOP = row["HDOP raw (io300)"]
        except Exception:
            HDOP = row.get("HDOP (hdop)", np.nan)

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

    # ===== Map state init =====
    # számoljuk a bounds-ot az összes eredeti pont alapján
    bounds = [
        [df["Latitude"].min(), df["Longitude"].min()],
        [df["Latitude"].max(), df["Longitude"].max()]
    ]

    if "map_state" not in st.session_state:
        st.session_state.map_state = {
            "center": None,
            "zoom": None,
            "bounds": bounds,
            "auto_fit_done": False
        }
    else:
        # ha új CSV-t töltöttél (más bounds), frissítjük a bounds-ot, de ne írjuk felül a felhasználói center/zoom-ot
        st.session_state.map_state["bounds"] = bounds

    # ===== Create map: ha van elmentett center+zoom használjuk, különben középre és később fit_bounds =====
    if st.session_state.map_state.get("center") and st.session_state.map_state.get("zoom") is not None:
        m = folium.Map(location=st.session_state.map_state["center"], zoom_start=st.session_state.map_state["zoom"])
    else:
        # középre állítás a bounding box közepére
        mid_lat = (bounds[0][0] + bounds[1][0]) / 2.0
        mid_lon = (bounds[0][1] + bounds[1][1]) / 2.0
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=12)

    # Polylines (checkboxok alapján)
    if show_original:
        folium.PolyLine(
            df[["Latitude", "Longitude"]].values.tolist(),
            color="red",
            weight=3,
            opacity=0.7,
            tooltip=folium.Tooltip("Original", sticky=False)
        ).add_to(m)

    if show_filtered:
        # ha nincs érvényes pont, ne próbáljuk megrajzolni
        filt_coords = df[df["valid"]][["lat_filtered", "lon_filtered"]].values.tolist()
        if len(filt_coords) > 1:
            folium.PolyLine(
                filt_coords,
                color="blue",
                weight=3,
                opacity=0.7,
                tooltip=folium.Tooltip("Filtered", sticky=False)
            ).add_to(m)

    # Invalid points with HTML tooltip (and prev altitude)
    for i, row in df[~df["valid"]].iterrows():
        try:
            HDOP = row["HDOP raw (io300)"]
        except Exception:
            HDOP = row.get("HDOP (hdop)", "")

        prev_alt = df.loc[i - 1, "Altitude"] if i > 0 else None

        tooltip_text = (
            f"<b>Time:</b> {row['Fixtime UTC']}<br>"
            f"<b>Satellites:</b> {row['Satelites (sat)']}<br>"
            f"<b>Speed:</b> {row['Speed']} km/h<br>"
            f"<b>Ignition:</b> {row.get('Custom Ignition (io409)', '')}<br>"
            f"<b>Altitude:</b> {row['Altitude']} m<br>"
            f"<b>Prev Altitude:</b> {prev_alt if prev_alt is not None else '-'} m<br>"
            f"<b>HDOP:</b> {HDOP}"
        )

        folium.CircleMarker(
            [row["lat_filtered"], row["lon_filtered"]],
            radius=4,
            color="yellow",
            fill=True,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip_text, sticky=True, parse_html=True)
        ).add_to(m)

    # Start & End
    folium.Marker([df["Latitude"].iloc[0], df["Longitude"].iloc[0]], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([df["Latitude"].iloc[-1], df["Longitude"].iloc[-1]], popup="End", icon=folium.Icon(color="red")).add_to(m)

    # Ha még nem futott le az auto-fit, alkalmazzuk a bounds-ot egyszer (első betöltés)
    if st.session_state.map_state.get("bounds") and not st.session_state.map_state.get("auto_fit_done"):
        try:
            m.fit_bounds(st.session_state.map_state["bounds"])
            st.session_state.map_state["auto_fit_done"] = True
        except Exception:
            # fallback: ha valamiért nem jó a bounds formátum
            b = [[df["Latitude"].min(), df["Longitude"].min()], [df["Latitude"].max(), df["Longitude"].max()]]
            m.fit_bounds(b)
            st.session_state.map_state["auto_fit_done"] = True

    # Megjelenítés (széles)
    st.subheader("Maps")
    st_data = st_folium(m, width=1200, height=700)

