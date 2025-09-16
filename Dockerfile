# Alap image Python 3.11
FROM python:3.11-slim

# Mappa létrehozása a konténerben
WORKDIR /app

# Függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# A teljes projekt másolása
COPY . .

# Streamlit futtatása a 9500-as porton, host 0.0.0.0
EXPOSE 9500
CMD ["streamlit", "run", "gps_app.py", "--server.port=9500", "--server.address=0.0.0.0", "--server.headless=true"]
