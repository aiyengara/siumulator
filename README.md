# 🛰 Live Satellite Tracker — Flask App

## Setup & Run

```bash
# 1. Install dependencies
pip install flask

# 2. Run the app
cd satellite_tracker
python app.py

# 3. Open in browser
# → http://localhost:5000
```

## Features
- **Real Earth map** (3 layers: Dark / Satellite / Terrain) via Leaflet.js
- **Live orbital propagation** — pure Python SGP4/J2 math, no external satellite libraries needed
- **11 satellites tracked**: ISS, Hubble, Terra, Aqua, 4× Starlink, 3× GPS
- **Orbit ground tracks** — 90-minute projected path shown on map
- **Signal beam line** — dotted line from your location to selected satellite
- **Distance & latency** — haversine + slant range → round-trip light-travel time
- **Auto-refresh every 3 seconds** via `/api/satellites` JSON endpoint
- **Filter by category**: All / ISS / Starlink / GPS / LEO

## API Endpoints
- `GET /api/satellites` — all satellites with positions, speeds, distances, latencies
- `GET /api/satellite/<id>` — single satellite detail

## Ground Station
Configured to **Arsikere, Karnataka, India** (13.31°N, 76.25°E).
To change, edit `GROUND_LAT` and `GROUND_LNG` in `app.py`.
