"""
Software Defined Antenna (SDA) Multi-Constellation & SatCom Simulator
Components:
  1. GNSS/IRNSS Simulator - multi-constellation with location deviation
  2. SatCom SDR Simulator - 300 MHz to 12 GHz
  3. SDA Capture Engine - MIMO signal capture
  4. Payload Processor & Display
"""

from flask import Flask, jsonify, request, render_template_string
import numpy as np
import json, math, random, time

app = Flask(__name__)

# ─── GNSS Constellation Definitions ───────────────────────────────────────────
CONSTELLATIONS = {
    "GPS_L1":    {"freq_mhz": 1575.42, "band": "L1",  "color": "#00FF88", "sats": 31, "sigma_m": 2.5,  "desc": "GPS L1 C/A"},
    "GPS_L5":    {"freq_mhz": 1176.45, "band": "L5",  "color": "#00CCFF", "sats": 18, "sigma_m": 1.2,  "desc": "GPS L5"},
    "GLONASS":   {"freq_mhz": 1602.00, "band": "L1",  "color": "#FF6B35", "sats": 24, "sigma_m": 3.8,  "desc": "GLONASS L1"},
    "GALILEO":   {"freq_mhz": 1575.42, "band": "E1",  "color": "#FFD700", "sats": 30, "sigma_m": 1.8,  "desc": "Galileo E1"},
    "BEIDOU":    {"freq_mhz": 1561.10, "band": "B1",  "color": "#FF4488", "sats": 35, "sigma_m": 2.1,  "desc": "BeiDou B1"},
    "NavIC":     {"freq_mhz": 1176.45, "band": "L5",  "color": "#AA88FF", "sats": 7,  "sigma_m": 5.0,  "desc": "NavIC L5 (IRNSS)"},
}

# ─── SatCom Band Definitions (300 MHz – 12 GHz) ──────────────────────────────
SATCOM_BANDS = [
    {"name": "UHF",      "min_mhz": 300,    "max_mhz": 1000,   "color": "#FF6B35", "use": "MILSATCOM, MUOS, UFO"},
    {"name": "L-Band",   "min_mhz": 1000,   "max_mhz": 2000,   "color": "#FFD700", "use": "Iridium, Inmarsat, GPS"},
    {"name": "S-Band",   "min_mhz": 2000,   "max_mhz": 4000,   "color": "#00FF88", "use": "Weather, Radar, NASA STDN"},
    {"name": "C-Band",   "min_mhz": 4000,   "max_mhz": 8000,   "color": "#00CCFF", "use": "FSS, BSS, VSAT, TV Broadcast"},
    {"name": "X-Band",   "min_mhz": 8000,   "max_mhz": 12000,  "color": "#FF4488", "use": "Military SatCom, SAR, Weather"},
]

# ─── SDA Antenna Element Config ──────────────────────────────────────────────
SDA_ELEMENTS = [
    {"id": "E1", "type": "Helix (Axial)",   "freq_range": "1.0–2.0 GHz",  "polarization": "RHCP", "mimo_port": 1},
    {"id": "E2", "type": "Helix (Normal)",  "freq_range": "0.3–1.0 GHz",  "polarization": "Linear-V", "mimo_port": 2},
    {"id": "E3", "type": "Linear Monopole","freq_range": "0.3–3.0 GHz",  "polarization": "Linear-V", "mimo_port": 3},
    {"id": "E4", "type": "Patch Array",     "freq_range": "1.1–2.0 GHz",  "polarization": "RHCP", "mimo_port": 4},
    {"id": "E5", "type": "Horn (SHF)",      "freq_range": "8.0–12.0 GHz", "polarization": "Linear-H", "mimo_port": 5},
    {"id": "E6", "type": "Spiral (Wideband)","freq_range":"1.0–8.0 GHz", "polarization": "LHCP/RHCP", "mimo_port": 6},
]

# ─── Helper: Generate Simulated Satellite Positions ──────────────────────────
def sim_satellites(constellation, lat, lon, n_sats):
    sats = []
    rng = np.random.default_rng(int(abs(lat*100+lon*10) + hash(constellation)) % 99999)
    for i in range(n_sats):
        el = float(rng.uniform(5, 85))
        az = float(rng.uniform(0, 360))
        snr = float(rng.uniform(28, 48))
        sats.append({"id": f"{constellation[:2]}{i+1:02d}", "el": round(el,1), "az": round(az,1), "snr": round(snr,1)})
    return sats

def position_deviation(constellation, lat, lon):
    """Simulate position error per constellation"""
    cfg = CONSTELLATIONS[constellation]
    rng = np.random.default_rng(int(time.time()*10) % 9999)
    sigma = cfg["sigma_m"]
    dlat_m = float(rng.normal(0, sigma))
    dlon_m = float(rng.normal(0, sigma))
    dlat_deg = dlat_m / 111320.0
    dlon_deg = dlon_m / (111320.0 * math.cos(math.radians(lat)))
    return {
        "lat": round(lat + dlat_deg, 7),
        "lon": round(lon + dlon_deg, 7),
        "dlat_m": round(dlat_m, 2),
        "dlon_m": round(dlon_m, 2),
        "error_m": round(math.sqrt(dlat_m**2 + dlon_m**2), 2),
    }

# ─── Helper: SDR Signal Simulation ──────────────────────────────────────────
def sim_sdr_spectrum(center_mhz, bw_mhz=200):
    """Simulate power spectral density around center freq"""
    freqs = np.linspace(center_mhz - bw_mhz/2, center_mhz + bw_mhz/2, 256)
    noise = np.random.normal(-95, 3, 256)
    # Add signal peaks
    for _ in range(random.randint(1,4)):
        fc = center_mhz + random.uniform(-bw_mhz*0.3, bw_mhz*0.3)
        bw = random.uniform(1, 20)
        amp = random.uniform(15, 40)
        noise += amp * np.exp(-0.5*((freqs-fc)/bw)**2)
    return {"freqs": freqs.tolist(), "power": noise.tolist()}

def sim_mimo_channels(n_tx=4, n_rx=4):
    """Simulate MIMO channel matrix H"""
    H_real = np.random.randn(n_rx, n_tx)
    H_imag = np.random.randn(n_rx, n_tx)
    H = H_real + 1j * H_imag
    # SVD for capacity
    sv = np.linalg.svd(H, compute_uv=False)
    snr_lin = 10.0
    capacity = float(np.sum(np.log2(1 + snr_lin * (sv**2) / n_tx)))
    return {
        "magnitude": np.abs(H).tolist(),
        "singular_values": sv.tolist(),
        "capacity_bps_hz": round(capacity, 2),
        "condition_number": round(float(sv[0]/sv[-1]), 2) if sv[-1] > 0 else 999,
    }

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/gnss", methods=["POST"])
def gnss_sim():
    data = request.json
    lat = float(data.get("lat", 13.0))
    lon = float(data.get("lon", 77.5))
    results = {}
    for name, cfg in CONSTELLATIONS.items():
        n = cfg["sats"]
        visible = random.randint(max(4, n//4), min(n, n//2 + 4))
        results[name] = {
            "freq_mhz": cfg["freq_mhz"],
            "band": cfg["band"],
            "color": cfg["color"],
            "desc": cfg["desc"],
            "visible_sats": visible,
            "total_sats": cfg["sats"],
            "satellites": sim_satellites(name, lat, lon, visible),
            "position": position_deviation(name, lat, lon),
            "pdop": round(random.uniform(1.2, 3.5), 2),
            "hdop": round(random.uniform(0.8, 2.2), 2),
        }
    return jsonify({"lat": lat, "lon": lon, "constellations": results})

@app.route("/api/satcom", methods=["POST"])
def satcom_sim():
    data = request.json
    freq_mhz = float(data.get("freq_mhz", 1550))
    # Find band
    band_info = next((b for b in SATCOM_BANDS if b["min_mhz"] <= freq_mhz <= b["max_mhz"]), SATCOM_BANDS[1])
    spectrum = sim_sdr_spectrum(freq_mhz, bw_mhz=float(data.get("bw_mhz", 200)))
    snr = round(random.uniform(8, 35), 1)
    return jsonify({
        "freq_mhz": freq_mhz,
        "band": band_info,
        "spectrum": spectrum,
        "snr_db": snr,
        "signal_power_dbm": round(random.uniform(-80, -40), 1),
        "noise_floor_dbm": round(random.uniform(-100, -90), 1),
        "modulation": random.choice(["BPSK","QPSK","8PSK","16APSK","32APSK"]),
        "coding_rate": random.choice(["1/2","2/3","3/4","5/6"]),
        "symbol_rate_msps": round(random.uniform(1, 50), 2),
    })

@app.route("/api/sda", methods=["POST"])
def sda_capture():
    data = request.json
    freq_mhz = float(data.get("freq_mhz", 1575))
    active = data.get("active_elements", [e["id"] for e in SDA_ELEMENTS])
    mimo = sim_mimo_channels(len(active), len(active))
    captures = []
    for eid in active:
        el = next((e for e in SDA_ELEMENTS if e["id"] == eid), SDA_ELEMENTS[0])
        captures.append({
            "element": el,
            "rssi_dbm": round(random.uniform(-85, -45), 1),
            "phase_deg": round(random.uniform(0, 360), 1),
            "gain_dbi": round(random.uniform(2, 8), 1),
            "status": "LOCKED" if random.random() > 0.1 else "SEARCHING",
        })
    return jsonify({
        "freq_mhz": freq_mhz,
        "captures": captures,
        "mimo": mimo,
        "timestamp": time.time(),
        "active_elements": len(active),
    })

@app.route("/api/payload", methods=["POST"])
def payload_process():
    data = request.json
    gnss_data = data.get("gnss", {})
    satcom_data = data.get("satcom", {})
    sda_data = data.get("sda", {})
    # Fuse GNSS positions
    positions = []
    for name, c in gnss_data.get("constellations", {}).items():
        positions.append(c["position"])
    if positions:
        fused_lat = sum(p["lat"] for p in positions) / len(positions)
        fused_lon = sum(p["lon"] for p in positions) / len(positions)
        errors = [p["error_m"] for p in positions]
        fused_error = round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 2)
    else:
        fused_lat = data.get("lat", 13.0)
        fused_lon = data.get("lon", 77.5)
        fused_error = 0
    return jsonify({
        "fused_position": {"lat": round(fused_lat,7), "lon": round(fused_lon,7), "error_m": fused_error},
        "gnss_health": "NOMINAL",
        "satcom_link": "ACTIVE" if satcom_data.get("snr_db", 0) > 10 else "DEGRADED",
        "sda_mimo_capacity": sda_data.get("mimo", {}).get("capacity_bps_hz", 0),
        "total_constellations": len(gnss_data.get("constellations", {})),
        "recommendation": "Use combined GPS L1+L5 + Galileo for optimal accuracy" if fused_error < 2 else "Enable additional constellations to improve fix",
    })

@app.route("/api/bands")
def get_bands():
    return jsonify({"satcom_bands": SATCOM_BANDS, "gnss_constellations": CONSTELLATIONS, "sda_elements": SDA_ELEMENTS})

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# ─── HTML Frontend ────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDA · Multi-Constellation & SatCom Simulator</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #020810;
  --bg2: #050f1e;
  --bg3: #071628;
  --panel: rgba(0,180,255,0.04);
  --border: rgba(0,180,255,0.18);
  --accent: #00b4ff;
  --accent2: #00ff88;
  --accent3: #ff6b35;
  --accent4: #ffd700;
  --text: #c8e8ff;
  --text2: #6a9fc0;
  --danger: #ff4488;
  --glow: 0 0 20px rgba(0,180,255,0.3);
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Rajdhani', sans-serif;
  font-size: 14px;
  min-height: 100vh;
  overflow-x: hidden;
}
/* Grid bg */
body::before {
  content:'';
  position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(0,180,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,180,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}
.header {
  position:relative; z-index:10;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(0,180,255,0.08) 0%, transparent 100%);
  display:flex; align-items:center; justify-content:space-between;
}
.header h1 {
  font-family:'Orbitron',monospace; font-size:22px; font-weight:900;
  color:#fff; letter-spacing:3px;
  text-shadow: 0 0 30px rgba(0,180,255,0.8);
}
.header h1 span { color:var(--accent); }
.header-sub { font-family:'Share Tech Mono'; font-size:11px; color:var(--text2); letter-spacing:2px; margin-top:4px; }
.status-bar { display:flex; gap:16px; align-items:center; }
.status-dot { width:8px; height:8px; border-radius:50%; background:var(--accent2); box-shadow:0 0 8px var(--accent2); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.status-label { font-family:'Share Tech Mono'; font-size:11px; color:var(--accent2); }

/* Tabs */
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); position:relative; z-index:10; }
.tab {
  padding: 12px 28px; cursor:pointer; font-family:'Orbitron'; font-size:11px;
  font-weight:700; letter-spacing:2px; color:var(--text2);
  border-bottom:2px solid transparent; transition:all 0.2s;
  border-right:1px solid var(--border);
}
.tab:hover { color:var(--accent); background:rgba(0,180,255,0.05); }
.tab.active { color:var(--accent); border-bottom-color:var(--accent); background:rgba(0,180,255,0.08); }

/* Main layout */
.main { position:relative; z-index:5; padding:24px; display:none; }
.main.active { display:block; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }

/* Panels */
.panel {
  background:var(--panel); border:1px solid var(--border);
  border-radius:6px; padding:20px;
  backdrop-filter:blur(4px);
  box-shadow: var(--glow);
}
.panel-title {
  font-family:'Orbitron'; font-size:11px; font-weight:700;
  letter-spacing:3px; color:var(--accent); margin-bottom:16px;
  padding-bottom:10px; border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center;
}
.panel-title .badge {
  font-size:9px; padding:2px 8px; border-radius:2px;
  background:rgba(0,180,255,0.15); color:var(--accent);
  border:1px solid var(--accent); letter-spacing:1px;
}

/* Inputs */
.input-row { display:flex; gap:12px; align-items:flex-end; flex-wrap:wrap; margin-bottom:16px; }
.input-group { display:flex; flex-direction:column; gap:6px; }
.input-group label { font-family:'Share Tech Mono'; font-size:10px; color:var(--text2); letter-spacing:1px; }
input[type=number], input[type=range], select {
  background: rgba(0,180,255,0.06);
  border:1px solid var(--border); border-radius:4px;
  color:var(--text); padding:8px 12px;
  font-family:'Share Tech Mono'; font-size:12px;
  outline:none; transition:border 0.2s;
}
input[type=number]:focus, select:focus { border-color:var(--accent); box-shadow:0 0 10px rgba(0,180,255,0.2); }
input[type=range] { padding:4px; cursor:pointer; accent-color:var(--accent); }
.btn {
  padding:10px 22px; border:1px solid var(--accent);
  background: rgba(0,180,255,0.12); color:var(--accent);
  font-family:'Orbitron'; font-size:10px; font-weight:700;
  letter-spacing:2px; cursor:pointer; border-radius:4px;
  transition:all 0.2s; white-space:nowrap;
}
.btn:hover { background:rgba(0,180,255,0.25); box-shadow:var(--glow); }
.btn-green { border-color:var(--accent2); color:var(--accent2); background:rgba(0,255,136,0.08); }
.btn-green:hover { background:rgba(0,255,136,0.2); box-shadow:0 0 20px rgba(0,255,136,0.3); }

/* Constellation cards */
.const-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.const-card {
  border:1px solid var(--border); border-radius:4px; padding:12px;
  background:rgba(0,0,0,0.3); transition:all 0.2s; cursor:pointer;
  position:relative; overflow:hidden;
}
.const-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.const-card:hover { border-color:var(--accent); transform:translateY(-2px); }
.const-card.active-const { background:rgba(0,180,255,0.08); }
.const-name { font-family:'Orbitron'; font-size:10px; font-weight:700; letter-spacing:1px; }
.const-freq { font-family:'Share Tech Mono'; font-size:10px; color:var(--text2); margin-top:4px; }
.const-sats { font-size:22px; font-weight:700; font-family:'Orbitron'; margin-top:6px; }
.const-error { font-family:'Share Tech Mono'; font-size:10px; margin-top:4px; }

/* Sky plot */
#skyPlot { width:100%; aspect-ratio:1; }

/* Map */
#gnssMap {
  width:100%; height:380px; border-radius:4px;
  border:1px solid var(--border); background:#0a1520;
  position:relative; overflow:hidden;
}

/* Band visualization */
.band-viz { width:100%; height:60px; border-radius:4px; overflow:hidden; margin-bottom:8px; display:flex; cursor:pointer; }
.band-seg {
  height:100%; display:flex; align-items:center; justify-content:center;
  font-family:'Orbitron'; font-size:9px; font-weight:700; letter-spacing:1px;
  transition:filter 0.2s; border-right:1px solid rgba(0,0,0,0.3);
  flex-direction:column; gap:2px;
}
.band-seg:hover { filter:brightness(1.3); }
.band-seg .band-freq { font-size:8px; font-family:'Share Tech Mono'; opacity:0.7; }

/* MIMO matrix */
.mimo-grid { display:grid; gap:3px; }
.mimo-cell {
  aspect-ratio:1; border-radius:2px; display:flex;
  align-items:center; justify-content:center;
  font-family:'Share Tech Mono'; font-size:9px;
  transition:all 0.3s;
}

/* Metric row */
.metric-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
.metric {
  flex:1; min-width:80px; padding:10px 14px;
  background:rgba(0,0,0,0.3); border:1px solid var(--border);
  border-radius:4px; text-align:center;
}
.metric-val { font-family:'Orbitron'; font-size:18px; font-weight:700; color:var(--accent2); }
.metric-label { font-family:'Share Tech Mono'; font-size:9px; color:var(--text2); margin-top:4px; letter-spacing:1px; }

/* Spectrum chart */
#spectrumChart, #satChart { max-height:200px; }
canvas { border-radius:4px; }

/* Map pins SVG */
.map-pin { cursor:pointer; transition:all 0.2s; }
.map-pin:hover .pin-circle { r:8; }

/* Deviation table */
.dev-table { width:100%; border-collapse:collapse; }
.dev-table th { font-family:'Orbitron'; font-size:9px; letter-spacing:1px; color:var(--text2); padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; }
.dev-table td { font-family:'Share Tech Mono'; font-size:11px; padding:6px 8px; border-bottom:1px solid rgba(0,180,255,0.06); }
.dev-table tr:hover td { background:rgba(0,180,255,0.05); }

/* Element status */
.elem-row { display:flex; gap:8px; align-items:center; padding:8px 10px; border-bottom:1px solid rgba(0,180,255,0.06); }
.elem-id { font-family:'Orbitron'; font-size:10px; font-weight:700; color:var(--accent4); min-width:28px; }
.elem-info { flex:1; }
.elem-type { font-size:12px; font-weight:600; }
.elem-freq { font-family:'Share Tech Mono'; font-size:10px; color:var(--text2); }
.elem-rssi { font-family:'Share Tech Mono'; font-size:11px; min-width:70px; text-align:right; }
.locked { color:var(--accent2); }
.searching { color:var(--accent3); animation:pulse 1.5s infinite; }

/* Payload fusion */
.fusion-display {
  text-align:center; padding:24px;
  background:rgba(0,0,0,0.4); border-radius:6px;
  border:1px solid var(--border);
}
.fusion-coords { font-family:'Orbitron'; font-size:24px; font-weight:900; color:#fff; margin:12px 0; }
.fusion-error { font-family:'Share Tech Mono'; font-size:13px; color:var(--accent2); }

.scrollable { max-height:260px; overflow-y:auto; }
.scrollable::-webkit-scrollbar { width:4px; }
.scrollable::-webkit-scrollbar-track { background:transparent; }
.scrollable::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

.spinner { display:inline-block; width:12px; height:12px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; margin-left:8px; vertical-align:middle; }
@keyframes spin { to { transform:rotate(360deg); } }

.gnss-map-svg { width:100%; height:100%; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>SDA · <span>MULTI-DOMAIN</span> RF SIMULATOR</h1>
    <div class="header-sub">SOFTWARE DEFINED ANTENNA · GNSS + SATCOM + MIMO · v2.0</div>
  </div>
  <div class="status-bar">
    <div class="status-dot"></div>
    <div class="status-label">SYSTEM NOMINAL</div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('gnss')">① GNSS / IRNSS</div>
  <div class="tab" onclick="showTab('satcom')">② SATCOM SDR</div>
  <div class="tab" onclick="showTab('sda')">③ SDA CAPTURE</div>
  <div class="tab" onclick="showTab('payload')">④ PAYLOAD PROCESSOR</div>
</div>

<!-- ═══════════════ TAB 1: GNSS ═══════════════ -->
<div id="tab-gnss" class="main active">
  <div class="input-row" style="margin-bottom:20px;">
    <div class="input-group">
      <label>LATITUDE (°N)</label>
      <input type="number" id="lat" value="13.0827" step="0.0001" style="width:140px;">
    </div>
    <div class="input-group">
      <label>LONGITUDE (°E)</label>
      <input type="number" id="lon" value="80.2707" step="0.0001" style="width:140px;">
    </div>
    <button class="btn btn-green" onclick="runGNSS()">▶ SIMULATE GNSS <span id="gnss-spin" style="display:none;" class="spinner"></span></button>
    <button class="btn" onclick="useMyLocation()">⊕ USE MY LOCATION</button>
  </div>

  <div class="grid2" style="margin-bottom:20px;">
    <!-- Constellation Cards -->
    <div class="panel">
      <div class="panel-title">CONSTELLATION STATUS <span class="badge">MULTI-FREQ</span></div>
      <div class="const-grid" id="constCards">
        <div style="color:var(--text2); font-family:'Share Tech Mono'; font-size:11px; grid-column:1/-1; text-align:center; padding:20px;">
          Enter coordinates and click SIMULATE GNSS
        </div>
      </div>
    </div>

    <!-- Map -->
    <div class="panel">
      <div class="panel-title">POSITION MAP · DEVIATION OVERLAY <span class="badge">LIVE</span></div>
      <div id="gnssMap">
        <svg id="mapSvg" class="gnss-map-svg" viewBox="0 0 500 380" xmlns="http://www.w3.org/2000/svg">
          <rect width="500" height="380" fill="#050f1e"/>
          <text x="250" y="190" fill="#1a3a5c" font-family="Orbitron" font-size="13" text-anchor="middle">AWAITING COORDINATES</text>
        </svg>
      </div>
    </div>
  </div>

  <div class="grid2">
    <!-- Deviation Table -->
    <div class="panel">
      <div class="panel-title">POSITION DEVIATION BY CONSTELLATION <span class="badge">CEP</span></div>
      <div class="scrollable">
        <table class="dev-table" id="devTable">
          <thead><tr>
            <th>CONSTELLATION</th><th>FREQ (MHz)</th><th>BAND</th>
            <th>ΔLat (m)</th><th>ΔLon (m)</th><th>ERROR (m)</th><th>PDOP</th>
          </tr></thead>
          <tbody id="devTableBody">
            <tr><td colspan="7" style="text-align:center;color:var(--text2);padding:20px;">No data</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Sky Plot -->
    <div class="panel">
      <div class="panel-title">SATELLITE SKY PLOT <span class="badge">ELEVATION/AZIMUTH</span></div>
      <canvas id="skyPlot"></canvas>
    </div>
  </div>
</div>

<!-- ═══════════════ TAB 2: SATCOM ═══════════════ -->
<div id="tab-satcom" class="main">
  <!-- Band Visualizer -->
  <div class="panel" style="margin-bottom:20px;">
    <div class="panel-title">RF BAND ALLOCATION · 300 MHz – 12 GHz <span class="badge">CLICKABLE</span></div>
    <div class="band-viz" id="bandViz"></div>
    <div style="display:flex; gap:24px; flex-wrap:wrap; margin-top:8px;" id="bandLegend"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <div class="panel-title">SDR TUNER CONTROL <span class="badge">300MHz–12GHz</span></div>
      <div class="input-row">
        <div class="input-group" style="flex:1;">
          <label>CENTER FREQUENCY (MHz)</label>
          <input type="range" id="freqSlider" min="300" max="12000" value="1550" oninput="updateFreqDisplay()">
        </div>
        <div class="input-group">
          <label>FREQ</label>
          <input type="number" id="freqMhz" value="1550" min="300" max="12000" style="width:110px;" oninput="syncSlider()">
        </div>
      </div>
      <div class="input-row">
        <div class="input-group">
          <label>BW (MHz)</label>
          <input type="number" id="bwMhz" value="200" min="10" max="500" style="width:90px;">
        </div>
        <button class="btn btn-green" onclick="runSatCom()">▶ TUNE SDR <span id="sdr-spin" style="display:none;" class="spinner"></span></button>
      </div>
      <div class="metric-row" id="sdrMetrics">
        <div class="metric"><div class="metric-val" id="m-snr">—</div><div class="metric-label">SNR (dB)</div></div>
        <div class="metric"><div class="metric-val" id="m-power">—</div><div class="metric-label">SIG POWER</div></div>
        <div class="metric"><div class="metric-val" id="m-noise">—</div><div class="metric-label">NOISE FLOOR</div></div>
        <div class="metric"><div class="metric-val" id="m-mod">—</div><div class="metric-label">MODULATION</div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">POWER SPECTRAL DENSITY <span class="badge">PSD</span></div>
      <canvas id="spectrumChart"></canvas>
      <div id="bandInfo" style="margin-top:12px; font-family:'Share Tech Mono'; font-size:11px; color:var(--text2); line-height:1.6;"></div>
    </div>
  </div>
</div>

<!-- ═══════════════ TAB 3: SDA ═══════════════ -->
<div id="tab-sda" class="main">
  <div class="grid2">
    <div class="panel">
      <div class="panel-title">SDA ELEMENT CONTROL <span class="badge">MIMO</span></div>
      <div class="input-row">
        <div class="input-group">
          <label>OPERATING FREQ (MHz)</label>
          <input type="number" id="sdaFreq" value="1575" min="300" max="12000" style="width:130px;">
        </div>
        <button class="btn btn-green" onclick="runSDA()">▶ CAPTURE <span id="sda-spin" style="display:none;" class="spinner"></span></button>
      </div>
      <div id="elemStatus" style="margin-top:8px;">
        <div style="color:var(--text2); font-family:'Share Tech Mono'; font-size:11px; padding:16px; text-align:center;">Click CAPTURE to activate elements</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">MIMO CHANNEL MATRIX <span class="badge">H-MATRIX</span></div>
      <div id="mimoMatrix" style="display:flex; justify-content:center; padding:10px;"></div>
      <div class="metric-row" style="margin-top:16px;" id="mimoMetrics">
        <div class="metric"><div class="metric-val" id="m-cap">—</div><div class="metric-label">CAPACITY bps/Hz</div></div>
        <div class="metric"><div class="metric-val" id="m-cond">—</div><div class="metric-label">CONDITION №</div></div>
        <div class="metric"><div class="metric-val" id="m-ports">—</div><div class="metric-label">ACTIVE PORTS</div></div>
      </div>
    </div>
  </div>

  <!-- SDA Band diagram -->
  <div class="panel" style="margin-top:20px;">
    <div class="panel-title">SDA FREQUENCY COVERAGE MAP <span class="badge">ALL ELEMENTS</span></div>
    <canvas id="sdaBandChart" style="max-height:180px;"></canvas>
  </div>
</div>

<!-- ═══════════════ TAB 4: PAYLOAD ═══════════════ -->
<div id="tab-payload" class="main">
  <div style="display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap;">
    <button class="btn btn-green" onclick="runFullChain()">▶ RUN FULL CHAIN <span id="chain-spin" style="display:none;" class="spinner"></span></button>
    <div style="font-family:'Share Tech Mono'; font-size:11px; color:var(--text2); display:flex; align-items:center;">
      Runs all 3 components and fuses results
    </div>
  </div>

  <div class="grid2" style="margin-bottom:20px;">
    <div class="panel">
      <div class="panel-title">FUSED NAVIGATION SOLUTION <span class="badge">MULTI-CONST</span></div>
      <div class="fusion-display">
        <div style="font-family:'Share Tech Mono'; font-size:10px; color:var(--text2); letter-spacing:2px;">FUSED POSITION</div>
        <div class="fusion-coords" id="fusedCoords">— ° N &nbsp; — ° E</div>
        <div class="fusion-error" id="fusedError">CEP: — m</div>
      </div>
      <div class="metric-row" style="margin-top:16px;">
        <div class="metric"><div class="metric-val" id="p-consts">—</div><div class="metric-label">CONSTELLATIONS</div></div>
        <div class="metric"><div class="metric-val" id="p-link">—</div><div class="metric-label">SATCOM LINK</div></div>
        <div class="metric"><div class="metric-val" id="p-mimo">—</div><div class="metric-label">MIMO bps/Hz</div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">CONSTELLATION ACCURACY COMPARISON <span class="badge">BAR CHART</span></div>
      <canvas id="satChart" style="max-height:220px;"></canvas>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">SYSTEM RECOMMENDATION ENGINE <span class="badge">AI ASSIST</span></div>
    <div id="recommendation" style="font-family:'Share Tech Mono'; font-size:12px; color:var(--accent2); padding:16px; background:rgba(0,255,136,0.04); border:1px solid rgba(0,255,136,0.15); border-radius:4px; line-height:1.8;">
      Run the full chain to generate recommendations.
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let gnssData = null, satcomData = null, sdaData = null;
let specChart = null, skyChart = null, satAccChart = null, sdaBandChart = null;

const BAND_COLORS = {
  "UHF":"#ff6b35","L-Band":"#ffd700","S-Band":"#00ff88","C-Band":"#00ccff","X-Band":"#ff4488"
};
const CONST_COLORS = {
  GPS_L1:"#00FF88",GPS_L5:"#00CCFF",GLONASS:"#FF6B35",GALILEO:"#FFD700",BEIDOU:"#FF4488",NavIC:"#AA88FF"
};

// ── Tab Switch ─────────────────────────────────────────────────────────────
function showTab(t) {
  document.querySelectorAll('.main').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  event.target.classList.add('active');
}

// ── Band Viz (SatCom) ──────────────────────────────────────────────────────
function buildBandViz() {
  const bands = [
    {name:"UHF",min:300,max:1000,color:"#ff6b35",use:"MILSATCOM / MUOS"},
    {name:"L-Band",min:1000,max:2000,color:"#ffd700",use:"Iridium / GPS"},
    {name:"S-Band",min:2000,max:4000,color:"#00ff88",use:"Weather / Radar"},
    {name:"C-Band",min:4000,max:8000,color:"#00ccff",use:"VSAT / FSS"},
    {name:"X-Band",min:8000,max:12000,color:"#ff4488",use:"Military / SAR"},
  ];
  const total = 12000-300;
  const viz = document.getElementById('bandViz');
  const leg = document.getElementById('bandLegend');
  viz.innerHTML=''; leg.innerHTML='';
  bands.forEach(b=>{
    const pct = (b.max-b.min)/total*100;
    const seg = document.createElement('div');
    seg.className='band-seg';
    seg.style.cssText=`width:${pct}%; background:${b.color}22; border-left:3px solid ${b.color};`;
    seg.innerHTML=`<span>${b.name}</span><span class="band-freq">${b.min}–${b.max} MHz</span>`;
    seg.onclick=()=>{ document.getElementById('freqMhz').value=Math.round((b.min+b.max)/2); syncSlider(); runSatCom(); };
    viz.appendChild(seg);
    const li = document.createElement('div');
    li.style.cssText=`display:flex;align-items:center;gap:6px;font-family:Share Tech Mono;font-size:10px;color:var(--text2);`;
    li.innerHTML=`<span style="width:10px;height:10px;border-radius:2px;background:${b.color};display:inline-block;"></span>${b.name}: ${b.use}`;
    leg.appendChild(li);
  });
}

// ── Freq Sync ──────────────────────────────────────────────────────────────
function updateFreqDisplay() {
  document.getElementById('freqMhz').value = document.getElementById('freqSlider').value;
}
function syncSlider() {
  document.getElementById('freqSlider').value = document.getElementById('freqMhz').value;
}

// ── Use My Location ────────────────────────────────────────────────────────
function useMyLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(p=>{
      document.getElementById('lat').value = p.coords.latitude.toFixed(4);
      document.getElementById('lon').value = p.coords.longitude.toFixed(4);
    });
  }
}

// ── GNSS Simulation ────────────────────────────────────────────────────────
async function runGNSS() {
  const lat = parseFloat(document.getElementById('lat').value);
  const lon = parseFloat(document.getElementById('lon').value);
  document.getElementById('gnss-spin').style.display='inline-block';
  try {
    const r = await fetch('/api/gnss', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({lat,lon})});
    gnssData = await r.json();
    renderConstellationCards(gnssData);
    renderDeviationTable(gnssData);
    renderMap(gnssData);
    renderSkyPlot(gnssData);
  } finally { document.getElementById('gnss-spin').style.display='none'; }
}

function renderConstellationCards(d) {
  const el = document.getElementById('constCards');
  el.innerHTML = '';
  for (const [name, c] of Object.entries(d.constellations)) {
    const div = document.createElement('div');
    div.className = 'const-card active-const';
    div.style.borderTopColor = c.color;
    div.innerHTML = `
      <div class="const-name" style="color:${c.color}">${name}</div>
      <div class="const-freq">${c.freq_mhz} MHz · ${c.band}</div>
      <div class="const-sats" style="color:${c.color}">${c.visible_sats}</div>
      <div style="font-size:10px;color:var(--text2)">/${c.total_sats} sats visible</div>
      <div class="const-error" style="color:var(--text2)">ERR: ${c.position.error_m}m · PDOP:${c.pdop}</div>`;
    div.style.setProperty('--card-color', c.color);
    div.querySelector('.const-card')?.setAttribute('style', `border-top: 2px solid ${c.color}`);
    el.appendChild(div);
  }
}

function renderDeviationTable(d) {
  const tbody = document.getElementById('devTableBody');
  tbody.innerHTML = '';
  for (const [name, c] of Object.entries(d.constellations)) {
    const p = c.position;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:${c.color};font-weight:700;">${name}</td>
      <td>${c.freq_mhz}</td>
      <td>${c.band}</td>
      <td style="color:${Math.abs(p.dlat_m)>3?'#ff6b35':'var(--accent2)'}">${p.dlat_m>0?'+':''}${p.dlat_m}</td>
      <td style="color:${Math.abs(p.dlon_m)>3?'#ff6b35':'var(--accent2)'}">${p.dlon_m>0?'+':''}${p.dlon_m}</td>
      <td style="color:${p.error_m>5?'#ff4488':'var(--accent2)'};font-weight:700;">${p.error_m}m</td>
      <td>${c.pdop}</td>`;
    tbody.appendChild(tr);
  }
}

function renderMap(d) {
  const svg = document.getElementById('mapSvg');
  const W=500, H=380;
  const lat0=d.lat, lon0=d.lon;
  // Lat/lon to pixel (simple linear map within ±0.001 deg)
  const scale = 40000; // pixels per degree
  function toXY(lat, lon) {
    const x = W/2 + (lon-lon0)*scale;
    const y = H/2 - (lat-lat0)*scale;
    return [x, y];
  }
  let markup = `<rect width="${W}" height="${H}" fill="#050f1e"/>`;
  // Grid
  for (let i=0;i<10;i++) {
    markup += `<line x1="${i*W/9}" y1="0" x2="${i*W/9}" y2="${H}" stroke="rgba(0,180,255,0.06)" stroke-width="1"/>`;
    markup += `<line x1="0" y1="${i*H/9}" x2="${W}" y2="${i*H/9}" stroke="rgba(0,180,255,0.06)" stroke-width="1"/>`;
  }
  // Deviation pins
  for (const [name, c] of Object.entries(d.constellations)) {
    const [x,y] = toXY(c.position.lat, c.position.lon);
    if (x<0||x>W||y<0||y>H) continue;
    markup += `<line x1="${W/2}" y1="${H/2}" x2="${x}" y2="${y}" stroke="${c.color}" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="4,3"/>`;
    markup += `<circle cx="${x}" cy="${y}" r="6" fill="${c.color}" fill-opacity="0.25" stroke="${c.color}" stroke-width="1.5"/>`;
    markup += `<circle cx="${x}" cy="${y}" r="3" fill="${c.color}"/>`;
    markup += `<text x="${x+9}" y="${y+4}" fill="${c.color}" font-family="Share Tech Mono" font-size="9">${name}</text>`;
  }
  // True position
  markup += `<circle cx="${W/2}" cy="${H/2}" r="18" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`;
  markup += `<circle cx="${W/2}" cy="${H/2}" r="8" fill="#ffffff22" stroke="white" stroke-width="2"/>`;
  markup += `<line x1="${W/2-14}" y1="${H/2}" x2="${W/2+14}" y2="${H/2}" stroke="white" stroke-width="1.5"/>`;
  markup += `<line x1="${W/2}" y1="${H/2-14}" x2="${W/2}" y2="${H/2+14}" stroke="white" stroke-width="1.5"/>`;
  markup += `<text x="${W/2+12}" y="${H/2-12}" fill="white" font-family="Orbitron" font-size="10" font-weight="700">TRUE POS</text>`;
  markup += `<text x="${W/2+12}" y="${H/2+2}" fill="rgba(255,255,255,0.6)" font-family="Share Tech Mono" font-size="9">${lat0.toFixed(4)}°N</text>`;
  markup += `<text x="${W/2+12}" y="${H/2+13}" fill="rgba(255,255,255,0.6)" font-family="Share Tech Mono" font-size="9">${lon0.toFixed(4)}°E</text>`;
  svg.innerHTML = markup;
}

function renderSkyPlot(d) {
  const canvas = document.getElementById('skyPlot');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.parentElement.clientWidth;
  const H = canvas.height = W;
  const cx=W/2, cy=H/2, r=W*0.42;
  ctx.fillStyle='#050f1e'; ctx.fillRect(0,0,W,H);
  // Rings
  [1,0.66,0.33].forEach((f,i)=>{
    ctx.beginPath(); ctx.arc(cx,cy,r*f,0,Math.PI*2);
    ctx.strokeStyle='rgba(0,180,255,0.15)'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle='rgba(0,180,255,0.3)'; ctx.font='9px Share Tech Mono';
    ctx.fillText(`${[90,60,30][i]}°`, cx+r*f+3, cy);
  });
  // Axes
  ctx.strokeStyle='rgba(0,180,255,0.1)'; ctx.lineWidth=1;
  [[cx-r,cy,cx+r,cy],[cx,cy-r,cx,cy+r]].forEach(([x1,y1,x2,y2])=>{
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  });
  ctx.fillStyle='rgba(0,180,255,0.4)'; ctx.font='10px Orbitron';
  ctx.fillText('N',cx-4,cy-r-5); ctx.fillText('S',cx-4,cy+r+14);
  ctx.fillText('E',cx+r+5,cy+4); ctx.fillText('W',cx-r-18,cy+4);
  // Satellites
  for (const [name, c] of Object.entries(d.constellations)) {
    c.satellites.forEach(sat=>{
      const el = sat.el * Math.PI/180;
      const az = sat.az * Math.PI/180;
      const dist = r * (1 - el/(Math.PI/2));
      const x = cx + dist*Math.sin(az);
      const y = cy - dist*Math.cos(az);
      ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2);
      ctx.fillStyle=c.color+'cc'; ctx.fill();
      ctx.strokeStyle=c.color; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle=c.color; ctx.font='7px Share Tech Mono';
      ctx.fillText(sat.id,x+5,y-3);
    });
  }
}

// ── SatCom Simulation ──────────────────────────────────────────────────────
async function runSatCom() {
  const freq = parseFloat(document.getElementById('freqMhz').value);
  const bw = parseFloat(document.getElementById('bwMhz').value);
  document.getElementById('sdr-spin').style.display='inline-block';
  try {
    const r = await fetch('/api/satcom', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({freq_mhz:freq,bw_mhz:bw})});
    satcomData = await r.json();
    document.getElementById('m-snr').textContent = satcomData.snr_db+'dB';
    document.getElementById('m-power').textContent = satcomData.signal_power_dbm+'dBm';
    document.getElementById('m-noise').textContent = satcomData.noise_floor_dbm+'dBm';
    document.getElementById('m-mod').textContent = satcomData.modulation;
    document.getElementById('bandInfo').innerHTML = `
      <b style="color:${satcomData.band.color}">${satcomData.band.name}</b> · ${satcomData.band.min_mhz}–${satcomData.band.max_mhz} MHz<br>
      USE: ${satcomData.band.use}<br>
      MODULATION: ${satcomData.modulation} &nbsp;|&nbsp; CODING: ${satcomData.coding_rate} &nbsp;|&nbsp; SR: ${satcomData.symbol_rate_msps} MSps`;
    renderSpectrum(satcomData);
  } finally { document.getElementById('sdr-spin').style.display='none'; }
}

function renderSpectrum(d) {
  const ctx = document.getElementById('spectrumChart').getContext('2d');
  if (specChart) specChart.destroy();
  specChart = new Chart(ctx, {
    type:'line',
    data:{
      labels: d.spectrum.freqs.map(f=>f.toFixed(0)),
      datasets:[{
        data: d.spectrum.power,
        borderColor: d.band.color,
        backgroundColor: d.band.color+'22',
        borderWidth:1.5, pointRadius:0, fill:true, tension:0.3
      }]
    },
    options:{
      responsive:true, animation:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#6a9fc0',maxTicksLimit:8,font:{family:'Share Tech Mono',size:9}},grid:{color:'rgba(0,180,255,0.06)'}},
        y:{ticks:{color:'#6a9fc0',font:{family:'Share Tech Mono',size:9}},grid:{color:'rgba(0,180,255,0.06)'},title:{display:true,text:'Power (dBm)',color:'#6a9fc0',font:{family:'Share Tech Mono',size:9}}}
      }
    }
  });
}

// ── SDA Simulation ─────────────────────────────────────────────────────────
async function runSDA() {
  const freq = parseFloat(document.getElementById('sdaFreq').value);
  document.getElementById('sda-spin').style.display='inline-block';
  try {
    const r = await fetch('/api/sda', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({freq_mhz:freq})});
    sdaData = await r.json();
    renderElements(sdaData);
    renderMIMO(sdaData);
    renderSDABands();
  } finally { document.getElementById('sda-spin').style.display='none'; }
}

function renderElements(d) {
  const el = document.getElementById('elemStatus');
  el.innerHTML = d.captures.map(c=>`
    <div class="elem-row">
      <div class="elem-id">${c.element.id}</div>
      <div class="elem-info">
        <div class="elem-type">${c.element.type}</div>
        <div class="elem-freq">${c.element.freq_range} · ${c.element.polarization} · MIMO Port ${c.element.mimo_port}</div>
      </div>
      <div class="elem-rssi" style="color:${c.rssi_dbm>-60?'var(--accent2)':'var(--accent3)'}">${c.rssi_dbm}dBm</div>
      <div style="margin-left:8px;font-family:'Share Tech Mono';font-size:10px;" class="${c.status==='LOCKED'?'locked':'searching'}">${c.status}</div>
    </div>`).join('');
}

function renderMIMO(d) {
  const n = d.captures.length;
  const H = d.mimo.magnitude;
  const mx = Math.max(...H.flat());
  const container = document.getElementById('mimoMatrix');
  const size = Math.min(220, container.clientWidth||220);
  const cell = Math.floor(size/n);
  let html = `<div style="display:grid;grid-template-columns:repeat(${n},${cell}px);gap:2px;">`;
  for (let i=0;i<n;i++) for(let j=0;j<n;j++){
    const v=H[i][j], norm=v/mx;
    const hue = norm*120; // green=high, red=low
    html+=`<div class="mimo-cell" style="width:${cell}px;background:hsl(${hue},80%,${15+norm*25}%);color:rgba(255,255,255,0.7)">${v.toFixed(1)}</div>`;
  }
  html+='</div>';
  container.innerHTML=html;
  document.getElementById('m-cap').textContent = d.mimo.capacity_bps_hz;
  document.getElementById('m-cond').textContent = d.mimo.condition_number;
  document.getElementById('m-ports').textContent = d.active_elements;
}

function renderSDABands() {
  const elements = [
    {id:'E1',label:'Helix Axial',min:1000,max:2000,color:'#ffd700'},
    {id:'E2',label:'Helix Normal',min:300,max:1000,color:'#ff6b35'},
    {id:'E3',label:'Linear Mono',min:300,max:3000,color:'#00ff88'},
    {id:'E4',label:'Patch Array',min:1100,max:2000,color:'#00ccff'},
    {id:'E5',label:'Horn SHF',min:8000,max:12000,color:'#ff4488'},
    {id:'E6',label:'Spiral WB',min:1000,max:8000,color:'#aa88ff'},
  ];
  const ctx = document.getElementById('sdaBandChart').getContext('2d');
  if (sdaBandChart) sdaBandChart.destroy();
  sdaBandChart = new Chart(ctx, {
    type:'bar',
    data:{
      labels: elements.map(e=>e.id+' '+e.label),
      datasets:[{
        label:'Start',
        data: elements.map(e=>e.min),
        backgroundColor:'transparent',
        borderColor:'transparent',
      },{
        label:'Coverage (MHz)',
        data: elements.map(e=>e.max-e.min),
        backgroundColor: elements.map(e=>e.color+'88'),
        borderColor: elements.map(e=>e.color),
        borderWidth:2,
        borderRadius:3,
      }]
    },
    options:{
      indexAxis:'y', responsive:true, animation:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{stacked:true,min:0,max:12000,ticks:{color:'#6a9fc0',font:{family:'Share Tech Mono',size:9}},grid:{color:'rgba(0,180,255,0.06)'},title:{display:true,text:'Frequency (MHz)',color:'#6a9fc0',font:{family:'Share Tech Mono'}}},
        y:{stacked:true,ticks:{color:'#c8e8ff',font:{family:'Share Tech Mono',size:10}},grid:{display:false}}
      }
    }
  });
}

// ── Full Chain / Payload ───────────────────────────────────────────────────
async function runFullChain() {
  document.getElementById('chain-spin').style.display='inline-block';
  const lat=parseFloat(document.getElementById('lat').value||13.0827);
  const lon=parseFloat(document.getElementById('lon').value||80.2707);
  try {
    // Run all 3
    const [gr, sr, sdar] = await Promise.all([
      fetch('/api/gnss',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat,lon})}),
      fetch('/api/satcom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({freq_mhz:1550,bw_mhz:200})}),
      fetch('/api/sda',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({freq_mhz:1575})}),
    ]);
    gnssData = await gr.json(); satcomData = await sr.json(); sdaData = await sdar.json();
    // Payload fusion
    const pr = await fetch('/api/payload',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lat,lon,gnss:gnssData,satcom:satcomData,sda:sdaData})});
    const pd = await pr.json();
    document.getElementById('fusedCoords').innerHTML = `${pd.fused_position.lat.toFixed(6)}° N &nbsp; ${pd.fused_position.lon.toFixed(6)}° E`;
    document.getElementById('fusedError').textContent = `CEP: ${pd.fused_position.error_m} m`;
    document.getElementById('p-consts').textContent = pd.total_constellations;
    document.getElementById('p-link').textContent = pd.satcom_link;
    document.getElementById('p-mimo').textContent = pd.sda_mimo_capacity;
    document.getElementById('recommendation').innerHTML = `
      ▶ GNSS HEALTH: <span style="color:var(--accent2)">${pd.gnss_health}</span><br>
      ▶ SATCOM LINK: <span style="color:${pd.satcom_link==='ACTIVE'?'var(--accent2)':'var(--accent3)'}">${pd.satcom_link}</span><br>
      ▶ MIMO CAPACITY: <span style="color:var(--accent)">${pd.sda_mimo_capacity} bps/Hz</span><br>
      ▶ RECOMMENDATION: <span style="color:var(--accent4)">${pd.recommendation}</span>`;
    renderAccuracyChart(gnssData);
  } finally { document.getElementById('chain-spin').style.display='none'; }
}

function renderAccuracyChart(d) {
  const ctx = document.getElementById('satChart').getContext('2d');
  if (satAccChart) satAccChart.destroy();
  const names=[], errors=[], colors=[];
  for(const [name,c] of Object.entries(d.constellations)){
    names.push(name); errors.push(c.position.error_m); colors.push(c.color);
  }
  satAccChart = new Chart(ctx,{
    type:'bar',
    data:{labels:names, datasets:[{label:'Position Error (m)',data:errors,backgroundColor:colors.map(c=>c+'88'),borderColor:colors,borderWidth:2,borderRadius:4}]},
    options:{
      responsive:true, animation:{duration:600},
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#c8e8ff',font:{family:'Share Tech Mono',size:10}},grid:{color:'rgba(0,180,255,0.06)'}},
        y:{ticks:{color:'#6a9fc0',font:{family:'Share Tech Mono',size:9}},grid:{color:'rgba(0,180,255,0.06)'},title:{display:true,text:'Error (m)',color:'#6a9fc0'}}
      }
    }
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
buildBandViz();
renderSDABands();
</script>
</body>
</html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
