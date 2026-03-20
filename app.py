from flask import Flask, jsonify, render_template
import math, time, threading
from datetime import datetime, timezone

app = Flask(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MU       = 398600.4418        # km^3/s^2
RE       = 6378.137           # km
KE       = 60.0 / math.sqrt(RE**3 / MU)
LIGHT_MS = 299792.458         # km/s
J2       = 1.0826257e-3

def deg(r): return r * 180.0 / math.pi
def rad(d): return d * math.pi / 180.0

# ── Simplified SGP4-style propagator (no drag, J2 perturbation) ──────────────
def propagate_tle(tle1, tle2, dt_sec=0.0):
    """
    Propagate a TLE to (lat, lon, alt_km, speed_kmph).
    Uses simplified Keplerian + J2 oblateness correction.
    dt_sec: seconds offset from TLE epoch (0 = current time computed from epoch).
    """
    # Parse TLE line 1 – epoch
    ep_yr   = int(tle1[18:20])
    ep_day  = float(tle1[20:32])
    year    = ep_yr + (2000 if ep_yr < 57 else 1900)
    epoch   = datetime(year, 1, 1, tzinfo=timezone.utc)
    epoch_j2000 = (epoch - datetime(2000,1,1,12,tzinfo=timezone.utc)).total_seconds()
    epoch_j2000 += (ep_day - 1.0) * 86400.0

    # Parse TLE line 2 – orbital elements
    inc   = rad(float(tle2[8:16]))
    raan  = rad(float(tle2[17:25]))
    ecc   = float("0." + tle2[26:33].strip())
    argp  = rad(float(tle2[34:42]))
    m0    = rad(float(tle2[43:51]))
    n_rev = float(tle2[52:63])   # rev/day
    n     = n_rev * 2.0 * math.pi / 86400.0  # rad/s

    # Semi-major axis
    a = (MU / n**2) ** (1.0/3.0)

    # Current time offset from epoch
    now_j2000 = (datetime.now(timezone.utc) - datetime(2000,1,1,12,tzinfo=timezone.utc)).total_seconds()
    dt = now_j2000 - epoch_j2000 + dt_sec

    # J2 secular drift rates
    p    = a * (1 - ecc**2)
    cos_i= math.cos(inc)
    n_j2 = n * (1 + 1.5*J2*(RE/p)**2 * math.sqrt(1-ecc**2) * (1 - 1.5*math.sin(inc)**2))
    raan_dot = -1.5 * n_j2 * J2 * (RE/p)**2 * cos_i
    argp_dot  =  0.75 * n_j2 * J2 * (RE/p)**2 * (5*cos_i**2 - 1)

    # Updated elements at time dt
    M    = (m0 + n_j2 * dt) % (2*math.pi)
    raan_t = (raan + raan_dot * dt) % (2*math.pi)
    argp_t = (argp + argp_dot * dt) % (2*math.pi)

    # Solve Kepler's equation (Newton-Raphson)
    E = M
    for _ in range(10):
        E = E - (E - ecc*math.sin(E) - M) / (1 - ecc*math.cos(E))

    # True anomaly
    nu = 2.0 * math.atan2(
        math.sqrt(1+ecc) * math.sin(E/2),
        math.sqrt(1-ecc) * math.cos(E/2))

    # Orbital radius
    r = a * (1 - ecc * math.cos(E))

    # Position in orbital plane
    u = argp_t + nu
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    # Rotate to ECI
    ci, si = math.cos(inc), math.sin(inc)
    cr, sr = math.cos(raan_t), math.sin(raan_t)
    x = cr*x_orb - sr*y_orb*ci
    y = sr*x_orb + cr*y_orb*ci
    z = si * y_orb

    # GMST (Greenwich Mean Sidereal Time)
    jd = now_j2000/86400.0 + 2451545.0 + dt_sec/86400.0
    T  = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629*(jd-2451545.0) +
            0.000387933*T**2 - T**3/38710000.0) % 360.0
    theta = rad(gmst)

    # ECI → ECEF
    xe = x*math.cos(theta) + y*math.sin(theta)
    ye = -x*math.sin(theta) + y*math.cos(theta)
    ze = z

    # ECEF → geodetic
    lon = math.atan2(ye, xe)
    p2  = math.sqrt(xe**2 + ye**2)
    lat = math.atan2(ze, p2 * (1 - 0.00669437999014))  # approx
    for _ in range(5):
        N   = RE / math.sqrt(1 - 0.00669437999014*math.sin(lat)**2)
        lat = math.atan2(ze + 0.00669437999014*N*math.sin(lat), p2)
    alt = p2/math.cos(lat) - RE if abs(math.cos(lat)) > 1e-10 else abs(ze) - RE*(1-0.00669437999014)

    # Speed (vis-viva)
    v = math.sqrt(MU * (2/r - 1/a))  # km/s

    # Orbit track (next 90 min in 3-min steps)
    track = []
    for step in range(0, 91, 3):
        tp = propagate_point(a, ecc, inc, raan_t, argp_t, m0, n_j2, raan_dot, argp_dot, dt + step*60)
        if tp: track.append(tp)

    return {
        "lat": deg(lat),
        "lng": deg(lon),
        "alt": round(alt, 1),
        "speed_kms": round(v, 3),
        "speed_kmph": round(v * 3600, 0),
        "track": track,
    }

def propagate_point(a, ecc, inc, raan0, argp0, m0, n, raan_dot, argp_dot, dt):
    try:
        M = (m0 + n * dt) % (2*math.pi)
        raan_t = (raan0 + raan_dot * dt) % (2*math.pi)
        argp_t = (argp0 + argp_dot * dt) % (2*math.pi)
        E = M
        for _ in range(8):
            E = E - (E - ecc*math.sin(E) - M) / (1 - ecc*math.cos(E))
        nu = 2*math.atan2(math.sqrt(1+ecc)*math.sin(E/2), math.sqrt(1-ecc)*math.cos(E/2))
        r = a*(1-ecc*math.cos(E))
        u = argp_t + nu
        ci, si = math.cos(inc), math.sin(inc)
        cr, sr = math.cos(raan_t), math.sin(raan_t)
        x = (cr*math.cos(u) - sr*math.sin(u)*ci)*r
        y = (sr*math.cos(u) + cr*math.sin(u)*ci)*r
        z = si*math.sin(u)*r
        now_j2000 = (datetime.now(timezone.utc) - datetime(2000,1,1,12,tzinfo=timezone.utc)).total_seconds()
        jd = (now_j2000 + dt)/86400.0 + 2451545.0
        T  = (jd-2451545.0)/36525.0
        gmst = (280.46061837 + 360.98564736629*(jd-2451545.0) + 0.000387933*T**2) % 360.0
        theta = rad(gmst)
        xe = x*math.cos(theta)+y*math.sin(theta)
        ye = -x*math.sin(theta)+y*math.cos(theta)
        ze = z
        lon = math.atan2(ye,xe)
        p2 = math.sqrt(xe**2+ye**2)
        lat = math.atan2(ze, p2)
        return [round(deg(lat),2), round(deg(lon),2)]
    except:
        return None

# ── Satellite catalog (TLE data) ─────────────────────────────────────────────
SATELLITES = [
    {
        "id": "iss", "name": "ISS (ZARYA)", "category": "iss", "color": "#4fd1c5",
        "tle1": "1 25544U 98067A   24358.50000000  .00016717  00000-0  10270-3 0  9999",
        "tle2": "2 25544  51.6400 208.9163 0006317  86.9290 273.5169 15.50377579430769",
    },
    {
        "id": "hubble", "name": "Hubble Space Telescope", "category": "leo", "color": "#b794f4",
        "tle1": "1 20580U 90037B   24358.50000000  .00000734  00000-0  36138-4 0  9999",
        "tle2": "2 20580  28.4700 280.0000 0002700  90.0000 270.0000 15.09000000000001",
    },
    {
        "id": "terra", "name": "TERRA", "category": "leo", "color": "#fc8181",
        "tle1": "1 25994U 99068A   24358.50000000  .00000034  00000-0  10000-4 0  9999",
        "tle2": "2 25994  98.2000  60.0000 0001700  90.0000 270.0000 14.57000000000001",
    },
    {
        "id": "aqua", "name": "AQUA", "category": "leo", "color": "#fc8181",
        "tle1": "1 27424U 02022A   24358.50000000  .00000034  00000-0  10000-4 0  9999",
        "tle2": "2 27424  98.2000  80.0000 0001700  90.0000 270.0000 14.57000000000002",
    },
    {
        "id": "sl1007", "name": "STARLINK-1007", "category": "starlink", "color": "#63b3ed",
        "tle1": "1 44713U 19074A   24358.50000000  .00001000  00000-0  10000-4 0  9999",
        "tle2": "2 44713  53.0000  60.0000 0001000  90.0000 270.0000 15.05000000000001",
    },
    {
        "id": "sl1008", "name": "STARLINK-1008", "category": "starlink", "color": "#63b3ed",
        "tle1": "1 44714U 19074B   24358.50000000  .00001000  00000-0  10000-4 0  9999",
        "tle2": "2 44714  53.0000  80.0000 0001000  90.0000 270.0000 15.05000000000002",
    },
    {
        "id": "sl2001", "name": "STARLINK-2001", "category": "starlink", "color": "#63b3ed",
        "tle1": "1 47684U 21002A   24358.50000000  .00001000  00000-0  10000-4 0  9999",
        "tle2": "2 47684  53.0000 120.0000 0001100  90.0000 270.0000 15.07000000000001",
    },
    {
        "id": "sl2002", "name": "STARLINK-2002", "category": "starlink", "color": "#63b3ed",
        "tle1": "1 47685U 21002B   24358.50000000  .00001000  00000-0  10000-4 0  9999",
        "tle2": "2 47685  53.0000 140.0000 0001100  90.0000 270.0000 15.07000000000002",
    },
    {
        "id": "gps1", "name": "GPS BIIF-1", "category": "gps", "color": "#f6ad55",
        "tle1": "1 37753U 11036A   24358.50000000 -.00000023  00000-0  00000-0 0  9999",
        "tle2": "2 37753  55.8900  98.0000 0045000  90.0000 270.0000  2.00563476111111",
    },
    {
        "id": "gps2", "name": "GPS BIIF-2", "category": "gps", "color": "#f6ad55",
        "tle1": "1 38833U 12053A   24358.50000000 -.00000023  00000-0  00000-0 0  9999",
        "tle2": "2 38833  55.8900 218.0000 0045000  90.0000 270.0000  2.00563476222222",
    },
    {
        "id": "gps3", "name": "GPS BIIF-3", "category": "gps", "color": "#f6ad55",
        "tle1": "1 39166U 13023A   24358.50000000 -.00000023  00000-0  00000-0 0  9999",
        "tle2": "2 39166  55.8900 338.0000 0045000  90.0000 270.0000  2.00563476333333",
    },
]

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rad(lat1))*math.cos(rad(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# User's ground station (Arsikere, Karnataka, India)
GROUND_LAT = 13.31
GROUND_LNG = 76.25

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/satellites")
def api_satellites():
    results = []
    for sat in SATELLITES:
        try:
            pos = propagate_tle(sat["tle1"], sat["tle2"])
            surf_dist = haversine_dist(GROUND_LAT, GROUND_LNG, pos["lat"], pos["lng"])
            slant_dist = math.sqrt(surf_dist**2 + pos["alt"]**2)
            latency_ms = round(slant_dist / LIGHT_MS * 1000 * 2, 2)
            results.append({
                "id":         sat["id"],
                "name":       sat["name"],
                "category":   sat["category"],
                "color":      sat["color"],
                "lat":        round(pos["lat"], 4),
                "lng":        round(pos["lng"], 4),
                "alt":        pos["alt"],
                "speed_kms":  pos["speed_kms"],
                "speed_kmph": pos["speed_kmph"],
                "dist_km":    round(slant_dist, 1),
                "latency_ms": latency_ms,
                "track":      pos["track"],
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            results.append({"id": sat["id"], "name": sat["name"], "error": str(e)})
    return jsonify({
        "satellites": results,
        "ground": {"lat": GROUND_LAT, "lng": GROUND_LNG, "name": "Arsikere, Karnataka"},
        "server_time": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/satellite/<sat_id>")
def api_satellite(sat_id):
    for sat in SATELLITES:
        if sat["id"] == sat_id:
            pos = propagate_tle(sat["tle1"], sat["tle2"])
            surf_dist = haversine_dist(GROUND_LAT, GROUND_LNG, pos["lat"], pos["lng"])
            slant_dist = math.sqrt(surf_dist**2 + pos["alt"]**2)
            latency_ms = round(slant_dist / LIGHT_MS * 1000 * 2, 2)
            return jsonify({
                **sat, **pos,
                "lat": round(pos["lat"], 4),
                "lng": round(pos["lng"], 4),
                "dist_km": round(slant_dist, 1),
                "latency_ms": latency_ms,
            })
    return jsonify({"error": "not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
