#!/usr/bin/env python3
"""
Mumbai hourly Air Quality forecasting + mapping.

What it does, every run:
  1. Pulls ~60 days of hourly air quality + weather for 18 points across Mumbai
     (Open-Meteo Air Quality + Forecast APIs: free, no API key).
  2. Trains a gradient-boosted model that forecasts PM2.5 1..48 hours ahead,
     using recent PM2.5 history, co-pollutants, and *forecast* weather
     (wind, rain, humidity, pressure, cloud) as inputs.
  3. Adds a likely range (3rd-97th percentile of the model) to every forecast.
  4. Converts PM2.5 to an Indian CPCB-style AQI and spatially interpolates
     (inverse-distance weighting) the point forecasts onto a grid.
  5. Backtests over 3 rolling weeks against two baselines and writes the results.
  6. Writes forecast.csv and a self-contained interactive map (index.html)
     with an hour slider.

Usage:
  python mumbai_aqi.py                 # run once
  python mumbai_aqi.py --loop          # run, then re-run at hh:05 every hour
  python mumbai_aqi.py --quick         # skip the backtest (faster)
  python mumbai_aqi.py --demo          # offline run on synthetic data (no internet)
  python mumbai_aqi.py --blend 0.7     # 70% own model + 30% CAMS forecast

Cron alternative (hourly at minute 5):
  5 * * * * cd /path/to/mumbai_aqi && python mumbai_aqi.py >> run.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TZ = "Asia/Kolkata"
HORIZONS = list(range(1, 49))          # forecast 1..48 h ahead
HISTORY_DAYS = 60                      # training history (API max is 92)
FOLD_DAYS = 7                          # each backtest fold validates on 7 days
N_FOLDS = 3                            # 3 rolling folds = last 21 days

# (lat, lon) of representative points across Mumbai. Add/remove freely.
LOCATIONS = {
    "Colaba":        (18.9067, 72.8147),
    "Fort":          (18.9398, 72.8355),
    "Byculla":       (18.9764, 72.8327),
    "Worli":         (19.0176, 72.8187),
    "Dadar":         (19.0178, 72.8478),
    "Sion":          (19.0390, 72.8619),
    "Bandra":        (19.0596, 72.8295),
    "Kurla":         (19.0728, 72.8826),
    "Chembur":       (19.0522, 72.9005),
    "Deonar":        (19.0440, 72.9258),
    "Ghatkopar":     (19.0860, 72.9081),
    "Andheri":       (19.1136, 72.8697),
    "Powai":         (19.1176, 72.9060),
    "Bhandup":       (19.1440, 72.9370),
    "Malad":         (19.1874, 72.8484),
    "Mulund":        (19.1726, 72.9425),
    "Borivali":      (19.2307, 72.8567),
    "Vashi":         (19.0771, 72.9986),
}

# Map grid
LAT_MIN, LAT_MAX = 18.88, 19.27
LON_MIN, LON_MAX = 72.77, 73.03
GRID_NY, GRID_NX = 34, 26
MAX_CELL_DISTANCE_KM = 5.0             # don't paint cells far from any point (sea, etc.)

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover"]
RENAME = {
    "pm2_5": "pm25", "pm10": "pm10", "nitrogen_dioxide": "no2", "ozone": "o3",
    "sulphur_dioxide": "so2", "carbon_monoxide": "co",
    "temperature_2m": "temp", "relative_humidity_2m": "rh", "wind_speed_10m": "ws",
    "wind_direction_10m": "wd", "surface_pressure": "pres", "precipitation": "precip",
    "cloud_cover": "cloud",
}

# CPCB (India) AQI categories
AQI_BINS = [0, 50, 100, 200, 300, 400, 500]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
# PM2.5 (ug/m3) breakpoints and matching AQI sub-index values
PM25_BP = [0, 30, 60, 90, 120, 250, 380]
AQI_BP = [0, 50, 100, 200, 300, 400, 500]


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict, retries: int = 3) -> list[dict]:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, list) else [js]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Request to {url} failed: {last}")


def fetch_live(history_days: int = HISTORY_DAYS, forecast_days: int = 3) -> dict[str, pd.DataFrame]:
    names = list(LOCATIONS)
    common = dict(
        latitude=",".join(str(LOCATIONS[n][0]) for n in names),
        longitude=",".join(str(LOCATIONS[n][1]) for n in names),
        past_days=history_days, forecast_days=forecast_days, timezone=TZ,
    )
    aq = _get_json(AQ_URL, {**common, "hourly": ",".join(AQ_VARS)})
    wx = _get_json(WX_URL, {**common, "hourly": ",".join(WX_VARS)})

    frames = {}
    for n, a, w in zip(names, aq, wx):
        da = pd.DataFrame(a["hourly"]).set_index("time")
        dw = pd.DataFrame(w["hourly"]).set_index("time")
        df = da.join(dw, how="inner").rename(columns=RENAME)
        df.index = pd.to_datetime(df.index)
        frames[n] = df.sort_index()
    return frames


def fetch_demo(now: pd.Timestamp, history_days: int = HISTORY_DAYS, forecast_days: int = 3):
    """Synthetic data with realistic structure, so the pipeline runs offline."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(now - pd.Timedelta(days=history_days),
                        now + pd.Timedelta(days=forecast_days), freq="h")
    n = len(idx)
    hod = idx.hour.values
    # shared weather
    wind = 8 + 4 * np.sin(np.arange(n) / 90) + rng.normal(0, 1.5, n).cumsum() * 0.05
    wind = np.clip(wind + rng.normal(0, 1, n), 1, 30)
    rain = np.clip(rng.normal(-1.2, 1.5, n), 0, None)
    temp = 28 + 3 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, .5, n)
    rh = np.clip(70 - 10 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, 4, n), 30, 100)
    regime = np.convolve(rng.normal(0, 1, n), np.ones(48) / 48, mode="same") * 6
    frames = {}
    for i, (name, (lat, lon)) in enumerate(LOCATIONS.items()):
        east = (lon - LON_MIN) / (LON_MAX - LON_MIN)
        base = 45 + 25 * east + 10 * rng.random()
        rush = 18 * np.exp(-((hod - 9) % 24) ** 2 / 8) + 22 * np.exp(-((hod - 20) % 24) ** 2 / 10)
        pm = base + rush + 12 * regime - 2.5 * wind - 6 * rain + rng.normal(0, 4, n)
        pm = np.clip(pd.Series(pm).rolling(3, min_periods=1).mean().values, 3, None)
        frames[name] = pd.DataFrame({
            "pm25": pm, "pm10": pm * 1.8 + rng.normal(0, 6, n),
            "no2": 20 + 0.3 * pm + rng.normal(0, 3, n), "o3": 40 - 0.2 * pm + rng.normal(0, 4, n),
            "so2": 8 + rng.normal(0, 1, n), "co": 400 + 6 * pm + rng.normal(0, 30, n),
            "temp": temp, "rh": rh, "ws": wind, "wd": 250 + rng.normal(0, 25, n),
            "pres": 1008 + rng.normal(0, 1, n).cumsum() * 0.05, "precip": rain,
            "cloud": np.clip(50 + rng.normal(0, 25, n), 0, 100),
        }, index=idx)
    return frames


# --------------------------------------------------------------------------- #
# AQI helpers
# --------------------------------------------------------------------------- #
def pm25_to_aqi(pm25):
    return np.interp(np.asarray(pm25, dtype=float), PM25_BP, AQI_BP)


def aqi_category(aqi):
    a = np.asarray(aqi, dtype=float)
    idx = np.clip(np.searchsorted(AQI_BINS[1:], a, side="left"), 0, len(AQI_LABELS) - 1)
    return np.array(AQI_LABELS)[idx]


# --------------------------------------------------------------------------- #
# Features + model
# --------------------------------------------------------------------------- #
def make_features(df: pd.DataFrame, h: int, lat: float, lon: float):
    """Features at anchor time t used to predict PM2.5 at t+h.
    Weather at t+h comes from the weather forecast, which is known at time t."""
    p = df["pm25"]
    X = pd.DataFrame(index=df.index)
    for lag in (0, 1, 3, 6, 12, 24):
        X[f"pm25_l{lag}"] = p.shift(lag)
    X["pm25_r6"] = p.rolling(6).mean()
    X["pm25_r24"] = p.rolling(24).mean()
    X["pm25_d3"] = p - p.shift(3)
    for c in ("pm10", "no2", "o3", "co", "so2"):
        X[c] = df[c]
    for c in ("temp", "rh", "ws", "pres"):
        X[f"{c}_now"] = df[c]
    X["rain24"] = df["precip"].rolling(24).sum()
    # Most recent observed value at the same hour-of-day as the target time
    # (captures the daily traffic/boundary-layer cycle).
    k = -(-h // 24)                                   # ceil(h / 24)
    X["pm25_same_tod"] = p.shift(24 * k - h)
    X["pm25_tod_gap"] = np.log1p(X["pm25_same_tod"]) - np.log1p(p)

    f = df.shift(-h)
    for c in ("temp", "rh", "ws", "pres", "precip", "cloud"):
        X[f"{c}_fut"] = f[c]
    rad = np.deg2rad(f["wd"])
    X["wind_u"] = -f["ws"] * np.sin(rad)
    X["wind_v"] = -f["ws"] * np.cos(rad)

    target_time = df.index + pd.Timedelta(hours=h)
    X["hod_sin"] = np.sin(2 * np.pi * target_time.hour.values / 24)
    X["hod_cos"] = np.cos(2 * np.pi * target_time.hour.values / 24)
    X["dow"] = target_time.dayofweek.values
    X["lat"], X["lon"], X["h"] = lat, lon, h

    # Target = change in log(PM2.5) relative to now. Predicting the *change* lets the
    # model extrapolate to levels it hasn't seen and keeps persistence as the anchor.
    y = np.log1p(p.shift(-h)) - np.log1p(p)
    return X, y


def build_dataset(frames, horizons, anchor_min=None, anchor_max=None, last_obs=None):
    Xs, ys, locs = [], [], []
    for name, df in frames.items():
        lat, lon = LOCATIONS[name]
        hist = df if last_obs is None else df[df.index <= last_obs]
        for h in horizons:
            X, y = make_features(hist, h, lat, lon)
            m = y.notna() & X["pm25_l24"].notna() & X["pm25_l0"].notna()
            if anchor_min is not None:
                m &= X.index > anchor_min
            if anchor_max is not None:
                m &= X.index <= anchor_max
            Xs.append(X[m]); ys.append(y[m]); locs += [name] * int(m.sum())
    return pd.concat(Xs), pd.concat(ys), np.array(locs)


def subsample(X, y, max_rows, seed=0):
    if len(X) <= max_rows:
        return X, y
    sel = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
    return X.iloc[sel], y.iloc[sel]


def predict_pm25(model, X: pd.DataFrame) -> np.ndarray:
    return np.expm1(np.log1p(X["pm25_l0"].values) + model.predict(X))


def new_model(quantile: float | None = None):
    # Deliberately conservative: rows from neighbouring points/horizons are highly
    # correlated, so a small, heavily regularised model generalises better.
    kw = dict(max_iter=150, learning_rate=0.05, max_depth=4,
              min_samples_leaf=500, l2_regularization=5.0, random_state=0)
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(**kw)


def backtest(frames, now, outdir: Path):
    """Rolling-origin backtest: for each of the last N_FOLDS weeks, train only on data
    before that week and predict the week. Compares with two baselines:
      persistence  = PM2.5 stays at its current value
      same_tod     = PM2.5 equals the last observed value at the same hour of day"""
    parts = []
    for k in range(N_FOLDS, 0, -1):
        cut = now - pd.Timedelta(days=FOLD_DAYS * k)
        end = cut + pd.Timedelta(days=FOLD_DAYS)
        Xtr, ytr, _ = build_dataset(frames, HORIZONS, last_obs=cut)
        Xtr, ytr = subsample(Xtr, ytr, 250_000)
        mid = new_model().fit(Xtr, ytr)
        Xva, yva, loc = build_dataset(frames, HORIZONS, anchor_min=cut, anchor_max=end, last_obs=now)
        base = np.log1p(Xva["pm25_l0"].values)
        d = pd.DataFrame({
            "fold": N_FOLDS - k + 1, "loc": loc, "h": Xva["h"].values,
            "true": np.expm1(yva.values + base),
            "ml": predict_pm25(mid, Xva),
            "persistence": Xva["pm25_l0"].values,
            "same_tod": Xva["pm25_same_tod"].values,
        })
        if k == 1:  # check interval coverage on the most recent fold
            lo = new_model(0.03).fit(Xtr, ytr); hi = new_model(0.97).fit(Xtr, ytr)
            d["lo"] = np.expm1(base + lo.predict(Xva)); d["hi"] = np.expm1(base + hi.predict(Xva))
        parts.append(d)
        print(f"  fold {N_FOLDS - k + 1}/{N_FOLDS} done ({cut:%d %b} -> {end:%d %b})")

    res = pd.concat(parts, ignore_index=True)
    models = ["ml", "persistence", "same_tod"]
    for m in models:
        res[f"ae_{m}"] = (res[m] - res["true"]).abs()
        res[f"se_{m}"] = (res[m] - res["true"]) ** 2
    ae = [f"ae_{m}" for m in models]
    res["bucket"] = pd.cut(res["h"], [0, 6, 12, 24, 48], labels=["1-6h", "7-12h", "13-24h", "25-48h"])

    by_h = res.groupby("bucket", observed=True)[ae].mean()
    by_h.loc["overall"] = res[ae].mean()
    by_h.columns = models
    by_fold = res.groupby("fold")[ae].mean(); by_fold.columns = models
    by_loc = res.groupby("loc")[ae].mean(); by_loc.columns = models
    rmse = {m: float(np.sqrt(res[f"se_{m}"].mean())) for m in models}
    skill = 1 - by_h["ml"] / by_h["persistence"]
    by_h["skill_vs_persistence_%"] = (skill * 100).round(1)

    outdir.mkdir(parents=True, exist_ok=True)
    by_h.round(2).to_csv(outdir / "backtest_by_horizon.csv")
    by_loc.round(2).to_csv(outdir / "backtest_by_location.csv")
    by_fold.round(2).to_csv(outdir / "backtest_by_fold.csv")

    print("\nBacktest MAE in ug/m3 (lower is better), last 3 weeks:")
    print(by_h.round(2).to_string())
    print("\nBy week (fold 3 = most recent):"); print(by_fold.round(2).to_string())
    print("\nOverall RMSE:", {k: round(v, 2) for k, v in rmse.items()})
    last = res[res["fold"] == N_FOLDS].dropna(subset=["lo", "hi"])
    cov = float(((last["true"] >= last["lo"]) & (last["true"] <= last["hi"])).mean())
    print(f"Interval coverage on the latest week: {cov:.0%} (target is about 80%)")


def fit_final(frames, now, max_rows=450_000):
    X, y, _ = build_dataset(frames, HORIZONS, last_obs=now)
    X, y = subsample(X, y, max_rows, seed=1)
    return new_model().fit(X, y), new_model(0.03).fit(X, y), new_model(0.97).fit(X, y)


def forecast(frames, models, now, blend: float = 1.0) -> pd.DataFrame:
    mid_m, lo_m, hi_m = models
    rows = []
    for name, df in frames.items():
        lat, lon = LOCATIONS[name]
        Xf = pd.concat([make_features(df, h, lat, lon)[0].loc[[now]] for h in HORIZONS])
        base = np.log1p(Xf["pm25_l0"].values)
        ml = np.expm1(base + mid_m.predict(Xf))
        lo = np.expm1(base + lo_m.predict(Xf)); hi = np.expm1(base + hi_m.predict(Xf))
        for i, h in enumerate(HORIZONS):
            t = now + pd.Timedelta(hours=h)
            cams = df["pm25"].get(t, np.nan)
            final = ml[i] if (np.isnan(cams) or blend >= 1) else blend * ml[i] + (1 - blend) * cams
            shift = final - ml[i]
            low = max(0.0, min(lo[i] + shift, final)); high = max(hi[i] + shift, final)
            rows.append((name, lat, lon, t, h, round(float(ml[i]), 1),
                         None if np.isnan(cams) else round(float(cams), 1),
                         round(float(final), 1), round(float(low), 1), round(float(high), 1)))
    out = pd.DataFrame(rows, columns=["location", "lat", "lon", "valid_time", "horizon_h",
                                      "pm25_model", "pm25_cams", "pm25_final", "pm25_low", "pm25_high"])
    out["aqi"] = pm25_to_aqi(out["pm25_final"]).round().astype(int)
    out["aqi_low"] = pm25_to_aqi(out["pm25_low"]).round().astype(int)
    out["aqi_high"] = pm25_to_aqi(out["pm25_high"]).round().astype(int)
    out["category"] = aqi_category(out["aqi"])
    return out


# --------------------------------------------------------------------------- #
# Spatial mapping
# --------------------------------------------------------------------------- #
def idw_weights(names, power=2.0):
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_NY)
    lons = np.linspace(LON_MIN, LON_MAX, GRID_NX)
    gy, gx = np.meshgrid(lats, lons, indexing="ij")
    cells = np.c_[gy.ravel(), gx.ravel()]
    st = np.array([LOCATIONS[n] for n in names])
    dy = (cells[:, None, 0] - st[None, :, 0]) * 110.57
    dx = (cells[:, None, 1] - st[None, :, 1]) * 111.32 * np.cos(np.deg2rad(cells[:, None, 0]))
    d = np.maximum(np.hypot(dx, dy), 0.3)
    W = 1.0 / d ** power
    W /= W.sum(axis=1, keepdims=True)
    mask = d.min(axis=1) <= MAX_CELL_DISTANCE_KM
    return lats, lons, W, mask


def build_map_payload(fc: pd.DataFrame, now: pd.Timestamp) -> dict:
    names = list(LOCATIONS)
    times = sorted(fc["valid_time"].unique())
    pm = np.vstack([fc[fc.location == n].sort_values("valid_time")["pm25_final"].values for n in names])
    lats, lons, W, mask = idw_weights(names)
    lo = np.vstack([fc[fc.location == n].sort_values("valid_time")["aqi_low"].values for n in names])
    hi = np.vstack([fc[fc.location == n].sort_values("valid_time")["aqi_high"].values for n in names])
    grid_pm = W @ pm                               # (cells, hours)
    grid_aqi = np.rint(pm25_to_aqi(grid_pm)).astype(int)
    grid_aqi[~mask, :] = -1
    return {
        "generated": now.strftime("%Y-%m-%d %H:%M IST"),
        "times": [pd.Timestamp(t).strftime("%a %d %b, %H:%M") for t in times],
        "stations": [{"name": n, "lat": LOCATIONS[n][0], "lon": LOCATIONS[n][1]} for n in names],
        "st_pm": np.round(pm, 1).tolist(),
        "st_aqi": np.rint(pm25_to_aqi(pm)).astype(int).tolist(),
        "st_lo": lo.tolist(), "st_hi": hi.tolist(),
        "grid": {"lats": lats.tolist(), "lons": lons.tolist(), "aqi": grid_aqi.T.tolist()},
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mumbai air quality forecast</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
  :root { --bg:#f6f7f8; --fg:#1b2024; --muted:#5d6670; --card:#ffffff; --line:#d9dde1; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14181b; --fg:#e8ebed; --muted:#9aa4ad; --card:#1d2227; --line:#2d343a; }
  }
  * { box-sizing: border-box; }
  html, body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         display:flex; flex-direction:column; }
  header { padding:12px 16px; display:flex; flex-wrap:wrap; gap:8px 24px; align-items:baseline;
           border-bottom:1px solid var(--line); background:var(--card); }
  h1 { font-size:18px; margin:0; font-weight:650; }
  .meta { color:var(--muted); font-size:13px; }
  #map { flex:1; min-height:320px; }
  .panel { padding:10px 16px 14px; background:var(--card); border-top:1px solid var(--line); }
  .row { display:flex; align-items:center; gap:12px; }
  #when { min-width:150px; font-weight:600; font-variant-numeric:tabular-nums; }
  input[type=range] { flex:1; accent-color:#4b6b88; }
  button { font:inherit; padding:5px 14px; border:1px solid var(--line); border-radius:6px;
           background:var(--bg); color:var(--fg); cursor:pointer; }
  button:focus-visible, input:focus-visible { outline:2px solid #4b6b88; outline-offset:2px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:12.5px; color:var(--muted); }
  .legend i { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  .note { margin-top:6px; font-size:12px; color:var(--muted); }
</style></head>
<body>
<header>
  <h1>Mumbai air quality, next 48 hours</h1>
  <span class="meta">Model run: <span id="gen"></span></span>
</header>
<div id="map"></div>
<div class="panel">
  <div class="row">
    <button id="play" aria-label="Play or pause the forecast animation">Play</button>
    <span id="when"></span>
    <input id="slider" type="range" min="0" value="0" step="1" aria-label="Forecast hour">
  </div>
  <div class="legend" id="legend"></div>
  <div class="note">AQI follows the Indian CPCB scale, computed from forecast PM2.5. The input data is coarse atmospheric-model output, so nearby points can look alike, and colour between points is interpolated. Treat the map as a regional estimate, not a measurement. Ranges are the likely span of PM2.5, not a guarantee.</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D = __DATA__;
const CATS = [[50,'Good','#3f9b4f'],[100,'Satisfactory','#9bc53d'],[200,'Moderate','#f0cf3a'],
              [300,'Poor','#f08c2e'],[400,'Very poor','#d9412f'],[500,'Severe','#8c1d2b']];
const col = a => { for (const c of CATS) if (a <= c[0]) return c[2]; return CATS[5][2]; };
const name = a => { for (const c of CATS) if (a <= c[0]) return c[1]; return 'Severe'; };

document.getElementById('gen').textContent = D.generated;
document.getElementById('legend').innerHTML = CATS.map((c,i) =>
  `<span><i style="background:${c[2]}"></i>${c[1]} (${i? CATS[i-1][0]+1 : 0}–${c[0]})</span>`).join('');

const map = L.map('map').setView([19.08, 72.89], 11);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 18, subdomains: 'abcd',
   attribution: '&copy; OpenStreetMap contributors &copy; CARTO'}).addTo(map);

// grid cells
const lats = D.grid.lats, lons = D.grid.lons;
const dLat = (lats[1]-lats[0])/2, dLon = (lons[1]-lons[0])/2;
const cells = [];
for (let i=0;i<lats.length;i++) for (let j=0;j<lons.length;j++) {
  const r = L.rectangle([[lats[i]-dLat, lons[j]-dLon],[lats[i]+dLat, lons[j]+dLon]],
    {stroke:false, fillOpacity:0.5, interactive:false}).addTo(map);
  cells.push({r, k:i*lons.length+j});
}
// station markers
const marks = D.stations.map((s,i) => {
  const m = L.circleMarker([s.lat, s.lon], {radius:8, weight:2, color:'#1b2024', fillOpacity:1}).addTo(map);
  return m;
});

const slider = document.getElementById('slider');
slider.max = D.times.length - 1;
function show(t) {
  const g = D.grid.aqi[t];
  cells.forEach(c => { const a = g[c.k]; if (a < 0) c.r.setStyle({fillOpacity:0}); else c.r.setStyle({fillColor:col(a), fillOpacity:0.5}); });
  marks.forEach((m,i) => {
    const a = D.st_aqi[i][t];
    m.setStyle({fillColor: col(a)});
    m.bindTooltip(`<b>${D.stations[i].name}</b><br>AQI ${a} (${name(a)})<br>Likely range ${D.st_lo[i][t]}–${D.st_hi[i][t]}<br>PM2.5 ${D.st_pm[i][t]} µg/m³`);
  });
  document.getElementById('when').textContent = D.times[t];
}
slider.addEventListener('input', () => show(+slider.value));
let timer = null;
const btn = document.getElementById('play');
btn.addEventListener('click', () => {
  if (timer) { clearInterval(timer); timer = null; btn.textContent = 'Play'; return; }
  btn.textContent = 'Pause';
  timer = setInterval(() => { slider.value = (+slider.value + 1) % D.times.length; show(+slider.value); }, 700);
});
show(0);
</script></body></html>
"""


PLAIN_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mumbai air quality forecast (no external files)</title>
<style>
  :root { --bg:#f6f7f8; --fg:#1b2024; --muted:#5d6670; --card:#ffffff; --line:#d9dde1; --sea:#e4edf3; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14181b; --fg:#e8ebed; --muted:#9aa4ad; --card:#1d2227; --line:#2d343a; --sea:#1a2530; }
  }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
         display:flex; flex-direction:column; }
  header { padding:12px 16px; display:flex; flex-wrap:wrap; gap:8px 24px; align-items:baseline;
           border-bottom:1px solid var(--line); background:var(--card); }
  h1 { font-size:18px; margin:0; font-weight:650; }
  .meta { color:var(--muted); font-size:13px; }
  #wrap { flex:1; min-height:320px; position:relative; background:var(--sea); }
  #tip { position:absolute; pointer-events:none; display:none; padding:6px 9px; font-size:13px;
         background:var(--card); border:1px solid var(--line); border-radius:6px; white-space:nowrap; }
  .panel { padding:10px 16px 14px; background:var(--card); border-top:1px solid var(--line); }
  .row { display:flex; align-items:center; gap:12px; }
  #when { min-width:150px; font-weight:600; font-variant-numeric:tabular-nums; }
  input[type=range] { flex:1; accent-color:#4b6b88; }
  button { font:inherit; padding:5px 14px; border:1px solid var(--line); border-radius:6px;
           background:var(--bg); color:var(--fg); cursor:pointer; }
  button:focus-visible, input:focus-visible { outline:2px solid #4b6b88; outline-offset:2px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px; font-size:12.5px; color:var(--muted); }
  .legend i { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  .note { margin-top:6px; font-size:12px; color:var(--muted); }
</style></head>
<body>
<header>
  <h1>Mumbai air quality, next 48 hours</h1>
  <span class="meta">Model run: <span id="gen"></span></span>
</header>
<div id="wrap"><canvas id="cv"></canvas><div id="tip"></div></div>
<div class="panel">
  <div class="row">
    <button id="play" aria-label="Play or pause the forecast animation">Play</button>
    <span id="when"></span>
    <input id="slider" type="range" min="0" value="0" step="1" aria-label="Forecast hour">
  </div>
  <div class="legend" id="legend"></div>
  <div class="note">Schematic map with no street background: west is left, north is up. Numbers in circles are AQI (Indian CPCB scale, from forecast PM2.5). Hover a circle for its name and likely range. Colours between points are interpolated estimates, and the source data is coarse, so nearby points can look alike.</div>
</div>
<script>
const D = __DATA__;
const CATS = [[50,'Good','#3f9b4f'],[100,'Satisfactory','#9bc53d'],[200,'Moderate','#f0cf3a'],
              [300,'Poor','#f08c2e'],[400,'Very poor','#d9412f'],[500,'Severe','#8c1d2b']];
const col = a => { for (const c of CATS) if (a <= c[0]) return c[2]; return CATS[5][2]; };
const cname = a => { for (const c of CATS) if (a <= c[0]) return c[1]; return 'Severe'; };
document.getElementById('gen').textContent = D.generated;
document.getElementById('legend').innerHTML = CATS.map((c,i) =>
  `<span><i style="background:${c[2]}"></i>${c[1]} (${i? CATS[i-1][0]+1 : 0}-${c[0]})</span>`).join('');

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const wrap = document.getElementById('wrap'), tip = document.getElementById('tip');
const slider = document.getElementById('slider');
slider.max = D.times.length - 1;
let t = 0, hover = -1, geo = null;
const cs = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function layout() {
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  cv.width = W*dpr; cv.height = H*dpr; cv.style.width = W+'px'; cv.style.height = H+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const la = D.grid.lats, lo = D.grid.lons;
  const k = Math.cos(19.08*Math.PI/180);
  const sc = Math.min((W-40)/((lo[lo.length-1]-lo[0])*k), (H-40)/(la[la.length-1]-la[0]));
  const w = (lo[lo.length-1]-lo[0])*k*sc, h = (la[la.length-1]-la[0])*sc;
  geo = {W, H, k, sc, x0:(W-w)/2, y0:(H-h)/2, latMax:la[la.length-1], lonMin:lo[0],
         cw:(lo[1]-lo[0])*k*sc, ch:(la[1]-la[0])*sc};
  draw();
}
const X = lon => geo.x0 + (lon-geo.lonMin)*geo.k*geo.sc;
const Y = lat => geo.y0 + (geo.latMax-lat)*geo.sc;

function draw() {
  ctx.clearRect(0,0,geo.W,geo.H);
  const g = D.grid.aqi[t], nx = D.grid.lons.length;
  ctx.globalAlpha = 0.65;
  for (let i=0;i<D.grid.lats.length;i++) for (let j=0;j<nx;j++) {
    const a = g[i*nx+j]; if (a < 0) continue;
    ctx.fillStyle = col(a);
    ctx.fillRect(X(D.grid.lons[j])-geo.cw/2, Y(D.grid.lats[i])-geo.ch/2, geo.cw+1, geo.ch+1);
  }
  ctx.globalAlpha = 1;
  ctx.font = '600 10px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  D.stations.forEach((s,i) => {
    const a = D.st_aqi[i][t], x = X(s.lon), y = Y(s.lat);
    ctx.beginPath(); ctx.arc(x, y, i===hover ? 14 : 12, 0, 6.2832);
    ctx.fillStyle = col(a); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = cs('--fg'); ctx.stroke();
    ctx.fillStyle = (a > 50 && a <= 200) ? '#1b2024' : '#ffffff';
    ctx.fillText(a, x, y);
  });
  document.getElementById('when').textContent = D.times[t];
}
function nearest(mx, my) {
  let best = -1, bd = 18*18;
  D.stations.forEach((s,i) => { const dx = X(s.lon)-mx, dy = Y(s.lat)-my, d = dx*dx+dy*dy; if (d < bd) { bd = d; best = i; } });
  return best;
}
cv.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect(), mx = e.clientX-r.left, my = e.clientY-r.top;
  const h = nearest(mx, my);
  if (h !== hover) { hover = h; draw(); }
  if (h < 0) { tip.style.display = 'none'; return; }
  const a = D.st_aqi[h][t];
  tip.innerHTML = `<b>${D.stations[h].name}</b><br>AQI ${a} (${cname(a)})<br>Likely range ${D.st_lo[h][t]}-${D.st_hi[h][t]}<br>PM2.5 ${D.st_pm[h][t]} &micro;g/m&sup3;`;
  tip.style.display = 'block';
  tip.style.left = Math.min(mx+14, geo.W-170)+'px'; tip.style.top = Math.max(my-30, 4)+'px';
});
cv.addEventListener('mouseleave', () => { hover = -1; tip.style.display = 'none'; draw(); });
slider.addEventListener('input', () => { t = +slider.value; draw(); });
let timer = null; const btn = document.getElementById('play');
btn.addEventListener('click', () => {
  if (timer) { clearInterval(timer); timer = null; btn.textContent = 'Play'; return; }
  btn.textContent = 'Pause';
  timer = setInterval(() => { t = (t+1) % D.times.length; slider.value = t; draw(); }, 700);
});
window.addEventListener('resize', layout);
layout();
</script></body></html>
"""


def write_map(payload: dict, path: Path):
    data = json.dumps(payload)
    path.write_text(HTML_TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    # Same map with no external files at all (no map tiles, no libraries): opens anywhere
    path.with_name("index_plain.html").write_text(PLAIN_TEMPLATE.replace("__DATA__", data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_once(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")

    print(f"[{now:%Y-%m-%d %H:%M}] fetching data ...")
    frames = fetch_demo(now) if args.demo else fetch_live()

    # Rows up to 'now' are history; later rows supply the weather forecast
    for n, df in frames.items():
        if now not in df.index:
            raise RuntimeError(f"No data row for {now} at {n}; got up to {df.index.max()}")
        frames[n] = df.ffill(limit=3)

    if not args.quick:
        print("running backtest ...")
        backtest(frames, now, out)

    print("\ntraining final models ...")
    models = fit_final(frames, now)

    fc = forecast(frames, models, now, blend=args.blend)
    stamp = now.strftime("%Y%m%d_%H")
    fc.to_csv(out / "forecast.csv", index=False)
    fc.to_csv(out / f"forecast_{stamp}.csv", index=False)
    write_map(build_map_payload(fc, now), out / "index.html")

    nxt = fc[fc.horizon_h == 1][["location", "pm25_final", "pm25_low", "pm25_high", "aqi", "category"]]
    print("\nNext-hour forecast:")
    print(nxt.to_string(index=False))
    print(f"\nSaved to the '{out}' folder: forecast.csv, index.html, index_plain.html, backtest_*.csv")


def main():
    ap = argparse.ArgumentParser(description="Hourly AQI forecast + map for Mumbai")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--loop", action="store_true", help="re-run every hour at hh:05")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--quick", action="store_true", help="skip the backtest (faster)")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="weight of own model vs CAMS forecast (1.0 = own model only)")
    # parse_known_args so this also works when pasted into Colab/Jupyter
    args, _ = ap.parse_known_args()

    if not args.loop:
        run_once(args)
        return
    while True:
        try:
            run_once(args)
        except Exception as e:  # noqa: BLE001
            print("run failed:", e)
        nxt = (datetime.now() + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
        wait = max(60, (nxt - datetime.now()).total_seconds())
        print(f"sleeping {wait/60:.0f} min until {nxt:%H:%M}\n")
        time.sleep(wait)


if __name__ == "__main__":
    main()
