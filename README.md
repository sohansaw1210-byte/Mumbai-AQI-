# Mumbai Hourly Air Quality Forecast and Map

Forecasts PM2.5 and the Indian CPCB-style AQI for 18 locations across Mumbai, 1 to 48 hours ahead, and shows the result on an interactive map. The script can refresh automatically every hour (`--loop`); the hosted page is a snapshot, and its "Model run" time shows when it was generated.

## How to run

```bash
pip install -r requirements.txt
python mumbai_aqi.py            # full run with backtest (about 2-4 minutes)
python mumbai_aqi.py --quick    # skip the backtest
python mumbai_aqi.py --loop     # refresh automatically at hh:05 every hour
python mumbai_aqi.py --demo     # offline test on synthetic data
```

Open `index.html` in a browser for the map. It has an hour slider and a play button.

**Live demo:** https://github.com/sohansaw1210-byte/Mumbai-AQI-

The script also runs in Google Colab: paste it into one cell and run it.

## Files

| File | Contents |
|---|---|
| `index.html` | Interactive forecast map with a 48-hour slider. It uses no external files, so it opens on any computer. (When you run the script yourself, this page is written as `index_plain.html`, and `index.html` is a street-map version that needs internet. Here the dependency-free page is uploaded as `index.html`.) |
| `forecast.csv` | Per location and hour: PM2.5 forecast, likely range, AQI, category |
| `backtest_by_horizon.csv` | Error of the model versus baselines, by forecast horizon |
| `backtest_by_location.csv` | The same, per location |
| `backtest_by_fold.csv` | The same, per backtest week |

## Method

1. **Data.** Hourly air quality and weather for 18 points, last 60 days plus the weather forecast, from the Open-Meteo Air Quality and Forecast APIs. No API key is needed.
2. **Model.** A gradient-boosted regressor (scikit-learn `HistGradientBoostingRegressor`) predicts the change in log PM2.5 between now and each target hour (1 to 48). Inputs:
   - recent PM2.5 (lags, rolling means, trend) and other pollutants,
   - the PM2.5 value at the same hour of day as the target (daily cycle),
   - the weather forecast for the target hour: wind, rain, humidity, pressure, cloud.
3. **Uncertainty.** Two extra quantile models give a likely range (3rd to 97th percentile) for every forecast.
4. **AQI.** PM2.5 is converted using CPCB breakpoints (Good to Severe).
5. **Map.** Point forecasts are spread over a grid by inverse-distance weighting. Cells farther than 5 km from any point are left blank.

## Evaluation

A rolling-origin backtest runs over the last three weeks. For each week, the model trains only on earlier data and predicts that week. It is compared with two baselines:

- **persistence**: PM2.5 stays at its current value,
- **same time of day**: PM2.5 equals the last observed value at that hour of day.

## Results

Backtest over the last three weeks (mean absolute error in µg/m³ of PM2.5, lower is better), run on 8 November 2026:

| Forecast horizon | Our model | Persistence | Same time of day | Improvement vs persistence |
|---|---|---|---|---|
| 1-6 h | 9.89 | 10.44 | 15.27 | 5.3% |
| 7-12 h | 12.96 | 18.49 | 15.20 | 29.9% |
| 13-24 h | 13.56 | 17.44 | 14.92 | 22.2% |
| 25-48 h | 16.50 | 19.22 | 17.32 | 14.2% |
| **Overall** | **14.43** | **17.53** | **16.17** | **17.7%** |

By week (fold 3 is the most recent): our model scored 20.65, 12.15 and 9.83, against persistence at 23.07, 13.71 and 15.51. The model beat persistence in all three weeks, and it was least accurate in the first week, which had much higher pollution levels and larger swings.

What this shows: the model clearly beats the simple baselines from about 7 hours out, and is only slightly better than persistence in the first few hours, where the current value is already a strong guess. The test period is three weeks, so these numbers should not be read as a seasonal guarantee.

## Limitations

- **Training data is atmospheric-model output, not sensor readings.** Open-Meteo's air quality values come from the CAMS model. The forecasts therefore estimate what that model will say, which is not the same as measured air quality at a street corner. Validating against CPCB or SAFAR station data is the main next step.
- **Coarse spatial resolution.** The source data is on a grid of tens of kilometres, so several of the 18 points fall in the same cell and can show near-identical values. Colour differences on the map are partly interpolation. The map is a regional estimate, not neighbourhood-level measurement.
- **The map does not model coastline, terrain, or local sources.**
- **The 48-hour forecast issued on 8 November 2026 shows little spread between locations.** Many neighbouring points have identical values (for example Worli, Dadar and Chembur), which confirms the resolution limit above. Forecast AQI stayed in the Satisfactory range, with a few hours reaching Moderate.
- **The backtest covers three weeks**, so it says little about seasonal behaviour (monsoon, winter, festival periods).
- **Prediction ranges are approximate.** They are tuned to cover about 80% of outcomes, not guaranteed to.

## Possible extensions

- Train and evaluate on real station readings (CPCB, SAFAR, or OpenAQ).
- Add a monsoon and winter backtest as more history is collected.
- Calibrate the likely ranges per horizon.
- Send an alert when forecast AQI crosses "Poor".

## Credits

Data: Open-Meteo (CAMS air quality, weather forecasts). Map tiles: OpenStreetMap contributors and CARTO. Map library: Leaflet.
