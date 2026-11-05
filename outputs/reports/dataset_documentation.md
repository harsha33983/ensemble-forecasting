# Dataset Documentation: UCI Bike Sharing Dataset

- **Source URL**: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Citation Required
- **Citation**: Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
- **Target Variable**: `cnt` (Total hourly count of rental bikes (sum of casual and registered users))
- **Granularity**: Hourly (17,379 observations) and Daily (731 observations)
- **Timeframe**: 2011-01-01 00:00:00 to 2012-12-31 23:00:00 (2 years)

## Feature Schema

| Column Name | Type | Description |
|-------------|------|-------------|
| `instant` | Feature | Record index (sequential) |
| `dteday` | Feature | Date of observation (YYYY-MM-DD) |
| `season` | Feature | Season (1: Winter/Spring start, 2: Summer, 3: Fall, 4: Winter) |
| `yr` | Feature | Year (0: 2011, 1: 2012) |
| `mnth` | Feature | Month of the year (1 to 12) |
| `hr` | Feature | Hour of the day (0 to 23) |
| `holiday` | Feature | Whether the day is a holiday (1: Yes, 0: No) |
| `weekday` | Feature | Day of the week (0: Sunday to 6: Saturday) |
| `workingday` | Feature | Working day indicator (1: Neither weekend nor holiday, 0: Weekend or holiday) |
| `weathersit` | Feature | Weather category (1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Ice/Fog) |
| `temp` | Feature | Normalized temperature in Celsius (divided by 41) |
| `atemp` | Feature | Normalized feeling temperature in Celsius (divided by 50) |
| `hum` | Feature | Normalized relative humidity (divided by 100) |
| `windspeed` | Feature | Normalized wind speed (divided by 67) |
| `casual` | Feature | Count of unregistered casual users (leakage component) |
| `registered` | Feature | Count of registered annual/monthly members (leakage component) |
| `cnt` | Feature | Count of total rental bikes (prediction target) |

## Known Dataset Limitations

- Aggregated city-level data for Washington, D.C.; cannot resolve spatial micro-movements.
- Occasional missing hours during extreme weather events or late night maintenance.
- Weather attributes are pre-normalized in the original UCI release.
- Strong structural break / secular growth between 2011 and 2012 due to system adoption.
