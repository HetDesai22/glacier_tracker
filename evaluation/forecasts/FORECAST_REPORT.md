# Samudra Tapu Glacier Forecast Report

## Dataset

- Historical observations used: 9
- Historical period: 2018–2026
- Target variable: U-Net-predicted area (km²)
- Method: Linear regression on annual observations

## Historical trend

- First observed area: 74.419 km²
- Last observed area: 57.285 km²
- Difference between first and last observations: -17.134 km²
- Fitted slope: -1.8759 km²/year
- Historical fitted R²: 0.4311

## Time-ordered validation

The final two historical observations were held out during validation.

- MAE: 4.677 km²
- RMSE: 5.262 km²
- R² on the two held-out years: 0.063

Only two validation observations are available, so these estimates are highly uncertain.

## Future trend extrapolation

Forecast period: 2027–2035.

|   year |   predicted_area_km2 | negative_area_warning   | interpretation                                                |
|-------:|---------------------:|:------------------------|:--------------------------------------------------------------|
|   2027 |              60.5242 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2028 |              58.6483 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2029 |              56.7724 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2030 |              54.8965 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2031 |              53.0206 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2032 |              51.1447 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2033 |              49.2689 | False                   | Experimental trend extrapolation; not validated physical melt |
|   2034 |              47.393  | False                   | Experimental trend extrapolation; not validated physical melt |
|   2035 |              45.5171 | False                   | Experimental trend extrapolation; not validated physical melt |

## Important limitations

- The model uses only nine or fewer annual observations.
- The target is U-Net-predicted area, not independently verified glacier area.
- The reference masks and predictions may contain systematic errors.
- Linear regression assumes a constant trend and does not model climate,
  snowfall, seasonal effects, glacier dynamics, or image differences.
- Historical R² is not proof that future forecasts are reliable.
- Forecasts must not be presented as scientifically established melt estimates.

Treat the results as an experimental baseline for the prototype, not a
validated prediction of physical glacier retreat.
