
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT_DIR / "evaluation" / "metrics_by_year.csv"
OUTPUT_DIR = ROOT_DIR / "evaluation" / "forecasts"

FORECAST_START_YEAR = 2027
FORECAST_END_YEAR = 2035

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate the historical dataset
# --------------------------------------------------

def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "year",
        "status",
        "prediction_area_km2",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Use only successful evaluation rows.
    df = df[df["status"] == "OK"].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["prediction_area_km2"] = pd.to_numeric(
        df["prediction_area_km2"], errors="coerce"
    )

    df = df.dropna(
        subset=["year", "prediction_area_km2"]
    )

    if df.empty:
        raise ValueError("No valid historical observations found.")

    if df["year"].duplicated().any():
        raise ValueError("Duplicate years found in the dataset.")

    if (df["prediction_area_km2"] <= 0).any():
        raise ValueError("Historical areas must be positive.")

    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            "At least five valid annual observations are required "
            "for this baseline workflow."
        )

    print("\nHISTORICAL DATA")
    print(
        df[["year", "prediction_area_km2"]].to_string(index=False)
    )
    print(f"\nValid observations: {len(df)}")
    print(
        f"Historical years: {df['year'].min()}–{df['year'].max()}"
    )

    return df


# --------------------------------------------------
# 3. Time-ordered validation
# --------------------------------------------------

def validate_model(df):
    # Hold out the last two years.
    train_df = df.iloc[:-2]
    test_df = df.iloc[-2:]

    X_train = train_df[["year"]].to_numpy()
    y_train = train_df["prediction_area_km2"].to_numpy()

    X_test = test_df[["year"]].to_numpy()
    y_test = test_df["prediction_area_km2"].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    # R² can be negative and is unstable with only two test points.
    test_r2 = (
        r2_score(y_test, test_predictions)
        if len(y_test) >= 2
        else float("nan")
    )

    validation_df = pd.DataFrame({
        "year": test_df["year"].to_numpy(),
        "actual_area_km2": y_test,
        "predicted_area_km2": test_predictions,
        "error_km2": test_predictions - y_test,
    })

    validation_path = OUTPUT_DIR / "validation_results.csv"
    validation_df.to_csv(validation_path, index=False)

    print("\nTIME-ORDERED VALIDATION")
    print(validation_df.to_string(index=False))
    print(f"\nMAE:  {mae:.3f} km²")
    print(f"RMSE: {rmse:.3f} km²")
    print(f"R² on two held-out years: {test_r2:.3f}")
    print(f"Validation results: {validation_path}")

    return {
        "mae": mae,
        "rmse": rmse,
        "test_r2": test_r2,
    }


# --------------------------------------------------
# 4. Train final baseline and forecast future years
# --------------------------------------------------

def create_forecast(df):
    X = df[["year"]].to_numpy()
    y = df["prediction_area_km2"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    historical_fitted = model.predict(X)

    future_years = np.arange(
        FORECAST_START_YEAR,
        FORECAST_END_YEAR + 1,
    )

    future_X = future_years.reshape(-1, 1)
    future_areas = model.predict(future_X)

    # Linear regression may extrapolate to impossible negative areas.
    # Preserve the raw result and flag it instead of silently clipping.
    forecast_df = pd.DataFrame({
        "year": future_years,
        "predicted_area_km2": future_areas,
    })

    forecast_df["negative_area_warning"] = (
        forecast_df["predicted_area_km2"] < 0
    )

    forecast_df["interpretation"] = (
        "Experimental trend extrapolation; not validated physical melt"
    )

    forecast_path = OUTPUT_DIR / "samudra_tapu_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # Record the fitted historical trend as well.
    fitted_df = pd.DataFrame({
        "year": df["year"].to_numpy(),
        "observed_area_km2": y,
        "fitted_area_km2": historical_fitted,
    })

    fitted_path = OUTPUT_DIR / "historical_fitted_trend.csv"
    fitted_df.to_csv(fitted_path, index=False)

    print("\nFULL-DATASET TREND")
    print(f"Slope: {model.coef_[0]:.4f} km²/year")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Historical fitted R²: {model.score(X, y):.4f}")

    print("\nFORECAST: 2027–2035")
    print(forecast_df.to_string(index=False))

    print(f"\nForecast CSV: {forecast_path}")
    print(f"Fitted trend CSV: {fitted_path}")

    return forecast_df, historical_fitted, model


# --------------------------------------------------
# 5. Generate historical and forecast plot
# --------------------------------------------------

def save_plot(df, forecast_df, historical_fitted):
    plt.figure(figsize=(11, 6))

    plt.plot(
        df["year"],
        df["prediction_area_km2"],
        marker="o",
        label="Historical U-Net-predicted area",
    )

    plt.plot(
        df["year"],
        historical_fitted,
        linestyle="--",
        label="Linear trend fitted to historical data",
    )

    plt.plot(
        forecast_df["year"],
        forecast_df["predicted_area_km2"],
        marker="o",
        linestyle=":",
        label="Experimental forecast",
    )

    plt.axvline(
        df["year"].max(),
        linestyle="--",
        label="Forecast boundary",
    )

    plt.xlabel("Year")
    plt.ylabel("Mapped area (km²)")
    plt.title("Samudra Tapu Glacier — Historical Area and Trend Forecast")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "forecast_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Forecast plot: {plot_path}")


# --------------------------------------------------
# 6. Save a summary report
# --------------------------------------------------

def save_report(df, validation, forecast_df, model):
    forecast_path = OUTPUT_DIR / "FORECAST_REPORT.md"

    historical_change = (
        df["prediction_area_km2"].iloc[-1]
        - df["prediction_area_km2"].iloc[0]
    )

    report = f"""# Samudra Tapu Glacier Forecast Report

## Dataset

- Historical observations used: {len(df)}
- Historical period: {df['year'].min()}–{df['year'].max()}
- Target variable: U-Net-predicted area (km²)
- Method: Linear regression on annual observations

## Historical trend

- First observed area: {df['prediction_area_km2'].iloc[0]:.3f} km²
- Last observed area: {df['prediction_area_km2'].iloc[-1]:.3f} km²
- Difference between first and last observations: {historical_change:+.3f} km²
- Fitted slope: {model.coef_[0]:+.4f} km²/year
- Historical fitted R²: {model.score(df[['year']].to_numpy(), df['prediction_area_km2'].to_numpy()):.4f}

## Time-ordered validation

The final two historical observations were held out during validation.

- MAE: {validation['mae']:.3f} km²
- RMSE: {validation['rmse']:.3f} km²
- R² on the two held-out years: {validation['test_r2']:.3f}

Only two validation observations are available, so these estimates are highly uncertain.

## Future trend extrapolation

Forecast period: {FORECAST_START_YEAR}–{FORECAST_END_YEAR}.

{forecast_df.to_markdown(index=False)}

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
"""

    forecast_path.write_text(report, encoding="utf-8")
    print(f"Forecast report: {forecast_path}")


# --------------------------------------------------
# 7. Main workflow
# --------------------------------------------------

def main():
    df = load_data()

    validation = validate_model(df)

    forecast_df, historical_fitted, model = create_forecast(df)

    save_plot(df, forecast_df, historical_fitted)

    save_report(df, validation, forecast_df, model)

    print("\nForecasting workflow completed.")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
