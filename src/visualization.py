"""
Visualization Utilities Module.

Generates high-resolution publication-quality plots for EDA,
residual diagnostics, feature importances, model comparisons, and forecasts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURE_DPI, PLOTS_DIR

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def save_current_figure(file_name: str, output_dir: Union[str, Path] = PLOTS_DIR) -> Path:
    """Save the current matplotlib figure and close it."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / file_name
    plt.tight_layout()
    plt.savefig(file_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return file_path


def plot_target_distribution(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Plot histogram and KDE with log-normal diagnostic of target demand."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard distribution
    sns.histplot(df[target_col], kde=True, ax=axes[0], color="#2b5c8f", bins=50)
    axes[0].set_title(f"Target Distribution: {target_col.upper()} (Hourly Rentals)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Hourly Rental Count", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    mean_val = df[target_col].mean()
    median_val = df[target_col].median()
    axes[0].axvline(mean_val, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean ({mean_val:.1f})")
    axes[0].axvline(median_val, color="#5cb85c", linestyle=":", linewidth=2.0, label=f"Median ({median_val:.1f})")
    axes[0].legend(frameon=True)

    # Log1p distribution
    log_target = np.log1p(df[target_col])
    sns.histplot(log_target, kde=True, ax=axes[1], color="#e67e22", bins=50)
    axes[1].set_title("Log-Transformed Target: log1p(cnt)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + cnt)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)

    return save_current_figure("target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Path:
    """Plot correlation matrix heatmap for numeric features."""
    plt.figure(figsize=(12, 10))
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter down if too many features to keep it clean and legible
        priority_cols = [
            "cnt", "temp", "atemp", "hum", "windspeed", "hr", "season",
            "yr", "mnth", "holiday", "workingday", "weathersit"
        ]
        columns = [c for c in priority_cols if c in numeric_cols]
        if len(columns) < 5:
            columns = numeric_cols[:15]

    corr = df[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
    return save_current_figure("correlation_heatmap.png")


def plot_feature_correlation_with_target(df: pd.DataFrame, target_col: str = "cnt", top_n: int = 15) -> Path:
    """Bar chart of Pearson correlations between features and target."""
    plt.figure(figsize=(10, 6))
    num_df = df.select_dtypes(include=[np.number])
    if target_col not in num_df.columns:
        return Path()

    corrs = num_df.corr()[target_col].drop(target_col, errors="ignore").sort_values()
    # Pick top positive and negative correlations
    top_corrs = pd.concat([corrs.head(top_n // 2), corrs.tail(top_n // 2)]).drop_duplicates().sort_values()

    colors = ["#d9534f" if val < 0 else "#2b5c8f" for val in top_corrs.values]
    bars = plt.barh(top_corrs.index, top_corrs.values, color=colors, edgecolor="#333333", alpha=0.85)
    plt.axvline(0, color="#333333", linewidth=1.0)
    plt.title(f"Top Feature Correlations with Target ({target_col.upper()})", fontsize=13, fontweight="bold")
    plt.xlabel("Pearson Correlation Coefficient", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    return save_current_figure("feature_correlation.png")


def plot_timeseries_trend(df: pd.DataFrame, datetime_col: str = "datetime", target_col: str = "cnt") -> Path:
    """Plot full longitudinal time-series with rolling 7-day trend line."""
    plt.figure(figsize=(16, 6))
    df_sorted = df.sort_values(datetime_col).copy()

    plt.plot(df_sorted[datetime_col], df_sorted[target_col], alpha=0.25, color="#7f8c8d", label="Hourly Count")
    rolling_7d = df_sorted[target_col].rolling(window=24 * 7, min_periods=24).mean()
    plt.plot(df_sorted[datetime_col], rolling_7d, color="#c0392b", linewidth=2.5, label="7-Day Moving Average")

    plt.title("Longitudinal Time-Series Demand Trend (2011 - 2012)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Rental Count", fontsize=11)
    plt.legend(frameon=True, loc="upper left")

    return save_current_figure("timeseries_trend.png")


def plot_monthly_demand(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Monthly rental demand comparison by year."""
    plt.figure(figsize=(11, 5))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if "yr" in df.columns:
        summary = df.groupby(["mnth", "yr"])[target_col].mean().reset_index()
        summary["Year"] = summary["yr"].map({0: "2011", 1: "2012"})
        sns.barplot(data=summary, x="mnth", y=target_col, hue="Year", palette=["#3498db", "#e74c3c"])
    else:
        summary = df.groupby("mnth")[target_col].mean().reset_index()
        sns.barplot(data=summary, x="mnth", y=target_col, color="#3498db")

    plt.xticks(ticks=range(12), labels=month_names)
    plt.title("Average Hourly Demand by Month (Seasonal Trajectory)", fontsize=13, fontweight="bold")
    plt.xlabel("Month", fontsize=11)
    plt.ylabel("Average Hourly Rentals", fontsize=11)
    plt.legend(title="Year", frameon=True)

    return save_current_figure("monthly_demand.png")


def plot_weekly_demand(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Weekly pattern across days of the week."""
    plt.figure(figsize=(10, 5))
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    summary = df.groupby("weekday")[target_col].agg(["mean", "median"]).reset_index()

    x = np.arange(len(day_names))
    width = 0.35

    plt.bar(x - width/2, summary["mean"], width, label="Mean Rentals", color="#2980b9", alpha=0.85)
    plt.bar(x + width/2, summary["median"], width, label="Median Rentals", color="#27ae60", alpha=0.85)

    plt.xticks(ticks=x, labels=day_names)
    plt.title("Average and Median Demand by Day of Week", fontsize=13, fontweight="bold")
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Hourly Rentals", fontsize=11)
    plt.legend(frameon=True)

    return save_current_figure("weekly_demand.png")


def plot_hourly_demand(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Hourly diurnal curves separated by working days vs non-working days."""
    plt.figure(figsize=(12, 6))

    if "workingday" in df.columns:
        summary = df.groupby(["hr", "workingday"])[target_col].mean().reset_index()
        summary["Day Type"] = summary["workingday"].map({1: "Working Day (Commute Pattern)", 0: "Weekend/Holiday (Leisure Pattern)"})
        palette = {"Working Day (Commute Pattern)": "#e74c3c", "Weekend/Holiday (Leisure Pattern)": "#2980b9"}
        sns.lineplot(data=summary, x="hr", y=target_col, hue="Day Type", palette=palette, marker="o", linewidth=2.5)
    else:
        summary = df.groupby("hr")[target_col].mean().reset_index()
        plt.plot(summary["hr"], summary[target_col], marker="o", color="#2980b9", linewidth=2.5)

    plt.title("Diurnal Demand Profile: Commuting Rush Hours vs Weekend Leisure", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (0 - 23)", fontsize=11)
    plt.ylabel("Average Rental Count", fontsize=11)
    plt.xticks(range(0, 24))
    plt.legend(frameon=True)

    return save_current_figure("hourly_demand.png")


def plot_outlier_analysis(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Boxplot outlier diagnostics across seasons and working days."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    season_map = {1: "Winter/Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    df_plot = df.copy()
    if "season" in df_plot.columns:
        df_plot["Season_Name"] = df_plot["season"].map(season_map)
        sns.boxplot(data=df_plot, x="Season_Name", y=target_col, ax=axes[0], palette="Set2")
        axes[0].set_title("Rental Distribution and Outliers across Seasons", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Season", fontsize=11)
        axes[0].set_ylabel("Hourly Count", fontsize=11)

    if "weathersit" in df_plot.columns:
        weather_map = {1: "Clear", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Ice"}
        df_plot["Weather_Name"] = df_plot["weathersit"].map(weather_map)
        sns.boxplot(data=df_plot, x="Weather_Name", y=target_col, ax=axes[1], palette="Blues_r")
        axes[1].set_title("Demand Outliers by Weather Severity", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Weather Condition", fontsize=11)
        axes[1].set_ylabel("Hourly Count", fontsize=11)

    return save_current_figure("outlier_analysis.png")


def plot_seasonal_patterns(df: pd.DataFrame, target_col: str = "cnt") -> Path:
    """Seasonal patterns showing temperature vs demand across seasons."""
    plt.figure(figsize=(11, 6))
    season_map = {1: "Winter/Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    df_plot = df.copy()
    df_plot["Season_Name"] = df_plot["season"].map(season_map)

    sns.scatterplot(
        data=df_plot,
        x="temp",
        y=target_col,
        hue="Season_Name",
        alpha=0.4,
        palette="tab10",
    )
    plt.title("Temperature vs Rental Demand Clustered by Season", fontsize=13, fontweight="bold")
    plt.xlabel("Normalized Temperature (temp)", fontsize=11)
    plt.ylabel("Hourly Rentals", fontsize=11)
    plt.legend(title="Season", frameon=True)

    return save_current_figure("seasonal_patterns.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, file_name: str) -> Path:
    """Plot residual diagnostics: residuals vs fitted and residual distribution."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.3, color="#2b5c8f", s=15)
    axes[0].axhline(0, color="#e74c3c", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"{model_name}: Residuals vs Predicted", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Values", fontsize=11)
    axes[0].set_ylabel("Residuals (Actual - Predicted)", fontsize=11)

    # Residual distribution histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color="#8e44ad", bins=50)
    axes[1].axvline(0, color="#e74c3c", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{model_name}: Residual Distribution (Zero-Centered Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual Error", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)

    return save_current_figure(file_name)


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, file_name: str, top_n: int = 15) -> Path:
    """Horizontal bar plot of top feature importances."""
    plt.figure(figsize=(10, 6))
    top = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.barh(top["feature"], top["importance"], color="#16a085", edgecolor="#2c3e50", alpha=0.85)
    plt.title(f"Top {top_n} Feature Importances: {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)

    return save_current_figure(file_name)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    timestamps: Optional[pd.Series] = None,
    file_name: str = "actual_vs_predicted.png",
    zoom_slice: int = 168,
) -> Path:
    """
    Plot actual versus predicted values for test set, including a zoomed 7-day window
    for crystal clear visual comparison of diurnal fit.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1.2]})

    # Top plot: Zoomed 1-week window (168 hours)
    n_sample = min(zoom_slice, len(y_true))
    idx = np.arange(n_sample)
    time_labels = timestamps.iloc[:n_sample] if timestamps is not None else idx

    axes[0].plot(time_labels, y_true[:n_sample], label="Actual Count", color="#2c3e50", linewidth=2.5, marker="o", markersize=3)
    colors = ["#e67e22", "#3498db", "#27ae60", "#9b59b6"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        axes[0].plot(time_labels, preds[:n_sample], label=f"{name}", color=color, linewidth=1.8, linestyle="--")

    axes[0].set_title(f"Holdout Test Set: 7-Day Zoomed Diurnal Fit ({n_sample} Hours)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Rental Count", fontsize=11)
    axes[0].legend(frameon=True, loc="upper right")

    # Bottom plot: Full test set scatter Actual vs Predicted for the best ensemble
    primary_name = list(predictions_dict.keys())[-1]
    primary_preds = predictions_dict[primary_name]

    axes[1].scatter(y_true, primary_preds, alpha=0.3, color="#2980b9", s=15, label=f"Predictions: {primary_name}")
    max_val = max(np.max(y_true), np.max(primary_preds))
    axes[1].plot([0, max_val], [0, max_val], color="#e74c3c", linestyle="--", linewidth=2.0, label="Ideal 1:1 Parity")
    axes[1].set_title(f"Parity Plot: Actual vs Predicted ({primary_name})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Actual Hourly Demand", fontsize=11)
    axes[1].set_ylabel("Predicted Hourly Demand", fontsize=11)
    axes[1].legend(frameon=True)

    return save_current_figure(file_name)


def plot_model_comparison_bar(comparison_df: pd.DataFrame, file_name: str = "models_comparison_bar.png") -> Path:
    """Bar chart comparing RMSE and MAE across all evaluated models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_df = comparison_df.sort_values("RMSE", ascending=True)

    # RMSE comparison
    sns.barplot(data=sorted_df, x="RMSE", y="Model", ax=axes[0], palette="viridis_r")
    axes[0].set_title("Model Comparison: RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Root Mean Squared Error (RMSE)", fontsize=11)

    # R2 comparison
    sns.barplot(data=sorted_df, x="R2", y="Model", ax=axes[1], palette="crest")
    axes[1].set_title("Model Comparison: R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Coefficient of Determination (R²)", fontsize=11)

    return save_current_figure(file_name)
