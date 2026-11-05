"""
Exploratory Data Analysis (EDA) Module.

Performs statistical profiling, time-series decomposition,
correlation analysis, and generates publication-grade visualizations.
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.config import PLOTS_DIR, REPORTS_DIR, TARGET_COLUMN
from src.utils import setup_logger, save_json
from src.visualization import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_feature_correlation_with_target,
    plot_timeseries_trend,
    plot_monthly_demand,
    plot_weekly_demand,
    plot_hourly_demand,
    plot_outlier_analysis,
    plot_seasonal_patterns,
)

logger = setup_logger("exploratory_analysis")


class ExploratoryAnalyzer:
    """Orchestrates comprehensive EDA and diagnostic report generation."""

    def __init__(self, plots_dir: Path = PLOTS_DIR, reports_dir: Path = REPORTS_DIR):
        self.plots_dir = Path(plots_dir)
        self.reports_dir = Path(reports_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_statistics(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
        """Compute complete descriptive statistics, quantiles, and skewness."""
        num_df = df.select_dtypes(include=[np.number])
        desc = num_df.describe().to_dict()

        target_stats = {}
        if target_col in df.columns:
            target_series = df[target_col]
            target_stats = {
                "count": int(len(target_series)),
                "mean": round(float(target_series.mean()), 2),
                "std": round(float(target_series.std()), 2),
                "min": float(target_series.min()),
                "q25": float(target_series.quantile(0.25)),
                "median": float(target_series.median()),
                "q75": float(target_series.quantile(0.75)),
                "max": float(target_series.max()),
                "skew": round(float(target_series.skew()), 3),
                "kurtosis": round(float(target_series.kurtosis()), 3),
                "iqr": float(target_series.quantile(0.75) - target_series.quantile(0.25)),
            }

        return {
            "dimensions": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "target_statistics": target_stats,
            "descriptive_statistics": desc,
        }

    def generate_all_eda_plots(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> List[str]:
        """Generate and save all 9 required EDA visualizations."""
        logger.info("Generating EDA visualization suite...")
        saved_plots = []

        # 1. Target distribution
        p1 = plot_target_distribution(df, target_col=target_col)
        saved_plots.append(str(p1.name))

        # 2. Correlation heatmap
        p2 = plot_correlation_heatmap(df)
        saved_plots.append(str(p2.name))

        # 3. Feature correlation with target
        p3 = plot_feature_correlation_with_target(df, target_col=target_col)
        saved_plots.append(str(p3.name))

        # 4. Longitudinal time-series trend
        if "datetime" in df.columns:
            p4 = plot_timeseries_trend(df, datetime_col="datetime", target_col=target_col)
            saved_plots.append(str(p4.name))

        # 5. Monthly demand patterns
        p5 = plot_monthly_demand(df, target_col=target_col)
        saved_plots.append(str(p5.name))

        # 6. Weekly demand patterns
        p6 = plot_weekly_demand(df, target_col=target_col)
        saved_plots.append(str(p6.name))

        # 7. Hourly diurnal demand profiles
        p7 = plot_hourly_demand(df, target_col=target_col)
        saved_plots.append(str(p7.name))

        # 8. Outlier diagnostics
        p8 = plot_outlier_analysis(df, target_col=target_col)
        saved_plots.append(str(p8.name))

        # 9. Seasonal meteorological patterns
        p9 = plot_seasonal_patterns(df, target_col=target_col)
        saved_plots.append(str(p9.name))

        logger.info("Generated %d EDA plots saved to %s", len(saved_plots), self.plots_dir)
        return saved_plots

    def run_full_eda(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
        """Execute full statistical analysis and plot generation pipeline."""
        stats = self.analyze_statistics(df, target_col=target_col)
        plots = self.generate_all_eda_plots(df, target_col=target_col)

        summary_report = {
            "dataset_dimensions": stats["dimensions"],
            "target_statistics": stats["target_statistics"],
            "generated_plots": plots,
        }

        json_out = self.reports_dir / "eda_summary.json"
        save_json(summary_report, json_out)

        # Generate markdown report
        md_out = self.reports_dir / "eda_summary.md"
        with open(md_out, "w", encoding="utf-8") as f:
            f.write("# Exploratory Data Analysis Summary Report\n\n")
            f.write(f"- **Total Observations**: {stats['dimensions']['rows']:,}\n")
            f.write(f"- **Total Features**: {stats['dimensions']['columns']}\n")
            f.write(f"- **Target Variable**: `{target_col}`\n\n")

            f.write("## Target Variable Distribution Summary\n\n")
            tstats = stats["target_statistics"]
            f.write(f"- **Mean**: {tstats.get('mean')} rentals/hour\n")
            f.write(f"- **Median**: {tstats.get('median')} rentals/hour\n")
            f.write(f"- **Std Dev**: {tstats.get('std')}\n")
            f.write(f"- **Range**: [{tstats.get('min')}, {tstats.get('max')}]\n")
            f.write(f"- **IQR**: {tstats.get('iqr')}\n")
            f.write(f"- **Skewness**: {tstats.get('skew')} (Right-skewed count distribution)\n\n")

            f.write("## Generated Visualizations\n\n")
            for plot in plots:
                f.write(f"- `outputs/plots/{plot}`\n")

        logger.info("EDA completed successfully. Reports saved to %s and %s", json_out, md_out)
        return summary_report

